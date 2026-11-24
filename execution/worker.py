"""One resident backbone and one cached CPU reference per worker."""

from utils.helper import configure_runtime


class Worker:
    def __init__(self, config, worker_id, device, bundle=None, initialize=True):
        configure_runtime(config["runtime"]["cpu_threads"])
        import torch

        torch.set_num_threads(config["runtime"]["cpu_threads"])
        if str(device).startswith("cuda"):
            torch.cuda.set_device(device)
        from algorithm.registry import load_plugins, build
        from utils.data import load_data
        from models.peft_adapter import PeftAdapter

        load_plugins(config["plugins"])
        self.bundle = bundle or load_data(config)
        initializer = (
            build("initialization", config["initialization"]) if initialize else None
        )
        self.net = PeftAdapter(
            config, self.bundle.info, device, initializer, initialize
        )
        self.reference = None
        if (
            worker_id == 0
            and initializer is not None
            and hasattr(initializer, "initialize")
        ):
            from algorithm.interfaces import context

            self.net.prepare(
                initializer.initialize(
                    self.net.export(),
                    self.net,
                    self.bundle,
                    context(config, "initialization"),
                )
            )
        self.worker_id = worker_id

    def execute(self, kind, payload):
        state, *arguments = payload
        if state is not None:
            self.reference = state
        if kind == "train":
            from models.Update import train_client

            return train_client(
                self.net, self.bundle, self.reference, *arguments, self.worker_id
            )
        if kind == "gradient":
            from training.loop import gradient_batch

            return gradient_batch(
                self.net, self.bundle, self.reference, *arguments, self.worker_id
            )
        if kind == "evaluate":
            from models.test import evaluate_statistics

            return evaluate_statistics(
                self.net, self.bundle, self.reference, *arguments
            )
        raise ValueError(f"Unknown worker job: {kind}")


def process_main(
    config, worker_id, device, inbox, outbox, bundle=None, initialize=True
):
    import traceback
    from execution.transport import send_message, receive_message

    try:
        worker = Worker(config, worker_id, device, bundle, initialize)
        state = worker.net.export() if worker_id == 0 and initialize else None
        send_message(outbox, (worker_id, "ready", state))
        while True:
            job_id, kind, payload = receive_message(inbox)
            if kind == "close":
                break
            try:
                response = ("ok", worker.execute(kind, payload))
            except Exception:
                response = ("error", traceback.format_exc())
            send_message(outbox, (worker_id, job_id, response))
    except Exception:
        send_message(outbox, (worker_id, "error", traceback.format_exc()))
