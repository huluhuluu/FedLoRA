"""Persistent local workers with one in-flight compute job per device."""

import multiprocessing as mp
import time
from queue import Empty
from execution import devices_for
from execution.pool import JobPool
from execution.worker import Worker, process_main
from execution.transport import send_message, receive_message
from utils.data import load_data


class SerialExecutor(JobPool):
    def __init__(self, config, bundle=None, initialize=True):
        device = devices_for(config)[0]
        self.bundle = bundle or load_data(config)
        self.initialize(config, self.bundle, [device])
        self.worker = Worker(config, 0, device, self.bundle, initialize)
        self.initial_state = self.worker.net.export() if initialize else None

    def send(self, worker, job_id, kind, payload):
        try:
            self.response = (worker, job_id, ("ok", self.worker.execute(kind, payload)))
        except Exception:
            import traceback

            self.response = (worker, job_id, ("error", traceback.format_exc()))

    def receive(self, timeout):
        return self.response

    def close(self, cancel=False):
        pass


class LocalExecutor(JobPool):
    def __init__(self, config, bundle=None, initialize=True):
        self.processes, self.inboxes, self.closed = [], [], False
        devices = devices_for(config)
        if (
            not devices
            or len(set(devices)) != len(devices)
            and not all(value == "cpu" for value in devices)
        ):
            raise ValueError("Local devices must be distinct; CPU workers may repeat")
        self.bundle = bundle or load_data(config)
        self.initialize(config, self.bundle, devices)
        context = mp.get_context("spawn")
        self.outbox = context.Queue()
        try:
            for worker, device in enumerate(devices):
                inbox = context.Queue()
                process = context.Process(
                    target=process_main,
                    args=(
                        config,
                        worker,
                        device,
                        inbox,
                        self.outbox,
                        self.bundle,
                        initialize,
                    ),
                )
                self.inboxes.append(inbox)
                process.start()
                self.processes.append(process)
            for _ in devices:
                worker, status, value = self.receive(None)
                if status != "ready":
                    raise RuntimeError(value)
                if worker == 0:
                    self.initial_state = value
        except BaseException as exc:
            try:
                self.close(cancel=True)
            except Exception as cleanup:
                exc.add_note(f"Executor cleanup failed: {cleanup}")
            raise

    def send(self, worker, job_id, kind, payload):
        send_message(self.inboxes[worker], (job_id, kind, payload))

    def receive(self, timeout):
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            interval = (
                min(1, max(0, deadline - time.monotonic()))
                if deadline is not None
                else 1
            )
            try:
                return receive_message(self.outbox, timeout=interval)
            except Empty:
                dead = [
                    (i, p.exitcode)
                    for i, p in enumerate(self.processes)
                    if p.exitcode is not None
                ]
                if dead:
                    raise RuntimeError(f"Worker process exited: {dead}") from None
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Worker response timed out") from None

    def close(self, cancel=False):
        if self.closed:
            return
        self.closed = True
        if cancel:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
        else:
            for inbox in self.inboxes:
                send_message(inbox, (0, "close", None))
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        for channel in [*self.inboxes, self.outbox]:
            channel.cancel_join_thread()
            channel.close()
