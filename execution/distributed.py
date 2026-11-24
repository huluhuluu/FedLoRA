"""Point-to-point rank services share the same task pool as local workers."""

import queue
import threading
import traceback
import torch.distributed as dist
from execution import rank_device
from execution.pool import JobPool
from execution.local import LocalExecutor
from execution.worker import Worker
from execution.transport import send_message, send_rank_message, receive_rank_message
from utils.data import load_data


class DistributedExecutor(JobPool):
    def __init__(self, config, bundle=None, initialize=True):
        self.local, self.threads, self.closed = None, [], False
        self.is_leader = False
        self.config = config
        try:
            dist.init_process_group("gloo")
            self.rank, self.size = dist.get_rank(), dist.get_world_size()
            self.is_leader = self.rank == 0
            device = rank_device(config)
            bundle = bundle or load_data(config)
            if self.is_leader:
                local = {
                    **config,
                    "execution": {**config["execution"], "devices": [device]},
                    "evaluation": {**config["evaluation"], "mode": "off"},
                }
                # A child owns rank 0's GPU, leaving the coordinator free to dispatch.
                self.local = LocalExecutor(local, bundle, initialize)
                self.initial_state = self.local.initial_state
                self.results = queue.Queue()
                devices = [device]
                for rank in range(1, self.size):
                    status, value = receive_rank_message(rank)
                    if status != "ready":
                        raise RuntimeError(value)
                    devices.append(value)
                self.initialize(config, bundle, devices)
                for rank in range(self.size):
                    thread = threading.Thread(
                        target=self._receive_rank, args=(rank,), daemon=True
                    )
                    thread.start()
                    self.threads.append(thread)
            else:
                try:
                    self.worker = Worker(config, self.rank, device, bundle, initialize)
                    send_rank_message(("ready", device), 0)
                except Exception:
                    send_rank_message(("error", traceback.format_exc()), 0)
                    raise
        except BaseException as exc:
            try:
                self.close(cancel=True)
            except Exception as cleanup:
                exc.add_note(f"Executor cleanup failed: {cleanup}")
            raise

    def _receive_rank(self, rank):
        try:
            while True:
                value = (
                    self.local.receive(None)
                    if rank == 0
                    else (rank, *receive_rank_message(rank))
                )
                self.results.put(value)
                if value[1] == "closed":
                    return
        except Exception:
            self.results.put((rank, "transport_error", traceback.format_exc()))

    def send(self, worker, job_id, kind, payload):
        if worker == 0:
            self.local.send(0, job_id, kind, payload)
        else:
            send_rank_message((job_id, kind, payload), worker)

    def receive(self, timeout):
        try:
            value = self.results.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("Distributed worker response timed out") from None
        if value[1] == "transport_error":
            raise RuntimeError(value[2])
        return value

    def serve(self):
        while True:
            job_id, kind, payload = receive_rank_message(0)
            if kind == "close":
                send_rank_message(("closed", None), 0)
                return
            try:
                response = ("ok", self.worker.execute(kind, payload))
            except Exception:
                response = ("error", traceback.format_exc())
            send_rank_message((job_id, response), 0)

    def close(self, cancel=False):
        if self.closed:
            return
        self.closed = True
        try:
            if self.is_leader and not cancel:
                for rank in range(1, self.size):
                    send_rank_message((0, "close", None), rank)
                for _ in range(1, self.size):
                    rank, status, _ = self.results.get(timeout=30)
                    if status != "closed":
                        raise RuntimeError(
                            f"Unexpected shutdown response from rank {rank}"
                        )
        finally:
            if self.local is not None:
                if self.threads:
                    send_message(self.local.outbox, (0, "closed", None))
                    self.threads[0].join(timeout=2)
                self.local.close(cancel=cancel)
            if dist.is_initialized():
                dist.destroy_process_group()
            for thread in self.threads:
                thread.join(timeout=1)
