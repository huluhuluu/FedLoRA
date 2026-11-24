"""Transport remains usable for many named adapter tensors under a small FD limit."""

import subprocess
import sys


def test_tensor_transport_has_no_per_tensor_file_descriptors():
    code = """
import resource
import multiprocessing as mp
import torch
from execution.transport import send_message,receive_message
_,hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE,(min(128,hard),hard))
channel = mp.get_context("spawn").Queue()
values = {str(i):torch.arange(4,dtype=torch.float32) for i in range(1200)}
for _ in range(3):
    send_message(channel,values)
    result = receive_message(channel,timeout=20)
    assert len(result)==1200 and torch.equal(result["1199"],values["1199"])
channel.close()
channel.join_thread()
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)


def test_idle_rank_can_wait_beyond_the_group_task_timeout():
    code = r"""
import multiprocessing as mp,queue,socket,time
from datetime import timedelta
import torch.distributed as dist
from execution.transport import send_rank_message,receive_rank_message
def participant(rank,port,out):
    try:
        dist.init_process_group("gloo",init_method=f"tcp://127.0.0.1:{port}",rank=rank,world_size=2,timeout=timedelta(seconds=1))
        if rank==0:
            assert receive_rank_message(1)=={"value":42}
            send_rank_message("closed",1)
        else:
            time.sleep(1.3)
            send_rank_message({"value":42},0)
            assert receive_rank_message(0)=="closed"
        out.put("ok")
    finally:
        if dist.is_initialized():dist.destroy_process_group()
with socket.socket() as listener:
    listener.bind(("127.0.0.1",0));port=listener.getsockname()[1]
ctx=mp.get_context("fork")
out=ctx.Queue()
children=[ctx.Process(target=participant,args=(rank,port,out)) for rank in range(2)]
try:
    for child in children:child.start()
    for child in children:child.join(timeout=12)
    assert all(child.exitcode==0 for child in children)
    assert [out.get(timeout=1) for _ in children]==["ok","ok"]
finally:
    for child in children:
        if child.is_alive():child.terminate();child.join(timeout=3)
    out.close();out.join_thread()
"""
    import os

    subprocess.run(
        [sys.executable, "-B", "-c", code],
        check=True,
        timeout=35,
        env={**os.environ, "GLOO_SOCKET_IFNAME": "lo"},
    )
