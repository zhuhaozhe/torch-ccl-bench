import torch
import extend_distributed as ext_dist
import torch.distributed as dist
import time_stamp

ITER=2000
N_TENSORS=1
NUMEL=(16384, 1024)

a_ = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float)
b_ = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float)


def cache_flush():
    # return
    # We assume the cache size is <= 512MB here.
    # a = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float)
    # b = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float)
    # a, b are initialized out of this function to avoid allocate memory every time
    global a_, b_
    a_ += b_



ext_dist.init_distributed()
ly_sparse = [torch.randn(NUMEL, dtype=torch.bfloat16) for _ in range(N_TENSORS)]
a = [torch.randn(NUMEL, dtype=torch.bfloat16) for _ in range(N_TENSORS)]
b = [torch.randn(NUMEL, dtype=torch.bfloat16) for _ in range(N_TENSORS)]

def send_and_wait():
    global ly_sparse
    a2a_req = ext_dist.alltoall(ly_sparse, per_rank_split_lengths=None, a2a_as_tensor_list=True)
    ly_sparse = a2a_req.wait()

def send_and_wait_no_wrapper():
    for i in range(N_TENSORS):
        dist.all_to_all_single(a[i], b[i], None, None, async_op=False)

def run_bench():
    for _ in range(ITER):
        time_stamp.cur_iters += 1
        if time_stamp.cur_iters == 1000:
            print("=======================finish warm up", flush=True)
        cache_flush()
        time_stamp.print_time(f"start all2all in model", time_stamp.cur_iters)
        send_and_wait_no_wrapper()
        time_stamp.print_time(f"finish all2all in model", time_stamp.cur_iters)

    # total_t = 0
    # for _ in range(ITER):
    #     cache_flush()
    #     start = time.time()
    #     send_and_wait_no_wrapper()
    #     total_t += time.time() - start
    # print("======================================", total_t, flush=True)

if __name__ == "__main__":
    run_bench()
