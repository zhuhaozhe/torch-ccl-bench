import time
start_time = time.time()
cur_iters = -1

def print_time(stage: str, iter: int = -1):
    global cur_iters
    should_print = cur_iters == -1 or (cur_iters >= 1000)
    if not should_print:
        return
    global start_time
    cur = time.time()
    print(f"Cur time: {cur}, Time usage: {cur - start_time}, at stage: {stage}, at iter: {cur_iters}", flush=True)

