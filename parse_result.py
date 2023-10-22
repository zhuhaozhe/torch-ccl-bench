import re

def filter_out_warmup(lines):
    for i in range(len(lines)):
        if "finish warm up" in lines[i]:
            return lines[i:]

def separate_ranks(lines):
    res = {}

    def valid_line(line):
        assert line.startswith('[')
        match = re.match("\[[0-9]{1,2}\]",line)
        assert match
        return match.group()

    def handle_exception_line(line):
        # [0] -----------------------------[1] Cur time: 1696835602.0989683, Time usage: 5.424879789352417, at stage: got sparse emb result in distribute_forward, at iter: 1012
        buf = []
        match = re.search("\[[0-9]{1,2}\]",line[3:])
        if match and match.span()[0] > 6:
            buf.append(line[:match.span()[0] + 3])
            buf.append(line[match.span()[0] + 3: ])
        return buf

    def create_or_append(line):
        rank_id = valid_line(line)
        if rank_id in res:
            res[rank_id].append(line)
        else:
            res[rank_id] = [line]
    for line in lines:
        buf = handle_exception_line(line)
        if len(buf) > 0:
            for _line in buf:
                create_or_append(_line)
        else:
            create_or_append(line)            
    return res


def get_time_usage(lines):
    results = {
        'model-level': {'spend': 0, 'repeat': 0},
        'torch-ccl-level': {'spend': 0, 'repeat': 0},
        'oneccl-verbose-level': {'spend': 0, 'repeat': 0},
    }
    time_stack = []
    def get_time_stamp(line):
        match = re.search("Cur time: [0-9]*\.[0-9]*,",line)
        assert match
        return float(match.group()[10:-1]) * 1e6

    for i in range(len(lines)):
        line = lines[i]
        if "start all2all in model" in line:
            assert len(time_stack) == 0
            time_stack.append(get_time_stamp(line))
            continue
        if "start to wait sparse" in line:
            assert len(time_stack) == 1
            time_stack.append(get_time_stamp(line))
            continue
        if "got sparse emb result" in line:
            assert len(time_stack) == 2
            cur_time = get_time_stamp(line)
            last_time = time_stack.pop()
            results['torch-ccl-level']['spend'] += (cur_time - last_time)
            results['torch-ccl-level']['repeat'] += 1
            continue
        if "finish all2all in model" in line:
            assert len(time_stack) == 1
            cur_time = get_time_stamp(line)
            last_time = time_stack.pop()
            results['model-level']['spend'] += (cur_time - last_time)
            results['model-level']['repeat'] += 1
            continue
        if "sched total" in line:
            assert len(time_stack) == 1
            if "A2AV_RECV" in lines[i + 4]:
                time_usage = float(re.search(" [0-9]*\.[0-9]*", lines[i + 4]).group())
                results['oneccl-verbose-level']['spend'] += time_usage
                results['oneccl-verbose-level']['repeat'] += 1
    return results


def display_time_usage(t, r):
    total_per_iter = t['model-level']['spend'] / t['model-level']['repeat']
    ccl_per_iter = t['oneccl-verbose-level']['spend'] / t['oneccl-verbose-level']['repeat']
    overhead = total_per_iter - ccl_per_iter
    print(f"for rank {r}, overhead take {total_per_iter - ccl_per_iter} us/iter, ratio = {overhead/total_per_iter}")


fp = open("./verbose.log")
lines = fp.readlines()
fp.close()
lines_per_rank = separate_ranks(lines)
for k, v in lines_per_rank.items():
    lines_per_rank[k] = filter_out_warmup(v)
    time_usage = get_time_usage(lines_per_rank[k])
    display_time_usage(time_usage, k)
