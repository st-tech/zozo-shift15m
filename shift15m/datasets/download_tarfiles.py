import os

try:
    import shift15m.constants as C
except ModuleNotFoundError as e:
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
    import shift15m.constants as C

import requests

FLIST_URL = (
    "https://research.zozo.com/data_release/shift15m/vgg16-features/filelist.txt"
)
jobs = []
children = {}


def spawn(cmd, *args):
    argv = [cmd] + list(args)
    pid = None
    args_str = " ".join(argv)
    try:
        pid = os.spawnlp(os.P_NOWAIT, cmd, *argv)
        children[pid] = {"pid": pid, "cmd": argv}
    except Exception as inst:
        print(f"'{args_str}': {str(inst)}")

    print(
        f"spawned pid {pid} of nproc={len(children)} njobs={len(jobs)} for '{args_str}'"
    )
    return pid


def main(root, processes):
    files = requests.get(FLIST_URL, stream=True)
    for line in files.iter_lines():
        url = line.decode()
        fname = os.path.basename(url)
        path = os.path.join(root, fname)

        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue

        cmd = f"wget -q -t 3 -nc {url} -O {path}"
        jobs.append((cmd.split(" "), fname))
    print(f"{len(jobs)} wget jobs queued")

    while len(jobs) > 0 and len(children) < processes:
        cmd, _ = jobs[0]
        if spawn(*cmd):
            del jobs[0]

    while len(jobs) > 0 or len(children):
        pid, status = os.wait()
        cmd = " ".join(children[pid]["cmd"])
        msg = f"pid {pid} exited. status={status}, nproc={len(children) - 1}, njobs={len(jobs)}, cmd={cmd}"
        print(msg)

        del children[pid]
        if len(children) < processes and len(jobs):
            cmd, _ = jobs[0]
            if spawn(*cmd):
                del jobs[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=C.ROOT)
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    args = parser.parse_args()
    main(args.root, args.processes)
