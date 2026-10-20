"""
run_experiments.py - launch all training runs (reward setting x seed) for the ablation.

Each run is a separate `python ppo.py ...` process with its own run name, so runs never
overwrite each other:  runs/<setting>_s<seed>/

    python run_experiments.py --list                         # show the reward settings
    python run_experiments.py --settings baseline,multi --seeds 0 --total-steps 2000000   # pilot
    python run_experiments.py --settings all --seeds 0,1,2 --parallel 3 --device cuda
    python run_experiments.py --settings all --seeds 0,1,2 --dry-run   # only print the commands

Re-running the same command is safe:
    - finished runs (final.pt exists) are skipped
    - interrupted runs (latest.pt exists) are resumed from their last checkpoint
    - anything else starts fresh

Edit SETTINGS below to change the weights (w1, w2, w3, w4). The names are used in the graphs.
"""

import argparse
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# name -> (w1 line clears, w2 new holes, w3 height/bumpiness, w4 survival)
# baseline + one term at a time + all four + a stronger-shaping variant.
# These values are starting points, not tuned values.
SETTINGS = {
    "baseline": (1.0, 0.0, 0.0, 0.0),  # R = r1 (single objective)
    "holes": (1.0, 0.5, 0.0, 0.0),  # r1 + r2
    "shape": (1.0, 0.0, 0.2, 0.0),  # r1 + r3
    "survival": (1.0, 0.0, 0.0, 0.1),  # r1 + r4
    "multi": (1.0, 0.5, 0.2, 0.1),  # all four terms
    "multi_strong": (1.0, 1.0, 0.5, 0.2),  # all four, heavier penalties
}


def run_status(out_dir: str, name: str) -> str:
    ckpt = os.path.join(out_dir, name, "checkpoints")
    if os.path.exists(os.path.join(ckpt, "final.pt")):
        return "done"
    if os.path.exists(os.path.join(ckpt, "latest.pt")):
        return "resume"
    return "new"


def build_cmd(args, name, weights, seed, resume_path=None):
    cmd = [
        sys.executable, os.path.join(HERE, "ppo.py"),
        "--run-name", name,
        "--out-dir", args.out_dir,
        "--weights", ",".join(f"{w:g}" for w in weights),
        "--seed", str(seed),
        "--total-steps", str(args.total_steps),
    ]
    if args.device:
        cmd += ["--device", args.device]
    if args.max_pieces:
        cmd += ["--max-pieces", str(args.max_pieces)]
    if resume_path:
        cmd += ["--resume", resume_path]
    if args.extra:
        cmd += shlex.split(args.extra)
    return cmd


def last_line(path: str) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 600))
            lines = f.read().decode(errors="ignore").strip().splitlines()
        return lines[-1] if lines else ""
    except OSError:
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show the reward settings and exit")
    ap.add_argument("--settings", default="baseline,multi", help="comma list of names, or 'all'")
    ap.add_argument("--seeds", default="0", help="comma list, e.g. 0,1,2")
    ap.add_argument("--total-steps", type=int, default=2_000_000, help="placements per run")
    ap.add_argument("--parallel", type=int, default=1, help="how many runs at the same time (about one per CPU core)")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--max-pieces", type=int, default=None, help="episode cap passed to ppo.py")
    ap.add_argument("--extra", default="", help='extra ppo.py flags, e.g. --extra "--gamma 0.9"')
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.list:
        print(f"{'name':14s} w1    w2    w3    w4")
        for k, w in SETTINGS.items():
            print(f"{k:14s} " + "  ".join(f"{x:<4g}" for x in w))
        return

    names = list(SETTINGS) if args.settings == "all" else [s.strip() for s in args.settings.split(",")]
    for n in names:
        if n not in SETTINGS:
            sys.exit(f"unknown setting '{n}'. Use --list to see the options.")
    seeds = [int(s) for s in args.seeds.split(",")]

    jobs = []
    for setting in names:
        for seed in seeds:
            name = f"{setting}_s{seed}"
            st = run_status(args.out_dir, name)
            if st == "done":
                print(f"skip   {name} (already finished)")
                continue
            resume = os.path.join(args.out_dir, name, "checkpoints", "latest.pt") if st == "resume" else None
            jobs.append((name, build_cmd(args, name, SETTINGS[setting], seed, resume), st))

    print(f"\n{len(jobs)} run(s) to do, {args.parallel} at a time, {args.total_steps:,} placements each\n")
    if args.dry_run:
        for name, cmd, st in jobs:
            print(f"[{st}] {' '.join(cmd)}")
        return

    child_env = dict(os.environ)
    if args.parallel > 1:  # avoid every process grabbing every CPU core
        child_env["OMP_NUM_THREADS"] = "1"
        child_env["MKL_NUM_THREADS"] = "1"

    queue, running, failed, finished = list(jobs), {}, [], []
    last_report = 0.0
    try:
        while queue or running:
            while queue and len(running) < args.parallel:
                name, cmd, st = queue.pop(0)
                os.makedirs(os.path.join(args.out_dir, name), exist_ok=True)
                log_path = os.path.join(args.out_dir, name, "train.log")
                log = open(log_path, "a")
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=child_env)
                running[name] = (proc, log, log_path, time.time())
                print(f"start  {name} ({'resuming' if st == 'resume' else 'new'})  log: {log_path}", flush=True)

            for name in list(running):
                proc, log, log_path, t0 = running[name]
                if proc.poll() is not None:
                    log.close()
                    del running[name]
                    mins = (time.time() - t0) / 60
                    if proc.returncode == 0:
                        finished.append(name)
                        print(f"done   {name} ({mins:.1f} min)", flush=True)
                    else:
                        failed.append(name)
                        print(f"FAILED {name} (exit {proc.returncode}), see {log_path}", flush=True)

            if running and time.time() - last_report > 120:
                last_report = time.time()
                print("--- progress ---")
                for name, (_, _, log_path, _) in running.items():
                    print(f"{name}: {last_line(log_path)[:140]}")
            time.sleep(3)
    except KeyboardInterrupt:
        print("\nstopping runs (re-run the same command later to resume them)")
        for proc, log, _, _ in running.values():
            proc.terminate()
        sys.exit(1)

    print(f"\nfinished: {len(finished)}, failed: {len(failed)}")
    if failed:
        print("failed runs:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
