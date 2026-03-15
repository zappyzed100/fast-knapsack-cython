import argparse
import datetime
import os
import subprocess
import sys
import time


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TIMEOUTS = [1, 10, 30, 60, 100]
DEFAULT_REPEATS = 20


def build_solver_jobs(no_full_output=True, stochastic_repeats=DEFAULT_REPEATS):
    """stochastic_repeats: Cython/Numbaの反復回数。MiniZinc等の決定論的ソルバーは常に1回。"""
    common_args = ["--no-full-output"] if no_full_output else []
    return [
        {
            "name": "cython",
            "script": os.path.join(PROJECT_ROOT, "scripts", "solve_with_cython.py"),
            "args": common_args,
            "repeats": stochastic_repeats,
        },
        {
            "name": "numba",
            "script": os.path.join(PROJECT_ROOT, "scripts", "solve_with_numba.py"),
            "args": common_args,
            "repeats": stochastic_repeats,
        },
        {
            # CBC / CP-SAT / Gecode は決定論的 → 1回のみ
            "name": "minizinc",
            "script": os.path.join(
                PROJECT_ROOT, "scripts", "solve_with_minizinc_solvers.py"
            ),
            "args": common_args,
            "repeats": 1,
        },
    ]


def run_command(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, check=False)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run timeout experiments for all solvers. "
            "Defaults: timeouts=1,10,30,60,100 and repeats=20."
        )
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Number of repeats for each timeout (default: {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--full-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass through full solver output (default: false)",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when any solver command fails",
    )
    parser.add_argument(
        "--solvers",
        type=str,
        default=None,
        help="Comma-separated list of solvers to run (e.g. cython,numba). Default: all",
    )
    parser.add_argument(
        "--timeouts",
        type=str,
        default=None,
        help="Comma-separated list of timeouts to run (e.g. 60,100). Default: all",
    )
    args = parser.parse_args()

    jobs = build_solver_jobs(
        no_full_output=not args.full_output,
        stochastic_repeats=args.repeats,
    )
    if args.solvers:
        allowed = {s.strip() for s in args.solvers.split(",")}
        jobs = [j for j in jobs if j["name"] in allowed]

    active_timeouts = TIMEOUTS
    if args.timeouts:
        active_timeouts = [int(t.strip()) for t in args.timeouts.split(",")]

    total = sum(len(active_timeouts) * job["repeats"] for job in jobs)

    print("=== Timeout experiment runner ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Timeouts: {active_timeouts}")
    for job in jobs:
        print(f"  {job['name']}: {job['repeats']} repeat(s) per timeout")
    print(f"Total runs: {total}")
    print("Results are appended under results/runs/*.txt by each solver script.")

    start_all = time.perf_counter()
    run_index = 0
    failures = []

    for timeout_sec in active_timeouts:
        for job in jobs:
            for repeat in range(1, job["repeats"] + 1):
                run_index += 1
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[{now_str}] ({run_index}/{total}) "
                    f"solver={job['name']} timeout={timeout_sec}s repeat={repeat}"
                )

                cmd = [
                    sys.executable,
                    job["script"],
                    "--timeout",
                    str(timeout_sec),
                    *job["args"],
                ]

                result = run_command(cmd, cwd=PROJECT_ROOT)
                if result.returncode != 0:
                    failures.append(
                        {
                            "solver": job["name"],
                            "timeout": timeout_sec,
                            "repeat": repeat,
                            "returncode": result.returncode,
                        }
                    )
                    print(
                        "  -> FAILED "
                        f"(solver={job['name']}, timeout={timeout_sec}, repeat={repeat}, "
                        f"code={result.returncode})"
                    )
                    if args.stop_on_error:
                        elapsed_all = time.perf_counter() - start_all
                        print(
                            f"Stopped by --stop-on-error. Elapsed: {elapsed_all:.1f}s"
                        )
                        sys.exit(result.returncode)
                    else:
                        continue

    elapsed_all = time.perf_counter() - start_all
    print("\n=== Experiment completed ===")
    print(f"Elapsed total: {elapsed_all:.1f} seconds")
    print(f"Failures: {len(failures)}")
    if failures:
        for f in failures:
            print(
                f"- solver={f['solver']} timeout={f['timeout']} "
                f"repeat={f['repeat']} code={f['returncode']}"
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
