"""LCB sandbox worker: runs ONE generated program against its tests with LiveCodeBench's official
lcb_runner/evaluation/testing_util.run_test (which installs reliability_guard + per-test SIGALRM timeouts).

Invoked by short_score.py as:  unshare -rn <venv python> -I lcb_worker.py RESULT_PATH  < job.json
(job = {"sample": {"input_output": str}, "code": str, "timeout": int, "mem_bytes": int, "cpu_secs": int,
        "testing_util": path of lcb_runner/evaluation/testing_util.py in the LiveCodeBench checkout})
Runs in a throwaway temp cwd, no network namespace, with RLIMIT_AS / RLIMIT_CPU / RLIMIT_FSIZE set here.
"""
import importlib.util, json, resource, sys


def main():
    res_path = sys.argv[1]
    job = json.loads(sys.stdin.buffer.read())
    sys.set_int_max_str_digits(50000)  # as lcb_runner/evaluation/compute_code_generation_metrics.py
    spec = importlib.util.spec_from_file_location("lcb_testing_util", job["testing_util"])
    tu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tu)
    out = open(res_path, "w")
    mem = int(job["mem_bytes"])
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    resource.setrlimit(resource.RLIMIT_CPU, (int(job["cpu_secs"]), int(job["cpu_secs"]) + 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    res, meta = tu.run_test(job["sample"], test=job["code"], debug=False, timeout=int(job["timeout"]))
    fixed = []
    for e in res:  # same normalisation as evaluate_generations_by_problem
        if hasattr(e, "item"):
            e = e.item()
        fixed.append(bool(e) if isinstance(e, bool) else e)
    try:
        meta = json.loads(json.dumps(meta, default=str))
    except Exception:
        meta = {"unserializable_metadata": True}
    out.write(json.dumps({"result": fixed, "metadata": meta}))
    out.flush()
    out.close()


if __name__ == "__main__":
    main()
