#!/usr/bin/env python3
"""runner.py GPU PORT  -- one queue per GPU.

Jobs are JSON files in $BENCH/queue/gpu<N>/ processed in name order. Each job:
  {"cfg": "dense|sparse|tier8|tier4", "env": {...server env overrides...},
   "reqs": "reqs/<suite>.jsonl", "shard": 0|1|null, "out": "out/<cfg>/<suite>.g<N>.jsonl",
   "conc": 1, "limit": {"<group key>": n} (optional), "ids": [...] (optional)}
The server is (re)started only when (cfg, env) changes. A finished job moves to queue/gpu<N>/done/.
queue/gpu<N>/STOP (or queue/STOP) makes the runner stop its server and exit after the current job.
Server processes are stopped by their process group (explicit PID), never by name.
"""
import json, os, signal, subprocess, sys, time, glob, asyncio, re
import urllib.request

BENCH = os.environ.get("BENCH", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BENCH, "src"))
import client  # noqa: E402

GPU, PORT = int(sys.argv[1]), int(sys.argv[2])
QD = os.path.join(BENCH, "queue", f"gpu{GPU}")
os.makedirs(os.path.join(QD, "done"), exist_ok=True)
LOGD = os.path.join(BENCH, "logs", f"gpu{GPU}")
os.makedirs(LOGD, exist_ok=True)
PROGRESS = os.path.join(BENCH, "PROGRESS.md")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}Z gpu{GPU}] {msg}"
    print(line, flush=True)


def progress(msg):
    with open(PROGRESS, "a") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M', time.gmtime())}Z gpu{GPU}: {msg}\n")


class Server:
    def __init__(self):
        self.proc = None
        self.key = None
        self.logpath = None

    def health(self):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def start(self, cfg, env):
        self.stop()
        e = dict(os.environ)
        e.update({k: str(v) for k, v in (env or {}).items()})
        e.update(HQ_GPU=str(GPU), PORT=str(PORT))
        ts = time.strftime("%m%d-%H%M%S")
        self.logpath = os.path.join(LOGD, f"server.{cfg}.{ts}.log")
        lf = open(self.logpath, "w")
        lf.write(f"# cfg={cfg} env={json.dumps(env or {})}\n"); lf.flush()
        self.proc = subprocess.Popen(["bash", os.path.join(BENCH, "src", "serve_bench.sh"), cfg],
                                     stdout=lf, stderr=subprocess.STDOUT, env=e, start_new_session=True)
        log(f"server {cfg} {env} pid {self.proc.pid} log {self.logpath}")
        t0 = time.time()
        while time.time() - t0 < 1200:
            if self.proc.poll() is not None:
                log(f"server {cfg} exited rc={self.proc.returncode} during boot")
                self.proc = None
                return False
            if self.health():
                log(f"server {cfg} healthy after {time.time()-t0:.0f}s")
                self.key = (cfg, json.dumps(env or {}, sort_keys=True))
                return True
            time.sleep(5)
        log("server boot timeout")
        self.stop()
        return False

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc is None:
            return
        pid = self.proc.pid
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(45):
            if self.proc.poll() is not None:
                break
            time.sleep(2)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        time.sleep(8)
        log(f"server pid {pid} stopped")
        self.proc = None
        self.key = None


def stop_requested():
    return os.path.exists(os.path.join(QD, "STOP")) or os.path.exists(os.path.join(BENCH, "queue", "STOP"))


def main():
    srv = Server()
    idle_since = None
    while True:
        if stop_requested():
            log("STOP file: exiting"); srv.stop(); return
        jobs = sorted(glob.glob(os.path.join(QD, "*.json")))
        if not jobs:
            if idle_since is None:
                idle_since = time.time(); log("queue empty, waiting")
                progress("queue empty, waiting for jobs")
            time.sleep(20)
            continue
        idle_since = None
        jp = jobs[0]
        job = json.load(open(jp))
        key = (job["cfg"], json.dumps(job.get("env") or {}, sort_keys=True))
        name = os.path.basename(jp)
        if srv.key != key or not srv.alive():
            ok = False
            for attempt in range(2):
                if srv.start(job["cfg"], job.get("env")):
                    ok = True; break
            if not ok:
                log(f"job {name}: server failed to boot twice; moving to failed/")
                os.makedirs(os.path.join(QD, "failed"), exist_ok=True)
                os.rename(jp, os.path.join(QD, "failed", name))
                progress(f"FAILED to boot {job['cfg']} for {name}")
                continue
        out = os.path.join(BENCH, job["out"])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        t0 = time.time()
        log(f"job {name}: {job['cfg']} {job['reqs']} shard={job.get('shard')} conc={job.get('conc',1)}")
        try:
            import importlib; importlib.reload(client)
            stats = asyncio.run(client.run_job(job, PORT, out, BENCH, srv.logpath, srv.alive))
        except Exception as ex:  # noqa: BLE001
            log(f"job {name} client exception {ex!r}")
            stats = {"error": repr(ex)}
        dt = time.time() - t0
        stats["secs"] = round(dt, 1)
        stats["server_log"] = os.path.relpath(srv.logpath, BENCH) if srv.logpath else None
        if stats.get("server_died"):
            log(f"job {name}: server died; restarting and retrying the job")
            srv.stop()
            job["_retries"] = job.get("_retries", 0) + 1
            if job["_retries"] <= 2:
                json.dump(job, open(jp, "w"))
                continue
        json.dump(stats, open(os.path.join(QD, "done", name + ".stats.json"), "w"), indent=1)
        os.rename(jp, os.path.join(QD, "done", name))
        progress(f"{job.get('label', name)} cfg={job['cfg']} shard={job.get('shard')} "
                 f"done {stats.get('n_done')}/{stats.get('n_total')} in {dt/60:.0f} min"
                 + (f" errors={stats.get('n_err')}" if stats.get('n_err') else ""))


if __name__ == "__main__":
    main()
