#!/usr/bin/env python
"""Self-test for short_score.py with crafted fake outputs (no model / GPU involved)."""
import json, os, sys, tempfile, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import short_score as S
from short_common import BENCH, decode_lcb_private

GT = os.path.join(BENCH, "data", "short_gt")
R = "Let me reason about this carefully...\n</think>\n\n"
U = {"prompt_tokens": 300, "completion_tokens": 1234}
fails = 0


def line(rid, text, finish="stop", error=None, cfg="selftest"):
    return {"id": rid, "cfg": cfg, "outputs": [text], "finish": [finish], "usage": [U], "error": error}


def check(name, got, want):
    global fails
    ok = got == want
    fails += not ok
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: got={got} want={want}")


def run(suite, lines, **kw):
    fd, p = tempfile.mkstemp(suffix=".jsonl", dir=f"{BENCH}/tmp")
    with os.fdopen(fd, "w") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")
    gt = json.load(open(f"{GT}/{suite}.json"))
    by = S.load_outputs([p])
    recs = S.score_records(suite, gt, by["selftest"], **kw)
    summ = S.summarize(suite, gt, recs, by["selftest"])
    sf = S.score_file(suite, gt, p, **kw) if suite != "lcb" else None
    os.unlink(p)
    return recs, summ, sf


os.makedirs(f"{BENCH}/tmp", exist_ok=True)

# ---------------- GPQA
print("== gpqa")
gt = json.load(open(f"{GT}/gpqa.json"))
ids = list(gt)
right = lambda i: gt[i]["answer"]
wrong = lambda i: "ABCD"[("ABCD".index(gt[i]["answer"]) + 1) % 4]
L = [line(ids[0], R + f"Step by step...\nAnswer: {right(ids[0])}"),
     line(ids[1], R + f"So the result.\n\n**Answer:** {right(ids[1])}"),
     line(ids[2], R + f"Final: $\\boxed{{{right(ids[2])}}}$"),
     line(ids[3], R + f"Answer: {wrong(ids[3])}"),
     line(ids[4], "Let me think forever about A, B... Answer: " + right(ids[4]), finish="length"),
     line(ids[5], R + "I am not sure."),
     line(ids[6], None, error="HTTP 500")]
recs, s, sf = run("gpqa", L)
check("answer line correct", recs[ids[0]]["score"], 1)
check("**Answer:** X correct", recs[ids[1]]["score"], 1)
check("boxed fallback correct", (recs[ids[2]]["score"], recs[ids[2]]["how"]), (1, "boxed"))
check("wrong letter", recs[ids[3]]["score"], 0)
check("truncated in reasoning (Answer inside reasoning ignored)", (recs[ids[4]]["score"], recs[ids[4]]["no_think"]), (0, True))
check("no answer", recs[ids[5]]["score"], 0)
check("error excluded", (ids[6] in recs, ids[6] in s["errors"]), (False, True))
check("summary n_scored/n_truncated/accuracy", (s["n_scored"], s["n_truncated"], round(s["accuracy"], 4)), (6, 1, 0.5))
check("missing count", s["n_missing"], len(ids) - 6)
check("score_file dict", sf, {i: recs[i]["score"] for i in recs})

# ---------------- MMLU-Pro
print("== mmlupro")
gt = json.load(open(f"{GT}/mmlupro.json"))
ids = list(gt)
a = lambda i: gt[i]["answer"]
w = lambda i: "ABCDEFGHIJ"[("ABCDEFGHIJ".index(gt[i]["answer"]) + 1) % gt[i]["n_options"]]
L = [line(ids[0], R + f"... so the answer is ({a(ids[0])})."),
     line(ids[1], R + f"**The answer is {a(ids[1])}**"),
     line(ids[2], R + f"Answer: {a(ids[2])}"),
     line(ids[3], R + f"The answer is ({w(ids[3])})"),
     line(ids[4], f"the answer is ({a(ids[4])}) hmm wait", finish="length"),
     line(ids[5], R + f"I pick option {a(ids[5])} here")]
recs, s, sf = run("mmlupro", L)
check("answer is (X)", recs[ids[0]]["score"], 1)
check("**The answer is X**", recs[ids[1]]["score"], 1)
check("Answer: X fallback", recs[ids[2]]["score"], 1)
check("wrong", recs[ids[3]]["score"], 0)
check("truncated", (recs[ids[4]]["score"], recs[ids[4]]["truncated"]), (0, True))
check("strict wrong / lenient extract_final right", (recs[ids[5]]["score"], recs[ids[5]]["score_lenient"]), (0, 1))
check("accuracy", round(s["accuracy"], 4), round(3 / 6, 4))

# ---------------- IFBench
print("== ifbench")
gt = json.load(open(f"{GT}/ifbench.json"))
k0 = "ifbench/0"  # count:keywords_multiple kaleidoscope x1, nebula x2, whisper x3, labyrinth x5, paradox x7
good = ("A kaleidoscope of thought. nebula nebula. whisper whisper whisper. "
        "labyrinth labyrinth labyrinth labyrinth labyrinth. paradox paradox paradox paradox paradox paradox paradox.")
bad = "kaleidoscope nebula whisper labyrinth paradox."
other = [k for k in gt if k != k0][:2]
L = [line(k0, R + good), line(other[0], R + bad), line(other[1], "thinking forever", finish="length")]
recs, s, sf = run("ifbench", L)
check("keywords exact -> strict&loose", (recs[k0]["strict"], recs[k0]["loose"]), (1, 1))
check("unrelated response fails", recs[other[0]]["strict"], 0)
check("truncated -> all instr False", (recs[other[1]]["strict"], recs[other[1]]["loose"], any(recs[other[1]]["loose_instr"])), (0, 0, False))
print("  summary:", {k: s[k] for k in ("prompt_level_strict", "prompt_level_loose", "instruction_level_strict",
                                      "instruction_level_loose", "instr_counts", "n_truncated")})
check("score_file ifbench shape", sf[k0], {"strict": 1, "loose": 1})

# ---------------- LCB
print("== lcb")
gt = json.load(open(f"{GT}/lcb.json"))
n_priv = 0
for k, v in gt.items():
    n = len(decode_lcb_private(v["private_test_cases"]))
    assert n == v["n_private"] and n > 0, k
    n_priv += n
print(f"  private tests decoded for all {len(gt)} problems: {n_priv} cases")
blk = lambda code: R + "Here is the solution:\n\n```python\n" + code + "\n```\n"
STDIN_OK = "D = input().strip()\nm = {'N':'S','S':'N','E':'W','W':'E','NE':'SW','SW':'NE','NW':'SE','SE':'NW'}\nprint(m[D])"
FUNC_OK = ("class Solution:\n    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:\n"
           "        return min(n * n, maxWeight // w)")
FUNC_BAD = "class Solution:\n    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:\n        return 0"
L = [line("lcb/abc391_a", blk(STDIN_OK)), line("lcb/3817", blk(FUNC_OK)), line("lcb/3723", blk(FUNC_BAD))]
# sandbox probes on other problems
probe = {
    "lcb/3753": "import os\nos.system('touch /tmp/PWNED_lcb')\nclass Solution:\n    def maxDifference(self, s): return 0",
    "lcb/3709": "import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=3)\nclass Solution:\n    def hasSpecialSubstring(self, s, k): return True",
    "lcb/3799": "x = bytearray(8 * 1024**3)\nclass Solution:\n    def totalNumbers(self, digits): return 0",
    "lcb/3788": "class Solution:\n    def maxSum(self, nums):\n        while True: pass",
}
for rid, code in probe.items():
    L.append(line(rid, blk(code)))
L.append(line("lcb/3832", R + "no code block here"))
L.append(line("lcb/3773", "```python\nprint(1)\n```", finish="length"))
t0 = time.time()
recs, s, _ = run("lcb", L, lcb_workers=6)
print(f"  lcb scoring wall time {time.time() - t0:.1f}s")
for rid in recs:
    r = recs[rid]
    print(f"   {rid}: score={r['score']} n_pass={r.get('n_pass')}/{r.get('n_tests')} first_fail={r.get('first_fail')} meta={str(r.get('meta'))[:160]}")
check("stdin hand-written correct passes all public+private", recs["lcb/abc391_a"]["score"], 1)
check("functional hand-written correct passes", recs["lcb/3817"]["score"], 1)
check("functional wrong fails", recs["lcb/3723"]["score"], 0)
check("os.system blocked -> fails, no file", (recs["lcb/3753"]["score"], os.path.exists("/tmp/PWNED_lcb")), (0, False))
check("network blocked -> fails", recs["lcb/3709"]["score"], 0)
check("8GB alloc blocked by RLIMIT_AS -> fails", recs["lcb/3799"]["score"], 0)
check("infinite loop -> timeout fail", recs["lcb/3788"]["score"], 0)
check("no code block -> fail", (recs["lcb/3832"]["score"], recs["lcb/3832"]["code_found"]), (0, False))
check("truncated -> fail", (recs["lcb/3773"]["score"], recs["lcb/3773"]["truncated"]), (0, True))
print("  by_difficulty:", s["by_difficulty"], "n_truncated", s["n_truncated"])

print(f"\nSELFTEST {'PASSED' if not fails else f'FAILED ({fails})'}")
sys.exit(1 if fails else 0)
