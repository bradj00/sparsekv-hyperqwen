"""Shared helpers for the short-benchmark suite (GPQA / LCB / IFBench / MMLU-Pro).

Kept dependency-free so both short_gen.py and short_score.py (and the LCB sandbox worker's parent)
can import it.
"""
import base64, hashlib, io, json, pickle, zlib

import os

# BENCH = work dir holding reqs/, data/, out/, ext/ (default: the parent of this harness directory)
BENCH = os.environ.get("BENCH", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LCB_REPO = f"{BENCH}/ext/LiveCodeBench"
IFBENCH_REPO = f"{BENCH}/ext/IFBench"
SUITES = ("gpqa", "lcb", "ifbench", "mmlupro")


def seed_for(rid: str) -> int:
    """Deterministic 31-bit seed per request id."""
    return int(hashlib.sha256(rid.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


class _NoGlobalsUnpickler(pickle.Unpickler):
    """LCB private tests are zlib+base64 pickles of a plain JSON string. Refuse any global lookup
    so a malicious pickle cannot execute code (only str/list/dict/number opcodes can succeed)."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"blocked global {module}.{name} in LCB test pickle")


def decode_lcb_private(s: str) -> list:
    """Same semantics as lcb_runner/benchmarks/code_generation.py CodeGenerationProblem.__post_init__:
    try json.loads; else json.loads(pickle.loads(zlib.decompress(base64.b64decode(s)))) -- with a
    restricted unpickler."""
    try:
        return json.loads(s)
    except Exception:
        raw = zlib.decompress(base64.b64decode(s.encode("utf-8")))
        return json.loads(_NoGlobalsUnpickler(io.BytesIO(raw)).load())


def split_final(text: str):
    """Return (final_answer_text, has_think_close). Final answer = text after the LAST </think>."""
    if text is None or "</think>" not in text:
        return None, False
    fin = text.rsplit("</think>", 1)[1]
    for tok in ("<|im_end|>", "<|endoftext|>"):
        fin = fin.replace(tok, "")
    return fin.strip(), True
