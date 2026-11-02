# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""Add env-gated (SPARSE_CG_STATS=1) cudagraph dispatch counters to vLLM v1/worker/gpu/cudagraph_utils.py."""
import sys

p = sys.argv[1]
s = open(p).read()
if "_CG_STATS" in s:
    print("already")
    sys.exit(0)


def rep(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new, 1)


rep("""class CudaGraphManager:
""", """import os as _os
import time as _time
from collections import Counter as _Counter

_CG_STATS = _os.environ.get("SPARSE_CG_STATS", "0") == "1"
_CG_COUNTS: "_Counter" = _Counter()
_CG_LAST = [0.0]


def _cg_note(key) -> None:
    \"\"\"Count one dispatch/replay; log the counters every 20 s (host only).\"\"\"
    _CG_COUNTS[key] += 1
    now = _time.monotonic()
    if now - _CG_LAST[0] > 20.0:
        _CG_LAST[0] = now
        logger.info("cg-stats: %s", dict(sorted(_CG_COUNTS.items(), key=lambda kv: str(kv[0]))))


class CudaGraphManager:
""")
rep("""                ):
                    return desc
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,""", """                ):
                    if _CG_STATS:
                        _cg_note((type(self).__name__, desc.cg_mode.name, num_tokens, num_reqs))
                    return desc
        if _CG_STATS:
            _cg_note((type(self).__name__, "NONE", num_tokens, num_reqs, uniform_token_count, max_query_len))
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,""")
rep("""        graph = self.graphs[desc]
        if self._sparse_switch is not None and self._sparse_switch.step_sparse():
            graph = self.graphs_sparse.get(desc, graph)
""", """        graph = self.graphs[desc]
        if self._sparse_switch is not None and self._sparse_switch.step_sparse():
            graph = self.graphs_sparse.get(desc, graph)
        if _CG_STATS:
            _cg_note((type(self).__name__, "replay", desc.num_tokens,
                      "sparse" if graph is not self.graphs[desc] else "dense"))
""")
open(p, "w").write(s)
print("patched cgstats")
