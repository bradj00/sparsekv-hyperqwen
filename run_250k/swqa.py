# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
import os, sys
sys.argv = ["x", "1", "/dev/null", sys.argv[1], "1"]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep.py")).read().split("docs = {}")[0]
exec(src)
for u in range(int(sys.argv[3]) if False else 4):
    d, qs = build(u); q, w, _ = qs["code"]; a = q.split(chr(10))[1]; i = d.find(a)
    print(u, "Q:", a[:80], "| WANT:", w[:80]); print("   ctx:", repr(d[i-150:i+len(a)+100]))
