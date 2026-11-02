# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
__file__ = os.path.join(HERE, "chat250.py")
src = open(__file__).read().split("if MODE ==")[0]
sys.argv = ["x", os.environ.get("PORT", "8082"), os.path.join(HERE, "r.gpu1_live.jsonl"), "none"]
exec(src.replace("\"model\": \"qwen3.8-27b\"", "\"model\": \"qwen3.8-27b-gpu1\""))
d = {"A": build(NTOK, False, "BLUE", "7418-2295", "amber-quartz-41")}
for name in ("needle", "code"):
    r = turn("A", d, name, {"mode": "gpu1-live"})
