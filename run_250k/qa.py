# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
import os, sys
__file__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat250.py")
src=open(__file__).read().split("if MODE ==")[0]
sys.argv=["x","1","/dev/null","none"]
exec(src)
for rev,tag in ((False,"BLUE"),(True,"RED")):
    d,qs=build(NTOK,rev,tag,"m","p")
    for k in ("code","code2"):
        q,w,_=qs[k]; a=q.split(chr(10))[1]; i=d.find(a)
        print(tag,k,"pos %.3f"%(i/len(d)),"WANT:",repr(w[:100])); print("  context:", repr(d[i-250:i+len(a)+120]))
