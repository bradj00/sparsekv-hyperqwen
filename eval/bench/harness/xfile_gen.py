#!/usr/bin/env python3
"""Exhaustive cross-file call-site retrieval ("find every place function X is called").

Builds packed source contexts from real Python repos, computes the ground-truth call sites of chosen functions
with the Python AST (cross-checked with a word-boundary grep), and writes request lines for client.py.

Usage: xfile_gen.py OUT.jsonl GT.json MODEL_DIR corpus=ROOT:subdir1,subdir2 ... [--per 8] [--seed 0]
For each corpus the 64K pack is a prefix of the 128K pack, which is a prefix of the 192K pack (same file order), so
one corpus' requests share their prompt prefix. The answer unit is (file path, innermost enclosing def name).
"""
import ast, json, os, random, re, sys
from collections import defaultdict
from transformers import AutoTokenizer

LENGTHS = {"64k": 65536, "128k": 131072, "192k": 196608}
HEAD = ("Below is the source code of many files from the {repo} repository. Each file starts with a line "
        "'### FILE: <path>'.\n\n")
QUESTION = ("\n\nTask: find EVERY place in the code above where the function `{name}` is called. Count each call site "
            "(a call such as `{name}(...)`, `self.{name}(...)` or `obj.{name}(...)`), not the definition and not imports. "
            "For each call, give the file path and the name of the innermost function or method that contains the call. "
            "List each (file, function) pair once, one pair per line, in exactly this format:\n"
            "path/to/file.py :: enclosing_function_name\n"
            "Output only the list, with no other text.")
SKIP_NAMES = {"__init__", "forward", "main", "run", "get", "set", "update", "apply", "call", "build", "load", "save",
              "reset", "close", "start", "stop", "step", "init", "setup", "wrapper", "inner", "decorator", "helper"}


def collect(root, subdirs):
    files = []
    for sd in subdirs:
        base = os.path.join(root, sd)
        for dp, dn, fn in os.walk(base):
            dn.sort()
            for f in sorted(fn):
                if f.endswith(".py"):
                    p = os.path.join(dp, f)
                    rel = os.path.relpath(p, os.path.dirname(root.rstrip("/")))
                    try:
                        src = open(p, encoding="utf-8").read()
                        ast.parse(src)
                    except Exception:
                        continue
                    if 200 < len(src) < 60000:
                        files.append((rel, src))
    return files


def analyze(files):
    """defs[name] = [(file, lineno)], calls[name] = set((file, enclosing def)), ncall[name] = call nodes,
    bad = names with a call outside any def (module or class body level)."""
    defs, calls, ncall, bad = defaultdict(list), defaultdict(set), defaultdict(int), set()

    class V(ast.NodeVisitor):
        def __init__(self, rel):
            self.rel, self.stack = rel, []

        def _def(self, node):
            defs[node.name].append((self.rel, node.lineno))
            for d in node.decorator_list:
                self.visit(d)
            self.visit(node.args)
            if node.returns is not None:
                self.visit(node.returns)
            self.stack.append(node.name)
            for b in node.body:
                self.visit(b)
            self.stack.pop()

        visit_FunctionDef = visit_AsyncFunctionDef = _def

        def visit_ClassDef(self, node):
            defs[node.name].append((self.rel, node.lineno))
            self.generic_visit(node)

        def visit_Call(self, node):
            f = node.func
            nm = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            if nm:
                ncall[nm] += 1
                if self.stack:
                    calls[nm].add((self.rel, self.stack[-1]))
                else:
                    bad.add(nm)
            self.generic_visit(node)

    for rel, src in files:
        V(rel).visit(ast.parse(src))
    return defs, calls, ncall, bad


def name_counts(files):
    """Counts of NAME tokens in code (comments and strings excluded) and of names inside import statements."""
    import io, tokenize
    from collections import Counter
    names, imps = Counter(), Counter()
    for rel, src in files:
        for t in tokenize.generate_tokens(io.StringIO(src).readline):
            if t.type == tokenize.NAME:
                names[t.string] += 1
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for al in node.names:
                    for part in al.name.split("."):
                        imps[part] += 1
                    if al.asname:
                        imps[al.asname] += 1
    return names, imps


def candidates(files):
    defs, calls, ncall, bad = analyze(files)
    names, imps = name_counts(files)
    out = []
    for name, dl in defs.items():
        if len(dl) != 1 or name in SKIP_NAMES or len(name) < 8 or name.startswith("__") or name not in calls:
            continue
        if name in bad:
            continue
        # cross-check: every code mention of the name (comments/strings excluded) is the def, an import, or a
        # call we counted -- no aliasing, callbacks or other references that would make the call set ambiguous
        if names[name] != 1 + imps[name] + ncall[name]:
            continue
        cs = sorted(calls[name])
        out.append({"name": name, "def": dl[0], "calls": cs, "n_pairs": len(cs), "n_calls": ncall[name],
                    "n_files": len({f for f, _ in cs})})
    return out


BINS = [(1, 1), (2, 3), (4, 7), (8, 14), (15, 1000)]


def pick(cands, per, rng, used):
    """Stratified by call-pair count: equal quota per bin; a bin's shortfall moves to the nearest bin that has spares
    (larger counts first). Larger bins prefer call sets that span >= 2 files."""
    by = defaultdict(list)
    for c in sorted(cands, key=lambda c: c["name"]):
        if c["name"] in used:
            continue
        for i, (lo, hi) in enumerate(BINS):
            if lo <= c["n_pairs"] <= hi:
                by[i].append(c)
    nb = len(BINS)
    quota = [per // nb + (1 if i < per % nb else 0) for i in range(nb)][::-1]  # extra slots to the larger bins
    for i in range(nb):
        short = quota[i] - min(quota[i], len(by[i]))
        quota[i] -= short
        order = sorted((j for j in range(nb) if j != i), key=lambda j: (abs(j - i), -j))
        for j in order:
            if short == 0:
                break
            spare = len(by[j]) - quota[j]
            if spare > 0:
                k = min(spare, short); quota[j] += k; short -= k
    chosen = []
    for i in range(nb):
        pool = by[i]
        cross = [c for c in pool if c["n_files"] >= 2] if i >= 1 else []
        rest = [c for c in pool if c not in cross]
        rng.shuffle(cross); rng.shuffle(rest)
        chosen += (cross + rest)[:quota[i]]
    return chosen


def main():
    out, gt_path, model = sys.argv[1:4]
    per, seed = 10, 0
    corpora = []
    for a in sys.argv[4:]:
        if a.startswith("--per="):
            per = int(a.split("=")[1])
        elif a.startswith("--seed="):
            seed = int(a.split("=")[1])
        else:
            name, spec = a.split("=", 1)
            root, subs = spec.split(":")
            corpora.append((name, root, subs.split(",")))
    tok = AutoTokenizer.from_pretrained(model)
    rng = random.Random(seed)
    reqs, gt = [], {}
    for ci, (cname, root, subs) in enumerate(corpora):
        files = collect(root, subs)
        repo = os.path.basename(root.rstrip("/"))
        head = HEAD.format(repo=repo)
        blocks = [f"### FILE: {rel}\n{src.rstrip()}\n\n" for rel, src in files]
        ntok = [len(tok(b, add_special_tokens=False)["input_ids"]) for b in blocks]
        used = set()
        for lname, L in LENGTHS.items():
            budget = L - 1600  # template + question + 1024-token answer budget
            tot, n = len(tok(head)["input_ids"]), 0
            while n < len(blocks) and tot + ntok[n] <= budget:
                tot += ntok[n]; n += 1
            pf = files[:n]
            cands = candidates(pf)
            chosen = pick(cands, per, rng, used)
            pack = head + "".join(blocks[:n])
            for q, c in enumerate(chosen):
                used.add(c["name"])
                user = pack.rstrip() + QUESTION.format(name=c["name"])
                prompt = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                rid = f"xfile/{cname}/{lname}/{q:02d}-{c['name']}"
                ptoks = len(tok(prompt, add_special_tokens=False)["input_ids"])
                reqs.append({"id": rid, "shard": ci % 2, "group": f"{cname}", "prompt": prompt, "max_tokens": 1024})
                gt[rid] = {"corpus": cname, "length": lname, "prompt_tokens": ptoks, "files_in_pack": n, **c}
            print(f"{cname} {lname}: {n} files, {tot} tokens, {len(cands)} candidates, chose "
                  f"{[(c['name'], c['n_pairs']) for c in chosen]}", flush=True)
    with open(out, "w") as f:
        for r in reqs:
            f.write(json.dumps(r) + "\n")
    json.dump(gt, open(gt_path, "w"), indent=1)


if __name__ == "__main__":
    main()
