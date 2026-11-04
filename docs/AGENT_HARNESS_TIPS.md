# Agent harness tips: make parallel agents share their prefix

When several coding agents start at the same time against one server, their first requests are mostly the same text:
- the system prompt;
- the tool definitions;
- the shared part of the task brief.

That is typically 15-17K tokens per agent. With prefix caching, the server prefills this once and every other agent
gets it from the cache, but only if the token sequences are **identical** up to that point. One differing token early
in the system prompt (a path, a model name, a counter) splits the prefix there, and everything after it is
prefilled again for each agent.

On the test hardware a cold ~16K-token first request costs about 8.5-10 s of GPU time. A cache hit on the shared part
brings it under ~1 s. The harness changes below are free: the text stays the same, only its order and a few strings
change.

## 1. Stable system prompt, per-agent text last

- **Put everything that differs between agents at the end of the request.** Order the first request as: system prompt
  → tools → shared brief → per-agent part (its task, its stream or slot label, its time budget, its working notes).
  Every per-agent token before the shared part ends cuts the shared prefix at that point.
- **Move run-specific facts out of the top of the system prompt.** Agent frameworks often put an environment block
  near the top: working directory, date, platform, git status. Move it to the end of the system prompt, or after the
  shared brief.
- **Write paths relative to the working directory** (`./src/main.ts`, not `/work/run-17/src/main.ts`).
- **Keep the order deterministic.** Tool lists, skill lists and file lists should come out in a fixed order. Avoid
  timestamps, random ids and counters anywhere before the per-agent part.
- **Watch for injected extras.** Some frameworks append host-specific skills, rules files or memory to every system
  prompt. If they are identical across agents they do not break sharing, but they make every prefill longer. Turn off
  the ones the agents do not need.

## 2. Identical working directory for every agent

Each agent usually has its own checkout, at a different path. The path ends up in the system prompt and in tool
output, so it splits the prefix. Give every agent the **same** path by running it in its own mount namespace. For
example, with bubblewrap (unprivileged, no root needed):

```bash
bwrap --dev-bind / / --tmpfs /srv --bind "$AGENT_WORKSPACE" /srv/ws --chdir /srv/ws --die-with-parent -- \
      opencode run "$FIRST_PROMPT" --dir /srv/ws ...
```

- Every agent then sees its own workspace at `/srv/ws`.
- The harness itself keeps using the real paths.
- `--die-with-parent` ties the agent's lifetime to the harness process.
- Use the same wrapper for every later turn of the session (resume, nudge, finalise), so the session directory stays
  the same.
- Allow the fixed path in the agent's permission settings.
- A container or `chroot` works the same way.

## 3. A model line that does not name the server

Many agent prompts state the served model id ("You are powered by the model named `<served-name>`. The exact model ID
is `<provider>/<id>`").
- If agents talk to several servers with the same weights, under different names or provider keys, that line differs
  per server. It sits near the top of the system prompt, so the prefix splits there.
- Moving the line to the end is not enough when the brief comes after the system prompt. Replace it with one neutral
  line, for example "You are powered by the model Qwen3.8-27B."

## 4. Example: opencode plugin

[`examples/opencode/stable-prefix.ts`](../examples/opencode/stable-prefix.ts) does steps 1 and 3 inside opencode,
through the `experimental.chat.system.transform` hook:
- it moves the `<env>` block to the end of the system prompt;
- it makes workspace paths `./relative`;
- it replaces the model line with a neutral one. Set `STABLE_PREFIX_MODEL` to the name you want stated.

Copy it to `~/.config/opencode/plugin/` (global) or `.opencode/plugin/` (per project). Combine it with step 2 (an
identical working directory) and with putting per-agent text last in your first prompt.

## 5. Keep the shared prefix warm on the server: the pin knob

- A shared prefix only helps if the server still has it when the next agent arrives. On a busy server, LRU eviction
  can drop it within minutes.
- In this hybrid model a hit also needs the recurrent (GDN) checkpoint at the hit boundary, which lives in a small
  slot store.
- The sparse-KV host tier can pin a warm prefix (`SPARSE_TIER_PIN_PREFIX_MAX`, default 2 in v1.1).

**How to use it:**
1. Find the shared prefix of your agents' first requests, in tokens.
2. Send one warm request whose prompt is exactly that prefix, ideally cut at a multiple of the cache block size (864
   tokens here), with `max_tokens: 1` and the extra argument below.

```json
POST /v1/completions
{"model": "...", "prompt": [<token ids of the shared prefix>], "max_tokens": 1, "vllm_xargs": {"sparse_pin_prefix": 1}}
```

Send token ids, not text. Get them by rendering one agent's first request with the chat template (for example
`POST /tokenize` with its `messages`) and cutting at the shared length. Then the warm prompt matches the agents'
requests token for token.

**What it does:**
- At finish, the server pins that request's host KV blocks and its GDN checkpoint at the highest saved boundary at or
  below the prompt end.
- Pinned entries are never evicted, so every later request starting with the same prefix hits it, however much
  traffic passes.
- Send `{"sparse_pin_prefix": "clear"}` to release all pins. Beyond the maximum, the oldest pin is released first.
- Cost per pin: one GDN slot (85 MB of pinned RAM) plus the prefix's host KV (~0.3 GB at 16.5K tokens, int4).
- With `SPARSE_TIER_PIN_PREFIX_MAX=0` the argument is ignored.

Re-send the warm request when the shared prefix changes, for example a new brief for a new round.

## Measured effect

| setup | common prefix of 8 parallel agents' first requests |
|---|---|
| as launched | **7,262 tokens**: split by the per-server model line |
| + plugin v1 (env block last, relative paths) | 7,262 across servers; 11,912 within one server |
| + identical working directory | 7,262 across servers; 15,785 within one server |
| + neutral model line + per-agent text last | **16,026 tokens** for all 8 (15,552 aligned to 864-token blocks) |

- **Server side:** after one pinned warm request of 15,552 tokens and 20 minutes of unrelated traffic, all 8 agents'
  first requests hit 15,552 cached tokens, with TTFT ≤ 1.11 s.
- **Live:** in a production round after the change, **5 of 8 builds hit 15,552 cached tokens** on their first step.
- Method and the other v1.1 measurements are in [`eval/prefill/README.md`](../eval/prefill/README.md#5-shared-stable-prefix-and-the-pin-knob).
