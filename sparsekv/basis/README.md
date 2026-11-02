# basis27.pt

Frozen rank-64 key-sketch basis for the 16 full-attention layers of Qwen3.8-27B: top-64 eigenvectors of the per-(layer, KV head) Gram of post-RoPE keys, fitted once offline by `sparsekv/tools/fit_basis27.py` (corpus and fit log: `fit.log`). sha256 in `basis27.pt.sha256`.

Derived from key activations of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0, Alibaba Cloud / Qwen team); it holds no model weights and no text. See `NOTICE`.
