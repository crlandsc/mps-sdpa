# mps-sdpa

**Fast drop-in `scaled_dot_product_attention` for Apple silicon (MPS).** Wraps
Apple's native `MPSGraph.scaledDotProductAttention` op with a zero-copy C++ /
Objective-C++ bridge. 5–7× inference speedup, 2–2.5× training speedup, 16–170×
less driver memory per call — with identical math and checkpoint compatibility.

```python
from mps_sdpa import sdpa_opt

# Drop-in replacement — same signature as torch.nn.functional.scaled_dot_product_attention
out = sdpa_opt(query, key, value,
               attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
               backend="auto")
```

---

## Why

PyTorch's MPS backend dispatches `torch.nn.functional.scaled_dot_product_attention`
to `sdpa_general_mps`, which builds a naive matmul → softmax → matmul graph.
It does **not** call Apple's dedicated `MPSGraph.scaledDotProductAttention` op,
which is present on macOS 15+ and is significantly faster.

`mps-sdpa` wraps that op directly, with an autograd.Function for training,
shape-threshold auto-calibration, graceful fallbacks, and thread-safe graph
caching. The C++ extension uses ATen's `getMTLBufferStorage` to hand torch
tensors to MPSGraph without CPU memcpy.

## Measured performance

All numbers on M4 / macOS 26.4.1 / torch 2.13 nightly, bfloat16.

### Inference (forward only, B=1, H=8, D=64)

| L | stock | mps-sdpa | speedup |
|---|---|---|---|
| 1024 | 5.78 ms | 0.90 ms | **6.42×** |
| 2048 | 19.0 ms | 3.82 ms | **4.97×** |
| 4096 | 76.1 ms | 11.79 ms | **6.45×** |
| 8192 | 317 ms | 44.3 ms | **7.17×** |

Weighted geomean across a realistic audio-model shape suite: **4.88×**.

### Training (forward + backward, same shapes)

| L | stock | mps-sdpa | speedup |
|---|---|---|---|
| 1024 | 9.93 ms | 5.06 ms | **1.96×** |
| 2048 | 38.6 ms | 17.1 ms | **2.25×** |
| 4096 | 154 ms | 64.8 ms | **2.38×** |
| 8192 | 608 ms | 247 ms | **2.46×** |

### Training with dropout (dropout_p=0.1)

| L | stock | mps-sdpa | speedup |
|---|---|---|---|
| 1024 | 14.19 ms | 7.63 ms | **1.86×** |
| 2048 | 55.83 ms | 28.49 ms | **1.96×** |
| 4096 | 228 ms | 101 ms | **2.26×** |

### Driver memory per call

Apple's fused op doesn't materialize the `[Lq, Lkv]` attention matrix. The
zero-copy C++ bridge removes the CPU-side intermediate buffer too.

| L | stock | mps-sdpa | reduction |
|---|---|---|---|
| 2048 | 1024 MB | <1 MB | **≫128×** |
| 4096 | 1024 MB | <1 MB | **≫64×** |
| 8192 | 1024 MB | 32 MB | **32×** |

## Install

```bash
pip install mps-sdpa
```

Requires macOS 15+ on Apple silicon (M1–M4) with PyTorch ≥ 2.11. `ninja` is
pulled in automatically for the zero-copy backend's JIT compile (~6s on first
import, then cached to `~/.cache/torch_extensions`).

Development install from source:
```bash
git clone https://github.com/crlandsc/mps-sdpa.git
cd mps-sdpa
pip install -e ".[dev]"
pytest tests/
```

## Usage

The API exactly mirrors `torch.nn.functional.scaled_dot_product_attention`:

```python
import torch
from mps_sdpa import sdpa_opt

q = torch.randn(1, 8, 2048, 64, dtype=torch.bfloat16, device="mps")
k = torch.randn(1, 8, 2048, 64, dtype=torch.bfloat16, device="mps")
v = torch.randn(1, 8, 2048, 64, dtype=torch.bfloat16, device="mps")

out = sdpa_opt(q, k, v)                                  # basic
out = sdpa_opt(q, k, v, is_causal=True)                  # causal mask
out = sdpa_opt(q, k, v, attn_mask=bool_mask)             # bool mask
out = sdpa_opt(q, k, v, dropout_p=0.1)                   # training dropout
out = sdpa_opt(q, k, v, backend="mpsgraph_zc")           # force specific backend
```

**As a drop-in swap inside an existing model:**

```python
import torch.nn as nn
import torch.nn.functional as F
from mps_sdpa import sdpa_opt

class MyAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = 0.1

    def forward(self, x):
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        p = self.dropout_p if self.training else 0.0  # standard pattern
        y = sdpa_opt(q, k, v, dropout_p=p)            # <— was F.scaled_dot_product_attention
        y = y.transpose(1, 2).reshape(B, L, self.d_model)
        return self.out(y)
```

**Gate the behavior at model-construction time** (useful for A/B comparison):

```python
class Attn(nn.Module):
    def __init__(self, ..., use_mps_sdpa_opt: bool = False):
        self.use_opt = use_mps_sdpa_opt

    def forward(self, q, k, v):
        if self.use_opt:
            return sdpa_opt(q, k, v, ...)
        return F.scaled_dot_product_attention(q, k, v, ...)
```

Checkpoints are fully interchangeable — `use_opt` changes behavior only, not
parameters.

## CLI

```bash
mps-sdpa self-test                      # quick validation (<1s)
mps-sdpa list-backends                  # show available backends
mps-sdpa correctness --backend mpsgraph_zc --device mps --suite realistic
mps-sdpa benchmark --backend mpsgraph_zc --baseline stock --device mps --suite realistic
```

## Backends

`backend="auto"` picks the best available. Available backends (in preference order):

| Name | Impl | Best for |
|---|---|---|
| `mpsgraph_zc` | Obj-C++ torch extension, getMTLBufferStorage zero-copy | **Default** — forward, training, masks, dropout |
| `mpsgraph` | pyobjc + CPU-copy bridge | Fallback when the ext can't build |
| `stock` | `torch.nn.functional.scaled_dot_product_attention` | Always available final fallback |
| `metal_proto` | naive single-thread Metal kernel via `torch.mps.compile_shader` | Reference / experimentation, not auto-selected |

Fallbacks are transparent — a call that can't go through `mpsgraph_zc` (e.g.
short seq, CPU device, unsupported dtype) routes down the list automatically.

## Correctness contract

Same tolerance bar as CUDA flash-attention vs math:
- fp32: atol=5e-6, rtol=5e-5
- fp16/bf16: atol=5e-3, rtol=5e-2
- gradients: 2× the forward tolerance

Not bitwise identical. Checkpoints trained with one backend load cleanly into
the other (verified).

## Scope / supported configs

| Category | Maintainer-tested | Should work (untested by maintainer) | Not supported |
|---|---|---|---|
| Apple silicon | M4 mini, M3 Max | M1, M2 (should work via auto-calibration) | — |
| macOS | 26.x | 15.x (all API surfaces present) | 14.x (op missing — backend registers unavailable) |
| torch | 2.11 stable + 2.13 nightly | — | — |
| dtypes | bf16, fp16, fp32 | — | fp64 (MPS doesn't support it) |

Maintainer testing is M-series Apple silicon on macOS 26+ only. Reports
from other configurations are welcome but no commitment to test them.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the internal structure: backend
registry, graph cache, threshold auto-calibration, C++ extension build system,
autograd wiring, dropout path.

## What doesn't work (yet)

- **GQA (Hq ≠ Hkv):** routes to stock with `repeat_interleave`. mpsgraph op
  is MHA-only. One-time warning.
- **Second-order gradients** (`create_graph=True` on our output's grad):
  raises a clear error. Backward uses MPSGraph which is opaque to torch
  autograd; true higher-order would require a differentiable backward impl.
- **macOS 14:** the `MPSGraph.scaledDotProductAttention` method doesn't exist
  on Sequoia's predecessor. Backend registers as unavailable with a clear
  reason; calls fall back to stock.

## What works

- **`torch.compile`:** supported via `torch.library.custom_op` +
  `register_autograd` + `register_fake` (added in v0.2.0). `sdpa_opt`
  traces cleanly under `torch.compile(..., dynamic=False)`; output is
  numerically identical to eager mode within `cross_impl_atol(dtype)`.
  Six dedicated equivalence tests in `tests/test_torch_compile.py` cover
  forward, backward, masked, dropout-entropy, and `opcheck` smoke.

## Correctness — what's tested

254 tests across 43 files. Highlights:

- **Shape matrix:** D ∈ {32, 64, 96, 128, 192, 256}; H ∈ {1..32}; B ∈ {1..32};
  Lq, Lkv ∈ {powers of 2, 777, 1345, 3141}.
- **Masks:** causal, bool, additive float, all-True, mostly-False,
  per-head [B,H,Lq,Lkv], per-batch, broadcast variants, dtype coercion.
- **Autograd:** partial requires_grad subsets, AMP autocast (bf16/fp16),
  `torch.utils.checkpoint` re-entry, retain_graph, grad accumulation,
  second-order detection, once_differentiable guard.
- **Training:** generic transformer smoke (forward + backward),
  1000-step long-horizon convergence, mid-run backend toggle, fp16 500-step AMP.
- **Numerical extremes:** Q/K std ∈ {10, 1e-4}, grad_out ∈ {1e3, 1e4} — all
  NaN/Inf-free.
- **Edge cases:** degenerate shapes, non-contiguous inputs, strided views,
  cache thrash (9 shapes), 200-call leak probe, 16k-seq OOM recovery.

See `tests/` for the full test suite.

## Citation / credit

- Apple's fused SDPA op: `MPSGraph.scaledDotProductAttention...` (macOS 15+).
- Related prior art: `philipturner/metal-flash-attention` (Swift, not vetted here);
  PyPI `mps-flash-attn` (8 stars, unverified). Neither is used or wrapped.

## License

MIT.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports with reproducible shapes
are the most helpful; the full-suite `pytest tests/` should pass on
any supported config.
