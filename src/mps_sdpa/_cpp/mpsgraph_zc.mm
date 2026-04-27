// Zero-copy MPSGraph SDPA — Objective-C++ extension.
//
// This is the C++ counterpart to backends/mpsgraph.py. The pyobjc-based path
// works but has to route tensor data via CPU memcpy because the pyobjc bridge
// can't safely reinterpret torch's opaque MPS storage as an id<MTLBuffer>.
// Here we do it the right way: getMTLBufferStorage() (static inline in ATen's
// OperationUtils.h) gives us the id<MTLBuffer> directly with zero copy.
//
// Exposes: mps_sdpa_ext.sdpa_forward(q, k, v, mask?, scale) -> out
//
// Graphs are cached per (dtype, B, H, Lq, Lkv, D, mask_kind, mask_shape).

#include <torch/extension.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <unordered_map>
#include <mutex>
#include <optional>
#include <tuple>

namespace {

// MPSDataType enum values (from MPSCoreTypes.h).
constexpr uint32_t MPS_FLOAT32 = 0x10000020;
constexpr uint32_t MPS_FLOAT16 = 0x10000010;
constexpr uint32_t MPS_BFLOAT16 = 0x90000010;

uint32_t dtype_to_mps(torch::Dtype d) {
  if (d == torch::kFloat32) return MPS_FLOAT32;
  if (d == torch::kFloat16) return MPS_FLOAT16;
  if (d == torch::kBFloat16) return MPS_BFLOAT16;
  TORCH_CHECK(false, "mps_sdpa_ext: unsupported dtype for mpsgraph SDPA");
}

// Cache key for compiled graphs.
struct GraphKey {
  uint32_t dtype_val;
  int64_t B, H, Lq, Lkv, D;
  int mask_kind;  // 0 = none, 1 = mask_present
  int64_t mask_B, mask_H;  // placeholder dims (only used if mask_kind != 0)
  bool dropout;   // unfused graph with explicit dropout mask input
  bool operator==(const GraphKey& o) const {
    return dtype_val == o.dtype_val && B == o.B && H == o.H &&
           Lq == o.Lq && Lkv == o.Lkv && D == o.D &&
           mask_kind == o.mask_kind && mask_B == o.mask_B && mask_H == o.mask_H &&
           dropout == o.dropout;
  }
};

struct GraphKeyHash {
  size_t operator()(const GraphKey& k) const noexcept {
    size_t h = std::hash<uint32_t>()(k.dtype_val);
    auto mix = [&](int64_t v) { h = h * 1315423911u ^ std::hash<int64_t>()(v); };
    mix(k.B); mix(k.H); mix(k.Lq); mix(k.Lkv); mix(k.D);
    h = h * 1315423911u ^ std::hash<int>()(k.mask_kind);
    mix(k.mask_B); mix(k.mask_H);
    h = h * 1315423911u ^ std::hash<bool>()(k.dropout);
    return h;
  }
};

// Compiled graph bundle. NSObjects are retained for the lifetime of the cache.
struct CachedGraph {
  MPSGraph* graph = nil;
  MPSGraphTensor* q_ph = nil;
  MPSGraphTensor* k_ph = nil;
  MPSGraphTensor* v_ph = nil;
  MPSGraphTensor* mask_ph = nil;  // nil if mask_kind=0
  MPSGraphTensor* drop_ph = nil;  // nil if dropout=false
  MPSGraphTensor* out_ph = nil;
};

std::unordered_map<GraphKey, CachedGraph*, GraphKeyHash> g_graph_cache;
std::mutex g_graph_cache_mutex;

CachedGraph* build_graph(const GraphKey& key) {
  CachedGraph* cg = new CachedGraph();
  cg->graph = [[MPSGraph alloc] init];

  NSArray<NSNumber*>* qShape = @[@(key.B), @(key.H), @(key.Lq), @(key.D)];
  NSArray<NSNumber*>* kShape = @[@(key.B), @(key.H), @(key.Lkv), @(key.D)];
  MPSDataType dt = (MPSDataType)key.dtype_val;

  cg->q_ph = [cg->graph placeholderWithShape:qShape dataType:dt name:@"q"];
  cg->k_ph = [cg->graph placeholderWithShape:kShape dataType:dt name:@"k"];
  cg->v_ph = [cg->graph placeholderWithShape:kShape dataType:dt name:@"v"];
  float scale = 1.0f / sqrtf((float)key.D);

  if (!key.dropout) {
    // Fast path: Apple's fused SDPA op (no dropout supported there).
    if (key.mask_kind != 0) {
      NSArray<NSNumber*>* maskShape = @[@(key.mask_B), @(key.mask_H),
                                         @(key.Lq), @(key.Lkv)];
      cg->mask_ph = [cg->graph placeholderWithShape:maskShape dataType:dt name:@"mask"];
      cg->out_ph = [cg->graph scaledDotProductAttentionWithQueryTensor:cg->q_ph
                                                             keyTensor:cg->k_ph
                                                           valueTensor:cg->v_ph
                                                            maskTensor:cg->mask_ph
                                                                 scale:scale
                                                                  name:@"sdpa_m"];
    } else {
      cg->out_ph = [cg->graph scaledDotProductAttentionWithQueryTensor:cg->q_ph
                                                             keyTensor:cg->k_ph
                                                           valueTensor:cg->v_ph
                                                                 scale:scale
                                                                  name:@"sdpa"];
    }
  } else {
    // Dropout: unfused graph. Apple's fused op has no dropout variant, so we
    // materialize the attention matrix explicitly.
    //   scores = (Q @ K^T) * scale   (+ mask)
    //   attn   = softmax(scores, -1)
    //   attn_d = attn * dropout_mask   (pre-scaled 1/(1-p))
    //   out    = attn_d @ V
    MPSGraphTensor* scale_t = [cg->graph constantWithScalar:scale dataType:dt];
    MPSGraphTensor* kT = [cg->graph transposeTensor:cg->k_ph
                                          dimension:-1 withDimension:-2 name:@"kT"];
    MPSGraphTensor* qk = [cg->graph matrixMultiplicationWithPrimaryTensor:cg->q_ph
                                                           secondaryTensor:kT name:@"qk"];
    MPSGraphTensor* scores = [cg->graph multiplicationWithPrimaryTensor:qk
                                                         secondaryTensor:scale_t
                                                                    name:@"scores"];
    if (key.mask_kind != 0) {
      NSArray<NSNumber*>* maskShape = @[@(key.mask_B), @(key.mask_H),
                                         @(key.Lq), @(key.Lkv)];
      cg->mask_ph = [cg->graph placeholderWithShape:maskShape dataType:dt name:@"mask"];
      scores = [cg->graph additionWithPrimaryTensor:scores
                                   secondaryTensor:cg->mask_ph
                                              name:@"scores_m"];
    }
    MPSGraphTensor* attn = [cg->graph softMaxWithTensor:scores axis:-1 name:@"attn"];
    NSArray<NSNumber*>* dropShape = @[@(key.B), @(key.H), @(key.Lq), @(key.Lkv)];
    cg->drop_ph = [cg->graph placeholderWithShape:dropShape dataType:dt name:@"drop_mask"];
    MPSGraphTensor* attn_d = [cg->graph multiplicationWithPrimaryTensor:attn
                                                         secondaryTensor:cg->drop_ph
                                                                    name:@"attn_d"];
    cg->out_ph = [cg->graph matrixMultiplicationWithPrimaryTensor:attn_d
                                                   secondaryTensor:cg->v_ph
                                                              name:@"out"];
  }
  return cg;
}

CachedGraph* get_or_build(const GraphKey& key) {
  {
    std::lock_guard<std::mutex> lk(g_graph_cache_mutex);
    auto it = g_graph_cache.find(key);
    if (it != g_graph_cache.end()) return it->second;
  }
  CachedGraph* cg = build_graph(key);
  {
    std::lock_guard<std::mutex> lk(g_graph_cache_mutex);
    auto it = g_graph_cache.find(key);
    if (it != g_graph_cache.end()) {
      // lost the race — drop ours, keep theirs
      [cg->graph release];
      delete cg;
      return it->second;
    }
    g_graph_cache[key] = cg;
  }
  return cg;
}

MPSGraphTensorData* make_tensor_data(const torch::Tensor& t, uint32_t dtype_val) {
  id<MTLBuffer> buf = at::native::mps::getMTLBufferStorage(t);
  NSMutableArray<NSNumber*>* shape = [NSMutableArray arrayWithCapacity:t.dim()];
  for (int64_t d = 0; d < t.dim(); ++d) {
    [shape addObject:@(t.size(d))];
  }
  return [[[MPSGraphTensorData alloc] initWithMTLBuffer:buf
                                                  shape:shape
                                               dataType:(MPSDataType)dtype_val]
           autorelease];
}

torch::Tensor sdpa_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                           std::optional<torch::Tensor> mask_opt,
                           std::optional<torch::Tensor> dropout_mask_opt) {
  TORCH_CHECK(q.device().is_mps(), "q must be on MPS device");
  TORCH_CHECK(k.device().is_mps(), "k must be on MPS device");
  TORCH_CHECK(v.device().is_mps(), "v must be on MPS device");
  TORCH_CHECK(q.dim() == 4, "q must be [B, H, Lq, D]");
  TORCH_CHECK(k.dim() == 4, "k must be [B, H, Lkv, D]");
  TORCH_CHECK(v.dim() == 4, "v must be [B, H, Lkv, D]");

  q = q.contiguous();
  k = k.contiguous();
  v = v.contiguous();

  int64_t B = q.size(0), H = q.size(1), Lq = q.size(2), D = q.size(3);
  int64_t Lkv = k.size(2);
  TORCH_CHECK(k.size(0) == B && k.size(1) == H && k.size(3) == D,
              "k shape must match q batch/heads/dim");
  TORCH_CHECK(v.sizes() == k.sizes(), "v shape must match k shape");
  uint32_t dtype_val = dtype_to_mps(q.scalar_type());

  bool dropout = dropout_mask_opt.has_value();
  GraphKey key{dtype_val, B, H, Lq, Lkv, D, 0, 0, 0, dropout};

  std::optional<torch::Tensor> mask_cast;
  if (mask_opt.has_value()) {
    auto m = mask_opt.value();
    TORCH_CHECK(m.device().is_mps(), "mask must be on MPS device");
    TORCH_CHECK(m.dim() == 4, "mask must be [B_m, H_m, Lq, Lkv] after broadcast");
    TORCH_CHECK(m.size(2) == Lq && m.size(3) == Lkv, "mask spatial shape mismatch");
    if (m.scalar_type() != q.scalar_type()) m = m.to(q.scalar_type());
    m = m.contiguous();
    mask_cast = m;
    key.mask_kind = 1;
    key.mask_B = m.size(0);
    key.mask_H = m.size(1);
  }

  std::optional<torch::Tensor> drop_cast;
  if (dropout) {
    auto dm = dropout_mask_opt.value();
    TORCH_CHECK(dm.device().is_mps(), "dropout_mask must be on MPS device");
    TORCH_CHECK(dm.dim() == 4 && dm.size(0) == B && dm.size(1) == H &&
                dm.size(2) == Lq && dm.size(3) == Lkv,
                "dropout_mask must be [B, H, Lq, Lkv]");
    if (dm.scalar_type() != q.scalar_type()) dm = dm.to(q.scalar_type());
    dm = dm.contiguous();
    drop_cast = dm;
  }

  CachedGraph* cg = get_or_build(key);

  auto out = torch::empty({B, H, Lq, D}, q.options());

  // CRITICAL: wrap Obj-C object lifetimes in an explicit autoreleasepool. Without
  // this, every make_tensor_data() autorelease accumulates (no implicit pool drain
  // in a Python-driven loop) -> Metal side "other allocations" balloons across
  // training steps -> OOM. One pool per call is sufficient and cheap.
  @autoreleasepool {
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
      [NSMutableDictionary dictionaryWithCapacity:5];
    feeds[cg->q_ph] = make_tensor_data(q, dtype_val);
    feeds[cg->k_ph] = make_tensor_data(k, dtype_val);
    feeds[cg->v_ph] = make_tensor_data(v, dtype_val);
    if (cg->mask_ph != nil) {
      feeds[cg->mask_ph] = make_tensor_data(mask_cast.value(), dtype_val);
    }
    if (cg->drop_ph != nil) {
      feeds[cg->drop_ph] = make_tensor_data(drop_cast.value(), dtype_val);
    }
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      cg->out_ph: make_tensor_data(out, dtype_val)
    };

    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    at::native::mps::runMPSGraph(stream, cg->graph, feeds, results);
  }

  return out;
}

// -------------- Backward -----------------
//
// Same math as the pyobjc backend's MPSGraph-native backward:
//   attn_raw   = softmax(Q @ K^T * scale)
//   dV         = attn_raw^T @ grad_out
//   d_attn_raw = grad_out @ V^T
//   d_scores   = attn_raw * (d_attn_raw - reduce_sum(attn_raw * d_attn_raw, -1))
//   d_scores_s = d_scores * scale
//   dQ         = d_scores_s @ K
//   dK         = d_scores_s^T @ Q

struct CachedBwdGraph {
  MPSGraph* graph = nil;
  MPSGraphTensor* q_ph = nil;
  MPSGraphTensor* k_ph = nil;
  MPSGraphTensor* v_ph = nil;
  MPSGraphTensor* go_ph = nil;  // grad_out
  MPSGraphTensor* mask_ph = nil;  // nil if mask_kind=0
  MPSGraphTensor* drop_ph = nil;  // nil if dropout=false
  MPSGraphTensor* dQ_ph = nil;
  MPSGraphTensor* dK_ph = nil;
  MPSGraphTensor* dV_ph = nil;
};

std::unordered_map<GraphKey, CachedBwdGraph*, GraphKeyHash> g_bwd_graph_cache;
std::mutex g_bwd_graph_cache_mutex;

CachedBwdGraph* build_bwd_graph(const GraphKey& key) {
  CachedBwdGraph* cg = new CachedBwdGraph();
  cg->graph = [[MPSGraph alloc] init];
  NSArray<NSNumber*>* qShape = @[@(key.B), @(key.H), @(key.Lq), @(key.D)];
  NSArray<NSNumber*>* kShape = @[@(key.B), @(key.H), @(key.Lkv), @(key.D)];
  MPSDataType dt = (MPSDataType)key.dtype_val;

  cg->q_ph = [cg->graph placeholderWithShape:qShape dataType:dt name:@"q"];
  cg->k_ph = [cg->graph placeholderWithShape:kShape dataType:dt name:@"k"];
  cg->v_ph = [cg->graph placeholderWithShape:kShape dataType:dt name:@"v"];
  cg->go_ph = [cg->graph placeholderWithShape:qShape dataType:dt name:@"grad_out"];

  if (key.mask_kind != 0) {
    NSArray<NSNumber*>* maskShape = @[@(key.mask_B), @(key.mask_H),
                                       @(key.Lq), @(key.Lkv)];
    cg->mask_ph = [cg->graph placeholderWithShape:maskShape dataType:dt name:@"mask"];
  }
  if (key.dropout) {
    NSArray<NSNumber*>* dropShape = @[@(key.B), @(key.H), @(key.Lq), @(key.Lkv)];
    cg->drop_ph = [cg->graph placeholderWithShape:dropShape dataType:dt name:@"drop_mask"];
  }

  float scale = 1.0f / sqrtf((float)key.D);
  MPSGraphTensor* scale_t = [cg->graph constantWithScalar:scale dataType:dt];

  // scores = Q @ K^T * scale  (+ mask)
  MPSGraphTensor* kT = [cg->graph transposeTensor:cg->k_ph
                                       dimension:-1 withDimension:-2 name:@"kT"];
  MPSGraphTensor* qk = [cg->graph matrixMultiplicationWithPrimaryTensor:cg->q_ph
                                                        secondaryTensor:kT name:@"qk"];
  MPSGraphTensor* scores = [cg->graph multiplicationWithPrimaryTensor:qk
                                                       secondaryTensor:scale_t name:@"scores"];
  if (cg->mask_ph != nil) {
    scores = [cg->graph additionWithPrimaryTensor:scores
                                 secondaryTensor:cg->mask_ph name:@"scores_m"];
  }

  // attn_raw = softmax(scores, -1)  (pre-dropout)
  MPSGraphTensor* attn_raw = [cg->graph softMaxWithTensor:scores axis:-1 name:@"attn_raw"];

  // Forward used attn_used = attn_raw (* dropout_mask when present).
  MPSGraphTensor* attn_used;
  if (cg->drop_ph != nil) {
    attn_used = [cg->graph multiplicationWithPrimaryTensor:attn_raw
                                           secondaryTensor:cg->drop_ph
                                                      name:@"attn_used"];
  } else {
    attn_used = attn_raw;
  }

  // dV = attn_used^T @ grad_out
  MPSGraphTensor* attn_T = [cg->graph transposeTensor:attn_used
                                            dimension:-1 withDimension:-2 name:@"attn_T"];
  cg->dV_ph = [cg->graph matrixMultiplicationWithPrimaryTensor:attn_T
                                                 secondaryTensor:cg->go_ph name:@"dV"];

  // d_attn_used = grad_out @ V^T
  MPSGraphTensor* vT = [cg->graph transposeTensor:cg->v_ph
                                        dimension:-1 withDimension:-2 name:@"vT"];
  MPSGraphTensor* d_attn_used = [cg->graph matrixMultiplicationWithPrimaryTensor:cg->go_ph
                                                                    secondaryTensor:vT
                                                                               name:@"d_attn_used"];

  // Chain through dropout: d_attn_raw = d_attn_used * dropout_mask
  MPSGraphTensor* d_attn_raw;
  if (cg->drop_ph != nil) {
    d_attn_raw = [cg->graph multiplicationWithPrimaryTensor:d_attn_used
                                             secondaryTensor:cg->drop_ph
                                                        name:@"d_attn_raw"];
  } else {
    d_attn_raw = d_attn_used;
  }

  // Softmax bwd: d_scores = attn_raw * (d_attn_raw - reduce_sum(attn_raw * d_attn_raw, -1))
  MPSGraphTensor* ad = [cg->graph multiplicationWithPrimaryTensor:attn_raw
                                                   secondaryTensor:d_attn_raw name:@"ad"];
  MPSGraphTensor* sum_ad = [cg->graph reductionSumWithTensor:ad axis:-1 name:@"sum_ad"];
  MPSGraphTensor* diff = [cg->graph subtractionWithPrimaryTensor:d_attn_raw
                                                  secondaryTensor:sum_ad name:@"diff"];
  MPSGraphTensor* d_scores = [cg->graph multiplicationWithPrimaryTensor:attn_raw
                                                         secondaryTensor:diff name:@"d_scores"];
  MPSGraphTensor* d_scores_s = [cg->graph multiplicationWithPrimaryTensor:d_scores
                                                           secondaryTensor:scale_t name:@"d_scores_s"];

  // dQ = d_scores_s @ K
  cg->dQ_ph = [cg->graph matrixMultiplicationWithPrimaryTensor:d_scores_s
                                                 secondaryTensor:cg->k_ph name:@"dQ"];
  // dK = d_scores_s^T @ Q
  MPSGraphTensor* dsT = [cg->graph transposeTensor:d_scores_s
                                         dimension:-1 withDimension:-2 name:@"dsT"];
  cg->dK_ph = [cg->graph matrixMultiplicationWithPrimaryTensor:dsT
                                                 secondaryTensor:cg->q_ph name:@"dK"];
  return cg;
}

CachedBwdGraph* get_or_build_bwd(const GraphKey& key) {
  {
    std::lock_guard<std::mutex> lk(g_bwd_graph_cache_mutex);
    auto it = g_bwd_graph_cache.find(key);
    if (it != g_bwd_graph_cache.end()) return it->second;
  }
  CachedBwdGraph* cg = build_bwd_graph(key);
  {
    std::lock_guard<std::mutex> lk(g_bwd_graph_cache_mutex);
    auto it = g_bwd_graph_cache.find(key);
    if (it != g_bwd_graph_cache.end()) {
      [cg->graph release];
      delete cg;
      return it->second;
    }
    g_bwd_graph_cache[key] = cg;
  }
  return cg;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sdpa_backward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
              torch::Tensor grad_out,
              std::optional<torch::Tensor> mask_opt,
              std::optional<torch::Tensor> dropout_mask_opt) {
  TORCH_CHECK(q.device().is_mps(), "q must be MPS");
  TORCH_CHECK(grad_out.device().is_mps(), "grad_out must be MPS");
  q = q.contiguous();
  k = k.contiguous();
  v = v.contiguous();
  grad_out = grad_out.contiguous();

  int64_t B = q.size(0), H = q.size(1), Lq = q.size(2), D = q.size(3);
  int64_t Lkv = k.size(2);
  uint32_t dtype_val = dtype_to_mps(q.scalar_type());

  bool dropout = dropout_mask_opt.has_value();
  GraphKey key{dtype_val, B, H, Lq, Lkv, D, 0, 0, 0, dropout};
  std::optional<torch::Tensor> mask_cast;
  if (mask_opt.has_value()) {
    auto m = mask_opt.value();
    if (m.scalar_type() != q.scalar_type()) m = m.to(q.scalar_type());
    m = m.contiguous();
    mask_cast = m;
    key.mask_kind = 1;
    key.mask_B = m.size(0);
    key.mask_H = m.size(1);
  }
  std::optional<torch::Tensor> drop_cast;
  if (dropout) {
    auto dm = dropout_mask_opt.value();
    if (dm.scalar_type() != q.scalar_type()) dm = dm.to(q.scalar_type());
    dm = dm.contiguous();
    drop_cast = dm;
  }

  CachedBwdGraph* cg = get_or_build_bwd(key);

  auto dQ = torch::empty({B, H, Lq, D}, q.options());
  auto dK = torch::empty({B, H, Lkv, D}, q.options());
  auto dV = torch::empty({B, H, Lkv, D}, q.options());

  // Same autoreleasepool treatment as forward — backward is called just as often,
  // same leak pattern otherwise.
  @autoreleasepool {
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
      [NSMutableDictionary dictionaryWithCapacity:6];
    feeds[cg->q_ph] = make_tensor_data(q, dtype_val);
    feeds[cg->k_ph] = make_tensor_data(k, dtype_val);
    feeds[cg->v_ph] = make_tensor_data(v, dtype_val);
    feeds[cg->go_ph] = make_tensor_data(grad_out, dtype_val);
    if (cg->mask_ph != nil) {
      feeds[cg->mask_ph] = make_tensor_data(mask_cast.value(), dtype_val);
    }
    if (cg->drop_ph != nil) {
      feeds[cg->drop_ph] = make_tensor_data(drop_cast.value(), dtype_val);
    }
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      cg->dQ_ph: make_tensor_data(dQ, dtype_val),
      cg->dK_ph: make_tensor_data(dK, dtype_val),
      cg->dV_ph: make_tensor_data(dV, dtype_val),
    };

    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    at::native::mps::runMPSGraph(stream, cg->graph, feeds, results);
  }
  return std::make_tuple(dQ, dK, dV);
}

int64_t graph_cache_size() {
  std::lock_guard<std::mutex> lk(g_graph_cache_mutex);
  return (int64_t)g_graph_cache.size();
}

int64_t bwd_graph_cache_size() {
  std::lock_guard<std::mutex> lk(g_bwd_graph_cache_mutex);
  return (int64_t)g_bwd_graph_cache.size();
}

void clear_graph_cache() {
  {
    std::lock_guard<std::mutex> lk(g_graph_cache_mutex);
    for (auto& kv : g_graph_cache) {
      [kv.second->graph release];
      delete kv.second;
    }
    g_graph_cache.clear();
  }
  {
    std::lock_guard<std::mutex> lk(g_bwd_graph_cache_mutex);
    for (auto& kv : g_bwd_graph_cache) {
      [kv.second->graph release];
      delete kv.second;
    }
    g_bwd_graph_cache.clear();
  }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sdpa_forward", &sdpa_forward,
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("mask") = py::none(),
        py::arg("dropout_mask") = py::none(),
        "Zero-copy MPSGraph SDPA forward. If dropout_mask is provided, uses an "
        "unfused graph (scores→softmax→*mask→matmul V) instead of Apple's fused op.");
  m.def("sdpa_backward", &sdpa_backward,
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("grad_out"),
        py::arg("mask") = py::none(),
        py::arg("dropout_mask") = py::none(),
        "Zero-copy MPSGraph SDPA backward; returns (dQ, dK, dV). dropout_mask "
        "chains correctly through the softmax backward.");
  m.def("graph_cache_size", &graph_cache_size);
  m.def("bwd_graph_cache_size", &bwd_graph_cache_size);
  m.def("clear_graph_cache", &clear_graph_cache);
}
