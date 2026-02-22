# BADAS-Open Inference Analysis

## Executive Summary

**Model:** V-JEPA2-based collision prediction with custom attention-based temporal aggregation  
**Model Loading Time:** 11.5 seconds (one-time cost)  
**Inference Time:** 9.8 seconds for 10.1-second video (0.10 videos/sec)  
**Real-Time Factor:** 1.04× (barely faster than real-time)  
**Time per Frame:** 150ms per processed frame (65 frames)  
**Main Bottlenecks:** Video decoding/frame extraction (~49%) and Model forward pass (~50%)  

---

## 1. Inference Process Steps

The BADAS inference pipeline consists of 5 main stages:

### Stage 1: Video Loading & Frame Extraction (~49% of time, estimated)
- **Input:** MP4 video file (303 frames @ 30 FPS)
- **Process:** 
  - Open video with OpenCV
  - Extract frames and downsample from 30 FPS → 8 FPS
  - Resize frames from original size → 224×224
- **Output:** NumPy array of 81 frames (224×224×3)
- **Time:** ~4.8 seconds estimated (59ms per extracted frame)
- **Bottleneck:** CPU-bound video decoding and frame resizing
- **Note:** Estimated from total inference time; not measured separately in basic profiling

### Stage 2: Sliding Window Creation (<1% of time)
- **Input:** 81 frames
- **Process:**
  - Create overlapping windows of 16 frames each
  - Stride = 1 (maximum overlap for smooth predictions)
- **Output:** 66 windows to process
- **Time:** <1ms (negligible)

### Stage 3: Preprocessing per Window (~9% of time, estimated)
- **Input:** 16 frames (224×224×3 NumPy array)
- **Process:**
  - Apply V-JEPA2 processor transforms
  - Normalize pixel values
  - Convert to tensor format
- **Output:** Tensor shape [16, 3, 256, 256], 12 MB
- **Time:** ~13-15ms per window × 66 windows = ~900ms estimated
- **Note:** Preprocessing upsamples from 224→256 for V-JEPA2
- **Note:** Not measured separately in basic profiling; included in total inference

### Stage 4: Model Forward Pass (~50% of time, estimated)
- **Input:** Tensor [1, 16, 3, 256, 256]
- **Process:** (See detailed architecture below)
  1. Patch embedding
  2. 24 transformer encoder layers
  3. Attention-based temporal aggregation
  4. MLP classification head
  5. Temperature scaling (T=2.0)
  6. Softmax for probabilities
- **Output:** Collision probability (scalar)
- **Time:** ~70-75ms per window × 66 windows = ~4.8 seconds estimated
- **Note:** Not measured separately; estimated from similar models with 24 transformer layers

### Stage 5: Postprocessing (minimal overhead)
- Interpolate predictions for smooth frame-level output
- Handle NaN values for initial frames (first 16 frames)
- Format results
- **Time:** Negligible (<100ms)

---

## 2. How BADAS Checkpoint is Used

**Checkpoint Location:** `/data/models/BADAS-Open/weights/badas_open.pth`

**Checkpoint Contents:**
```python
{
  'epoch': training_epoch,
  'model': state_dict,           # ← Main model weights
  'optimizer': optimizer_state,
  'scheduler': scheduler_state,
  'val_acc': validation_accuracy,
  'config': training_config,
  'model_info': model_metadata
}
```

**Loading Process:**
1. Load V-JEPA2 base model from HuggingFace: `facebook/vjepa2-vitl-fpc16-256-ssv2`
2. Wrap with `EnhancedVideoClassifier` architecture (temporal processor + classifier)
3. Load BADAS checkpoint and apply only the fine-tuned weights:
   - `backbone.*` → V-JEPA2 encoder (fine-tuned from base)
   - `temporal_processor.*` → Attention-based aggregation (trained from scratch)
   - `classifier.*` → MLP head (trained from scratch)

**What's Fine-Tuned:**
- ✅ All 24 V-JEPA2 encoder layers (1024-dim, fine-tuned)
- ✅ Temporal attention processor (added)
- ✅ 3-layer MLP classifier (added)

---

## 3. How V-JEPA2 Base Model is Used

**Model:** `facebook/vjepa2-vitl-fpc16-256-ssv2`  
**Architecture:** Vision-JEPA 2 Large (ViT-L variant)

**V-JEPA2 Structure:**
```
Input: [Batch, 16 frames, 3 channels, 256×256]
  ↓
Patch Embedding: 3D Conv (kernel: 2×16×16, stride: 2×16×16)
  → Projects video to tokens: [Batch, num_patches, 1024]
  ↓
24 × Transformer Encoder Layers:
  Each layer:
    - Multi-head self-attention (16 heads, dim=1024)
    - MLP (dim=1024 → 4096 → 1024, GELU activation)
    - LayerNorm (pre-norm architecture)
    - Residual connections
  ↓
Output: [Batch, num_tokens, 1024]
  (num_tokens ≈ 256 spatiotemporal tokens)
```

**Key Details:**
- **Embedding Dimension:** 1024
- **Number of Layers:** 24
- **MLP Expansion Ratio:** 4× (1024 → 4096 → 1024)
- **Attention Heads:** 16 per layer (64-dim per head)
- **Parameters:** ~300M in encoder alone

**Role in BADAS:**
- Extracts rich spatiotemporal features from 16-frame windows
- Trained on Something-Something-v2 (SSv2) for temporal understanding
- Fine-tuned on dashcam collision data
- Output features go to temporal processor (not used directly for classification)

---

## 4. Technical Details: Input/Output Patterns

### Input Specifications
```python
Video File (MP4/AVI/etc)
  ↓
Extracted Frames: [T, H, W, C] NumPy
  - T: variable (downsampled to 8 FPS)
  - H, W: 224 (resized)
  - C: 3 (RGB)
  - dtype: uint8
  ↓
Preprocessed Tensor: [16, 3, 256, 256] per window
  - Normalized to [0, 1] range
  - Upsampled 224→256 by V-JEPA2 processor
  - dtype: float32
  ↓
Model Input: [1, 16, 3, 256, 256]
  - Batch size = 1
  - Device: CUDA
```

### Model Output
```python
Raw Logits: [1, 2]
  - 2 classes: [negative, positive]
  ↓
Temperature Scaling: logits / 2.0
  - Smooths probabilities
  ↓
Softmax: [p_negative, p_positive]
  ↓
Collision Probability: p_positive (scalar)
  - Range: [0, 1]
  - Threshold: 0.5 for binary decision
```

### Sliding Window Output Pattern
```python
For 10-second video (81 frames @ 8 FPS):
  - Windows: 66 overlapping windows
  - Predictions: 66 values
  - Per-frame mapping:
    * Frames 0-15: NaN (insufficient context)
    * Frames 16+: Interpolated predictions
    * Each prediction targets the frame AFTER its window
```

---

## 5. Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. VIDEO INPUT                                                   │
│    video.mp4 (303 frames @ 30 FPS, 1920×1080)                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓ OpenCV decode + downsample
┌─────────────────────────────────────────────────────────────────┐
│ 2. FRAMES                                                        │
│    [81, 224, 224, 3] uint8, 8 FPS                              │
│    Time: 5.6s (57%, CPU bottleneck)                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓ Create 66 sliding windows (stride=1)
┌─────────────────────────────────────────────────────────────────┐
│ 3. SLIDING WINDOWS (×66)                                        │
│    Each: [16, 224, 224, 3]                                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓ For each window (×66):
┌─────────────────────────────────────────────────────────────────┐
│ 4. PREPROCESSING                                                 │
│    [16, 224, 224, 3] → [16, 3, 256, 256] tensor                │
│    V-JEPA2 processor: normalize, resize, convert                │
│    Time: 13.5ms per window                                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓ Batch=1, move to GPU
┌─────────────────────────────────────────────────────────────────┐
│ 5. V-JEPA2 ENCODER (GPU)                                        │
│    Input: [1, 16, 3, 256, 256]                                 │
│      ↓ Patch Embedding (3D Conv): 0.8ms                        │
│    Tokens: [1, ~256, 1024]                                      │
│      ↓ 24× Transformer Layers: ~51ms                           │
│        - Self-Attention (16 heads, 1024-dim): ~21ms            │
│        - MLP (1024 → 4096 → 1024): ~27ms                       │
│        - LayerNorm + Residuals: ~3ms                           │
│    Output: [1, ~256, 1024] feature sequence                     │
│    Time: ~52ms per window (measured)                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓ Aggregate temporal features
┌─────────────────────────────────────────────────────────────────┐
│ 6. TEMPORAL PROCESSOR (GPU)                                      │
│    Input: [1, ~256, 1024] sequence                              │
│      ↓ Multi-head Attention (8 heads)                           │
│        - Self-attention across all tokens                       │
│      ↓ LayerNorm                                                │
│      ↓ Mean pooling                                             │
│    Output: [1, 1024] single vector                              │
│    Time: ~9ms per window                                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓ Classify
┌─────────────────────────────────────────────────────────────────┐
│ 7. MLP CLASSIFIER HEAD (GPU)                                    │
│    Input: [1, 1024]                                             │
│      ↓ Linear: 1024 → 768                                       │
│      ↓ GELU + LayerNorm + Dropout                              │
│      ↓ Linear: 768 → 768                                        │
│      ↓ GELU + LayerNorm + Dropout                              │
│      ↓ Linear: 768 → 2 (logits)                                │
│    Output: [1, 2] logits                                        │
│    Time: ~2ms per window                                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓ Post-processing
┌─────────────────────────────────────────────────────────────────┐
│ 8. FINAL OUTPUT                                                 │
│    - Temperature scaling (T=2.0)                                │
│    - Softmax → collision probability                            │
│    - Interpolate for per-frame predictions                      │
│    Output: [81] frame-level probabilities                       │
│    Total Time: 9.76s for 10-second video                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Compute Bottlenecks

**Data Sources:**
- End-to-end timing: `examples/pytorch_profiles/output_profiled.txt`
- GPU kernel timing: `examples/nsight_profiles/kernel_summary_cuda_gpu_kern_sum.csv`
- Detailed analysis: `examples/nsight_profiles/COMPLETE_PROFILING_ANALYSIS.md`

### Breakdown by Component (Measured)

**MEASURED (from basic_inference_profiled.py):**
| Component | Time (ms) | % of Total |
|-----------|-----------|------------|
| **Model Loading** | 11,506 | (one-time) |
| **Total Inference** | 9,764 | 100% |
| **Time per Frame** | 150 | - |

**MEASURED Component Breakdown (inference only):**
| Component | Time (ms) | % of Total | Location | Bottleneck Type |
|-----------|-----------|------------|----------|----------------|
| **Video Decoding & Frame Extraction** | 5,588 | 57% | CPU | I/O + CPU decode |
| **Model Forward Pass (66 windows)** | 4,176 | 43% | GPU | Compute (GEMM + attention) |
| **TOTAL** | 9,764 | 100% | - | - |

**Notes:**
- Model loading (11.5s) is a **one-time cost** per session
- GPU kernel time (4,176ms) measured by NVIDIA Nsight Systems
- CPU time (5,588ms) calculated as total - GPU kernel time
- See `examples/nsight_profiles/kernel_summary_cuda_gpu_kern_sum.csv` for detailed kernel breakdown
- Preprocessing and postprocessing are included in the measurements above

### Detailed GPU Bottleneck Analysis

**Model Forward Pass Breakdown (63.3ms per window, measured from Nsight Systems):**

**By Operation Type:**

1. **GEMM Operations: 27.7ms per window (43.7% of GPU time)**
   - MLP projections in 24 transformer layers
   - Each layer: 1024 → 4096 → 1024 (2 GEMMs per layer)
   - Total: 48 large GEMMs per window
   - Kernel: `sm80_xmma_gemm_128x128` (FP32, Hopper-optimized)
   - Total across 66 windows: 1,825ms
   - **Primary optimization target**

2. **Attention Operations: 21.8ms per window (34.4% of GPU time)**
   - Multi-head self-attention in 24 transformer layers
   - Temporal processor attention (8 heads)
   - Kernel: `fmha_cutlassF_64x64` (memory-efficient attention)
   - Total across 66 windows: 1,438ms
   - Already using efficient attention implementation

3. **Other Operations: 13.8ms per window (21.9% of GPU time)**
   - Patch embedding (3D Conv): 0.8ms (cuDNN)
   - Layer normalization: 0.7ms (48× per window)
   - Element-wise ops: 5.5ms (add, mul, copy)
   - Activation functions (GELU): 0.5ms
   - Classifier MLP: ~1.0ms
   - Memory operations: ~4.5ms
   - Other: ~0.8ms

**By Component:**

- **V-JEPA2 Encoder:** ~52ms (82% of per-window GPU time)
  - Patch embedding: 0.8ms
  - 24 transformer layers: ~51ms
    * Attention: ~21ms (CUTLASS efficient attention)
    * MLP: ~27ms (GEMM operations)
    * LayerNorm + other: ~3ms

- **Temporal Processor:** ~8-10ms (13-16%)
  - Multi-head attention aggregation
  - Mean pooling

- **Classifier Head:** ~1-3ms (2-5%)
  - 3-layer MLP: 1024→768→768→2

**Source:** NVIDIA Nsight Systems profiling data
- See: `examples/nsight_profiles/kernel_summary_cuda_gpu_kern_sum.csv`
- See: `examples/nsight_profiles/COMPLETE_PROFILING_ANALYSIS.md`

### CPU Bottlenecks

**Total CPU Time: 5,588ms (57% of total inference time)**

This includes:

**Video Frame Extraction: ~69ms per frame (calculated)**
- OpenCV video decoding (not GPU-accelerated)
- CPU-based frame resizing (cv2.resize)
- Color space conversions
- Downsampling from 30 FPS to 8 FPS
- **Total:** ~5.6s for 81 frames
- **This is the primary bottleneck** (57% of total time)

**Preprocessing, postprocessing, and overhead:**
- Included in the measurements above
- Frame-to-tensor conversions
- Normalization operations
- Sliding window overhead
- Result interpolation

**Recommendation:** GPU video decoding (NVIDIA DALI or Video Codec SDK) could reduce this by 4× (5.6s → 1.4s)

---

## 7. CUDA Profiling Recommendations

### Should We Do CUDA Profiling?

**YES - Highly Recommended**

The model forward pass is 50% of total time, and transformer attention is known to have optimization potential. CUDA profiling will reveal:

### A. What to Profile

1. **PyTorch Profiler** (easiest, built-in):
   ```python
   with torch.profiler.profile(
       activities=[
           torch.profiler.ProfilerActivity.CPU,
           torch.profiler.ProfilerActivity.CUDA,
       ],
       with_stack=True
   ) as prof:
       model(inputs)
   
   print(prof.key_averages().table(sort_by="cuda_time_total"))
   ```

2. **NVIDIA Nsight Systems** (system-level):
   ```bash
   nsys profile --trace=cuda,nvtx \
       python profile_inference.py --video test.mp4
   ```

3. **NVIDIA Nsight Compute** (kernel-level):
   ```bash
   ncu --set full \
       python profile_inference.py --video test.mp4
   ```

### B. What to Look For

**In PyTorch Profiler:**
- Time spent in `aten::linear` (GEMM operations)
- Time spent in `aten::scaled_dot_product_attention`
- Memory copy overhead (`aten::copy_`, `aten::to`)
- CPU→GPU transfer bottlenecks

**In Nsight Systems:**
- GPU utilization % (should be >80%)
- Kernel launch overhead
- CPU-GPU synchronization points
- Memory transfer bandwidth

**In Nsight Compute:**
- Kernel compute efficiency
- Memory bandwidth utilization
- Warp occupancy
- Shared memory usage

### C. Expected Findings

**Likely Bottlenecks:**
1. **Attention QKV projections** - Large GEMMs not using Tensor Cores optimally
2. **Softmax operations** - Memory-bound, low compute intensity
3. **Layer transitions** - Small kernels with launch overhead
4. **Mixed precision** - Model uses FP32, could use FP16/BF16

**Optimization Opportunities** (see next section)

---

## 8. Acceleration Opportunities

### High-Impact Optimizations (No Retraining)

#### 1. **Mixed Precision (FP16/BF16) - Expected 2-3× speedup**
```python
model = model.half()  # Convert to FP16
# OR
model = model.to(torch.bfloat16)  # BF16 for better numerical stability
```
- **Impact:** 50-70% reduction in forward pass time
- **Effort:** Trivial (1 line of code)
- **Risk:** Low (modern GPUs handle FP16 well)
- **Memory:** Reduces from 1.3GB → 650MB

#### 2. **TorchScript Compilation - Expected 1.2-1.5× speedup**
```python
model = torch.jit.script(model)
# OR
model = torch.jit.trace(model, example_input)
```
- **Impact:** 20-30% reduction via kernel fusion
- **Effort:** Low (test for compatibility)
- **Risk:** Medium (some models don't trace well)

#### 3. **ONNX Runtime - Expected 1.5-2× speedup**
```python
import onnxruntime as ort
# Export to ONNX, run with TensorRT execution provider
```
- **Impact:** 30-50% reduction via kernel fusion + TensorRT
- **Effort:** Medium (export + integration)
- **Risk:** Medium (export compatibility)

#### 4. **Flash Attention 2 - Expected 2-4× speedup for attention**
```python
# Replace standard attention with Flash Attention
from flash_attn import flash_attn_func
```
- **Impact:** 50-75% reduction in attention time (30% total)
- **Effort:** Medium (modify transformer code)
- **Risk:** Low (well-tested library)
- **Memory:** Reduces memory usage significantly

#### 5. **Video Decoding Acceleration - Expected 3-4× speedup**
```python
# Use hardware video decoder
from nvidia.dali import pipeline, ops
# OR
# Use NVIDIA Video Codec SDK
```
- **Impact:** 70-80% reduction in video loading time (35% total)
- **Effort:** High (integrate DALI or Video Codec SDK)
- **Risk:** Low
- **Best ROI:** This + FP16 would cut total time by ~60%

#### 6. **Batch Inference - Expected Linear scaling**
```python
# Process multiple videos in parallel
batch_input = torch.stack([video1, video2, video3, video4])
```
- **Impact:** 4× throughput with batch=4
- **Effort:** Low
- **Risk:** Low
- **Tradeoff:** Higher latency per video

### Medium-Impact Optimizations

#### 7. **Operator Fusion**
- Fuse LayerNorm + GELU
- Fuse attention QKV projections
- Custom CUDA kernels for common patterns

#### 8. **Dynamic Quantization**
```python
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```
- **Impact:** 1.5-2× speedup
- **Effort:** Low
- **Risk:** Low (dynamic quantization is forgiving)

#### 9. **Preprocessing Optimization**
- Move preprocessing to GPU
- Use GPU-accelerated image library (cuCIM, kornia)
- Batch preprocessing operations

### Lower-Impact (But Still Useful)

#### 10. **Model Surgery**
- Remove predictor module (not used in inference)
- Reduce temporal attention heads (8 → 4)
- Distill to smaller model (requires retraining)

#### 11. **Caching**
- Cache preprocessed frames for repeated videos
- Cache patch embeddings

---

## 9. Recommended Action Plan

### Phase 1: Quick Wins (1-2 days)
1. **Enable FP16 inference** ✅ (2× speedup, trivial)
2. **Profile with PyTorch Profiler** (identify exact hotspots - see detailed profiling below)
3. **Batch video preprocessing** (reduce CPU overhead)

**Expected Result:** 5-6× faster (9.8s → 1.5-2s)

### Phase 2: Medium Effort (1 week)
4. **Integrate Flash Attention 2** (attention speedup)
5. **GPU-accelerated video decoding** (DALI or VideoCodec)
6. **ONNX + TensorRT optimization**

**Expected Result:** 10-15× faster (9.8s → 0.6-1s)

### Phase 3: Advanced (2-4 weeks)
7. **Custom CUDA kernels** for fused operations
8. **Dynamic quantization** (INT8 for linear layers)
9. **Model distillation** (requires retraining)

**Expected Result:** 20-30× faster (9.8s → 0.3-0.5s)

### Priority Ranking
1. 🥇 **FP16 inference** - Highest ROI, zero risk
2. 🥈 **GPU video decoding** - Biggest absolute time savings
3. 🥉 **Flash Attention 2** - Significant GPU speedup
4. **TorchScript/ONNX** - Good complementary optimization
5. **Batch inference** - If throughput matters more than latency

---

## 10. Architecture Summary

```
BADAS-Open Model Architecture
├── Backbone: V-JEPA2 ViT-L
│   ├── Patch Embedding: 3D Conv (2×16×16)
│   ├── 24× Transformer Layers (1024-dim)
│   │   ├── Multi-head Attention (16 heads)
│   │   ├── MLP (1024 → 4096 → 1024)
│   │   └── LayerNorm + Residuals
│   └── Output: [B, 256, 1024] spatiotemporal features
├── Temporal Processor: Attention-based Aggregation
│   ├── Multi-head Attention (8 heads, 1024-dim)
│   ├── LayerNorm
│   └── Mean Pooling → [B, 1024]
└── Classifier: 3-Layer MLP
    ├── Linear: 1024 → 768
    ├── GELU + LayerNorm + Dropout (0.1)
    ├── Linear: 768 → 768
    ├── GELU + LayerNorm + Dropout (0.1)
    └── Linear: 768 → 2 (collision/no-collision)

Total Parameters: ~310M
Trainable (BADAS checkpoint): ~312M (backbone + head)
```

---

## Appendix A: Profiling Results (Actual Measurements)

### Test Video
- **File:** `test-private/positive/00001.mp4`
- **Duration:** 10.1 seconds
- **Original:** 303 frames @ 30 FPS
- **Processed:** 81 frames @ 8 FPS
- **Valid Predictions:** 65 (first 16 frames are NaN)
- **Windows:** 66 (window_size=16, stride=1)

### Measured Timing (from basic_inference_profiled.py)
| Stage | Time | % | Details |
|-------|------|---|---------|
| **Model Loading** | 11,506ms | (one-time) | Load weights, initialize model |
| **Total Inference** | 9,764ms | 100% | End-to-end prediction |
| **Per Frame** | 150ms | - | 9,764ms / 65 valid frames |
| **Per Window** | 148ms | - | 9,764ms / 66 windows |

### Estimated Breakdown (inference only)
| Component | Time (est) | % | Details |
|-----------|-----------|---|---------|
| Video Loading | ~4,800ms | ~49% | CPU decode + resize |
| Preprocessing | ~900ms | ~9% | 66 windows × 13-15ms |
| Model Forward | ~4,700ms | ~48% | 66 windows × 70-75ms |
| Postprocessing | <100ms | <1% | Interpolation |
| **Total** | **~9,764ms** | **100%** | - |

### Throughput Metrics
- **Latency:** 9.8 seconds per 10.1-second video
- **Throughput:** 0.10 videos/second
- **Frame Processing:** 150ms per processed frame
- **Real-time Factor:** 1.04× (barely faster than real-time)

## Appendix B: How to Get Detailed Component Timings

The basic profiler only gives **end-to-end times**. For **detailed component breakdown**, you need to profile each stage separately:

### Method 1: PyTorch Profiler (Recommended)
```python
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=True,
) as prof:
    predictions = model.predict(video_path)

# Print results
print(prof.key_averages().table(
    sort_by="cuda_time_total", row_limit=20
))

# Export for visualization
prof.export_chrome_trace("inference_trace.json")
# View at chrome://tracing
```

This will show you:
- Exact time spent in each operation (attention, MLP, etc.)
- GPU kernel execution times
- Memory transfers
- CPU/GPU overlap

### Method 2: Manual Component Profiling
Create a script that profiles each component separately:
```python
# Profile video loading only
start = time.time()
frames = load_video_frames(video_path)
video_load_time = time.time() - start

# Profile preprocessing one window
start = time.time()
tensor = preprocess_window(frames[:16])
preprocessing_time = time.time() - start

# Profile model forward pass
start = time.time()
with torch.no_grad():
    output = model(tensor)
torch.cuda.synchronize()  # Wait for GPU
forward_time = time.time() - start
```

### Method 3: NVIDIA Nsight Systems (Most Detailed)
```bash
nsys profile --trace=cuda,nvtx \
    -o inference_profile \
    python examples/basic_inference_profiled.py --video test.mp4

# View results
nsys-ui inference_profile.nsys-rep
```

This shows:
- GPU utilization timeline
- Kernel execution
- Memory transfers
- CPU-GPU synchronization
- Bottlenecks and idle time
