# BADAS Profiling Results - Quick Navigation

**Date:** February 3, 2026  
**Test Video:** `/data/karthik_data/badas_data/test-private/positive/00001.mp4`  
**GPU:** NVIDIA GH100 (GPU 4)

---

## 📊 Main Summary Document

**👉 START HERE:** [`COMPLETE_PROFILING_ANALYSIS.md`](./COMPLETE_PROFILING_ANALYSIS.md)

This is the comprehensive profiling report containing:
- Executive summary with performance metrics
- Detailed model architecture analysis
- Complete kernel-level breakdown
- Bottleneck identification with root causes
- Optimization recommendations with implementation code
- Roadmap with expected speedups (3.1× → 10× faster)

**File size:** 23 KB (human-readable)

---

## 🔬 Raw Profiling Data

### Nsight Systems (System-Level GPU Timeline)
- **`badas_nsys_profile.nsys-rep`** (41 MB) - Main report file
  - View with: `nsys-ui badas_nsys_profile.nsys-rep`
  - Shows: GPU timeline, kernel launches, memory transfers, CPU-GPU sync
  
- **`badas_nsys_profile.sqlite`** (535 MB) - SQLite database
  - Query with: `sqlite3 badas_nsys_profile.sqlite`
  - Contains: All raw profiling data, queryable

- **`kernel_summary_cuda_gpu_kern_sum.csv`** - Kernel statistics
  - Top kernels ranked by time
  - Easy to import into Excel/pandas

### Nsight Compute (Kernel-Level Details)
- **`badas_ncu_profile.ncu-rep`** (44 MB) - Detailed kernel metrics
  - View with: `ncu-ui badas_ncu_profile.ncu-rep`
  - Shows: SM utilization, memory throughput, warp states, bottlenecks

### PyTorch Profiler
- **`../output_pytorch_profiled.txt`** - Full PyTorch profiling output
- **`../PYTORCH_PROFILING_SUMMARY.txt`** - Condensed summary
- **`../inference_trace.json`** - Chrome trace (view at `chrome://tracing`)

### Logs
- **`full_profile_run2.txt`** - Complete profiling run output
- **`full_profile_complete_log.txt`** - Backup of full log

---

## 🎯 Key Findings (TL;DR)

### Current Performance
```
Total Inference Time: 9.8 seconds
├─ Video Decoding (CPU): 4.8s (49%)
├─ GPU Computation:      4.2s (43%)
└─ Memory Transfers:     0.6s ( 6%)

Real-time Factor: 1.04× (barely faster than real-time)
```

### Main Bottleneck
```
GPU Compute Breakdown:
├─ GEMM (Matrix Multiply):    1,825ms (43.7%)  ← FP32, optimize to FP16
├─ Attention:                  1,438ms (34.4%)  ← Already efficient (CUTLASS)
└─ Other:                        913ms (21.9%)
```

### Optimization Potential

| Phase | Changes | Expected Time | Speedup |
|-------|---------|---------------|---------|
| **Current** | - | 9.8s | 1.0× |
| **Phase 1** | FP16 + GPU decode | 3.2s | **3.1×** |
| **Phase 2** | + TensorRT/Flash Attn | 2.2s | **4.5×** |
| **Phase 3** | + INT8/Distillation | 1.0-1.5s | **6-10×** |

---

## 🚀 Recommended Next Steps

### Week 1: Quick Wins (Expected: 3.1× faster)

1. **Implement FP16 inference** (1 day)
   ```python
   # Add this to basic_inference.py
   model = BADASModel(...).half()  # Convert to FP16
   ```

2. **Integrate GPU video decoding** (2-3 days)
   ```bash
   pip install nvidia-dali-cuda120  # or use Video Codec SDK
   ```

3. **Profile and validate** (1 day)
   ```bash
   bash examples/profile_nsight_quick.sh test_video.mp4
   ```

---

## 📁 File Organization

```
nsight_profiles/
├── README.md (this file)
├── COMPLETE_PROFILING_ANALYSIS.md ← Main report
│
├── Nsight Systems
│   ├── badas_nsys_profile.nsys-rep
│   ├── badas_nsys_profile.sqlite
│   └── kernel_summary_cuda_gpu_kern_sum.csv
│
├── Nsight Compute
│   └── badas_ncu_profile.ncu-rep
│
├── Text Summaries
│   ├── quick_profile_stats_*.txt (from quick run)
│   └── badas_full_kernel_summary_*.txt
│
└── Logs
    ├── full_profile_run2.txt
    ├── full_profile_complete_log.txt
    └── quick_profile_output.txt (from quick run)
```

---

## 🔍 How to Use Profiling Tools

### View Nsight Systems Timeline
```bash
# GUI (recommended) - shows visual timeline
nsys-ui badas_nsys_profile.nsys-rep

# Generate text reports
nsys stats badas_nsys_profile.nsys-rep --report cuda_gpu_kern_sum
nsys stats badas_nsys_profile.nsys-rep --report cuda_api_sum
```

### View Nsight Compute Details
```bash
# GUI (recommended) - detailed kernel metrics
ncu-ui badas_ncu_profile.ncu-rep

# Export to CSV
ncu --import badas_ncu_profile.ncu-rep --page details --csv > details.csv
```

### View PyTorch Profiler
```bash
# Text output
cat ../output_pytorch_profiled.txt

# Chrome trace (visual timeline)
# 1. Open Chrome
# 2. Go to: chrome://tracing
# 3. Load: ../inference_trace.json
```

### Query SQLite Database
```bash
sqlite3 badas_nsys_profile.sqlite

# Example queries:
sqlite> .schema CUPTI_ACTIVITY_KIND_KERNEL
sqlite> SELECT shortName, COUNT(*), SUM(end-start)/1e6 as total_ms 
        FROM CUPTI_ACTIVITY_KIND_KERNEL 
        GROUP BY shortName 
        ORDER BY total_ms DESC 
        LIMIT 10;
```

---

## 📖 Understanding the Results

### Key Metrics Explained

**GEMM (General Matrix Multiply):**
- What: Matrix multiplication operations (A × B = C)
- Where: MLP layers in transformer (1024→4096→1024)
- Why slow: Running in FP32 (32-bit floating point)
- Fix: Use FP16 (16-bit) for 2-3× speedup

**Efficient Attention:**
- What: Fused multi-head self-attention
- Implementation: CUTLASS library (memory-efficient)
- Status: Already optimized
- Further optimization: FP16 gives ~1.5-2× speedup

**SM Utilization:**
- What: GPU compute core usage
- Current: ~57% (room for improvement)
- Target: >80% with optimizations

**Memory Throughput:**
- What: Memory bandwidth usage
- Current: ~25% (not memory-bound)
- Good news: We're compute-bound, not memory-bound

---

## 🎓 Additional Resources

### NVIDIA Documentation
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### PyTorch Resources
- [PyTorch Profiler Tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [TorchScript Documentation](https://pytorch.org/docs/stable/jit.html)

### Optimization Guides
- [NVIDIA DALI Documentation](https://docs.nvidia.com/deeplearning/dali/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)

---

**Questions?** Check [`COMPLETE_PROFILING_ANALYSIS.md`](./COMPLETE_PROFILING_ANALYSIS.md) for detailed explanations!
