#!/bin/bash
#
# NVIDIA Nsight Profiling for BADAS Inference
# =============================================
# Profiles GPU kernels, memory transfers, and system-level metrics
#

set -e

# Configuration
VIDEO_PATH="${1:-/data/karthik_data/badas_data/test-private/positive/00001.mp4}"
OUTPUT_DIR="$(dirname $0)/nsight_profiles"
GPU_ID="${CUDA_VISIBLE_DEVICES:-4}"

echo "======================================================================="
echo "NVIDIA Nsight Profiling - BADAS Inference"
echo "======================================================================="
echo "Video: $VIDEO_PATH"
echo "GPU: $GPU_ID"
echo "Output directory: $OUTPUT_DIR"
echo "======================================================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Change to project directory
cd "$(dirname $0)/.."

echo ""
echo "======================================================================="
echo "1. Nsight Systems Profiling (System-level)"
echo "======================================================================="
echo "Captures: Kernel timeline, GPU utilization, memory transfers, CPU-GPU sync"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --gpu-metrics-device=$GPU_ID \
    --output="$OUTPUT_DIR/badas_nsys_profile" \
    --force-overwrite=true \
    --export=sqlite \
    conda run -n badas_open python examples/basic_inference.py \
        "$VIDEO_PATH" \
        --device cuda

echo ""
echo "✅ Nsight Systems profile saved:"
echo "   - $OUTPUT_DIR/badas_nsys_profile.nsys-rep (main report)"
echo "   - $OUTPUT_DIR/badas_nsys_profile.sqlite (SQLite export)"
echo ""
echo "View with: nsys-ui $OUTPUT_DIR/badas_nsys_profile.nsys-rep"
echo ""

echo "======================================================================="
echo "2. Nsight Compute Profiling (Kernel-level - Top 10 kernels)"
echo "======================================================================="
echo "Captures: Detailed metrics for individual GPU kernels"
echo "Note: This is VERY slow, only profiling first few windows"
echo ""

# Profile only a few iterations to keep it reasonable
CUDA_VISIBLE_DEVICES=$GPU_ID ncu \
    --set full \
    --target-processes all \
    --kernel-name-base mangled \
    --launch-count 10 \
    -o "$OUTPUT_DIR/badas_ncu_profile" \
    --force-overwrite \
    conda run -n badas_open python examples/basic_inference.py \
        "$VIDEO_PATH" \
        --device cuda

echo ""
echo "✅ Nsight Compute profile saved:"
echo "   - $OUTPUT_DIR/badas_ncu_profile.ncu-rep (main report)"
echo ""
echo "View with: ncu-ui $OUTPUT_DIR/badas_ncu_profile.ncu-rep"
echo ""

echo "======================================================================="
echo "3. Extracting Summary Statistics"
echo "======================================================================="
echo ""

# Extract stats from Nsight Systems using SQLite
echo "Nsight Systems Summary:"
echo "----------------------"
sqlite3 "$OUTPUT_DIR/badas_nsys_profile.sqlite" <<EOF
.mode column
.headers on
SELECT 
    'Total Runtime (ms)' as Metric,
    ROUND(MAX(end - start) / 1000000.0, 2) as Value
FROM CUPTI_ACTIVITY_KIND_RUNTIME;

SELECT 
    'GPU Kernels Executed' as Metric,
    COUNT(*) as Value
FROM CUPTI_ACTIVITY_KIND_KERNEL;

SELECT 
    'Total GPU Time (ms)' as Metric,
    ROUND(SUM(end - start) / 1000000.0, 2) as Value
FROM CUPTI_ACTIVITY_KIND_KERNEL;

SELECT 
    'Memory Transfers' as Metric,
    COUNT(*) as Value
FROM CUPTI_ACTIVITY_KIND_MEMCPY;
EOF

echo ""
echo "Top 10 GPU Kernels by Time:"
echo "---------------------------"
sqlite3 "$OUTPUT_DIR/badas_nsys_profile.sqlite" <<EOF
.mode column
.headers on
SELECT 
    shortName as Kernel,
    COUNT(*) as Calls,
    ROUND(SUM(end - start) / 1000000.0, 2) as 'Total_ms',
    ROUND(AVG(end - start) / 1000000.0, 3) as 'Avg_ms'
FROM CUPTI_ACTIVITY_KIND_KERNEL
GROUP BY shortName
ORDER BY SUM(end - start) DESC
LIMIT 10;
EOF

echo ""
echo "======================================================================="
echo "Profiling Complete!"
echo "======================================================================="
echo ""
echo "Files generated:"
echo "  1. $OUTPUT_DIR/badas_nsys_profile.nsys-rep"
echo "  2. $OUTPUT_DIR/badas_nsys_profile.sqlite"
echo "  3. $OUTPUT_DIR/badas_ncu_profile.ncu-rep"
echo ""
echo "View reports:"
echo "  nsys-ui $OUTPUT_DIR/badas_nsys_profile.nsys-rep   # System timeline"
echo "  ncu-ui $OUTPUT_DIR/badas_ncu_profile.ncu-rep      # Kernel details"
echo ""
echo "Or generate text reports:"
echo "  nsys stats $OUTPUT_DIR/badas_nsys_profile.nsys-rep"
echo "  ncu --import $OUTPUT_DIR/badas_ncu_profile.ncu-rep --page details"
echo ""
