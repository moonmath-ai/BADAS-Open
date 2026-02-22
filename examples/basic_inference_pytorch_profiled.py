#!/usr/bin/env python3
"""
Basic Inference with PyTorch Profiler
======================================
Runs full inference pipeline with detailed PyTorch profiling.
Shows exact time breakdown for every operation.
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from badas import BADASModel


def main():
    parser = argparse.ArgumentParser(description="BADAS inference with PyTorch profiling")
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--threshold', type=float, default=0.8, help='Collision probability threshold')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], 
                       help='Device to run inference on')
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help='Path to model checkpoint (optional)')
    parser.add_argument('--export-trace', action='store_true',
                       help='Export Chrome trace for visualization')
    parser.add_argument('--trace-file', type=str, default='inference_trace.json',
                       help='Trace file name')
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    print("=" * 70)
    print("BADAS Inference with PyTorch Profiler")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Device: {args.device}")
    print(f"Threshold: {args.threshold}")
    print("=" * 70)
    
    # =========================================================================
    # MODEL LOADING
    # =========================================================================
    print(f"\nLoading BADAS model...")
    
    # Use local weights if no checkpoint specified
    if args.checkpoint is None:
        local_weights = Path(__file__).parent.parent / "badas" / "weights" / "badas_open.pth"
        if local_weights.exists():
            args.checkpoint = str(local_weights)
            print(f"Using local weights: {args.checkpoint}")
    
    model = BADASModel(
        device=args.device,
        confidence_threshold=args.threshold,
        checkpoint_path=args.checkpoint
    )
    
    print(f"Processing video: {video_path}")
    print(f"Threshold: {args.threshold}")
    print("-" * 70)
    
    # =========================================================================
    # INFERENCE WITH PYTORCH PROFILER
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("Running inference with PyTorch Profiler...")
    print(f"{'=' * 70}\n")
    
    # Configure profiler
    activities = [torch.profiler.ProfilerActivity.CPU]
    if args.device == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    
    predictions = None
    
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,  # Disable stack traces for cleaner output
        with_flops=False,  # Disable FLOPS calculation (can be slow)
    ) as prof:
        try:
            predictions = model.predict(str(video_path))
        except Exception as e:
            print(f"Error during inference: {e}")
            sys.exit(1)
    
    # =========================================================================
    # DISPLAY PROFILING RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("PROFILING RESULTS - Top Operations by Time")
    print("=" * 70 + "\n")
    
    # Sort by CUDA time if GPU, otherwise CPU time
    sort_key = "cuda_time_total" if args.device == 'cuda' else "cpu_time_total"
    
    # Get top operations
    print("Top 30 Most Time-Consuming Operations:")
    print("-" * 70)
    print(prof.key_averages().table(
        sort_by=sort_key,
        row_limit=30,
        max_name_column_width=50
    ))
    
    # Group by operation type
    print("\n" + "=" * 70)
    print("PROFILING RESULTS - By Operation Category")
    print("=" * 70 + "\n")
    
    key_averages = prof.key_averages()
    
    # Categorize operations
    categories = {
        'Attention': [],
        'Linear/GEMM': [],
        'Normalization': [],
        'Activation': [],
        'Convolution': [],
        'Memory': [],
        'Other': []
    }
    
    for item in key_averages:
        name = item.key.lower()
        time_val = item.self_device_time_total if args.device == 'cuda' else item.self_cpu_time_total
        
        if time_val == 0:
            continue
            
        if 'attention' in name or 'sdpa' in name or 'bmm' in name:
            categories['Attention'].append((item.key, time_val))
        elif 'linear' in name or 'addmm' in name or 'mm' in name or 'matmul' in name:
            categories['Linear/GEMM'].append((item.key, time_val))
        elif 'norm' in name or 'layer_norm' in name:
            categories['Normalization'].append((item.key, time_val))
        elif 'gelu' in name or 'relu' in name or 'sigmoid' in name or 'softmax' in name:
            categories['Activation'].append((item.key, time_val))
        elif 'conv' in name:
            categories['Convolution'].append((item.key, time_val))
        elif 'copy' in name or 'to(' in name or 'clone' in name:
            categories['Memory'].append((item.key, time_val))
        else:
            categories['Other'].append((item.key, time_val))
    
    # Display category summaries
    total_time = sum(item.self_device_time_total if args.device == 'cuda' else item.self_cpu_time_total 
                     for item in key_averages)
    
    print("Time by Operation Category:")
    print("-" * 70)
    for category, ops in categories.items():
        if ops:
            cat_time = sum(t for _, t in ops)
            cat_pct = (cat_time / total_time * 100) if total_time > 0 else 0
            print(f"{category:20s}: {cat_time/1000:>10.1f}ms ({cat_pct:>5.1f}%) - {len(ops)} ops")
    
    print(f"\n{'Total Time':20s}: {total_time/1000:>10.1f}ms (100.0%)")
    
    # Show top operations in each category
    print("\n" + "=" * 70)
    print("Top 3 Operations in Each Category:")
    print("=" * 70)
    for category, ops in categories.items():
        if ops:
            print(f"\n{category}:")
            sorted_ops = sorted(ops, key=lambda x: x[1], reverse=True)[:3]
            for name, time_val in sorted_ops:
                pct = (time_val / total_time * 100) if total_time > 0 else 0
                print(f"  {time_val/1000:>8.1f}ms ({pct:>4.1f}%) - {name[:60]}")
    
    # Export trace if requested
    if args.export_trace:
        trace_path = Path(__file__).parent / args.trace_file
        prof.export_chrome_trace(str(trace_path))
        print(f"\n{'=' * 70}")
        print(f"Chrome trace exported to: {trace_path}")
        print(f"View at: chrome://tracing")
        print(f"{'=' * 70}")
    
    # =========================================================================
    # INFERENCE RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("INFERENCE RESULTS")
    print("=" * 70)
    
    print(f"\nResults:")
    print(f"Total predictions: {len(predictions)}")
    
    # Find high-risk moments
    high_risk_frames = []
    for i, prob in enumerate(predictions):
        timestamp = i * 0.125  # 8 FPS
        
        if prob >= args.threshold:
            high_risk_frames.append((timestamp, prob))
            print(f"⚠️  High collision risk at {timestamp:.1f}s: {prob:.2%}")
        elif prob >= 0.5:
            print(f"⚡ Medium risk at {timestamp:.1f}s: {prob:.2%}")
    
    # Summary
    print("\n" + "=" * 50)
    if high_risk_frames:
        print(f"🚨 COLLISION WARNING: {len(high_risk_frames)} high-risk moments detected!")
        
        first_collision_time = high_risk_frames[0][0]
        print(f"First high-risk event at: {first_collision_time:.1f} seconds")
        
        peak_risk = max(high_risk_frames, key=lambda x: x[1])
        print(f"Peak collision probability: {peak_risk[1]:.2%} at {peak_risk[0]:.1f}s")
    else:
        print("✅ No high collision risk detected in this video")
    
    # Statistics
    valid_predictions = [p for p in predictions if not np.isnan(p)]
    
    if valid_predictions:
        avg_risk = sum(valid_predictions) / len(valid_predictions)
        max_risk = max(valid_predictions)
        print(f"\nStatistics:")
        print(f"  Average risk: {avg_risk:.2%}")
        print(f"  Maximum risk: {max_risk:.2%}")
        print(f"  Valid predictions: {len(valid_predictions)}/{len(predictions)}")
    else:
        print(f"\nStatistics:")
        print(f"  Warning: All predictions are NaN")
    
    print("\n" + "=" * 70)
    print("Profiling complete!")
    if args.export_trace:
        print(f"Tip: Load {args.trace_file} in chrome://tracing for visual timeline")
    print("=" * 70)


if __name__ == "__main__":
    main()
