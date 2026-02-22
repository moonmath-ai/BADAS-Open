#!/usr/bin/env python3
"""
Basic Inference with Detailed Profiling
========================================
Runs the EXACT same inference as basic_inference.py but with timing added.
"""

import sys
import argparse
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from badas import BADASModel


def main():
    parser = argparse.ArgumentParser(description="BADAS collision prediction")
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--threshold', type=float, default=0.8, help='Collision probability threshold')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], 
                       help='Device to run inference on')
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help='Path to model checkpoint (optional)')
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # =========================================================================
    # MODEL LOADING (with timing)
    # =========================================================================
    t_start_load = time.time()
    
    print(f"Loading BADAS model...")
    
    # Use local weights if no checkpoint specified
    if args.checkpoint is None:
        # Try local weights first
        local_weights = Path(__file__).parent.parent / "badas" / "weights" / "badas_open.pth"
        if local_weights.exists():
            args.checkpoint = str(local_weights)
            print(f"Using local weights: {args.checkpoint}")
    
    model = BADASModel(
        device=args.device,
        confidence_threshold=args.threshold,
        checkpoint_path=args.checkpoint
    )
    
    t_load = (time.time() - t_start_load) * 1000
    
    print(f"Processing video: {video_path}")
    print(f"Threshold: {args.threshold}")
    print("-" * 50)
    
    # =========================================================================
    # INFERENCE (with timing)
    # =========================================================================
    t_start_inference = time.time()
    
    # Run inference
    try:
        predictions = model.predict(str(video_path))
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)
    
    t_inference = (time.time() - t_start_inference) * 1000
    
    # =========================================================================
    # RESULTS (EXACT same output as basic_inference.py)
    # =========================================================================
    # Analyze results
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
        
        # Estimate time to first collision
        first_collision_time = high_risk_frames[0][0]
        print(f"First high-risk event at: {first_collision_time:.1f} seconds")
        
        # Find peak risk
        peak_risk = max(high_risk_frames, key=lambda x: x[1])
        print(f"Peak collision probability: {peak_risk[1]:.2%} at {peak_risk[0]:.1f}s")
    else:
        print("✅ No high collision risk detected in this video")
    
    # Additional statistics
    # Filter out NaN values from predictions
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
    
    # =========================================================================
    # TIMING SUMMARY (ADDED - not in original)
    # =========================================================================
    print("\n" + "=" * 50)
    print("⏱️  TIMING SUMMARY")
    print("=" * 50)
    
    total_time = t_load + t_inference
    
    print(f"\nModel Loading:        {t_load:>8.1f}ms  ({100*t_load/total_time:>5.1f}%)")
    print(f"Inference:            {t_inference:>8.1f}ms  ({100*t_inference/total_time:>5.1f}%)")
    print(f"{'-' * 50}")
    print(f"TOTAL:                {total_time:>8.1f}ms  (100.0%)")
    
    # Throughput metrics
    video_duration = len(predictions) / 8.0  # Assuming 8 FPS
    print(f"\nThroughput Metrics:")
    print(f"  Video duration:       {video_duration:.1f}s")
    print(f"  Processing time:      {t_inference/1000:.1f}s")
    print(f"  Real-time factor:     {video_duration/(t_inference/1000):.2f}×")
    if valid_predictions:
        print(f"  Frames processed:     {len(valid_predictions)}")
        print(f"  Time per frame:       {t_inference/len(valid_predictions):.1f}ms")


if __name__ == "__main__":
    main()
