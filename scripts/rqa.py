#!/usr/bin/env python3
"""
Compute Percent Recurrence (%RQA) for musical intervals.

Following methodology from Trulla et al. (2018) "Computational Approach to Musical Consonance":
- Pure sinusoidal tones (no harmonics)
- Sampling rate 8000 Hz (as per paper)
- Root frequency 400 Hz (as per paper)
- Embedding dimension = 5, delay = 3 (as per paper: "The embedding dimension was 5 and the delay was 3 points")
- Epsilon threshold: 5-10% of AVERAGE pairwise distance
- Window=480, shift=48 as specified in paper

RQA parameters from paper: window=480, shift=48, emb_dim=5, delay=3

Outputs a CSV at results/rqa_intervals.csv with mean and std of %Recurrence across windows for each interval.
"""
import os
import argparse
import math
from typing import List, Tuple
import numpy as np


# Just intonation ratios for intervals (as used in paper)
JUST_INTERVALS = {
    "unison": [1/1, 1/1],           # 1:1
    "octave": [1/1, 2/1],           # 1:2
    "fifth": [1/1, 3/2],            # 2:3
    "fourth": [1/1, 4/3],           # 3:4
    "major6th": [1/1, 5/3],         # 3:5
    "major3rd": [1/1, 5/4],         # 4:5
    "minor3rd": [1/1, 6/5],         # 5:6
    "minor6th": [1/1, 8/5],         # 5:8
    "minor7th": [1/1, 9/5],         # 5:9
    "major7th": [1/1, 15/8],        # 8:15
    "tritone": [1/1, 45/32],        # 32:45
    "minor2nd": [1/1, 16/15],       # 15:16
    "major2nd": [1/1, 9/8],         # 8:9
}


def time_delay_embed(x: np.ndarray, emb_dim: int, delay: int) -> np.ndarray:
    N = len(x)
    L = N - (emb_dim - 1) * delay
    if L <= 0:
        return np.empty((0, emb_dim))
    M = np.empty((L, emb_dim))
    for i in range(emb_dim):
        M[:, i] = x[i * delay: i * delay + L]
    return M


def condensed_pairwise_distances(V: np.ndarray) -> np.ndarray:
    # Computes condensed distances (upper triangle) without extra deps
    # V shape: (N, m)
    if V.shape[0] < 2:
        return np.array([])
    sum_sq = np.sum(V * V, axis=1)
    D2 = sum_sq[:, None] + sum_sq[None, :] - 2.0 * (V @ V.T)
    # numerical safety
    D2[D2 < 0] = 0.0
    # take upper triangle
    iu = np.triu_indices(V.shape[0], k=1)
    d2_upper = D2[iu]
    return np.sqrt(d2_upper)


def percent_recurrence_single_window(
    x: np.ndarray,
    emb_dim: int,
    delay: int,
    eps: float = None,
    eps_factor: float = 0.1,
    eps_method: str = "average",
    eps_percentile: float = 5.0,
) -> float:
    """Return recurrence as a fraction in [0,1] for the given window.

    eps_method options:
    - 'average': eps = eps_factor * AVERAGE pairwise distance (paper's method)
    - 'factor': eps = eps_factor * max_distance
    - 'percentile': eps = percentile(dists, eps_percentile)
    """
    V = time_delay_embed(x, emb_dim, delay)
    Nvec = V.shape[0]
    if Nvec < 2:
        return 0.0

    # NO z-normalization - use raw embedded vectors as in paper
    dists = condensed_pairwise_distances(V)
    
    if eps is None:
        if eps_method == "average":
            # Paper: "r is usually set to 5-10% of the average pairwise distances"
            avg_dist = np.mean(dists) if dists.size > 0 else 0.0
            eps = eps_factor * avg_dist
        elif eps_method == "percentile":
            eps = float(np.percentile(dists, eps_percentile)) if dists.size > 0 else 0.0
        else:  # factor method uses max
            maxd = np.max(dists) if dists.size > 0 else 0.0
            eps = eps_factor * maxd

    # count recurrences (non-trivial pairs only in condensed form)
    rec_count = np.sum(dists <= eps)
    max_pairs = Nvec * (Nvec - 1) / 2.0
    return float(rec_count / max_pairs) if max_pairs > 0 else 0.0


def sliding_rqa(
    signal: np.ndarray,
    window: int,
    shift: int,
    emb_dim: int,
    delay: int,
    eps: float = None,
    eps_factor: float = 0.1,
    eps_method: str = "average",
    eps_percentile: float = 5.0,
) -> np.ndarray:
    results = []
    i = 0
    while i + window <= len(signal):
        w = signal[i: i + window]
        pr = percent_recurrence_single_window(
            w,
            emb_dim,
            delay,
            eps=eps,
            eps_factor=eps_factor,
            eps_method=eps_method,
            eps_percentile=eps_percentile,
        )
        results.append(pr)
        i += shift
    return np.array(results)


def analyze_intervals(
    root_freq: float = 400.0,  # Paper uses 400 Hz
    sr: int = 8000,
    duration: float = 10.0,
    window: int = 480,
    shift: int = 48,
    emb_dim: int = 5,  # Paper: "embedding dimension was 5"
    delay: int = 3,    # Paper: "delay was 3 points"
    eps: float = None,
    eps_factor: float = 0.1,
    eps_method: str = "average",
    eps_percentile: float = 5.0,
) -> List[Tuple[str, float, float, float, float, float, float, float, int]]:
    """Return list of tuples per interval:
    (name, mean_frac, mean_percent, normalized_percent, p5_percent, median_percent, p95_percent, std_percent, n_windows)
    
    normalized_percent is scaled so that unison = 100%
    """
    raw_results = []
    
    for name, interval_ratios in JUST_INTERVALS.items():
        print("Analyzing:", name)
        
        t = np.arange(0, duration, 1.0 / sr)
        
        # interval_ratios is a list of frequency ratios [1/1, ratio]
        sig = np.zeros_like(t)
        for ratio in interval_ratios:
            f = root_freq * ratio
            sig += np.sin(2.0 * math.pi * f * t)
        
        sig = sig / np.max(np.abs(sig))

        vals = sliding_rqa(
            sig,
            window=window,
            shift=shift,
            emb_dim=emb_dim,
            delay=delay,
            eps=eps,
            eps_factor=eps_factor,
            eps_method=eps_method,
            eps_percentile=eps_percentile,
        )
        n = int(vals.size)
        mean_frac = float(np.mean(vals)) if vals.size else 0.0
        mean_perc = mean_frac * 100.0
        std_perc = float(np.std(vals) * 100.0) if vals.size else 0.0
        median_perc = float(np.median(vals) * 100.0) if vals.size else 0.0
        p5 = float(np.percentile(vals, 5) * 100.0) if vals.size else 0.0
        p95 = float(np.percentile(vals, 95) * 100.0) if vals.size else 0.0
        raw_results.append((name, mean_frac, mean_perc, p5, median_perc, p95, std_perc, n))
    
    # Normalize: find unison's mean_frac and scale all to make unison = 100%
    unison_frac = None
    for name, mean_frac, *_ in raw_results:
        if name == "unison":
            unison_frac = mean_frac
            break
    
    if unison_frac is None or unison_frac == 0:
        # If no unison found, use the first chord as reference
        unison_frac = raw_results[0][1] if raw_results else 1.0
    
    # Build output with normalized percentage
    out = []
    for name, mean_frac, mean_perc, p5, median_perc, p95, std_perc, n in raw_results:
        normalized_perc = (mean_frac / unison_frac) * 100.0
        out.append((name, mean_frac, mean_perc, normalized_perc, p5, median_perc, p95, std_perc, n))
    
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=8000)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--window", type=int, default=480)
    parser.add_argument("--shift", type=int, default=48)
    parser.add_argument("--emb", type=int, default=5, help="Embedding dimension (paper: 5)")
    parser.add_argument("--delay", type=int, default=3, help="Time delay for embedding (paper: 3)")
    parser.add_argument("--eps", type=float, default=None, help="absolute epsilon; if omitted eps = eps_factor * avg_dist (or max_dist)")
    parser.add_argument("--eps-factor", type=float, default=0.1)
    parser.add_argument("--eps-method", type=str, default="average", choices=["average", "factor", "percentile"], help="Method to choose eps: 'average' (paper), 'factor', or 'percentile'")
    parser.add_argument("--eps-percentile", type=float, default=5.0, help="Percentile to use for eps when --eps-method percentile (e.g. 5)")
    parser.add_argument("--root", type=float, default=400.0, help="Root frequency in Hz (paper: 400)")
    parser.add_argument("--out", type=str, default="results/rqa_intervals.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results = analyze_intervals(
        root_freq=args.root,
        sr=args.sr,
        duration=args.duration,
        window=args.window,
        shift=args.shift,
        emb_dim=args.emb,
        delay=args.delay,
        eps=args.eps,
        eps_factor=args.eps_factor,
        eps_method=args.eps_method,
        eps_percentile=args.eps_percentile,
    )

    # write CSV header for expanded stats (now includes normalized_percent)
    with open(args.out, "w") as f:
        f.write("interval,mean_frac,mean_percent,normalized_percent,p5_percent,median_percent,p95_percent,std_percent,n_windows\n")
        for name, mean_frac, mean_perc, norm_perc, p5, median_perc, p95, std_perc, n in results:
            f.write(f"{name},{mean_frac:.6f},{mean_perc:.6f},{norm_perc:.1f},{p5:.6f},{median_perc:.6f},{p95:.6f},{std_perc:.6f},{n}\n")

    # print brief summary with normalized values
    print("%RQA results written to:", args.out)
    print(f"{'Interval':15s} | {'Normalized':>10s} | {'Raw %':>8s}")
    print("-" * 40)
    for name, mean_frac, mean_perc, norm_perc, p5, median_perc, p95, std_perc, n in results:
        print(f"{name:15s} | {norm_perc:>10.1f} | {mean_perc:>8.3f}%")


if __name__ == "__main__":
    main()
