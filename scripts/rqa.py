#!/usr/bin/env python3
"""
Compute Percent Recurrence (%RQA) for common triads and 7th chords.

Following methodology from Trulla et al. (2018) "Computational Approach to Musical Consonance":
- Pure sinusoidal tones (no harmonics)
- Sampling rate 8000 Hz (as per paper)
- Root frequency 400 Hz (as per paper)
- Embedding dimension = 5, delay = 3 (as per paper: "The embedding dimension was 5 and the delay was 3 points")
- Epsilon threshold: 5-10% of AVERAGE pairwise distance
- Window=480, shift=48 as specified in paper

RQA parameters from paper: window=480, shift=48, emb_dim=5, delay=3

Outputs a CSV at results/rqa_chords.csv with mean and std of %Recurrence across windows for each chord.
"""
import os
import argparse
import math
from typing import List, Tuple
import numpy as np


# Just intonation ratios (as used in paper)
JUST_RATIOS = {
    "unison": 1/1,      # 1.0
    "octave": 2/1,      # 2.0
    "fifth": 3/2,       # 1.5
    "fourth": 4/3,      # 1.333...
    "major6th": 5/3,    # 1.666...
    "major3rd": 5/4,    # 1.25
    "minor3rd": 6/5,    # 1.2
    "minor6th": 8/5,    # 1.6
    "minor7th": 9/5,    # 1.8
    "major7th": 15/8,   # 1.875
}

# Semitone-based definitions (equal temperament) - kept for reference
CHORD_DEFS = {
    # intervals - unison must be first for normalization
    "unison": [0, 0],
    "octave": [0, 12],
    "fifth": [0, 7],
    "fourth": [0, 5],
    "major6th": [0, 9],
    "major3rd": [0, 4], 
    "minor3rd": [0, 3],
    "minor6th": [0, 8],
    "minor7th": [0, 10],
    "major7th": [0, 11],
    # triads
    # "major": [0, 4, 7],
    # "minor": [0, 3, 7],
    # "diminished": [0, 3, 6],
    # "augmented": [0, 4, 8],
    # # sevenths
    # "major7": [0, 4, 7, 11],
    # "dominant7": [0, 4, 7, 10],
    # "minor7": [0, 3, 7, 10],
    # "half-diminished7": [0, 3, 6, 10],
    # "diminished7": [0, 3, 6, 9],
}


def semitone_to_freq(root_freq: float, semitone_offset: float) -> float:
    return root_freq * (2.0 ** (semitone_offset / 12.0))


def generate_note(freq: float, duration: float, sr: int, harmonics: int = 6) -> np.ndarray:
    t = np.arange(0, duration, 1.0 / sr)
    sig = np.zeros_like(t)
    for h in range(1, harmonics + 1):
        sig += (1.0 / h) * np.sin(2.0 * math.pi * freq * h * t)
    # normalize
    sig = sig / np.max(np.abs(sig))
    return sig


def generate_chord(semitones: List[int], root_freq: float, duration: float, sr: int) -> np.ndarray:
    parts = []
    for st in semitones:
        f = semitone_to_freq(root_freq, st)
        parts.append(generate_note(f, duration, sr))
    chord = np.sum(parts, axis=0)
    chord = chord / np.max(np.abs(chord))
    return chord


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


def analyze_chords(
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
    harmonics: int = 1,
    use_just_intonation: bool = True,
) -> List[Tuple[str, float, float, float, float, float, float, float, int]]:
    """Return list of tuples per chord:
    (name, mean_frac, mean_percent, normalized_percent, p5_percent, median_percent, p95_percent, std_percent, n_windows)
    
    normalized_percent is scaled so that unison = 100%
    """
    raw_results = []
    
    intervals = JUST_RATIOS if use_just_intonation else CHORD_DEFS
    
    for name, interval_def in intervals.items():
        print("Analyzing interval:", name)
        
        if use_just_intonation:
            # interval_def is a ratio
            ratio = interval_def
            f1 = root_freq
            f2 = root_freq * ratio
            t = np.arange(0, duration, 1.0 / sr)
            # Generate two pure tones and sum them
            sig1 = np.sin(2.0 * math.pi * f1 * t)
            sig2 = np.sin(2.0 * math.pi * f2 * t)
            sig = sig1 + sig2
        else:
            # interval_def is list of semitones
            parts = []
            for st in interval_def:
                f = semitone_to_freq(root_freq, st)
                parts.append(generate_note(f, duration, sr, harmonics=harmonics))
            sig = np.sum(parts, axis=0)
        
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
        unison_frac = 1.0  # fallback to avoid division by zero
    
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
    parser.add_argument("--harmonics", type=int, default=1, help="Number of harmonics/partials per note (1=pure tone as in paper)")
    parser.add_argument("--root", type=float, default=400.0, help="Root frequency in Hz (paper: 400)")
    parser.add_argument("--out", type=str, default="results/rqa_chords.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results = analyze_chords(
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
        harmonics=args.harmonics,
    )

    # write CSV header for expanded stats (now includes normalized_percent)
    with open(args.out, "w") as f:
        f.write("chord,mean_frac,mean_percent,normalized_percent,p5_percent,median_percent,p95_percent,std_percent,n_windows\n")
        for name, mean_frac, mean_perc, norm_perc, p5, median_perc, p95, std_perc, n in results:
            f.write(f"{name},{mean_frac:.6f},{mean_perc:.6f},{norm_perc:.1f},{p5:.6f},{median_perc:.6f},{p95:.6f},{std_perc:.6f},{n}\n")

    # print brief summary with normalized values
    print("%RQA results written to:", args.out)
    print(f"{'Chord':15s} | {'Normalized':>10s} | {'Raw %':>8s}")
    print("-" * 40)
    for name, mean_frac, mean_perc, norm_perc, p5, median_perc, p95, std_perc, n in results:
        print(f"{name:15s} | {norm_perc:>10.1f} | {mean_perc:>8.3f}%")


if __name__ == "__main__":
    main()
