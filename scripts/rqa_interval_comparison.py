#!/usr/bin/env python3
"""
RQA Interval Comparison: 12-TET vs Just Intonation

Demonstrates the impact of snapping frequency ratios to simple integer ratios
on waveform recurrence (stability). For a set of common musical intervals,
computes RQA recurrence using:
  - Equal-tempered ratio: 2^(semitones/12)
  - Just intonation ratio: canonical simple fraction (e.g., 3/2, 5/4)

Higher recurrence indicates more periodic, stable signals. This script shows
that snapping to just ratios generally increases recurrence.
"""

import math
from fractions import Fraction
from typing import List, Dict

import numpy as np
from scipy.spatial.distance import pdist

# Local ratio snapping for consistency with project
from ratio_snapping import snap_to_just_intonation

# RQA parameters (aligned with chord_progression_setup.py)
BASELINE_FREQ = 400.0
BASELINE_SR = 8000
SAMPLES_PER_CYCLE = BASELINE_SR / BASELINE_FREQ  # = 20

RQA_DURATION = 0.3
RQA_WINDOW = 480
RQA_SHIFT = 48
RQA_EMB_DIM = 5
RQA_DELAY = 3
RQA_EPS_FACTOR = 0.1


def get_sr_for_freq(root_freq: float) -> int:
    """Scale sample rate so each root frequency gets same samples/cycle as baseline."""
    return int(SAMPLES_PER_CYCLE * root_freq)


def time_delay_embed(x: np.ndarray, emb_dim: int, delay: int) -> np.ndarray:
    N = len(x)
    L = N - (emb_dim - 1) * delay
    if L <= 0:
        return np.empty((0, emb_dim))
    M = np.empty((L, emb_dim))
    for i in range(emb_dim):
        M[:, i] = x[i * delay: i * delay + L]
    return M


def percent_recurrence_single_window(x: np.ndarray) -> float:
    V = time_delay_embed(x, RQA_EMB_DIM, RQA_DELAY)
    Nvec = V.shape[0]
    if Nvec < 2:
        return 0.0
    dists = pdist(V, metric="euclidean")
    avg_dist = np.mean(dists) if dists.size > 0 else 0.0
    eps = RQA_EPS_FACTOR * avg_dist
    rec_count = np.sum(dists <= eps)
    max_pairs = Nvec * (Nvec - 1) / 2.0
    return float(rec_count / max_pairs) if max_pairs > 0 else 0.0


def compute_interval_rqa(root_freq: float, ratio: float) -> float:
    """Compute RQA recurrence for a two-tone interval: root and root*ratio."""
    sr = get_sr_for_freq(root_freq)
    t = np.arange(0, RQA_DURATION, 1.0 / sr)
    sig = (
        np.sin(2.0 * math.pi * root_freq * t)
        + np.sin(2.0 * math.pi * (root_freq * ratio) * t)
    )
    max_val = np.max(np.abs(sig))
    if max_val > 0:
        sig = sig / max_val

    results = []
    i = 0
    while i + RQA_WINDOW <= len(sig):
        w = sig[i : i + RQA_WINDOW]
        pr = percent_recurrence_single_window(w)
        results.append(pr)
        i += RQA_SHIFT
    return float(np.mean(results)) if results else 0.0


def analyze_intervals(root_freq: float = 220.0) -> List[Dict]:
    """Return comparison rows for common intervals under ET vs Just ratios, with octave normalization."""
    intervals = [
        {"name": "Octave", "semitones": 12, "just": Fraction(2, 1)},
        {"name": "Perfect Fifth", "semitones": 7, "just": Fraction(3, 2)},
        {"name": "Perfect Fourth", "semitones": 5, "just": Fraction(4, 3)},
        {"name": "Major Third", "semitones": 4, "just": Fraction(5, 4)},
        {"name": "Minor Third", "semitones": 3, "just": Fraction(6, 5)},
        {"name": "Major Sixth", "semitones": 9, "just": Fraction(5, 3)},
        {"name": "Minor Sixth", "semitones": 8, "just": Fraction(8, 5)},
        {"name": "Major Second", "semitones": 2, "just": Fraction(9, 8)},
        {"name": "Minor Seventh", "semitones": 10, "just": Fraction(9, 5)},
        {"name": "Major Seventh", "semitones": 11, "just": Fraction(15, 8)},
        {"name": "Tritone", "semitones": 6, "just": Fraction(45, 32)},
    ]

    # Compute octave baseline RQA (ET and Just are the same for 12 semitones)
    rqa_octave = compute_interval_rqa(root_freq, 2.0)

    rows: List[Dict] = []
    for iv in intervals:
        ratio_et = 2.0 ** (iv["semitones"] / 12.0)
        ratio_just = float(iv["just"])  # canonical just value

        snapped_from_et = snap_to_just_intonation(ratio_et)

        rqa_et = compute_interval_rqa(root_freq, ratio_et)
        rqa_just = compute_interval_rqa(root_freq, ratio_just)
        rqa_snapped = compute_interval_rqa(root_freq, snapped_from_et)

        # Normalize by octave baseline
        if rqa_octave > 0.0:
            rqa_et_norm = rqa_et / rqa_octave
            rqa_just_norm = rqa_just / rqa_octave
            rqa_snapped_norm = rqa_snapped / rqa_octave
        else:
            rqa_et_norm = 0.0
            rqa_just_norm = 0.0
            rqa_snapped_norm = 0.0

        rows.append(
            {
                "interval": iv["name"],
                "semitones": iv["semitones"],
                "ratio_et": ratio_et,
                "ratio_just": ratio_just,
                "ratio_snapped": snapped_from_et,
                "rqa_et": rqa_et,
                "rqa_just": rqa_just,
                "rqa_snapped": rqa_snapped,
                "rqa_octave": rqa_octave,
                "rqa_et_norm": rqa_et_norm,
                "rqa_just_norm": rqa_just_norm,
                "rqa_snapped_norm": rqa_snapped_norm,
                "delta_just_minus_et": rqa_just - rqa_et,
                "delta_snapped_minus_et": rqa_snapped - rqa_et,
            }
        )

    # Sort by normalized snapped RQA (descending)
    rows.sort(key=lambda r: r["rqa_snapped_norm"], reverse=True)
    return rows


def print_table(rows: List[Dict], root_freq: float):
    print("\n" + "=" * 110)
    print(f"RQA Interval Stability — Root {root_freq:.2f} Hz (Octave-normalized)")
    print("=" * 110)
    hdr = (
        f"{'Interval':<16} | {'ET':>10} | {'Just':>10} | {'Snap':>10} | "
        f"{'RQA_ET':>10} | {'RQA_Just':>10} | {'RQA_Snap':>10} | "
        f"{'Norm_ET':>10} | {'Norm_Just':>10} | {'Norm_Snap':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['interval']:<16} | "
            f"{r['ratio_et']:>10.6f} | {r['ratio_just']:>10.6f} | {r['ratio_snapped']:>10.6f} | "
            f"{r['rqa_et']:>10.6f} | {r['rqa_just']:>10.6f} | {r['rqa_snapped']:>10.6f} | "
            f"{r['rqa_et_norm']:>10.6f} | {r['rqa_just_norm']:>10.6f} | {r['rqa_snapped_norm']:>10.6f}"
        )

    avg_et = float(np.mean([r["rqa_et"] for r in rows]))
    avg_just = float(np.mean([r["rqa_just"] for r in rows]))
    avg_snap = float(np.mean([r["rqa_snapped"] for r in rows]))
    avg_norm_et = float(np.mean([r["rqa_et_norm"] for r in rows]))
    avg_norm_just = float(np.mean([r["rqa_just_norm"] for r in rows]))
    avg_norm_snap = float(np.mean([r["rqa_snapped_norm"] for r in rows]))
    octave_rqa = rows[0]["rqa_octave"] if rows else 0.0
    print("\nSummary:")
    print(f"  Octave RQA baseline : {octave_rqa:.2f}")
    print(f"  Mean RQA (ET)       : {avg_et:.2f}")
    print(f"  Mean RQA (Just)     : {avg_just:.2f}")
    print(f"  Mean RQA (Snap)     : {avg_snap:.2f}")
    print(f"  Mean Norm (ET)      : {avg_norm_et:.2f}")
    print(f"  Mean Norm (Just)    : {avg_norm_just:.2f}")
    print(f"  Mean Norm (Snap)    : {avg_norm_snap:.2f}")


def main():
    root = 220.0  # A3; choose any — SR normalizes per root
    rows = analyze_intervals(root)
    print_table(rows, root)


if __name__ == "__main__":
    main()
