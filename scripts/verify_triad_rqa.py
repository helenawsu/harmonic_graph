#!/usr/bin/env python3
"""
Compare RQA for C3-E3-G3 vs F3-A3-C4 chords STANDALONE (no home comparison)
Just analyzing the triads themselves
"""

import numpy as np
import sys
import os
from typing import List, Tuple

sys.path.append(os.path.dirname(__file__))
from ratio_snapping import snap_to_just_intonation
from chord_progression_setup import (
    calculate_bass_stability_weight,
    identify_chord_root,
    CHORD_ROOT_BOOST,
    RQA_DURATION,
    BASELINE_SR,
    BASELINE_FREQ,
    SAMPLES_PER_CYCLE,
    RQA_WINDOW,
    RQA_SHIFT,
    percent_recurrence_single_window
)


def get_sr_for_freq(freq: float) -> int:
    """Scale sample rate so each frequency gets same samples/cycle as baseline."""
    return int(SAMPLES_PER_CYCLE * freq)


def compute_triad_rqa_standalone(
    chord_frequencies: List[float],
    chord_name: str,
    boost_root: bool = True,
    verbose: bool = False
) -> Tuple[float, str]:
    """
    Compute RQA for a triad by itself (no home note comparison).
    
    Returns:
        (rqa_score, identified_root_name)
    """
    # Use lowest frequency as reference for sample rate
    root_freq = min(chord_frequencies)
    sr = get_sr_for_freq(root_freq)
    
    if verbose:
        print(f"\n  {chord_name}")
        print(f"  {'='*66}")
        print(f"    Sample rate: {sr} Hz (proportional to {root_freq:.2f} Hz)")
        print(f"    Samples per cycle: {sr / root_freq:.2f}")
    
    # Identify chord root if boosting
    if boost_root:
        chord_root_freq = identify_chord_root(chord_frequencies)
        root_note = f"{chord_root_freq:.2f} Hz"
    else:
        chord_root_freq = chord_frequencies[0]  # Assume bass
        root_note = f"{chord_frequencies[0]:.2f} Hz (bass)"
    
    # Create signal with all chord notes
    t = np.arange(0, RQA_DURATION, 1.0 / sr)
    signal = np.zeros_like(t)
    
    for freq in chord_frequencies:
        # IMPORTANT: For standalone comparison, do NOT change the waveform when
        # computing bass-stability-weighted values. We keep the synthesis equal-
        # amplitude and apply bass stability as a post-factor to the raw RQA.
        amplitude = 1.0 if not (boost_root and abs(freq - chord_root_freq) < 0.1) else 1.0
        if verbose:
            action = "Adding"
            print(f"    {action} {freq:.2f} Hz at 1x amplitude")
        signal += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    # Compute RQA
    results = []
    i = 0
    while i + RQA_WINDOW <= len(signal):
        w = signal[i: i + RQA_WINDOW]
        pr = percent_recurrence_single_window(w)
        results.append(pr)
        i += RQA_SHIFT
    
    rqa_score = float(np.mean(results)) if results else 0.0
    
    return rqa_score, root_note


def main():
    # Define chord frequencies
    c3 = 130.81
    e3 = c3 * 5/4
    g3 = c3 * 3/2
    ceg_freqs = [c3, e3, g3]

    c3_base = 130.81
    f3 = c3_base * 4/3
    a3 = c3_base * 5/3
    cfa_freqs = [c3_base, f3, a3]

    # Compute raw RQA values (equal-amplitude synthesis)
    rqa1_raw, _ = compute_triad_rqa_standalone(ceg_freqs, "C3-E3-G3 (raw)", boost_root=False, verbose=False)
    rqa2_raw, _ = compute_triad_rqa_standalone(cfa_freqs, "C3-F3-A3 (raw)", boost_root=False, verbose=False)

    # Compute bass stability weights using the BASS as anchor (penalize inversions)
    bass1 = min(ceg_freqs)
    bass2 = min(cfa_freqs)
    bass_w1 = calculate_bass_stability_weight(bass1, ceg_freqs)
    bass_w2 = calculate_bass_stability_weight(bass2, cfa_freqs)

    # Apply bass stability as a multiplicative post-weight to the same raw RQA
    # This ensures that if bass_w == 1.0 (e.g., root position, power-of-two LCM),
    # the weighted RQA equals the raw RQA.
    rqa1_weighted = rqa1_raw * bass_w1
    rqa2_weighted = rqa2_raw * bass_w2

    # Table header
    header = [
        "Chord",
        "Without Bass Stability",
        "With Bass Stability (bass × weight)",
    ]

    rows = [
        ["C3-E3-G3 (4:5:6)", f"{rqa1_raw:.3f}", f"{rqa1_weighted:.3f}"],
        ["C3-F3-A3 (3:4:5)", f"{rqa2_raw:.3f}", f"{rqa2_weighted:.3f}"],
    ]

    # Compute column widths
    col_widths = [
        max(len(header[0]), *(len(r[0]) for r in rows)),
        max(len(header[1]), *(len(r[1]) for r in rows)),
        max(len(header[2]), *(len(r[2]) for r in rows)),
    ]

    def fmt_row(cols):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cols, col_widths)) + " |"

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"

    # Print table
    print(sep)
    print(fmt_row(header))
    print(sep)
    for r in rows:
        print(fmt_row(r))
    print(sep)


if __name__ == "__main__":
    main()
