#!/usr/bin/env python3
"""
Compute Percent Recurrence (%RQA) for all chords across all 12 root notes in C3-C4 octave.
Outputs a CSV with normalized percentages (unison=100) for each chord and root note.
"""
import os
import math
import numpy as np
from scipy.spatial.distance import pdist

# C3 = 130.81 Hz (MIDI note 48)
C3_FREQ = 130.81

# 12 notes in the octave
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Just intonation ratios for chords (pure frequency ratios)
JUST_CHORDS = {
    "unison": [1/1, 1/1],                    # Reference
    "Major": [1/1, 5/4, 3/2],                # 4:5:6
    "minor": [1/1, 6/5, 3/2],                # 10:12:15
    "suspended 4": [1/1, 4/3, 3/2],          # 6:8:9
    "suspended 2": [1/1, 9/8, 3/2],          # 8:9:12
    "diminished": [1/1, 6/5, 36/25],         # 25:30:36
    "Augmented": [1/1, 5/4, 25/16],          # 16:20:25
    "Major 7": [1/1, 5/4, 3/2, 15/8],        # 8:10:12:15
    "Dominant 7": [1/1, 5/4, 3/2, 9/5],      # 20:25:30:36
    "minor 7": [1/1, 6/5, 3/2, 9/5],         # 10:12:15:18
    "minor Major 7": [1/1, 6/5, 3/2, 15/8],
    "diminished 7": [1/1, 6/5, 36/25, 216/125],
    "half diminished 7": [1/1, 6/5, 36/25, 9/5],
    "Major 6": [1/1, 5/4, 3/2, 5/3],
    "minor 6": [1/1, 6/5, 3/2, 5/3],
    "Augmented Major 7": [1/1, 5/4, 25/16, 15/8],
}

# RQA parameters from paper
# Baseline: 400 Hz at 8000 SR = 20 samples per cycle
BASELINE_FREQ = 400.0
BASELINE_SR = 8000
SAMPLES_PER_CYCLE = BASELINE_SR / BASELINE_FREQ  # = 20

DURATION = 6.0  # seconds
WINDOW = 480
SHIFT = 48
EMB_DIM = 5
DELAY = 3
EPS_FACTOR = 0.1


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
    V = time_delay_embed(x, EMB_DIM, DELAY)
    Nvec = V.shape[0]
    if Nvec < 2:
        return 0.0
    dists = pdist(V, metric='euclidean')  # Much faster than manual computation
    avg_dist = np.mean(dists) if dists.size > 0 else 0.0
    eps = EPS_FACTOR * avg_dist
    rec_count = np.sum(dists <= eps)
    max_pairs = Nvec * (Nvec - 1) / 2.0
    return float(rec_count / max_pairs) if max_pairs > 0 else 0.0


def compute_chord_recurrence(root_freq: float, ratios: list) -> float:
    """Compute mean recurrence for a chord at given root frequency."""
    # Scale sample rate to maintain constant samples-per-cycle
    sr = get_sr_for_freq(root_freq)
    t = np.arange(0, DURATION, 1.0 / sr)
    sig = np.zeros_like(t)
    for ratio in ratios:
        f = root_freq * ratio
        sig += np.sin(2.0 * math.pi * f * t)
    sig = sig / np.max(np.abs(sig))
    
    # Sliding window RQA
    results = []
    i = 0
    while i + WINDOW <= len(sig):
        w = sig[i: i + WINDOW]
        pr = percent_recurrence_single_window(w)
        results.append(pr)
        i += SHIFT
    
    return float(np.mean(results)) if results else 0.0


def main():
    # Output file
    out_path = "results/rqa_all_roots.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Store results: {chord_name: {note_name: normalized_percent}}
    results = {}
    
    chord_names = list(JUST_CHORDS.keys())
    
    # For each root note in C3-C4 octave
    for note_idx, note_name in enumerate(NOTE_NAMES):
        root_freq = C3_FREQ * (2.0 ** (note_idx / 12.0))
        print(f"Analyzing root: {note_name}3 ({root_freq:.2f} Hz)")
        
        # Compute recurrence for each chord
        raw_recurrences = {}
        for chord_name, ratios in JUST_CHORDS.items():
            rec = compute_chord_recurrence(root_freq, ratios)
            raw_recurrences[chord_name] = rec
        
        # Normalize to unison = 100
        unison_rec = raw_recurrences.get("unison", 1.0)
        if unison_rec == 0:
            unison_rec = 1.0
        
        for chord_name in chord_names:
            normalized = (raw_recurrences[chord_name] / unison_rec) * 100.0
            if chord_name not in results:
                results[chord_name] = {}
            results[chord_name][note_name] = normalized
    
    # Write CSV
    with open(out_path, "w") as f:
        # Header: chord, C, C#, D, D#, E, F, F#, G, G#, A, A#, B
        header = "chord," + ",".join(NOTE_NAMES)
        f.write(header + "\n")
        
        # Data rows
        for chord_name in chord_names:
            row_values = [f"{results[chord_name][note]:.1f}" for note in NOTE_NAMES]
            f.write(f"{chord_name}," + ",".join(row_values) + "\n")
    
    print(f"\nResults written to: {out_path}")
    
    # Print summary table
    print("\nNormalized Recurrence (%) - Unison = 100:")
    print("-" * 100)
    print(f"{'Chord':20s} | " + " | ".join([f"{n:>4s}" for n in NOTE_NAMES]))
    print("-" * 100)
    for chord_name in chord_names:
        values = [f"{results[chord_name][note]:4.1f}" for note in NOTE_NAMES]
        print(f"{chord_name:20s} | " + " | ".join(values))


if __name__ == "__main__":
    main()
