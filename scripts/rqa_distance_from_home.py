#!/usr/bin/env python3
"""
Compute RQA-based "Distance from Home" for all chords.

This script measures how "distant" each chord sounds from the home ROOT NOTE
by computing the recurrence when playing each chord SIMULTANEOUSLY with
just the root note (e.g., C alone, not C Major chord).

The idea: When a chord is consonant with the root, they create a stable
waveform with high recurrence. Dissonant combinations create chaotic
waveforms with low recurrence.

Distance = 1.0 - (normalized_recurrence)
- Unison (root + root): Distance = 0.0 (maximum consonance)
- Very dissonant chord: Distance → 1.0 (low recurrence)

Outputs: results/rqa_distance_from_home.csv
"""
import os
import math
import numpy as np
from typing import List, Dict, Tuple

# Try to use scipy for faster distance computation
try:
    from scipy.spatial.distance import pdist
    USE_SCIPY = True
except ImportError:
    USE_SCIPY = False
    print("scipy not found, using numpy (slower)")

# Note names
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Chord types (triads only - Major and minor)
CHORD_TYPES = ["Major", "minor"]

# Just intonation ratios for triads
JUST_RATIOS = {
    "Major": [1/1, 5/4, 3/2],    # 4:5:6
    "minor": [1/1, 6/5, 3/2],    # 10:12:15
}

# Just intonation ratios for each semitone from C
# These are the pure frequency ratios relative to the root (C)
JUST_SEMITONE_RATIOS = {
    0: 1/1,       # C (unison)
    1: 16/15,     # C# (minor second)
    2: 9/8,       # D (major second)
    3: 6/5,       # D# (minor third)
    4: 5/4,       # E (major third)
    5: 4/3,       # F (perfect fourth)
    6: 45/32,     # F# (tritone)
    7: 3/2,       # G (perfect fifth)
    8: 8/5,       # G# (minor sixth)
    9: 5/3,       # A (major sixth)
    10: 9/5,      # A# (minor seventh)
    11: 15/8,     # B (major seventh)
}

# RQA parameters (from paper)
C3_FREQ = 130.81  # C3 = 130.81 Hz
BASELINE_FREQ = 400.0
BASELINE_SR = 8000
SAMPLES_PER_CYCLE = BASELINE_SR / BASELINE_FREQ  # = 20

DURATION = 6.0  # seconds
WINDOW = 480
SHIFT = 48
EMB_DIM = 5
DELAY = 3
EPS_FACTOR = 0.1

# Amplitude boost for chord root to emphasize root-to-home relationship
# Higher values make the chord root ↔ home note relationship dominate
# At 10x, the chord root's relationship to home becomes the primary factor
CHORD_ROOT_BOOST = 10.0  # Chord root is 10x louder than other chord tones


def get_sr_for_freq(root_freq: float) -> int:
    """Scale sample rate so each root frequency gets same samples/cycle as baseline."""
    return int(SAMPLES_PER_CYCLE * root_freq)


def semitone_to_freq(root_freq: float, semitones: int) -> float:
    """Convert semitone offset to frequency."""
    return root_freq * (2.0 ** (semitones / 12.0))


def time_delay_embed(x: np.ndarray, emb_dim: int, delay: int) -> np.ndarray:
    """Time delay embedding."""
    N = len(x)
    L = N - (emb_dim - 1) * delay
    if L <= 0:
        return np.empty((0, emb_dim))
    M = np.empty((L, emb_dim))
    for i in range(emb_dim):
        M[:, i] = x[i * delay: i * delay + L]
    return M


def condensed_pairwise_distances(V: np.ndarray) -> np.ndarray:
    """Compute condensed pairwise distances."""
    if USE_SCIPY:
        return pdist(V, metric='euclidean')
    else:
        # Manual computation
        if V.shape[0] < 2:
            return np.array([])
        sum_sq = np.sum(V * V, axis=1)
        D2 = sum_sq[:, None] + sum_sq[None, :] - 2.0 * (V @ V.T)
        D2[D2 < 0] = 0.0
        iu = np.triu_indices(V.shape[0], k=1)
        return np.sqrt(D2[iu])


def percent_recurrence_single_window(x: np.ndarray) -> float:
    """Compute percent recurrence for a single window."""
    V = time_delay_embed(x, EMB_DIM, DELAY)
    Nvec = V.shape[0]
    if Nvec < 2:
        return 0.0
    
    dists = condensed_pairwise_distances(V)
    avg_dist = np.mean(dists) if dists.size > 0 else 0.0
    eps = EPS_FACTOR * avg_dist
    
    rec_count = np.sum(dists <= eps)
    max_pairs = Nvec * (Nvec - 1) / 2.0
    return float(rec_count / max_pairs) if max_pairs > 0 else 0.0


def generate_combined_signal(
    home_root: int,
    chord_root: int,
    chord_type: str,
    base_freq: float,
    duration: float,
    sr: int,
) -> np.ndarray:
    """
    Generate signal of home ROOT NOTE + test chord playing simultaneously.
    Uses just intonation for all frequencies.
    
    All notes are added as separate sine waves, even if they are the same pitch.
    This means if C appears in both the home note and the chord, both sine waves
    are added (creating double amplitude at that frequency).
    
    Args:
        home_root: Root note (semitones from C) - played as single note
        chord_root: Root of test chord (semitones from C)
        chord_type: Type of test chord ("Major" or "minor")
        base_freq: Base frequency (C)
        duration: Duration in seconds
        sr: Sample rate
    
    Returns:
        Combined normalized signal
    """
    t = np.arange(0, duration, 1.0 / sr)
    sig = np.zeros_like(t)
    
    # Add home ROOT NOTE using just intonation ratio
    home_ratio = JUST_SEMITONE_RATIOS[home_root % 12]
    home_freq = base_freq * home_ratio
    sig += np.sin(2.0 * math.pi * home_freq * t)
    
    # Add test chord tones using just intonation
    # First get the chord root frequency using just intonation
    chord_root_ratio = JUST_SEMITONE_RATIOS[chord_root % 12]
    chord_base = base_freq * chord_root_ratio
    
    # Then add each chord tone (using just intonation ratios within the chord)
    # BOOST the chord root to emphasize the chord root ↔ home note relationship
    chord_ratios = JUST_RATIOS[chord_type]
    for i, ratio in enumerate(chord_ratios):
        freq = chord_base * ratio
        # First ratio (1/1) is the chord root - boost it
        amplitude = CHORD_ROOT_BOOST if i == 0 else 1.0
        sig += amplitude * np.sin(2.0 * math.pi * freq * t)
    
    # Normalize
    if np.max(np.abs(sig)) > 0:
        sig = sig / np.max(np.abs(sig))
    
    return sig


def compute_combined_recurrence(
    home_root: int,
    chord_root: int,
    chord_type: str,
    base_freq: float = C3_FREQ,
) -> float:
    """
    Compute mean recurrence when playing chord simultaneously with home root note.
    """
    sr = get_sr_for_freq(base_freq)
    
    sig = generate_combined_signal(
        home_root=home_root,
        chord_root=chord_root,
        chord_type=chord_type,
        base_freq=base_freq,
        duration=DURATION,
        sr=sr,
    )
    
    # Sliding window RQA
    results = []
    i = 0
    while i + WINDOW <= len(sig):
        w = sig[i: i + WINDOW]
        pr = percent_recurrence_single_window(w)
        results.append(pr)
        i += SHIFT
    
    return float(np.mean(results)) if results else 0.0


def format_chord_name(root: int, chord_type: str) -> str:
    """Format chord name like 'C_Maj' or 'A_min'."""
    note = NOTE_NAMES[root % 12]
    type_abbrev = "Maj" if chord_type == "Major" else "min"
    return f"{note}_{type_abbrev}"


def generate_distance_table(
    home_root: int = 0,
) -> Dict[str, Dict]:
    """
    Generate distance-from-home table for all chords.
    Uses single root note as reference, not a chord.
    
    Returns:
        Dictionary: {chord_name: {recurrence, normalized_recurrence, distance}}
    """
    print(f"Computing distances from home root: {NOTE_NAMES[home_root]}")
    print("=" * 60)
    
    # First, compute recurrence for home root + home Major chord (reference)
    # This gives us the "most consonant" baseline
    print(f"Computing reference ({NOTE_NAMES[home_root]} root + {NOTE_NAMES[home_root]} Major)...")
    
    home_recurrence = compute_combined_recurrence(
        home_root=home_root,
        chord_root=home_root,
        chord_type="Major",
    )
    print(f"  {NOTE_NAMES[home_root]} + {NOTE_NAMES[home_root]}_Maj: recurrence = {home_recurrence:.6f}")
    
    # Compute for all chords
    results = {}
    
    for root in range(12):
        for chord_type in CHORD_TYPES:
            chord_name = format_chord_name(root, chord_type)
            print(f"Computing: {NOTE_NAMES[home_root]} + {chord_name}...", end=" ")
            
            recurrence = compute_combined_recurrence(
                home_root=home_root,
                chord_root=root,
                chord_type=chord_type,
            )
            
            # Normalize: home_recurrence -> 1.0
            if home_recurrence > 0:
                normalized = recurrence / home_recurrence
            else:
                normalized = 0.0
            
            # Distance: 1.0 - normalized (so home = 0.0, dissonant = 1.0)
            distance = 1.0 - normalized
            distance = max(0.0, min(1.0, distance))  # Clamp to [0, 1]
            
            results[chord_name] = {
                "root": root,
                "type": chord_type,
                "recurrence": recurrence,
                "normalized_recurrence": normalized,
                "distance": distance,
            }
            
            print(f"rec={recurrence:.4f}, norm={normalized:.4f}, dist={distance:.4f}")
    
    return results


def save_distance_table(
    results: Dict[str, Dict],
    home_root: int,
    output_path: str = "results/rqa_distance_from_home.csv",
):
    """Save distance table to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(f"# Distance from home root: {NOTE_NAMES[home_root]}\n")
        f.write("chord,root,type,recurrence,normalized_recurrence,distance\n")
        
        # Sort by distance
        sorted_chords = sorted(results.items(), key=lambda x: x[1]["distance"])
        
        for chord_name, data in sorted_chords:
            f.write(f"{chord_name},{data['root']},{data['type']},"
                    f"{data['recurrence']:.6f},{data['normalized_recurrence']:.6f},"
                    f"{data['distance']:.6f}\n")
    
    print(f"\nResults saved to: {output_path}")


def print_summary_table(results: Dict[str, Dict], home_root: int):
    """Print a formatted summary table."""
    print("\n" + "=" * 70)
    print(f"DISTANCE FROM HOME ROOT: {NOTE_NAMES[home_root]}")
    print("=" * 70)
    print(f"{'Chord':12s} | {'Recurrence':>12s} | {'Normalized':>12s} | {'Distance':>10s}")
    print("-" * 70)
    
    # Sort by distance
    sorted_chords = sorted(results.items(), key=lambda x: x[1]["distance"])
    
    for chord_name, data in sorted_chords:
        print(f"{chord_name:12s} | {data['recurrence']:>12.6f} | "
              f"{data['normalized_recurrence']:>12.4f} | {data['distance']:>10.4f}")


def generate_python_lookup(results: Dict[str, Dict], home_root: int, output_path: str):
    """Generate a Python file with the lookup table for use in optimizer."""
    
    with open(output_path, "w") as f:
        f.write('"""Auto-generated RQA distance-from-home lookup table."""\n\n')
        f.write(f"# Distance from {NOTE_NAMES[home_root]} root note (not chord)\n")
        f.write("# Higher distance = more dissonant with home root\n")
        f.write("RQA_DISTANCE_FROM_HOME = {\n")
        
        # Sort by chord name for readability
        for chord_name in sorted(results.keys()):
            data = results[chord_name]
            f.write(f'    "{chord_name}": {data["distance"]:.4f},\n')
        
        f.write("}\n")
    
    print(f"Python lookup table saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute RQA-based distance from home root note for all chords"
    )
    parser.add_argument(
        "--home-root", type=int, default=0,
        help="Home root note (0=C, 1=C#, etc.). Default: 0 (C)"
    )
    parser.add_argument(
        "--output", type=str, default="results/rqa_distance_from_home.csv",
        help="Output CSV path. Default: results/rqa_distance_from_home.csv"
    )
    parser.add_argument(
        "--python-output", type=str, default="results/distance_lookup.py",
        help="Output Python lookup table path. Default: results/distance_lookup.py"
    )
    
    args = parser.parse_args()
    
    # Generate distance table
    results = generate_distance_table(
        home_root=args.home_root,
    )
    
    # Save CSV
    save_distance_table(
        results=results,
        home_root=args.home_root,
        output_path=args.output,
    )
    
    # Save Python lookup
    generate_python_lookup(results, args.home_root, args.python_output)
    
    # Print summary
    print_summary_table(results, args.home_root)


if __name__ == "__main__":
    main()
