#!/usr/bin/env python3
"""
Script to compute normalized RQA distance from home (C3 as bass) for all major and minor triads in the chromatic scale (C3 as root), using chord_progression_setup.py methods.

- I (C3 major) is considered 0 distance (normalized recurrence = 1.0)
- For each chromatic root (C3, C#3, ..., B3), compute both major and minor triads
- Print a table of Roman numeral, chord type, notes, frequencies, normalized RQA recurrence, and distance from home
- Sort table by descending normalized RQA recurrence (most consonant first)
"""

import sys
import math
import numpy as np
from scipy.spatial.distance import pdist
from chord_progression_setup import (
    get_chromatic_scale_frequencies,
    compute_distance_from_home_with_boost,
    compute_triad_rqa,
    freq_to_note_name,
    time_delay_embed,
    snap_to_just_intonation,
    SAMPLES_PER_CYCLE,
    RQA_DURATION,
    RQA_WINDOW,
    RQA_SHIFT,
    RQA_EMB_DIM,
    RQA_DELAY,
    RQA_EPS_FACTOR
)

# Roman numerals for 7 diatonic degrees in C major
ROMAN_NUMERALS = [
    "I",  "II",  "III", "IV", "V",  "VI",  "VII"
]

# Diatonic scale degree indices in chromatic scale (C major)
DIATONIC_INDICES = [0, 2, 4, 5, 7, 9, 11]  # C, D, E, F, G, A, B


def percent_recurrence_single_window(x: np.ndarray) -> float:
    """Compute percent recurrence for a single window."""
    V = time_delay_embed(x, RQA_EMB_DIM, RQA_DELAY)
    Nvec = V.shape[0]
    if Nvec < 2:
        return 0.0
    dists = pdist(V, metric='euclidean')
    avg_dist = np.mean(dists) if dists.size > 0 else 0.0
    eps = RQA_EPS_FACTOR * avg_dist
    rec_count = np.sum(dists <= eps)
    max_pairs = Nvec * (Nvec - 1) / 2.0
    return float(rec_count / max_pairs) if max_pairs > 0 else 0.0


def get_major_triad_indices(root_idx):
    # Major triad: root, major third, perfect fifth (0, +4, +7)
    return [root_idx, (root_idx + 4) % 12, (root_idx + 7) % 12]

def get_minor_triad_indices(root_idx):
    # Minor triad: root, minor third, perfect fifth (0, +3, +7)
    return [root_idx, (root_idx + 3) % 12, (root_idx + 7) % 12]


def compute_chord_rqa_normalized(fixed_root_freq: float, chord_root_freq: float, all_frequencies: List[float]) -> Tuple[float, float]:
    """
    Compute RQA for a chord using FIXED sample rate (for consistency across all chords),
    but preserve the chord's interval structure relative to its actual root.
    
    Also computes bass stability weight to penalize inversions.
    
    This ensures:
    - All major triads have identical RQA values (same interval structure)
    - Sample rate stays fixed for fair comparison
    - Inversions are penalized based on bass stability
    
    Args:
        fixed_root_freq: The reference frequency for sample rate normalization (e.g., home note)
        chord_root_freq: The actual root of the chord
        all_frequencies: All frequencies in the chord
    
    Returns:
        Tuple of (rqa_score, bass_weight)
        - rqa_score: RQA recurrence (0-1, higher = more stable)
        - bass_weight: Bass stability weight (0.4-1.0, higher = more stable in root position)
    """
    if len(all_frequencies) < 2:
        return 0.0, 0.0
    
    # Use FIXED root frequency for sample rate, ensuring consistency
    sr = int(SAMPLES_PER_CYCLE * fixed_root_freq)
    
    # Convert frequencies to ratios relative to CHORD ROOT (not fixed_root_freq)
    raw_ratios = [f / chord_root_freq for f in all_frequencies]
    ratios = [snap_to_just_intonation(r) for r in raw_ratios]
    
    t = np.arange(0, RQA_DURATION, 1.0 / sr)
    sig = np.zeros_like(t)
    
    # Sum all frequency components using SNAPPED ratios
    # Use the actual chord root frequency to synthesize the signal
    for i, ratio in enumerate(ratios):
        f = chord_root_freq * ratio
        sig += np.sin(2.0 * math.pi * f * t)
    
    # Normalize
    max_val = np.max(np.abs(sig))
    if max_val > 0:
        sig = sig / max_val
    
    # Sliding window RQA
    results = []
    i = 0
    while i + RQA_WINDOW <= len(sig):
        w = sig[i: i + RQA_WINDOW]
        pr = percent_recurrence_single_window(w)
        results.append(pr)
        i += RQA_SHIFT
    
    rqa_score = float(np.mean(results)) if results else 0.0
    
    # Calculate bass stability weight
    # The bass note (lowest frequency) determines voice leading stability
    # Root position (bass = root): weight = 1.0
    # Inversions (bass ≠ root): weight < 1.0 (penalized)
    bass_freq = min(all_frequencies)
    # Use 1% tolerance for floating-point comparison
    is_root_position = abs(bass_freq - chord_root_freq) / chord_root_freq < 0.01
    bass_weight = 1.0 if is_root_position else 0.7
    
    return rqa_score, bass_weight


def main():
    frequencies, note_names = get_chromatic_scale_frequencies(octave=3)
    home_freq = frequencies[0]  # C3
    home_name = note_names[0]

    results = []

    # CRITICAL: Use a FIXED root frequency for all RQA computations to ensure consistent sample rate
    # This ensures all major triads have the same RQA value (same interval structure)
    fixed_root_freq = home_freq  # Use home as the fixed reference for sample rate normalization

    # Step 1: Compute baseline RQA values for chord I (major on C3)
    maj_indices_I = get_major_triad_indices(0)
    maj_freqs_I = [frequencies[j] for j in maj_indices_I]
    chord_root_I = maj_freqs_I[0]  # C3 (actual root)
    
    # RQA of chord I by itself (no home note) - using FIXED sample rate for consistency
    rqa_I_chord_alone, bass_weight_I = compute_chord_rqa_normalized(fixed_root_freq, chord_root_I, maj_freqs_I)
    
    # RQA of chord I with home note added
    rqa_I_chord_with_home, _ = compute_distance_from_home_with_boost(
        home_freq, maj_freqs_I[0], maj_freqs_I, home_note_name=home_name
    )
    if rqa_I_chord_with_home <= 0:
        rqa_I_chord_with_home = 1e-12

    for idx, i in enumerate(DIATONIC_INDICES):
        # Major triad
        maj_indices = get_major_triad_indices(i)
        maj_freqs_unsorted = [frequencies[j] for j in maj_indices]
        maj_names = [note_names[j] for j in maj_indices]
        roman = ROMAN_NUMERALS[idx]
        chord_type = "Major"
        
        # The chord root is the harmonic root (the i-th degree of chromatic scale)
        # NOT necessarily the lowest note (which would be an inversion)
        harmonic_root = maj_freqs_unsorted[0]  # Root of the chord (e.g., A for VI)
        
        # Sort frequencies for consistent voicing in RQA calculation
        maj_freqs = sorted(maj_freqs_unsorted)
        bass_note = maj_freqs[0]  # Lowest frequency is the bass
        
        # RQA of chord by itself (without home) - using FIXED sample rate for consistency
        # Use harmonic root for ratio calculation to maintain interval structure
        rqa_chord_alone, bass_weight = compute_chord_rqa_normalized(fixed_root_freq, harmonic_root, maj_freqs)
        
        # RQA of chord with home note added
        # CRITICAL: Use harmonic_root for CHORD_ROOT_BOOST, not bass_note
        rqa_chord_with_home, distance = compute_distance_from_home_with_boost(
            home_freq, harmonic_root, maj_freqs, home_note_name=home_name
        )
        
        # Apply bass weight to RQA+home as well (penalize inversions)
        weighted_rqa_with_home = rqa_chord_with_home * bass_weight
        
        # Normalize based on RQA AFTER home is added (measures "distance from home")
        norm_rqa_maj = weighted_rqa_with_home / rqa_I_chord_with_home if rqa_I_chord_with_home > 0 else weighted_rqa_with_home
        
        # Show how much adding home affects the RQA (delta)
        rqa_delta = rqa_chord_with_home / rqa_chord_alone if rqa_chord_alone > 0 else 0
        
        results.append({
            "roman": roman,
            "type": chord_type,
            "notes": maj_names,
            "norm_rqa": norm_rqa_maj,
            "distance": distance,
            "rqa_chord_alone": rqa_chord_alone,
            "bass_weight": bass_weight,
            "rqa_with_home": rqa_chord_with_home,
            "home_effect": rqa_delta
        })

        # Minor triad
        min_indices = get_minor_triad_indices(i)
        min_freqs_unsorted = [frequencies[j] for j in min_indices]
        min_names = [note_names[j] for j in min_indices]
        roman_minor = roman.lower()  # Use lowercase for minor
        chord_type_minor = "minor"
        
        # The chord root is the harmonic root (the i-th degree)
        harmonic_root = min_freqs_unsorted[0]
        
        # Sort frequencies for consistent voicing
        min_freqs = sorted(min_freqs_unsorted)
        
        # RQA of chord by itself (without home) - using FIXED sample rate for consistency
        rqa_chord_alone, bass_weight = compute_chord_rqa_normalized(fixed_root_freq, harmonic_root, min_freqs)
        
        # RQA of chord with home note added
        rqa_chord_with_home, distance = compute_distance_from_home_with_boost(
            home_freq, harmonic_root, min_freqs, home_note_name=home_name
        )
        
        # Apply bass weight to RQA+home as well (penalize inversions)
        weighted_rqa_with_home = rqa_chord_with_home * bass_weight
        
        # Normalize based on RQA AFTER home is added (measures "distance from home")
        norm_rqa_min = weighted_rqa_with_home / rqa_I_chord_with_home if rqa_I_chord_with_home > 0 else weighted_rqa_with_home
        
        # Show how much adding home affects the RQA (delta)
        rqa_delta = rqa_chord_with_home / rqa_chord_alone if rqa_chord_alone > 0 else 0
        
        results.append({
            "roman": roman_minor,
            "type": chord_type_minor,
            "notes": min_names,
            "norm_rqa": norm_rqa_min,
            "distance": distance,
            "rqa_chord_alone": rqa_chord_alone,
            "bass_weight": bass_weight,
            "rqa_with_home": rqa_chord_with_home,
            "home_effect": rqa_delta
        })

    # Sort by normalized RQA (all major triads should have same RQA by themselves)
    results.sort(key=lambda x: x["norm_rqa"], reverse=True)

    # Print table with clear labels
    # NormToI is NOW normalized after home boost is applied (measures "distance from home")
    print(f"{'Roman':<6} {'Type':<8} {'Notes':<20} {'RQA(alone)':<12} {'RQA(with home)':<12}  {'Norm RQA with home':<12}")
    print("-" * 120)
    for row in results:
        notes_str = "-".join(row["notes"])
        print(
            f"{row['roman']:<6} {row['type']:<8} {notes_str:<20} "
            f"{row.get('rqa_chord_alone', 0):<12.3f} "
            f"{row.get('rqa_with_home', 0):<12.3f} "
             f"{row['norm_rqa']:<12.3f}"
        )

if __name__ == "__main__":
    main()
