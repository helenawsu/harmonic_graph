#!/usr/bin/env python3
"""
Tension Resolution Analysis

Analyzes the change in tension when resolving from all diatonic major and minor 
chords in C major to the tonic (I - C major).

Tension is defined by:
- RQA-based distance from home (C3)
- Weighted by bass stability (root position vs inversions)
- Normalized by I chord's RQA value

Tension change is normalized by voice leading distance using a soft denominator:
    tension_change_rate = (tension_chord - tension_I) / sqrt(1 + semitone_distance^2)

This penalizes large voice leading jumps while allowing comparison across different resolutions.
"""

#!/usr/bin/env python3
import sys
import math
import numpy as np
from typing import List, Tuple, Dict
from scipy.spatial.distance import pdist

from chord_progression_setup import (
    get_chromatic_scale_frequencies,
    compute_distance_from_home_with_boost,
    time_delay_embed,
    snap_to_just_intonation,
    SAMPLES_PER_CYCLE,
    RQA_DURATION,
    RQA_WINDOW,
    RQA_SHIFT,
    RQA_EMB_DIM,
    RQA_DELAY,
    RQA_EPS_FACTOR,
    CHORD_ROOT_BOOST
)

from chord_progression_optimizer import voice_leading_cost_freq, soft_voice_leading_denominator

# Roman numerals for 7 diatonic degrees in C major
ROMAN_NUMERALS = ["I", "II", "III", "IV", "V", "VI", "VII"]

# Diatonic scale degree indices in chromatic scale (C major)
DIATONIC_INDICES = [0, 2, 4, 5, 7, 9, 11]  # C, D, E, F, G, A, B


def get_major_triad_indices(root_idx: int) -> List[int]:
    """Major triad: root, major third, perfect fifth (0, +4, +7)"""
    return [root_idx, (root_idx + 4) % 12, (root_idx + 7) % 12]


def get_minor_triad_indices(root_idx: int) -> List[int]:
    """Minor triad: root, minor third, perfect fifth (0, +3, +7)"""
    return [root_idx, (root_idx + 3) % 12, (root_idx + 7) % 12]


def calculate_voice_leading_distance(chord_freqs: List[float], target_freqs: List[float]) -> float:
    """
    Calculate the total semitone distance for optimal voice leading between two chords.
    The voice_leading_cost_freq function already handles octave equivalence through
    pitch class (mod 12) calculation, so no manual octave transposition is needed.
    
    Args:
        chord_freqs: List of frequencies (Hz) in source chord (unsorted, in voicing order)
        target_freqs: List of frequencies (Hz) in target chord (unsorted, in voicing order)
    
    Returns:
        Total semitone distance across all voices (optimal greedy matching)
    """
    return voice_leading_cost_freq(chord_freqs, target_freqs)


def calculate_tension(
    home_freq: float,
    chord_root_freq: float,
    chord_frequencies: List[float],
    fixed_root_freq: float
) -> Tuple[float, float, float]:
    """
    Calculate tension for a chord using RQA-based distance from home.
    
    Pipeline:
    1. Calculate raw RQA with home note boost (home + chord)
    2. Apply bass weight (penalize inversions)
    3. Return raw weighted RQA (to be normalized by I chord later)
    
    Args:
        home_freq: Home reference frequency (C3)
        chord_root_freq: The harmonic root of the chord
        chord_frequencies: All frequencies in the chord
        fixed_root_freq: Fixed reference for sample rate normalization
    
    Returns:
        Tuple of (raw_rqa_with_home, weighted_rqa, bass_weight)
        - raw_rqa_with_home: Raw RQA recurrence with home boost (before bass weight)
        - weighted_rqa: RQA after applying bass weight
        - bass_weight: Bass stability weight (1.0 = root position)
    """
    # Calculate bass stability weight
    bass_freq = min(chord_frequencies)
    is_root_position = abs(bass_freq - chord_root_freq) / chord_root_freq < 0.01
    bass_weight = 1.0 if is_root_position else 0.7
    
    # Compute RQA with home note added (returns raw RQA recurrence)
    rqa_with_home, _ = compute_distance_from_home_with_boost(
        home_freq, chord_root_freq, chord_frequencies
    )
    
    # Apply bass weight to RQA
    weighted_rqa = rqa_with_home * bass_weight
    
    return rqa_with_home, weighted_rqa, bass_weight


def main():
    print("=" * 100)
    print("TENSION RESOLUTION ANALYSIS")
    print("Analyzing tension changes for diatonic chords resolving to I in C major")
    print("=" * 100)
    
    # Get chromatic scale
    frequencies, note_names = get_chromatic_scale_frequencies(octave=3)
    home_freq = frequencies[0]  # C3
    home_name = note_names[0]
    
    # Fixed root for sample rate normalization
    fixed_root_freq = home_freq
    
    # Get I chord (C major) as the resolution target
    I_indices = get_major_triad_indices(0)
    I_freqs_unsorted = [frequencies[j] for j in I_indices]
    I_root = I_freqs_unsorted[0]  # C3
    I_freqs = sorted(I_freqs_unsorted)  # For RQA calculation
    I_names = [note_names[j] for j in I_indices]
    
    # Calculate RQA for I chord (baseline for normalization)
    I_rqa_raw, I_rqa_weighted, I_bass_weight = calculate_tension(home_freq, I_root, I_freqs, fixed_root_freq)
    
    # Normalized tension for I is 0 (by definition - no tension at home)
    I_tension_normalized = 0.0
    
    print(f"\nTonic (I - {'-'.join(I_names)}):")
    print(f"  Raw RQA with home: {I_rqa_raw:.3f}")
    print(f"  Weighted RQA: {I_rqa_weighted:.3f}")
    print(f"  Bass Weight: {I_bass_weight:.2f}")
    print(f"  Normalized Tension: {I_tension_normalized:.3f} (baseline)")
    print("\n" + "=" * 100)
    
    results = []
    
    # Analyze all diatonic chords (major and minor)
    for idx, chromatic_idx in enumerate(DIATONIC_INDICES):
        roman = ROMAN_NUMERALS[idx]
        
        # Major triad
        maj_indices = get_major_triad_indices(chromatic_idx)
        maj_freqs_unsorted = [frequencies[j] for j in maj_indices]
        maj_root = maj_freqs_unsorted[0]
        maj_freqs_sorted = sorted(maj_freqs_unsorted)  # For RQA calculation
        maj_names = [note_names[j] for j in maj_indices]
        
        # Calculate RQA with home boost (uses sorted frequencies for consistency)
        maj_rqa_raw, maj_rqa_weighted, maj_bass_weight = calculate_tension(
            home_freq, maj_root, maj_freqs_sorted, fixed_root_freq
        )
        
        # Normalize by I chord: tension = 1 - (weighted_rqa / I_weighted_rqa)
        # Higher RQA = more consonant = less tension
        # I chord has tension 0, chords far from I have high tension
        maj_tension_normalized = 1.0 - (maj_rqa_weighted / I_rqa_weighted) if I_rqa_weighted > 0 else 0.0
        
        # Calculate voice leading distance to I (using UNSORTED frequencies to preserve voicing)
        vl_distance = calculate_voice_leading_distance(maj_freqs_unsorted, I_freqs_unsorted)
        
        # Tension change with soft denominator (normalized tension difference)
        # Negative = tension decreases when resolving to I
        tension_change = I_tension_normalized - maj_tension_normalized
        soft_denom = soft_voice_leading_denominator(vl_distance)
        tension_change_rate = tension_change / soft_denom if soft_denom > 0 else 0.0
        
        results.append({
            "roman": roman,
            "type": "Major",
            "notes": maj_names,
            "rqa_raw": maj_rqa_raw,
            "rqa_weighted": maj_rqa_weighted,
            "tension": maj_tension_normalized,
            "bass_weight": maj_bass_weight,
            "tension_change": tension_change,
            "vl_distance": vl_distance,
            "tension_change_rate": tension_change_rate
        })
        
        # Minor triad
        min_indices = get_minor_triad_indices(chromatic_idx)
        min_freqs_unsorted = [frequencies[j] for j in min_indices]
        min_root = min_freqs_unsorted[0]
        min_freqs_sorted = sorted(min_freqs_unsorted)  # For RQA calculation
        min_names = [note_names[j] for j in min_indices]
        
        # Calculate RQA with home boost (uses sorted frequencies for consistency)
        min_rqa_raw, min_rqa_weighted, min_bass_weight = calculate_tension(
            home_freq, min_root, min_freqs_sorted, fixed_root_freq
        )
        
        # Normalize by I chord: tension = 1 - (weighted_rqa / I_weighted_rqa)
        min_tension_normalized = 1.0 - (min_rqa_weighted / I_rqa_weighted) if I_rqa_weighted > 0 else 0.0
        
        # Calculate voice leading distance to I (using UNSORTED frequencies to preserve voicing)
        vl_distance = calculate_voice_leading_distance(min_freqs_unsorted, I_freqs_unsorted)
        
        # Tension change with soft denominator (normalized tension difference)
        # Negative = tension decreases when resolving to I
        tension_change = I_tension_normalized - min_tension_normalized
        soft_denom = soft_voice_leading_denominator(vl_distance)
        tension_change_rate = tension_change / soft_denom if soft_denom > 0 else 0.0
        
        results.append({
            "roman": roman.lower(),
            "type": "minor",
            "notes": min_names,
            "rqa_raw": min_rqa_raw,
            "rqa_weighted": min_rqa_weighted,
            "tension": min_tension_normalized,
            "bass_weight": min_bass_weight,
            "tension_change": tension_change,
            "vl_distance": vl_distance,
            "tension_change_rate": tension_change_rate
        })
    
    # Sort by tension change rate (descending = most negative first = strongest resolution)
    results.sort(key=lambda x: x["tension_change_rate"])
    
    # Print results table
    print(f"\n{'Roman':<8} {'Type':<8} {'Notes':<20} {'RQA(raw)':<10} {'RQA(wtd)':<10} {'Tension':<10} {'Bass':<6} {'ΔT':<10} {'VL':<6} {'ΔT Rate':<10}")
    print("-" * 110)
    
    for row in results:
        notes_str = "-".join(row["notes"])
        print(
            f"{row['roman']:<8} {row['type']:<8} {notes_str:<20} "
            f"{row['rqa_raw']:<10.3f} {row['rqa_weighted']:<10.3f} "
            f"{row['tension']:<10.3f} {row['bass_weight']:<6.2f} "
            f"{row['tension_change']:<10.3f} {row['vl_distance']:<6.1f} "
            f"{row['tension_change_rate']:<10.3f}"
        )
    
    print("\n" + "=" * 110)
    print("Legend:")
    print("  Roman     : Roman numeral (uppercase = major, lowercase = minor)")
    print("  Type      : Chord quality (Major or minor)")
    print("  Notes     : Note names in the chord")
    print("  RQA(raw)  : Raw RQA recurrence with home boost (before bass weight)")
    print("  RQA(wtd)  : RQA after applying bass weight")
    print("  Tension   : Normalized tension = 1 - (RQA_weighted / I_RQA_weighted)")
    print("              0 = no tension (at home), higher = more tension")
    print("  Bass      : Bass stability weight (1.0 = root position, 0.7 = inversion)")
    print("  ΔT        : Tension change when resolving to I (I tension - chord tension)")
    print("              Negative = tension decreases (resolution), positive = tension increases")
    print("  VL        : Voice leading distance in semitones (optimal voice leading)")
    print("  ΔT Rate   : Tension change rate = ΔT / soft_denom")
    print("              soft_denom = (log2(VL) + 1) / 4 (logarithmic scaling)")
    print("=" * 110)
    
    # Print resolution table showing each chord resolving to I
    print("\n" + "=" * 110)
    print("RESOLUTION TO I - TENSION CHANGES")
    print("=" * 110)
    print(f"\n{'Progression':<25} {'From Tension':<15} {'To Tension':<15} {'ΔT':<10} {'VL Dist':<10} {'ΔT Rate':<10}")
    print("-" * 110)
    
    for row in results:
        if row['roman'] == 'I' and row['type'] == 'Major':
            continue  # Skip I → I
        
        progression = f"{row['roman']} ({'-'.join(row['notes'])}) → I"
        from_tension = row['tension']
        to_tension = I_tension_normalized
        delta_t = row['tension_change']
        vl_dist = row['vl_distance']
        delta_t_rate = row['tension_change_rate']
        
        print(
            f"{progression:<25} {from_tension:<15.3f} {to_tension:<15.3f} "
            f"{delta_t:<10.3f} {vl_dist:<10.1f} {delta_t_rate:<10.3f}"
        )
    
    print("=" * 110)
    
    # Find most interesting resolutions
    print("\n" + "=" * 110)
    print("ANALYSIS INSIGHTS")
    print("=" * 110)
    
    # Strongest resolution (highest tension change rate)
    strongest = min(results, key=lambda x: x["tension_change_rate"])  # Most negative = strongest
    print(f"\nStrongest Resolution (most negative ΔT rate):")
    print(f"  {strongest['roman']} ({'-'.join(strongest['notes'])}) → I")
    print(f"  ΔT Rate = {strongest['tension_change_rate']:.3f}")
    print(f"  Tension: {strongest['tension']:.3f} → {I_tension_normalized:.3f}")
    print(f"  Voice leading: {strongest['vl_distance']:.1f} semitones")
    
    # Most efficient resolution (high tension change, low voice leading)
    efficiency_score = [(abs(r["tension_change"]) / (1 + r["vl_distance"]), r) for r in results]
    efficiency_score.sort(key=lambda x: x[0], reverse=True)
    most_efficient = efficiency_score[0][1]
    print(f"\nMost Efficient Resolution (high |ΔT|, low VL):")
    print(f"  {most_efficient['roman']} ({'-'.join(most_efficient['notes'])}) → I")
    print(f"  ΔT = {most_efficient['tension_change']:.3f}")
    print(f"  Voice leading: {most_efficient['vl_distance']:.1f} semitones")
    print(f"  Efficiency = {efficiency_score[0][0]:.3f}")
    
    # Smoothest resolution (lowest voice leading distance among high-tension chords)
    high_tension = [r for r in results if r["tension"] > I_tension_normalized + 0.01]
    if high_tension:
        smoothest = min(high_tension, key=lambda x: x["vl_distance"])
        print(f"\nSmoothest High-Tension Resolution:")
        print(f"  {smoothest['roman']} ({'-'.join(smoothest['notes'])}) → I")
        print(f"  Tension: {smoothest['tension']:.3f} → {I_tension_normalized:.3f}")
        print(f"  Voice leading: {smoothest['vl_distance']:.1f} semitones (minimal)")
    
    # Export to CSV
    import csv
    from pathlib import Path
    
    output_path = "results/tension_resolution_analysis.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Roman", "Type", "Notes", "RQA Raw", "RQA Weighted", "Tension", "Bass Weight",
            "Tension Change", "VL Distance", "Tension Change Rate"
        ])
        
        for row in results:
            writer.writerow([
                row["roman"],
                row["type"],
                "-".join(row["notes"]),
                f"{row['rqa_raw']:.3f}",
                f"{row['rqa_weighted']:.3f}",
                f"{row['tension']:.3f}",
                f"{row['bass_weight']:.2f}",
                f"{row['tension_change']:.3f}",
                f"{row['vl_distance']:.1f}",
                f"{row['tension_change_rate']:.3f}"
            ])
    
    print(f"\n✓ Results exported to: {output_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
