#!/usr/bin/env python3
"""
Arbitrary Chord Progression Analyzer

Extends chord_progression_optimizer.py to work with arbitrary frequency inputs.
Given a set of frequencies (max 12 due to exponential runtime), this module:
1. Finds all possible triads (n choose 3 combinations)
2. Scores each triad by the stability of its frequency ratios
3. Identifies the "Major" equivalent (most stable) and "Minor" equivalent (2nd most stable)

Stability is based on how close frequency ratios are to simple integer ratios
like 2:3 (perfect fifth), 3:4 (perfect fourth), 4:5 (major third), 5:6 (minor third).

Two-stage approach:
1. Ratio-based pruning: Get top 3 candidates quickly
2. RQA refinement: Use recurrence analysis to rank the top 3
"""

import math
import numpy as np
from itertools import combinations
from typing import List, Tuple, Dict
from scipy.spatial.distance import pdist

# Simple integer ratios considered "stable" in just intonation
# Format: (numerator, denominator, name, stability_score)
# Lower stability_score = more consonant/stable
STABLE_RATIOS = [
    (1, 1, "unison", 0.0),
    (2, 1, "octave", 0.1),
    (3, 2, "perfect fifth", 0.2),
    (4, 3, "perfect fourth", 0.3),
    (5, 4, "major third", 0.4),
    (6, 5, "minor third", 0.5),
    (5, 3, "major sixth", 0.6),
    (8, 5, "minor sixth", 0.7),
    (9, 8, "major second", 0.8),
    (16, 15, "minor second", 0.9),
    (15, 8, "major seventh", 0.95),
    (16, 9, "minor seventh", 0.85),
]

# Just the ratios for quick lookup (for snapping)
JUST_RATIOS = {ratio[0]/ratio[1]: (ratio[0], ratio[1], ratio[2]) for ratio in STABLE_RATIOS}

# Tolerance for matching ratios (e.g., 1.5 vs 1.501 should still match 3:2)
RATIO_TOLERANCE = 0.02  # 2% tolerance


def snap_to_just_intonation(ratio: float, tolerance: float = RATIO_TOLERANCE) -> float:
    """
    Snap a frequency ratio to the nearest just intonation ratio within tolerance.
    
    Args:
        ratio: The frequency ratio to snap (should be >= 1.0)
        tolerance: Maximum deviation allowed (default 2%)
    
    Returns:
        The snapped just intonation ratio, or original if no match
    """
    # Octave-reduce to 1.0-2.0 range
    octave_reduced = ratio
    octaves = 0
    while octave_reduced >= 2.0:
        octave_reduced /= 2.0
        octaves += 1
    while octave_reduced < 1.0:
        octave_reduced *= 2.0
        octaves -= 1
    
    best_match = None
    best_deviation = float('inf')
    
    for just_ratio in JUST_RATIOS.keys():
        if just_ratio < 1.0 or just_ratio >= 2.0:
            continue  # Only check ratios in 1.0-2.0 range
        
        deviation = abs(octave_reduced - just_ratio) / just_ratio
        if deviation <= tolerance and deviation < best_deviation:
            best_deviation = deviation
            best_match = just_ratio
    
    if best_match is not None:
        # Return the snapped ratio, restored to original octave
        return best_match * (2.0 ** octaves)
    
    return ratio  # No match, return original


# =============================================================================
# RQA (Recurrence Quantification Analysis) - for final ranking
# Adapted from rqa_all_roots.py with 1 second duration for speed
# =============================================================================

# RQA parameters
BASELINE_FREQ = 400.0
BASELINE_SR = 8000
SAMPLES_PER_CYCLE = BASELINE_SR / BASELINE_FREQ  # = 20

RQA_DURATION = 1.0  # 1 second (reduced from 6 for speed)
RQA_WINDOW = 480
RQA_SHIFT = 48
RQA_EMB_DIM = 5
RQA_DELAY = 3
RQA_EPS_FACTOR = 0.1


def get_sr_for_freq(root_freq: float) -> int:
    """Scale sample rate so each root frequency gets same samples/cycle as baseline."""
    return int(SAMPLES_PER_CYCLE * root_freq)


def time_delay_embed(x: np.ndarray, emb_dim: int, delay: int) -> np.ndarray:
    """Create time-delay embedding matrix."""
    N = len(x)
    L = N - (emb_dim - 1) * delay
    if L <= 0:
        return np.empty((0, emb_dim))
    M = np.empty((L, emb_dim))
    for i in range(emb_dim):
        M[:, i] = x[i * delay: i * delay + L]
    return M


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


def compute_triad_rqa(root_freq: float, all_frequencies: List[float]) -> float:
    """
    Compute RQA recurrence for a triad with explicit root frequency.
    Snaps ratios to just intonation before synthesizing (RQA is physically sensitive!).
    Higher recurrence = more stable/consonant.
    
    Args:
        root_freq: The root frequency in Hz (used for sample rate normalization)
        all_frequencies: List of all frequencies in the chord (including root)
    
    Returns:
        Mean percent recurrence (0-1, higher = more stable)
    """
    if len(all_frequencies) < 2:
        return 0.0
    
    # Normalize sample rate to root frequency (key insight from rqa_all_roots.py!)
    sr = get_sr_for_freq(root_freq)
    
    # Convert frequencies to ratios relative to root, then SNAP to just intonation!
    raw_ratios = [f / root_freq for f in all_frequencies]
    ratios = [snap_to_just_intonation(r) for r in raw_ratios]
    
    t = np.arange(0, RQA_DURATION, 1.0 / sr)
    sig = np.zeros_like(t)
    
    # Sum all frequency components using SNAPPED ratios
    for ratio in ratios:
        f = root_freq * ratio
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
    
    return float(np.mean(results)) if results else 0.0


def ratio_stability_score(freq1: float, freq2: float, tolerance: float = RATIO_TOLERANCE) -> Tuple[float, str]:
    """
    Calculate how "stable" the ratio between two frequencies is.
    
    Returns:
        Tuple of (stability_score, ratio_name)
        Lower score = more stable/consonant
    """
    if freq1 <= 0 or freq2 <= 0:
        return (float('inf'), "invalid")
    
    # Ensure freq1 <= freq2 for consistent ratio calculation
    if freq1 > freq2:
        freq1, freq2 = freq2, freq1
    
    actual_ratio = freq2 / freq1
    
    # Octave-reduce: bring ratio into 1.0 - 2.0 range
    octave_reduced = actual_ratio
    while octave_reduced >= 2.0:
        octave_reduced /= 2.0
    while octave_reduced < 1.0:
        octave_reduced *= 2.0
    
    best_score = float('inf')
    best_name = "complex"
    
    for num, denom, name, base_score in STABLE_RATIOS:
        target_ratio = num / denom
        
        # Only check the octave-reduced ratio (no inverse!)
        # This ensures we match the actual interval, not its complement
        deviation = abs(octave_reduced - target_ratio) / target_ratio
        
        if deviation <= tolerance:
            # Penalize slightly for deviation within tolerance
            adjusted_score = base_score + deviation * 0.5
            if adjusted_score < best_score:
                best_score = adjusted_score
                best_name = name
    
    return (best_score, best_name)


def triad_stability_score_rooted(root_freq: float, other_freqs: List[float]) -> Tuple[float, Dict]:
    """
    Calculate stability score for a triad with a FIXED ROOT.
    
    ALL THREE pairwise intervals are weighted equally:
    - Root to note 1
    - Root to note 2  
    - Note 1 to note 2 (this is critical - sus4 fails here with M2!)
    
    Args:
        root_freq: The root/bass frequency (fixed)
        other_freqs: List of 2 other frequencies
    
    Returns:
        Tuple of (total_score, details_dict)
        Lower score = more stable/consonant triad
    """
    if len(other_freqs) != 2:
        raise ValueError("Must provide exactly 2 other frequencies")
    
    sorted_others = sorted(other_freqs)
    all_freqs = [root_freq] + sorted_others
    
    intervals = []
    
    # Interval 1: Root to lower note
    score1, name1 = ratio_stability_score(root_freq, sorted_others[0])
    intervals.append({"from": root_freq, "to": sorted_others[0], "score": score1, "name": name1})
    
    # Interval 2: Root to upper note
    score2, name2 = ratio_stability_score(root_freq, sorted_others[1])
    intervals.append({"from": root_freq, "to": sorted_others[1], "score": score2, "name": name2})
    
    # Interval 3: Between the two other notes (EQUALLY IMPORTANT!)
    # This is why sus4 (C-F-G) fails - F to G is a major 2nd (dissonant)
    # While major (C-E-G) has E to G as minor 3rd (consonant)
    score3, name3 = ratio_stability_score(sorted_others[0], sorted_others[1])
    intervals.append({"from": sorted_others[0], "to": sorted_others[1], "score": score3, "name": name3})
    
    # Equal weight for all three intervals
    total_score = score1 + score2 + score3
    
    details = {
        "root": root_freq,
        "frequencies": all_freqs,
        "intervals": intervals,
        "total_score": total_score,
    }
    
    return (total_score, details)


def triad_stability_score(freqs: List[float]) -> Tuple[float, Dict]:
    """
    Calculate the overall stability score for a triad (3 frequencies).
    
    Returns:
        Tuple of (total_score, details_dict)
        Lower score = more stable/consonant triad
    """
    if len(freqs) != 3:
        raise ValueError("Triad must have exactly 3 frequencies")
    
    # Sort frequencies for consistent processing
    sorted_freqs = sorted(freqs)
    
    # Calculate stability for all three intervals
    intervals = []
    
    # Interval 1: lowest to middle
    score1, name1 = ratio_stability_score(sorted_freqs[0], sorted_freqs[1])
    intervals.append({"from": sorted_freqs[0], "to": sorted_freqs[1], "score": score1, "name": name1})
    
    # Interval 2: middle to highest
    score2, name2 = ratio_stability_score(sorted_freqs[1], sorted_freqs[2])
    intervals.append({"from": sorted_freqs[1], "to": sorted_freqs[2], "score": score2, "name": name2})
    
    # Interval 3: lowest to highest (the "outer" interval)
    score3, name3 = ratio_stability_score(sorted_freqs[0], sorted_freqs[2])
    intervals.append({"from": sorted_freqs[0], "to": sorted_freqs[2], "score": score3, "name": name3})
    
    # Total score is sum of all interval scores
    # We could weight them differently (e.g., outer interval more important)
    total_score = score1 + score2 + score3
    
    details = {
        "frequencies": sorted_freqs,
        "intervals": intervals,
        "total_score": total_score,
    }
    
    return (total_score, details)


def find_best_triads_per_root(
    frequencies: List[float],
    note_names: List[str] = None,
    top_n: int = 5,
    use_rqa: bool = True,
    ratio_candidates: int = 3
) -> Dict[str, List[Dict]]:
    """
    For each note as root, find the best triads using two-stage approach:
    1. Ratio-based pruning: Get top 'ratio_candidates' quickly
    2. RQA refinement: Use recurrence analysis to rank them (if use_rqa=True)
    
    Args:
        frequencies: List of frequencies in Hz (max 12 for reasonable runtime)
        note_names: Optional list of names for each frequency
        top_n: Number of top triads per root (default 2 for major/minor equivalent)
        use_rqa: Whether to use RQA for final ranking (slower but more accurate)
        ratio_candidates: Number of candidates to pass to RQA stage
    
    Returns:
        Dict mapping root note name -> list of top triads for that root
    """
    if len(frequencies) > 12:
        raise ValueError(f"Maximum 12 frequencies allowed, got {len(frequencies)}")
    
    if note_names is None:
        note_names = [f"Note{i}" for i in range(len(frequencies))]
    
    if len(note_names) != len(frequencies):
        raise ValueError("note_names must match length of frequencies")
    
    results = {}
    
    # For each note as root
    for root_idx, (root_freq, root_name) in enumerate(zip(frequencies, note_names)):
        # Get all other notes (excluding root)
        other_notes = [
            (i, frequencies[i], note_names[i])
            for i in range(len(frequencies))
            if i != root_idx
        ]
        
        triads_for_root = []
        
        # Stage 1: Generate all 2-combinations and score by ratio stability
        for combo in combinations(other_notes, 2):
            other_indices = [c[0] for c in combo]
            other_freqs = [c[1] for c in combo]
            other_names = [c[2] for c in combo]
            
            score, details = triad_stability_score_rooted(root_freq, other_freqs)
            
            triad_info = {
                "root_name": root_name,
                "root_freq": root_freq,
                "other_names": other_names,
                "other_freqs": other_freqs,
                "chord_name": f"{root_name}-{other_names[0]}-{other_names[1]}",
                "ratio_score": score,
                "score": score,  # Will be updated by RQA if enabled
                "details": details,
            }
            triads_for_root.append(triad_info)
        
        # Sort by ratio score (lower = more stable)
        triads_for_root.sort(key=lambda x: x["ratio_score"])
        
        # Stage 2: Apply RQA to top candidates for final ranking
        if use_rqa and len(triads_for_root) > 0:
            # Take top candidates based on ratio score
            candidates = triads_for_root[:ratio_candidates]
            
            # Compute RQA for each candidate (pass root_freq explicitly!)
            for triad in candidates:
                all_freqs = [triad["root_freq"]] + triad["other_freqs"]
                rqa_score = compute_triad_rqa(triad["root_freq"], all_freqs)
                # RQA: higher = more stable, so we negate for consistent sorting
                triad["rqa_score"] = rqa_score
                triad["score"] = -rqa_score  # Negate so lower = more stable
            
            # Re-sort candidates by RQA score
            candidates.sort(key=lambda x: x["score"])
            results[root_name] = candidates[:top_n]
        else:
            results[root_name] = triads_for_root[:top_n]
    
    return results


def find_best_triads(
    frequencies: List[float],
    note_names: List[str] = None,
    top_n: int = 10
) -> List[Dict]:
    """
    Find the best (most stable) triads from a set of frequencies.
    
    Args:
        frequencies: List of frequencies in Hz (max 12 for reasonable runtime)
        note_names: Optional list of names for each frequency
        top_n: Number of top triads to return
    
    Returns:
        List of top triads sorted by stability score (most stable first)
    """
    if len(frequencies) > 12:
        raise ValueError(f"Maximum 12 frequencies allowed, got {len(frequencies)}")
    
    if note_names is None:
        note_names = [f"Note{i}" for i in range(len(frequencies))]
    
    if len(note_names) != len(frequencies):
        raise ValueError("note_names must match length of frequencies")
    
    # Create indexed list for tracking which notes form each triad
    indexed_freqs = list(enumerate(zip(frequencies, note_names)))
    
    all_triads = []
    
    # Generate all 3-combinations
    for combo in combinations(indexed_freqs, 3):
        indices = [c[0] for c in combo]
        freqs = [c[1][0] for c in combo]
        names = [c[1][1] for c in combo]
        
        score, details = triad_stability_score(freqs)
        
        triad_info = {
            "indices": indices,
            "names": names,
            "frequencies": freqs,
            "score": score,
            "details": details,
        }
        all_triads.append(triad_info)
    
    # Sort by stability score (lower = more stable)
    all_triads.sort(key=lambda x: x["score"])
    
    return all_triads[:top_n]


def freq_to_note_name(freq: float, reference_a4: float = 440.0) -> str:
    """Convert frequency to nearest note name."""
    if freq <= 0:
        return "?"
    
    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
    # Calculate semitones from A4
    semitones_from_a4 = 12 * math.log2(freq / reference_a4)
    midi_note = round(69 + semitones_from_a4)
    
    note_name = NOTE_NAMES[midi_note % 12]
    octave = (midi_note // 12) - 1
    
    return f"{note_name}{octave}"

def get_c_major_scale_frequencies(octave: int = 3) -> Tuple[List[float], List[str]]:
    """
    Get frequencies for C major scale in a given octave.
    Uses equal temperament (12-TET).
    
    Args:
        octave: Starting octave (3 = C3, 4 = C4, etc.)
    
    Returns:
        Tuple of (frequencies, note_names)
    """
    # C major scale intervals in semitones from C
    scale_intervals = [0, 2, 4, 5, 7, 9, 11]  # C, D, E, F, G, A, B
    note_names = ["C", "D", "E", "F", "G", "A", "B"]
    
    # C3 = MIDI 48, C4 = MIDI 60
    base_midi = 12 * (octave + 1)  # C of the given octave
    
    frequencies = []
    full_names = []
    
    for interval, name in zip(scale_intervals, note_names):
        midi_note = base_midi + interval
        freq = 440.0 * (2 ** ((midi_note - 69) / 12))
        frequencies.append(freq)
        full_names.append(f"{name}{octave}")
    
    return frequencies, full_names


def get_chromatic_scale_frequencies(octave: int = 3) -> Tuple[List[float], List[str]]:
    """
    Get frequencies for chromatic scale (all 12 notes) in a given octave.
    Uses equal temperament (12-TET).
    
    Args:
        octave: Starting octave (3 = C3, 4 = C4, etc.)
    
    Returns:
        Tuple of (frequencies, note_names)
    """
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
    # C3 = MIDI 48, C4 = MIDI 60
    base_midi = 12 * (octave + 1)  # C of the given octave
    
    frequencies = []
    full_names = []
    
    for i, name in enumerate(note_names):
        midi_note = base_midi + i
        freq = 440.0 * (2 ** ((midi_note - 69) / 12))
        frequencies.append(freq)
        full_names.append(f"{name}{octave}")
    
    return frequencies, full_names


def print_triad_analysis(triads: List[Dict], title: str = "Triad Analysis"):
    """Pretty print the triad analysis results."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    
    for i, triad in enumerate(triads):
        rank_label = ""
        if i == 0:
            rank_label = " [MAJOR EQUIVALENT - Most Stable]"
        elif i == 1:
            rank_label = " [MINOR EQUIVALENT - 2nd Most Stable]"
        
        print(f"\n#{i+1}{rank_label}")
        print(f"  Notes: {' - '.join(triad['names'])}")
        print(f"  Frequencies: {[f'{f:.2f} Hz' for f in triad['frequencies']]}")
        print(f"  Total Stability Score: {triad['score']:.4f} (lower = more stable)")
        
        print("  Intervals:")
        for interval in triad['details']['intervals']:
            print(f"    {interval['from']:.2f} -> {interval['to']:.2f} Hz: "
                  f"{interval['name']} (score: {interval['score']:.3f})")


def main():
    """Test with 12-note chromatic scale - find best triads per root."""
    print("=" * 70)
    print("ARBITRARY CHORD PROGRESSION ANALYZER")
    print("Testing with 12-Note Chromatic Scale (C3 to B3)")
    print("Two-stage approach: Ratio pruning -> RQA refinement")
    print("=" * 70)
    
    # Get chromatic scale frequencies
    frequencies, note_names = get_chromatic_scale_frequencies(octave=3)
    
    print("\nInput Scale (Chromatic):")
    print("-" * 40)
    for name, freq in zip(note_names, frequencies):
        print(f"  {name}: {freq:.2f} Hz")
    
    n = len(frequencies)
    combos_per_root = math.comb(n - 1, 2)
    print(f"\nTotal combinations: {n} roots × {combos_per_root} pairs = {n * combos_per_root}")
    print("Stage 1: Ratio-based pruning to top 3 candidates per root")
    print("Stage 2: RQA refinement (1 second duration) to rank top 2")
    
    # Find best triads per root with RQA
    print("\nAnalyzing triads for each root note...")
    results = find_best_triads_per_root(
        frequencies, note_names, 
        top_n=5, 
        use_rqa=True, 
        ratio_candidates=5  # Increased to ensure we get both major and minor
    )
    
    print("\n" + "=" * 70)
    print("BEST TRIADS PER ROOT (Major & Minor Equivalents)")
    print("=" * 70)
    
    for root_name in note_names:
        triads = results[root_name]
        print(f"\n{root_name} as ROOT:")
        
        for i, triad in enumerate(triads):
            label = "MAJOR" if i == 0 else "MINOR"
            chord_notes = [triad['root_name']] + triad['other_names']
            
            # Get interval names
            intervals = triad['details']['intervals']
            interval_str = f"{intervals[0]['name']}, {intervals[1]['name']}"
            
            # Show RQA score if available
            rqa_str = ""
            if "rqa_score" in triad:
                rqa_str = f" | RQA: {triad['rqa_score']:.4f}"
            
            print(f"  {label}: {'-'.join(chord_notes):15s} | Ratio: {triad['ratio_score']:.3f}{rqa_str} | {interval_str}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Root':<6} | {'Major Equivalent':<20} | {'Minor Equivalent':<20}")
    print("-" * 55)
    
    for root_name in note_names:
        triads = results[root_name]
        major = triads[0]
        minor = triads[1] if len(triads) > 1 else None
        
        major_chord = f"{major['root_name']}-{'-'.join(major['other_names'])}"
        minor_chord = f"{minor['root_name']}-{'-'.join(minor['other_names'])}" if minor else "N/A"
        
        print(f"{root_name:<6} | {major_chord:<20} | {minor_chord:<20}")
    
    # Show what traditional music theory expects
    print("\n" + "-" * 70)
    print("Expected from Traditional Music Theory (12-TET):")
    print("  - Major triads: Root + Major 3rd (4 semitones) + Perfect 5th (7 semitones)")
    print("  - Minor triads: Root + Minor 3rd (3 semitones) + Perfect 5th (7 semitones)")
    print("-" * 70)


if __name__ == "__main__":
    main()
