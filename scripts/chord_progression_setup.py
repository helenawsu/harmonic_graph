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
from fractions import Fraction

# Import ratio snapping functions from dedicated module
from ratio_snapping import snap_to_just_intonation, ratio_stability_score, RATIO_TOLERANCE

# =============================================================================
# Bass Stability Weighting - Prioritize Root Position and Strong Inversions
# =============================================================================

def calculate_bass_stability_weight(root_freq: float, all_freqs: List[float]) -> float:
    """
    Calculates a weight (0.0 to 1.0) based on how well the bass note 
    supports the chord's virtual fundamental.
    
    The key insight: In root position (C-E-G), the bass note IS the root,
    and the virtual fundamental frequency is the root itself. This creates
    maximum stability.
    
    In inversions, the bass note is a harmonic overtone of the root, and
    the virtual fundamental is derived from the LCM of the harmonic series.
    This creates less stability.
    
    SNAPPING: First, we snap all frequency ratios to just intonation to get
    clean integer ratios before calculating LCM. This avoids spurious high-
    denominator fractions from 12-TET equal temperament.
    
    Returns:
        1.0 if Bass is the true root (root position - most stable).
        0.7 if Bass is a strong harmonic (e.g., 3rd harmonic - 2nd inversion).
        0.5 if Bass is a weak harmonic (e.g., 5th harmonic - 1st inversion).
        Lower for higher harmonics (less stable).
    
    Args:
        root_freq: The root frequency (anchor note)
        all_freqs: All frequencies in the chord (should include root_freq as lowest)
    """
    if len(all_freqs) < 2:
        return 1.0
    
    # 1. Convert all frequencies to ratios relative to root, then SNAP to just intonation
    raw_ratios = [f / root_freq for f in all_freqs]
    snapped_ratios = [snap_to_just_intonation(r) for r in raw_ratios]
    
    # 2. Convert snapped ratios to Fractions for LCM calculation
    ratios = []
    for r in snapped_ratios:
        # The snapped ratio should already be clean, but use limit_denominator to be safe
        frac = Fraction(r).limit_denominator(100)
        ratios.append(frac)
    
    # 3. Find the Least Common Multiple (LCM) of all denominators
    # This represents the "virtual fundamental" of the harmonic series
    lcm = 1
    for frac in ratios:
        lcm = math.lcm(lcm, frac.denominator)
    
    # 4. Calculate bass harmonic index
    # Bass harmonic index tells us how many times the virtual fundamental
    # fits into the bass note frequency
    bass_harmonic_index = lcm
    
    # 5. Check if bass_harmonic_index is a power of 2 (0.5, 1, 2, 4, 8, 16...)
    # If it is, the bass note is "pure" relative to root (root position).
    # If not, the bass is a harmonic overtone of the virtual fundamental.
    
    if bass_harmonic_index <= 0:
        return 0.5
    
    log_val = math.log2(bass_harmonic_index)
    
    # Check if log_val is close to an integer (power of 2)
    if abs(log_val - round(log_val)) < 0.001:
        return 1.0  # PERFECT STABILITY (Root Position)
    
    # If not a power of 2, penalize based on distance from root
    # Lower harmonics (3, 5, 7) are more stable than higher (11, 13, etc.)
    # Penalize using logarithmic scale
    
    penalty = 0.2 * math.log2(bass_harmonic_index)
    return max(0.4, 1.0 - penalty)


# =============================================================================
# RQA (Recurrence Quantification Analysis) - for final ranking
# Adapted from rqa_all_roots.py with 1 second duration for speed
# =============================================================================

# RQA parameters
BASELINE_FREQ = 400.0
BASELINE_SR = 8000
SAMPLES_PER_CYCLE = BASELINE_SR / BASELINE_FREQ  # = 20

RQA_DURATION = 0.3  # Reduced for speed (was 1.0)
RQA_WINDOW = 480
RQA_SHIFT = 48
RQA_EMB_DIM = 5
RQA_DELAY = 3
RQA_EPS_FACTOR = 0.1

# Amplitude boost for chord root to emphasize root-to-home relationship
# Similar to rqa_distance_from_home.py: CHORD_ROOT_BOOST = 10.0
# Higher values make the chord root ↔ home note relationship dominate
# At 10x, the chord root's relationship to home becomes the primary factor
CHORD_ROOT_BOOST = 10.0  # Chord root is 10x louder than other chord tones


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


def compute_triad_rqa(root_freq: float, all_frequencies: List[float]) -> Tuple[float, float]:
    """
    Compute RQA recurrence for a triad with explicit root frequency.
    Snaps ratios to just intonation before synthesizing (RQA is physically sensitive!).
    Higher recurrence = more stable/consonant.
    
    UPDATED: Now also computes bass stability weight. Root position (bass = root)
    gets maximum weight (1.0). Inversions are penalized based on harmonic distance.
    
    Args:
        root_freq: The root frequency in Hz (used for sample rate normalization)
        all_frequencies: List of all frequencies in the chord (including root)
    
    Returns:
        Tuple of (rqa_score, bass_weight)
        - rqa_score: Mean percent recurrence (0-1, higher = more stable)
        - bass_weight: Bass stability weight (0.4-1.0, higher = more stable)
    """
    if len(all_frequencies) < 2:
        return 0.0, 0.0
    
    # Normalize sample rate to root frequency (key insight from rqa_all_roots.py!)
    sr = get_sr_for_freq(root_freq)
    
    # Convert frequencies to ratios relative to root, then SNAP to just intonation!
    raw_ratios = [f / root_freq for f in all_frequencies]
    ratios = [snap_to_just_intonation(r) for r in raw_ratios]
    
    t = np.arange(0, RQA_DURATION, 1.0 / sr)
    sig = np.zeros_like(t)
    
    # Sum all frequency components using SNAPPED ratios
    # Weight root (first component at ratio 1.0) 3x more heavily to anchor the chord
    for i, ratio in enumerate(ratios):
        f = root_freq * ratio
        weight = 1.0 if (ratio == 1.0 or i == 0) else 1.0
        sig += weight * np.sin(2.0 * math.pi * f * t)
    
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
    bass_weight = calculate_bass_stability_weight(root_freq, all_frequencies)
    
    return rqa_score, bass_weight


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
    use_rqa: bool = True
) -> Dict[str, List[Dict]]:
    """
    For each note as root, find the best triads using a two-stage approach:
    
    Stage 1: Find the most stable triad (C-E-G for C root)
    Stage 2: Find minor equivalent by:
      - Identify least stable note in the major triad
      - Replace with nearest neighbor (higher or lower)
      - Compare both candidates with RQA + bass weight
      - Select the one with better combined score as minor
    
    Args:
        frequencies: List of frequencies in Hz (max 12 for reasonable runtime)
        note_names: Optional list of names for each frequency
        top_n: Number of top triads per root (default 2 for major/minor equivalent)
        use_rqa: Whether to use RQA for ranking (required for this approach)
    
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
        print(root_name)
        other_notes = [
            (i, frequencies[i], note_names[i])
            for i in range(len(frequencies))
            if i != root_idx
        ]
        
        triads_for_root = []
        
        # STAGE 1: Find all candidates and score them
        triad_candidates = []
        
        # Generate all 2-combinations and prepare triad info
        for combo in combinations(other_notes, 2):
            other_indices = [c[0] for c in combo]
            other_freqs = [c[1] for c in combo]
            other_names = [c[2] for c in combo]
            
            # Get ratio details for display
            _, details = triad_stability_score_rooted(root_freq, other_freqs)
            
            triad_info = {
                "root_name": root_name,
                "root_freq": root_freq,
                "other_names": other_names,
                "other_freqs": other_freqs,
                "other_indices": other_indices,
                "chord_name": f"{root_name}-{other_names[0]}-{other_names[1]}",
                "score": 0.0,
                "details": details,
            }
            triad_candidates.append(triad_info)
        
        # Score all candidates with RQA + bass weight
        if use_rqa and len(triad_candidates) > 0:
            for triad in triad_candidates:
                all_freqs = [triad["root_freq"]] + triad["other_freqs"]
                rqa_score, bass_weight = compute_triad_rqa(triad["root_freq"], all_freqs)
                
                # Combine RQA and bass weight
                # Higher RQA = more stable/consonant, so we maximize it
                # Bass weight bonus: root position (1.0) is most stable
                bonus_factor = 0.05
                combined_score = rqa_score + bass_weight * bonus_factor
                
                triad["rqa_score"] = rqa_score
                triad["bass_weight"] = bass_weight
                triad["score"] = combined_score
            
            # Sort to find most stable (highest score)
            triad_candidates.sort(key=lambda x: x["score"], reverse=True)
            major_triad = triad_candidates[0]
            
            # STAGE 2: Find minor equivalent by replacing least stable note
            # Identify which note in the major triad is least stable
            intervals = major_triad["details"]["intervals"]
            
            # Find the interval with highest stability score (least stable)
            least_stable_interval_idx = max(range(len(intervals)), 
                                           key=lambda i: intervals[i]["score"])
            
            # The "least stable note" is the one not in the most stable interval pair
            all_notes_in_major = [major_triad["root_freq"]] + major_triad["other_freqs"]
            all_names_in_major = [major_triad["root_name"]] + major_triad["other_names"]
            
            # intervals are: [root-note1, root-note2, note1-note2]
            # Find which note appears in the least stable interval
            least_stable_interval = intervals[least_stable_interval_idx]
            freq_a = least_stable_interval["from"]
            freq_b = least_stable_interval["to"]
            
            # The note to replace is the one that's NOT the root (if interval involves root)
            # or either note if it doesn't
            if freq_a == major_triad["root_freq"]:
                freq_to_replace = freq_b
                name_to_replace = major_triad["other_names"][0] if major_triad["other_freqs"][0] == freq_b else major_triad["other_names"][1]
                idx_to_replace = major_triad["other_indices"][0] if major_triad["other_freqs"][0] == freq_b else major_triad["other_indices"][1]
            elif freq_b == major_triad["root_freq"]:
                freq_to_replace = freq_a
                name_to_replace = major_triad["other_names"][0] if major_triad["other_freqs"][0] == freq_a else major_triad["other_names"][1]
                idx_to_replace = major_triad["other_indices"][0] if major_triad["other_freqs"][0] == freq_a else major_triad["other_indices"][1]
            else:
                # Interval is between the two other notes, replace the first one
                freq_to_replace = freq_a
                name_to_replace = major_triad["other_names"][0]
                idx_to_replace = major_triad["other_indices"][0]
            
            # Find nearest neighbors (higher and lower in the original frequency list)
            idx_neighbors = []
            
            # Lower neighbor
            lower_candidates = [i for i in range(len(frequencies)) 
                               if i != root_idx and frequencies[i] < freq_to_replace]
            if lower_candidates:
                idx_neighbors.append(max(lower_candidates, key=lambda i: frequencies[i]))
            
            # Higher neighbor
            higher_candidates = [i for i in range(len(frequencies)) 
                                if i != root_idx and frequencies[i] > freq_to_replace]
            if higher_candidates:
                idx_neighbors.append(min(higher_candidates, key=lambda i: frequencies[i]))
            
            # Create minor candidates by replacing with neighbors
            minor_candidates = [major_triad]  # Keep major as first choice
            
            for neighbor_idx in idx_neighbors:
                neighbor_freq = frequencies[neighbor_idx]
                neighbor_name = note_names[neighbor_idx]
                
                # Create new chord with neighbor note
                new_other_freqs = list(major_triad["other_freqs"])
                new_other_names = list(major_triad["other_names"])
                new_other_indices = list(major_triad["other_indices"])
                
                # Find and replace the old note
                for j, idx in enumerate(new_other_indices):
                    if idx == idx_to_replace:
                        new_other_freqs[j] = neighbor_freq
                        new_other_names[j] = neighbor_name
                        new_other_indices[j] = neighbor_idx
                        break
                
                # Sort the other notes
                sorted_pairs = sorted(zip(new_other_freqs, new_other_names, new_other_indices))
                new_other_freqs = [p[0] for p in sorted_pairs]
                new_other_names = [p[1] for p in sorted_pairs]
                new_other_indices = [p[2] for p in sorted_pairs]
                
                # Calculate RQA for minor candidate
                _, details = triad_stability_score_rooted(root_freq, new_other_freqs)
                
                all_freqs = [root_freq] + new_other_freqs
                rqa_score, bass_weight = compute_triad_rqa(root_freq, all_freqs)
                
                penalty_factor = 0.05
                combined_score = (-rqa_score) + (1.0 - bass_weight) * penalty_factor
                
                minor_info = {
                    "root_name": root_name,
                    "root_freq": root_freq,
                    "other_names": new_other_names,
                    "other_freqs": new_other_freqs,
                    "other_indices": new_other_indices,
                    "chord_name": f"{root_name}-{'-'.join(new_other_names)}",
                    "rqa_score": rqa_score,
                    "bass_weight": bass_weight,
                    "score": combined_score,
                    "details": details,
                }
                minor_candidates.append(minor_info)
            
            # Select best major and minor
            results[root_name] = [major_triad]
            if len(minor_candidates) > 1:
                # Sort minor candidates by RQA score (lower = more stable/consonant)
                # The combined_score includes bass weight penalty, but for minor we prioritize
                # the actual RQA recurrence score (lower RQA = better waveform stability)
                minor_candidates[1:] = sorted(minor_candidates[1:], key=lambda x: x["rqa_score"])
                results[root_name].append(minor_candidates[1])
        else:
            # Fallback: sort all by ratio score if RQA disabled
            for triad in triad_candidates:
                triad["score"] = triad["details"]["total_score"]
            triad_candidates.sort(key=lambda x: x["score"])
            results[root_name] = triad_candidates[:top_n]
    
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


def identify_chord_root(chord_frequencies: List[float]) -> float:
    """
    Identify the true harmonic root of a chord using bass stability weight.
    
    For each frequency in the chord, calculate how stable it is as a bass note.
    The frequency with the highest bass stability weight (closest to 1.0) is
    the chord root.
    
    Example:
    - C-E-G: C has weight 1.0 (root position) → C is the root
    - E-G-C: C has weight 1.0 (root position) → C is the root (not C at bottom!)
    - F-C-A: C has weight 1.0 (root position) → C is the root (F-C-A is F major in first inversion)
    
    Args:
        chord_frequencies: List of frequencies in the chord
    
    Returns:
        The frequency identified as the chord root
    """
    if len(chord_frequencies) == 0:
        return 0.0
    if len(chord_frequencies) == 1:
        return chord_frequencies[0]
    
    # Try each frequency as a potential root
    best_root = chord_frequencies[0]
    best_weight = -1.0
    
    for candidate_root in chord_frequencies:
        # Calculate bass stability weight for this candidate
        weight = calculate_bass_stability_weight(candidate_root, chord_frequencies)
        
        if weight > best_weight:
            best_weight = weight
            best_root = candidate_root
    
    return best_root


def compute_distance_from_home_with_boost(
    home_freq: float,
    chord_root_freq: float,
    chord_frequencies: List[float],
    home_note_name: str = "Home"
) -> Tuple[float, float]:
    """
    Compute RQA-based distance from home note with CHORD_ROOT_BOOST.
    
    This function:
    1. Creates signal: home note + chord frequencies
    2. Boosts the chord root frequency (CHORD_ROOT_BOOST = 10x amplitude)
    3. Computes RQA recurrence and compares to home-only baseline
    
    KEY INSIGHT: 
    - C + C-E-G: Has double-C, both boosted → highest recurrence (closest to home)
    - C + F-A (no C): Has no matching root → lower recurrence (farther from home)
    
    The identified chord root is what gets boosted, so:
    - C-E-G: C is root (weight 1.0) → C gets boosted
    - F-C-A: C is root (weight 1.0, not F!) → C gets boosted
    
    Args:
        home_freq: The home reference frequency in Hz
        chord_root_freq: The identified chord root frequency (to be boosted)
        chord_frequencies: List of all chord frequencies
        home_note_name: Name of the home note (for display)
    
    Returns:
        Tuple of (recurrence_score, distance_from_home)
        - recurrence_score: RQA recurrence (0-1, higher = more stable)
        - distance_from_home: 1.0 - normalized_recurrence (0-1, lower = closer to home)
    """
    # Compute RQA of home note playing alone (baseline)
    home_only_rqa, _ = compute_triad_rqa(home_freq, [home_freq])
    
    # Compute RQA of home note + chord with boosted chord root
    sr = get_sr_for_freq(home_freq)
    
    # Convert all frequencies to ratios relative to home, then SNAP to just intonation
    all_freqs_for_signal = [home_freq] + chord_frequencies
    raw_ratios = [f / home_freq for f in all_freqs_for_signal]
    snapped_ratios = [snap_to_just_intonation(r) for r in raw_ratios]
    
    t = np.arange(0, RQA_DURATION, 1.0 / sr)
    sig = np.zeros_like(t)
    
    # Add home note at BOOSTED amplitude (10x) to emphasize home-to-chord relationship
    # This makes chords containing the home note (e.g., I, IV, vi) have high recurrence
    # And chords with notes close to home (e.g., bII with C# near C) show dissonance
    home_ratio = snapped_ratios[0]  # Should be 1.0
    sig += CHORD_ROOT_BOOST * np.sin(2.0 * math.pi * home_freq * home_ratio * t)
    
    # Add chord frequencies at normal amplitude
    for i, freq in enumerate(chord_frequencies):
        ratio = snapped_ratios[i + 1]
        f = home_freq * ratio
        
        # All chord tones at normal amplitude (home is the one boosted)
        amplitude = 1.0
        
        sig += amplitude * np.sin(2.0 * math.pi * f * t)
    
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
    
    chord_with_home_rqa = float(np.mean(results)) if results else 0.0
    
    # Normalize by home-only RQA to get distance metric
    # If chord_root == home_freq, the signal has double-strength home frequency
    # which creates high recurrence
    if home_only_rqa > 0.0001:
        normalized_recurrence = chord_with_home_rqa / home_only_rqa
        # Don't cap - let matching roots produce > 1.0 for maximum consonance
    else:
        normalized_recurrence = chord_with_home_rqa
    
    # Distance is inverse of recurrence: higher recurrence = lower distance (closer to home)
    distance_from_home = 1.0 / (1.0 + normalized_recurrence)  # Map to 0-1 range
    
    return chord_with_home_rqa, distance_from_home


def compute_distance_from_home(
    home_freq: float,
    chord_frequencies: List[float],
    home_note_name: str = "Home"
) -> Tuple[float, float]:
    """
    Compute RQA-based distance from home note (wrapper).
    
    Identifies the true chord root and applies CHORD_ROOT_BOOST.
    
    Args:
        home_freq: The home reference frequency in Hz
        chord_frequencies: List of frequencies in the chord
        home_note_name: Name of the home note (for display)
    
    Returns:
        Tuple of (recurrence_score, distance_from_home)
    """
    # Identify the true chord root using bass stability weight
    chord_root = identify_chord_root(chord_frequencies)
    
    # Compute distance with chord root boost
    return compute_distance_from_home_with_boost(
        home_freq, chord_root, chord_frequencies, home_note_name
    )


def analyze_chord_palette(
    frequencies: List[float],
    note_names: List[str],
    home_note_idx: int = 0
) -> Dict:
    """
    Analyze a chord palette (set of frequencies) by:
    1. Finding the best major/minor triads for each root
    2. Computing each chord's distance from the home note using RQA
    3. Ranking chords by consonance with home
    
    Args:
        frequencies: List of frequencies in Hz
        note_names: List of names for each frequency
        home_note_idx: Index of home note (default 0 = lowest frequency)
    
    Returns:
        Dictionary with analysis results including distance rankings
    """
    if home_note_idx >= len(frequencies):
        raise ValueError(f"Home note index {home_note_idx} out of range [0, {len(frequencies)-1}]")
    
    home_freq = frequencies[home_note_idx]
    home_name = note_names[home_note_idx]
    
    print(f"\n{'='*70}")
    print(f"HOME NOTE: {home_name} ({home_freq:.2f} Hz)")
    print(f"{'='*70}")
    
    # Find best triads per root
    triads_by_root = find_best_triads_per_root(
        frequencies, note_names,
        top_n=5,
        use_rqa=True
    )
    
    # Compute distance from home for each chord
    all_chords = []
    
    for root_name, triads in triads_by_root.items():
        for chord_idx, triad in enumerate(triads):
            # Just use the note names without Major/Minor labels
            chord_name = triad['chord_name']
            
            # Compute distance from home
            # Pass full chord including root (not just other_freqs)
            full_chord_freqs = [triad['root_freq']] + triad['other_freqs']
            rqa_with_home, distance = compute_distance_from_home(
                home_freq,
                full_chord_freqs,
                home_name
            )
            
            chord_info = {
                "chord_name": chord_name,
                "root_name": root_name,
                "frequencies": [triad['root_freq']] + triad['other_freqs'],
                "all_freqs_str": "-".join(triad['other_names']),
                "rqa_recurrence": triad['rqa_score'],
                "bass_weight": triad['bass_weight'],
                "rqa_with_home": rqa_with_home,
                "distance_from_home": distance,
            }
            all_chords.append(chord_info)
    
    # Sort by distance from home (ascending = closest to home first)
    all_chords.sort(key=lambda x: x["distance_from_home"])
    
    return {
        "home_note": home_name,
        "home_freq": home_freq,
        "all_chords": all_chords,
        "triads_by_root": triads_by_root,
    }


def print_chord_palette_analysis(analysis: Dict):
    """Print the chord palette analysis in a readable format."""
    print(f"\n{'='*100}")
    print("CHORD PALETTE ANALYSIS - DISTANCE FROM HOME")
    print(f"{'='*100}")
    
    print(f"\n{'Rank':<5} | {'Chord Name':<25} | {'RQA':<8} | {'Bass':<6} | {'RQA+Home':<10} | {'Distance':<10} | {'Consonance':<12}")
    print("-" * 100)
    
    for i, chord in enumerate(analysis["all_chords"][:20]):  # Show top 20
        rank = i + 1
        
        # Consonance indicator (inverse of distance)
        consonance = 1.0 - chord["distance_from_home"]
        consonance_pct = f"{consonance*100:.1f}%"
        
        print(
            f"{rank:<5} | {chord['chord_name']:<25} | "
            f"{chord['rqa_recurrence']:.6f} | {chord['bass_weight']:.2f} | "
            f"{chord['rqa_with_home']:.6f} | {chord['distance_from_home']:.6f} | {consonance_pct:<12}"
        )
    
    print("\n" + "-" * 100)
    print(f"Total chords analyzed: {len(analysis['all_chords'])}")
    print(f"Most consonant with home: {analysis['all_chords'][0]['chord_name']} "
          f"(distance: {analysis['all_chords'][0]['distance_from_home']:.6f})")
    print(f"Most dissonant with home: {analysis['all_chords'][-1]['chord_name']} "
          f"(distance: {analysis['all_chords'][-1]['distance_from_home']:.6f})")


def export_chord_palette_to_csv(analysis: Dict, output_path: str = None):
    """
    Export chord palette analysis to CSV file.
    
    Args:
        analysis: Dictionary with analysis results
        output_path: Path to save CSV (default: results/chord_palette_<home_note>.csv)
    """
    if output_path is None:
        home_name = analysis["home_note"].replace("/", "_")
        output_path = f"results/chord_palette_{home_name}_analysis.csv"
    
    import csv
    from pathlib import Path
    
    # Ensure results directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            "Rank",
            "Chord Name",
            "Root Note",
            "Chord Type",
            "RQA Recurrence",
            "Bass Weight",
            "RQA with Home",
            "Distance from Home",
            "Consonance %"
        ])
        
        # Write data rows
        for i, chord in enumerate(analysis["all_chords"]):
            consonance = 1.0 - chord["distance_from_home"]
            writer.writerow([
                i + 1,
                chord["chord_name"],
                chord["root_name"],
                chord["chord_type"],
                f"{chord['rqa_recurrence']:.6f}",
                f"{chord['bass_weight']:.2f}",
                f"{chord['rqa_with_home']:.6f}",
                f"{chord['distance_from_home']:.6f}",
                f"{consonance*100:.2f}%"
            ])
    
    print(f"\n✓ Chord palette analysis exported to: {output_path}")


def main():
    """Test with 12-note chromatic scale - find best triads per root."""
    print("=" * 70)
    print("ARBITRARY CHORD PROGRESSION ANALYZER")
    print("Testing with 12-Note Chromatic Scale (C3 to B3)")
    print("RQA-only approach: Direct RQA analysis on all triad candidates")
    print("=" * 70)
    
    # Get chromatic scale frequencies
    frequencies, note_names = get_chromatic_scale_frequencies(octave=3)
    
    print("\nInput Scale (Chromatic):")
    print("-" * 40)
    for i, (name, freq) in enumerate(zip(note_names, frequencies)):
        print(f"  [{i}] {name}: {freq:.2f} Hz")
    
    n = len(frequencies)
    combos_per_root = math.comb(n - 1, 2)
    print(f"\nTotal combinations: {n} roots × {combos_per_root} pairs = {n * combos_per_root}")
    print("RQA analysis (0.3 second duration) on all candidates per root")
    
    # Analyze chord palette with home note (default = C3, the lowest)
    home_note_idx = 0  # C3 (lowest frequency)
    
    print(f"\nAnalyzing chord palette with home note index {home_note_idx} ({note_names[home_note_idx]})...")
    analysis = analyze_chord_palette(
        frequencies, note_names,
        home_note_idx=home_note_idx
    )
    
    # Print detailed analysis
    print_chord_palette_analysis(analysis)
    
    # Export to CSV
    export_chord_palette_to_csv(analysis)


if __name__ == "__main__":
    main()
