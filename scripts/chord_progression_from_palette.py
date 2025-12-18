#!/usr/bin/env python3
"""
Chord Progression Generator from Arbitrary Chord Palette

Given:
1. A list of input frequencies (chord palette)
2. A home note from that palette
3. Pre-calculated distance-from-home for each chord

Generate optimal chord progressions that follow a target tension curve
while balancing voice leading smoothness and chord consonance.

Three Independent Weights (all components normalized 0-1):
    - Voice Leading: 0 = smooth (no movement), 1 = jumpy (max movement)
    - Roughness: Based on RQA recurrence of the chord (0 = consonant, 1 = dissonant)
    - Distance from Home: 0 = familiar (home key), 1 = remote (far from home)

Where:
    - Tension = (Roughness + Distance) / 2
    - Voice Leading: Minimum semitone movement between chord voicings
"""
import os
import sys
import random
import math
from typing import List, Tuple, Dict
from itertools import combinations

sys.path.insert(0, os.path.dirname(__file__))

from audio_utils import generate_progression_audio, save_audio


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


def get_chromatic_scale_frequencies(octave: int = 3) -> Tuple[List[float], List[str]]:
    """Get frequencies for chromatic scale (all 12 notes) in a given octave."""
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
    base_midi = 12 * (octave + 1)
    
    frequencies = []
    full_names = []
    
    for i, name in enumerate(note_names):
        midi_note = base_midi + i
        freq = 440.0 * (2 ** ((midi_note - 69) / 12))
        frequencies.append(freq)
        full_names.append(f"{name}{octave}")
    
    return frequencies, full_names


def analyze_chord_palette(frequencies: List[float], note_names: List[str], home_note_idx: int = 0) -> Dict:
    """
    Analyze a chord palette by computing RQA-based distance from home for each chord.
    This is a wrapper that calls the main analysis from chord_progression_setup.
    """
    # Import the actual function from chord_progression_setup
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "chord_progression_setup",
        os.path.join(os.path.dirname(__file__), "chord_progression_setup.py")
    )
    cps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cps)
    
    return cps.analyze_chord_palette(frequencies, note_names, home_note_idx)


def freq_to_continuous_semitone(freq: float) -> float:
    """
    Converts Hz to a continuous semitone scale (relative to MIDI 0).
    Formula: 69 + 12 * log2(freq / 440)
    """
    if freq <= 0:
        return 0
    return 69 + 12 * math.log2(freq / 440.0)


def voice_leading_cost_freq(chord1_freqs: List[float], chord2_freqs: List[float]) -> float:
    """
    Calculates voice leading distance for raw frequencies (Hz).
    Handles microtones and non-standard tuning.
    
    Returns:
        Total semitone movement (as float, handles microtones)
    """
    # Convert Hz to Continuous Semitones
    pitch1 = [freq_to_continuous_semitone(f) for f in chord1_freqs]
    pitch2 = [freq_to_continuous_semitone(f) for f in chord2_freqs]

    # Convert to Pitch Class (0.0 to 11.999...)
    pc1 = [p % 12.0 for p in pitch1]
    pc2 = [p % 12.0 for p in pitch2]

    total_movement = 0.0
    remaining_targets = pc2.copy()

    # Greedy Matching
    for p1 in pc1:
        best_dist = 100.0
        best_target_idx = -1

        for i, p2 in enumerate(remaining_targets):
            # Calculate Circular Distance
            diff = abs(p1 - p2)
            dist = min(diff, 12.0 - diff)

            if dist < best_dist:
                best_dist = dist
                best_target_idx = i

        total_movement += best_dist

        if best_target_idx != -1:
            remaining_targets.pop(best_target_idx)

    return total_movement


def soft_voice_leading_denominator(voice_distance: float) -> float:
    """
    Soft denominator for tension rate calculation.
    Uses logarithmic scaling to avoid punishing larger voice movements too hard.
    
    Formula: (log2(voice_distance + 1)) + 1
    """
    if voice_distance < 1e-5:
        return 1.0
    return (math.log2(voice_distance + 1)) + 1


def build_chord_database_from_palette(
    analysis: Dict,
) -> List[Dict]:
    """
    Build a chord database from the palette analysis.
    
    Args:
        analysis: Result from analyze_chord_palette()
    
    Returns:
        List of chord dicts with:
        - name: chord name (e.g., "C-E-G")
        - frequencies: list of frequencies
        - tension: normalized distance from home (0=close to home, 1=far from home)
    """
    chords = []
    
    # Get raw RQA values and normalize to 0-1 range within this palette
    # Higher RQA = more consonant with home = lower tension
    rqa_values = [c["rqa_with_home"] for c in analysis["all_chords"]]
    min_rqa = min(rqa_values)
    max_rqa = max(rqa_values)
    rqa_range = max_rqa - min_rqa if max_rqa > min_rqa else 1.0
    
    for chord_info in analysis["all_chords"]:
        # Normalize tension: highest RQA = 0 tension, lowest RQA = 1 tension
        normalized_rqa = (chord_info["rqa_with_home"] - min_rqa) / rqa_range
        tension = 1.0 - normalized_rqa  # Invert so high RQA = low tension
        
        chord = {
            "name": chord_info["chord_name"],
            "frequencies": chord_info["frequencies"],
            "tension": tension,
            "rqa_recurrence": chord_info["rqa_recurrence"],
            "bass_weight": chord_info["bass_weight"],
        }
        chords.append(chord)
    
    return chords


def normalize_chord_frequencies(frequencies: List[float]) -> Tuple[float, ...]:
    """
    Normalize chord by sorting its frequencies.
    This allows comparison of chords regardless of voicing order.
    
    Args:
        frequencies: List of frequencies in the chord
    
    Returns:
        Tuple of sorted frequencies (hashable for comparison)
    """
    return tuple(sorted(frequencies))


def chords_are_identical(chord1: Dict, chord2: Dict) -> bool:
    """
    Check if two chords are identical (same pitch classes, regardless of voicing/order).
    
    Args:
        chord1: First chord dict with 'frequencies' key
        chord2: Second chord dict with 'frequencies' key
    
    Returns:
        True if chords have the same notes (ignoring order), False otherwise
    """
    normalized1 = normalize_chord_frequencies(chord1["frequencies"])
    normalized2 = normalize_chord_frequencies(chord2["frequencies"])
    
    # Check if same number of notes and all frequencies match (within small tolerance)
    if len(normalized1) != len(normalized2):
        return False
    
    tolerance = 0.5  # Hz tolerance for matching frequencies
    for f1, f2 in zip(normalized1, normalized2):
        if abs(f1 - f2) > tolerance:
            return False
    
    return True


def brute_force_optimize(
    chords: List[Dict],
    target_curve: List[float],
    voice_weight: float = 0.5,
    tension_weight: float = 1.0,
    temperature: float = 0.0,
) -> Tuple[List[Dict], float]:
    """
    Brute force search for the optimal 4-chord progression.
    
    Finds a path through chord space that:
    1. Matches the target tension curve as closely as possible
    2. Minimizes voice leading distance (smooth transitions)
    3. Prefers natural, music-theory-compliant progressions
    4. Never uses the same chord twice (regardless of voicing order)
    
    Args:
        chords: List of available chords
        target_curve: Target tension values [t1, t2, t3, t4] (4 chords)
        voice_weight: Weight for voice leading smoothness
        tension_weight: Weight for matching target tension
        temperature: Randomness factor (0 = deterministic)
    
    Returns:
        Tuple of (best_path, best_score)
        - best_path: List of 4 chord dicts
        - best_score: Total optimization score (lower is better)
    """
    if len(target_curve) != 4:
        raise ValueError("Target curve must have exactly 4 tension values")
    
    if len(chords) < 4:
        raise ValueError(f"Need at least 4 chords, got {len(chords)}")
    
    # Compute target slopes (ΔTension between consecutive chords)
    target_slopes = [target_curve[i] - target_curve[i-1] for i in range(1, len(target_curve))]
    
    best_path = None
    best_score = float('inf')
    all_candidates = []
    
    # Generate all 4-combinations of chords (order matters)
    n_chords = len(chords)
    
    for c1 in range(n_chords):
        for c2 in range(n_chords):
            # Skip consecutive repeats
            if c2 == c1:
                continue
            # Skip if chord is already used (regardless of order)
            if chords_are_identical(chords[c1], chords[c2]):
                continue
            for c3 in range(n_chords):
                # Skip consecutive repeats
                if c3 == c2:
                    continue
                # Skip if chord is already used in positions 1 or 2 (regardless of order)
                if chords_are_identical(chords[c1], chords[c3]) or chords_are_identical(chords[c2], chords[c3]):
                    continue
                for c4 in range(n_chords):
                    # Skip consecutive repeats
                    if c4 == c3:
                        continue
                    # Skip if chord is already used in positions 1, 2, or 3 (regardless of order)
                    if (chords_are_identical(chords[c1], chords[c4]) or 
                        chords_are_identical(chords[c2], chords[c4]) or
                        chords_are_identical(chords[c3], chords[c4])):
                        continue
                    
                    path = [chords[c1], chords[c2], chords[c3], chords[c4]]
                    path_cost = 0.0
                    
                    # Iterate through the transitions (Steps 1, 2, 3)
                    for i in range(1, len(path)):
                        
                        # A. Get the Target Slope for this specific step
                        target_slope = target_slopes[i-1]
                        
                        # B. Calculate Actual Change in Tension
                        d_tension = path[i]["tension"] - path[i-1]["tension"]
                        
                        # C. Calculate Voice Leading Distance (in semitones)
                        freq1 = path[i-1]["frequencies"]
                        freq2 = path[i]["frequencies"]
                        d_voice = voice_leading_cost_freq(freq1, freq2)
                        
                        # D. Use soft denominator to avoid punishing larger moves too hard
                        soft_denom = soft_voice_leading_denominator(d_voice)
                        
                        # E. Calculate Actual Rate with soft denominator
                        actual_rate = d_tension / soft_denom
                        
                        # F. Add penalty for deviation from target slope
                        path_cost += abs(actual_rate - target_slope)
                    
                    # G. Check if this is the best "flow" so far
                    if temperature > 0:
                        jitter = random.gauss(0, temperature)
                        jittered_score = path_cost + jitter
                        all_candidates.append((path, path_cost, jittered_score))
                    else:
                        if path_cost < best_score:
                            best_score = path_cost
                            best_path = path
    
    if temperature > 0:
        # Sort by jittered score and pick the best
        all_candidates.sort(key=lambda x: x[2])
        best_path, best_score, _ = all_candidates[0]
    
    return best_path, best_score


def print_progression(path: List[Dict], target_curve: List[float]):
    """Print the chord progression with rate-matching details."""
    print("\n" + "=" * 90)
    print("OPTIMAL CHORD PROGRESSION")
    print("=" * 90)
    
    print("\nTarget Tension Curve:", target_curve)
    print("Tension = (Roughness + Distance) / 2")
    
    # Compute target slopes
    target_slopes = [target_curve[i] - target_curve[i-1] for i in range(1, len(target_curve))]
    print("Target Slopes (ΔTension):", [f"{s:+.3f}" for s in target_slopes])
    
    print("\nBest Path Found:")
    print("-" * 90)
    
    result_strings = []
    
    # First chord
    chord = path[0]
    result_strings.append(f'"{chord["name"]} (T: {chord["tension"]:.2f})"')
    print(f"  Step 1: {chord['name']:25s} | Tension: {chord['tension']:.3f} | "
          f"Roughness: {chord['roughness']:.3f} | Distance: {chord['distance']:.3f}")
    
    # Transitions: show rate matching
    print("-" * 90)
    print("  Transitions (using soft denominator: log2(ΔV + 1) + 1):")
    total_rate_cost = 0.0
    
    for i in range(1, len(path)):
        chord = path[i]
        prev_chord = path[i-1]
        
        target_slope = target_slopes[i-1]
        d_tension = chord["tension"] - prev_chord["tension"]
        freq1 = prev_chord["frequencies"]
        freq2 = chord["frequencies"]
        d_voice = voice_leading_cost_freq(freq1, freq2)
        soft_denom = soft_voice_leading_denominator(d_voice)
        actual_rate = d_tension / soft_denom
        rate_diff = abs(actual_rate - target_slope)
        total_rate_cost += rate_diff
        
        result_strings.append(f'"{chord["name"]} (T: {chord["tension"]:.2f})"')
        print(f"    {prev_chord['name']:25s} -> {chord['name']:25s} | "
              f"ΔT: {d_tension:+.3f} | ΔV: {d_voice:.1f}s (soft: {soft_denom:.2f}) | "
              f"Rate: {actual_rate:+.3f} (target: {target_slope:+.3f}, diff: {rate_diff:.3f})")
    
    print("-" * 90)
    print(f"  Total rate cost: {total_rate_cost:.3f}")
    print(f"\nProgression (Chord Names): {' - '.join([c['name'] for c in path])}")
    print(f"\nFormatted Output:")
    print(f"  [{', '.join(result_strings)}]")


def print_chord_tensions(chords: List[Dict]):
    """Print all chord properties for reference."""
    print("\n" + "=" * 100)
    print("CHORD PALETTE PROPERTIES")
    print("=" * 100)
    print("\nAll values normalized to 0-1:")
    print("  - Tension = (Roughness + Distance) / 2")
    print("  - Roughness: 0 = consonant, 1 = dissonant (inverse of RQA recurrence)")
    print("  - Distance:  0 = familiar (home), 1 = remote (far from home)")
    
    # Sort by tension
    chords_sorted = sorted(chords, key=lambda c: c["tension"])
    
    print(f"\n{'Chord':25s} | {'Tension':>8s} | {'Roughness':>10s} | {'Distance':>10s} | {'RQA':>8s}")
    print("-" * 100)
    
    for chord in chords_sorted:
        print(f"{chord['name']:25s} | {chord['tension']:>8.3f} | {chord['roughness']:>10.3f} | "
              f"{chord['distance']:>10.3f} | {chord['rqa_recurrence']:>8.6f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Chord Progression Generator from Arbitrary Chord Palette"
    )
    parser.add_argument(
        "--frequencies", type=float, nargs="+",
        help="Input frequencies (Hz) for chord palette. Default: chromatic scale C3-B3"
    )
    parser.add_argument(
        "--home-idx", type=int, default=0,
        help="Index of home note in frequency list (default: 0 = lowest)"
    )
    parser.add_argument(
        "--curve", type=float, nargs=4, default=[0.0, 0.2, 0.65, -0.1],
        help="Target tension curve (4 values, 0-1). Default: [0.0, 0.2, 0.65, -0.1]"
    )
    parser.add_argument(
        "--voice-weight", type=float, default=0.0,
        help="Weight for voice leading smoothness (default: 0.0)"
    )
    parser.add_argument(
        "--tension-weight", type=float, default=1.0,
        help="Weight for tension change matching (default: 1.0)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.01,
        help="Randomness/jitter factor. 0.0 = deterministic (default: 0.01)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--show-tensions", action="store_true",
        help="Show all chord properties (tension breakdown) for reference"
    )
    parser.add_argument(
        "--output", type=str, default="results/progression_from_palette.mp3",
        help="Output audio file path (default: results/progression_from_palette.mp3)"
    )
    parser.add_argument(
        "--chord-duration", type=float, default=1.0,
        help="Duration of each chord in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="Skip audio file generation"
    )
    
    args = parser.parse_args()
    
    # Get input frequencies
    if args.frequencies:
        frequencies = args.frequencies
        note_names = [freq_to_note_name(f) for f in frequencies]
    else:
        # Use chromatic scale C3-B3 as default
        frequencies, note_names = get_chromatic_scale_frequencies(octave=3)
    
    home_idx = args.home_idx % len(frequencies)
    home_freq = frequencies[home_idx]
    home_name = note_names[home_idx]
    
    print(f"\nChord Progression Generator from Arbitrary Palette")
    print(f"Input frequencies: {len(frequencies)} notes")
    print(f"Home note: {home_name} ({home_freq:.2f} Hz, index {home_idx})")
    print(f"Target Tension Curve: {args.curve}")
    print(f"\nWeights (all components normalized 0-1):")
    print(f"  Voice Leading: {args.voice_weight}")
    print(f"  Tension:       {args.tension_weight}")
    print(f"Temperature: {args.temperature}")
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random Seed: {args.seed}")
    
    # Analyze chord palette
    print("\nAnalyzing chord palette...")
    analysis = analyze_chord_palette(frequencies, note_names, home_idx)
    
    # Build chord database from analysis
    chords = build_chord_database_from_palette(analysis)
    
    print(f"Available chords: {len(chords)}")
    
    if args.show_tensions:
        print_chord_tensions(chords)
    
    # Run optimizer
    print("\nSearching for optimal progression...")
    best_path, best_score = brute_force_optimize(
        chords=chords,
        target_curve=args.curve,
        voice_weight=args.voice_weight,
        tension_weight=args.tension_weight,
        temperature=args.temperature,
    )
    
    print(f"Best Total Score: {best_score:.4f}")
    print_progression(best_path, args.curve)
    
    # Generate audio
    if not args.no_audio:
        print("\nGenerating audio...")
        
        # Convert frequencies to MIDI notes for audio generation
        # (audio_utils expects MIDI notes)
        midi_notes = []
        for freq in best_path[0]["frequencies"]:
            semitones_from_a4 = 12 * math.log2(freq / 440.0)
            midi_note = round(69 + semitones_from_a4)
            midi_notes.append(midi_note)
        
        # Create path compatible with audio_utils
        audio_path = []
        for chord in best_path:
            midi_notes = []
            for freq in chord["frequencies"]:
                semitones_from_a4 = 12 * math.log2(freq / 440.0)
                midi_note = round(69 + semitones_from_a4)
                midi_notes.append(midi_note)
            
            audio_path.append({
                "name": chord["name"],
                "notes": midi_notes,
                "tension": chord["tension"],
            })
        
        audio = generate_progression_audio(
            path=audio_path,
            chord_duration=args.chord_duration,
        )
        
        save_audio(audio, args.output)


if __name__ == "__main__":
    main()
