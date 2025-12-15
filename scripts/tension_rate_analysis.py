#!/usr/bin/env python3
"""
Tension Rate of Change Analysis

Calculates the rate of change in tension using a soft denominator:
    Rate = ΔTension / (log2(VoiceLeadingDistance + 1) + 1)

This metric represents: "How much tension change per unit of (soft) voice movement?"
- Positive rate: tension increases (moving away from home)
- Negative rate: tension decreases (resolving toward home)
- Soft denominator: avoids punishing larger voice movements too hard
"""

from typing import List, Dict
import pandas as pd

from chord_progression_optimizer import (
    NOTE_NAMES,
    build_chord_database,
    voice_leading_cost,
    soft_voice_leading_denominator,
)


def calculate_all_transition_rates(home_root: int = 0) -> List[Dict]:
    """
    Calculate tension rate of change for all possible chord transitions.
    
    Rate = ΔTension / soft_denominator(ΔVoiceLeading)
    
    Where:
        ΔTension = chord2.tension - chord1.tension
        soft_denominator = log2(voice_distance + 1) + 1
    
    Returns:
        List of dicts with transition details
    """
    chords = build_chord_database(home_root)
    transitions = []
    
    for chord1 in chords:
        for chord2 in chords:
            if chord1["name"] == chord2["name"]:
                continue
            
            d_tension = chord2["tension"] - chord1["tension"]
            d_voice = voice_leading_cost(chord1["notes"], chord2["notes"])
            soft_denom = soft_voice_leading_denominator(d_voice)
            
            rate = d_tension / soft_denom
            
            transitions.append({
                "from_chord": chord1["name"],
                "from_roman": chord1["roman"],
                "from_tension": chord1["tension"],
                "to_chord": chord2["name"],
                "to_roman": chord2["roman"],
                "to_tension": chord2["tension"],
                "delta_tension": d_tension,
                "voice_leading_cost": d_voice,
                "soft_denom": soft_denom,
                "rate": rate,
                "abs_rate": abs(rate),
            })
    
    return transitions


def build_rate_table(home_root: int = 0) -> pd.DataFrame:
    """
    Build an N x N table of tension rates with Roman numeral labels.
    Rows = From chord, Columns = To chord
    Positive = tension increases, Negative = tension decreases (resolution)
    Uses soft denominator: log2(voice_distance + 1) + 1
    """
    chords = build_chord_database(home_root)
    
    # Sort by tension (most consonant first)
    chords = sorted(chords, key=lambda c: c["tension"])
    
    # Build rate lookup using roman numerals
    rates = {}
    for c1 in chords:
        for c2 in chords:
            if c1["name"] == c2["name"]:
                rates[(c1["roman"], c2["roman"])] = None
            else:
                d_tension = c2["tension"] - c1["tension"]
                d_voice = voice_leading_cost(c1["notes"], c2["notes"])
                soft_denom = soft_voice_leading_denominator(d_voice)
                rates[(c1["roman"], c2["roman"])] = d_tension / soft_denom
    
    # Get roman numerals in tension order
    romans = [c["roman"] for c in chords]
    
    # Build DataFrame
    data = []
    for from_roman in romans:
        row = {}
        for to_roman in romans:
            row[to_roman] = rates.get((from_roman, to_roman))
        data.append(row)
    
    df = pd.DataFrame(data, index=romans)
    return df


def analyze_transitions(transitions: List[Dict]) -> Dict:
    """Compute summary statistics for the transitions."""
    rates = [t["rate"] for t in transitions]
    abs_rates = [t["abs_rate"] for t in transitions]
    
    return {
        "total_transitions": len(transitions),
        "min_rate": min(rates),
        "max_rate": max(rates),
        "mean_rate": sum(rates) / len(rates),
        "mean_abs_rate": sum(abs_rates) / len(abs_rates),
        "positive_rates": len([r for r in rates if r > 0]),
        "negative_rates": len([r for r in rates if r < 0]),
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze tension rate of change for all chord transitions"
    )
    parser.add_argument(
        "--home", type=int, default=0,
        help="Home root note (0=C, 1=C#, 2=D, etc.). Default: 0 (C)"
    )
    parser.add_argument(
        "--output", type=str, default="results/tension_rate_table.csv",
        help="Output CSV file path. Default: results/tension_rate_table.csv"
    )
    parser.add_argument(
        "--top-n", type=int, default=15,
        help="Show top N transitions. Default: 15"
    )
    
    args = parser.parse_args()
    home_root = args.home % 12
    
    print(f"\nTension Rate of Change Analysis")
    print(f"Home Key: {NOTE_NAMES[home_root]} Major")
    print(f"=" * 70)
    
    # Calculate all transitions
    transitions = calculate_all_transition_rates(home_root)
    
    # Summary statistics
    stats = analyze_transitions(transitions)
    print(f"\nSummary:")
    print(f"  Total transitions: {stats['total_transitions']}")
    print(f"  Rate range: [{stats['min_rate']:.4f}, {stats['max_rate']:.4f}]")
    print(f"  Positive rates (builds tension): {stats['positive_rates']}")
    print(f"  Negative rates (resolves): {stats['negative_rates']}")
    
    # Build and display the table
    df = build_rate_table(home_root)
    
    print(f"\n{'='*70}")
    print("TENSION RATE TABLE (ΔTension / ΔVoiceLeading)")
    print("Rows = From, Columns = To")
    print("Positive = builds tension, Negative = resolves")
    print(f"{'='*70}\n")
    print(df.round(3).to_string())
    
    # Top resolving transitions
    print(f"\n{'='*70}")
    print(f"TOP {args.top_n} RESOLVING TRANSITIONS (Most Negative Rate)")
    print(f"{'='*70}")
    
    sorted_neg = sorted(transitions, key=lambda x: x["rate"])
    print(f"\n{'From':>8} -> {'To':>8} | {'Rate':>8} | {'ΔT':>7} | {'ΔV':>4}")
    print("-" * 50)
    for t in sorted_neg[:args.top_n]:
        print(f"{t['from_roman']:>8} -> {t['to_roman']:>8} | {t['rate']:>+8.4f} | "
              f"{t['delta_tension']:>+7.4f} | {t['voice_leading_cost']:>4.0f}")
    
    # Top tension-building transitions
    print(f"\n{'='*70}")
    print(f"TOP {args.top_n} TENSION-BUILDING TRANSITIONS (Most Positive Rate)")
    print(f"{'='*70}")
    
    sorted_pos = sorted(transitions, key=lambda x: x["rate"], reverse=True)
    print(f"\n{'From':>8} -> {'To':>8} | {'Rate':>8} | {'ΔT':>7} | {'ΔV':>4}")
    print("-" * 50)
    for t in sorted_pos[:args.top_n]:
        print(f"{t['from_roman']:>8} -> {t['to_roman']:>8} | {t['rate']:>+8.4f} | "
              f"{t['delta_tension']:>+7.4f} | {t['voice_leading_cost']:>4.0f}")
    
    # Common progressions
    print(f"\n{'='*70}")
    print("COMMON PROGRESSIONS")
    print(f"{'='*70}")
    
    common = [
        ("I", "IV"), ("IV", "V"), ("V", "I"), ("I", "V"),
        ("I", "vi"), ("vi", "IV"), ("IV", "I"), ("ii", "V"), ("V", "vi"),
    ]
    
    print(f"\n{'Progression':>12} | {'Rate':>8} | Notes")
    print("-" * 40)
    for from_r, to_r in common:
        for t in transitions:
            if t["from_roman"] == from_r and t["to_roman"] == to_r:
                note = "builds" if t["rate"] > 0 else "resolves"
                print(f"{from_r:>5} → {to_r:<5} | {t['rate']:>+8.4f} | {note}")
                break
    
    # Detailed breakdown for I, iii, V, VII transitions
    print(f"\n{'='*70}")
    print("DETAILED BREAKDOWN: I, iii, V, VII TRANSITIONS")
    print(f"{'='*70}")
    
    chords = build_chord_database(home_root)
    chord_lookup = {c["roman"]: c for c in chords}
    
    # Get I, iii, V, VII chords (tertian stack from I)
    primary_chords = ["I", "iii", "V", "VII"]
    
    print("\n--- CHORD PROPERTIES ---")
    print(f"{'Chord':>6} | {'Roughness':>10} | {'Distance':>10} | {'Tension':>10}")
    print("-" * 50)
    for roman in primary_chords:
        c = chord_lookup[roman]
        print(f"{roman:>6} | {c['roughness']:>10.4f} | {c['distance']:>10.4f} | {c['tension']:>10.4f}")
    
    print("\n--- TRANSITION BREAKDOWN ---")
    primary_transitions = [
        ("I", "iii"), ("iii", "V"), ("V", "VII"), ("VII", "I"),
        ("I", "V"), ("V", "I"), ("I", "VII"), ("VII", "V"),
        ("iii", "I"), ("iii", "VII"), ("VII", "iii"),
    ]
    
    print(f"\n{'From':>4} -> {'To':>4} | {'From_T':>7} | {'To_T':>7} | {'ΔT':>7} | {'ΔV':>4} | {'SoftD':>6} | {'Rate':>8} | Direction")
    print("-" * 85)
    
    for from_r, to_r in primary_transitions:
        c1 = chord_lookup[from_r]
        c2 = chord_lookup[to_r]
        d_tension = c2["tension"] - c1["tension"]
        d_voice = voice_leading_cost(c1["notes"], c2["notes"])
        soft_denom = soft_voice_leading_denominator(d_voice)
        rate = d_tension / soft_denom
        direction = "builds ↑" if rate > 0 else "resolves ↓"
        
        print(f"{from_r:>4} -> {to_r:>4} | {c1['tension']:>7.4f} | {c2['tension']:>7.4f} | "
              f"{d_tension:>+7.4f} | {d_voice:>4.0f} | {soft_denom:>6.2f} | {rate:>+8.4f} | {direction}")
    
    print("\n--- FORMULA EXPLANATION ---")
    print("Tension = Distance (100% distance from home)")
    print("Rate = ΔTension / SoftDenominator")
    print("SoftDenominator = log2(VoiceLeadingDistance + 1) + 1")
    print("  - Distance: 0=home (I), 1=remote (from RQA spectral distance)")
    print("  - VoiceLeading: semitones of movement between chord voicings")
    print("  - SoftD: log2(ΔV + 1) + 1 (reduces penalty for larger moves)")
    
    # Save table
    df.round(4).to_csv(args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
