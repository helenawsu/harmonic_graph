"""
Tension Curve Analysis for Top 10 Chord Progressions

This script analyzes the tension curves of the top 10 most common
4-bar chord progressions using rate-of-change analysis.

The rate-of-change approach:
- First chord: uses local tension value
- Subsequent chords: calculates voice leading cost and computes
  the rate of tension change per semitone of voice movement
  (ΔTension / ΔVoiceLeading)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List

# Distance from home lookup table - keyed by Roman numerals
# Higher distance = more dissonant/tension with home root
# Based on RQA distance from C (I) for each scale degree
ROMAN_TENSION = {
    # Major chords (uppercase)
    'I': 0.0000,       # C_Maj - home/tonic
    '♭II': 0.5036,     # C#_Maj (Neapolitan)
    'II': 0.4459,      # D_Maj
    '♭III': 0.4172,    # D#/Eb_Maj
    'III': 0.4327,     # E_Maj
    'IV': 0.1422,      # F_Maj - subdominant
    '♯IV': 0.5745,     # F#_Maj
    'V': 0.5160,       # G_Maj - dominant
    '♭VI': 0.4823,     # G#/Ab_Maj
    'VI': 0.3579,      # A_Maj
    '♭VII': 0.7697,    # A#/Bb_Maj
    'VII': 0.7930,     # B_Maj
    
    # Minor chords (lowercase)
    'i': 0.0544,       # C_min
    '♭ii': 0.3243,     # C#_min
    'ii': 0.4353,      # D_min
    '♭iii': 0.4120,    # D#/Eb_min
    'iii': 0.4562,     # E_min
    'iv': 0.4336,      # F_min
    '♯iv': 0.5325,     # F#_min
    'v': 0.4820,       # G_min
    '♭vi': 0.6620,     # G#/Ab_min
    'vi': 0.6417,      # A_min - relative minor
    '♭vii': 0.7385,    # A#/Bb_min
    'vii': 0.7854,     # B_min
}

# Semitone intervals for chord types (for voice leading calculation)
CHORD_INTERVALS = {
    "Major": [0, 4, 7],  # Root, major 3rd, perfect 5th
    "minor": [0, 3, 7],  # Root, minor 3rd, perfect 5th
}

# Roman numeral to root semitone mapping (relative to home)
ROMAN_TO_SEMITONE = {
    'I': 0, 'i': 0,
    '♭II': 1, '♭ii': 1,
    'II': 2, 'ii': 2,
    '♭III': 3, '♭iii': 3,
    'III': 4, 'iii': 4,
    'IV': 5, 'iv': 5,
    '♯IV': 6, '♯iv': 6,
    'V': 7, 'v': 7,
    '♭VI': 8, '♭vi': 8,
    'VI': 9, 'vi': 9,
    '♭VII': 10, '♭vii': 10,
    'VII': 11, 'vii': 11,
}


def get_tension(chord: str) -> float:
    """Get the tension value for a Roman numeral chord."""
    chord = chord.strip()
    if chord in ROMAN_TENSION:
        return ROMAN_TENSION[chord]
    else:
        print(f"Warning: Chord '{chord}' not found in lookup table")
        return 0.0


def is_minor_chord(chord: str) -> bool:
    """Check if a Roman numeral chord is minor (lowercase)."""
    chord = chord.strip()
    # Minor chords are lowercase (excluding accidentals)
    base = chord.lstrip('♭♯')
    return base[0].islower() if base else False


def get_chord_notes(chord: str) -> List[int]:
    """
    Get MIDI note numbers for a Roman numeral chord.
    Returns notes in octave 4 (C4=60).
    """
    chord = chord.strip()
    
    # Get root semitone from home
    if chord not in ROMAN_TO_SEMITONE:
        print(f"Warning: Chord '{chord}' not found in semitone lookup")
        return [60, 64, 67]  # Default to C Major
    
    root_semitone = ROMAN_TO_SEMITONE[chord]
    base = 48 + root_semitone  # C3 + root offset
    
    # Determine chord type
    chord_type = "minor" if is_minor_chord(chord) else "Major"
    intervals = CHORD_INTERVALS[chord_type]
    
    return [base + interval for interval in intervals]


def voice_leading_cost(chord1_notes: List[int], chord2_notes: List[int]) -> float:
    """
    Calculates the 'Smart Pianist' distance in raw semitones.
    Assumes the player finds the closest inversion (minimized movement).
    
    Returns:
        Total semitone movement (not normalized)
    """
    # Convert everything to Pitch Classes (0-11)
    pc1 = [n % 12 for n in chord1_notes]
    pc2 = [n % 12 for n in chord2_notes]
    
    total_movement = 0.0
    
    # Create a copy of pc2 so we can "consume" notes as we match them
    remaining_targets = pc2.copy()
    
    for n1 in pc1:
        # Find the closest note in the remaining targets
        best_dist = 100
        best_target_idx = -1
        
        for i, n2 in enumerate(remaining_targets):
            # Calculate distance on the circle (0-6)
            diff = abs(n1 - n2)
            dist = min(diff, 12 - diff)
            
            if dist < best_dist:
                best_dist = dist
                best_target_idx = i
        
        total_movement += best_dist
        
        if best_target_idx != -1:
            remaining_targets.pop(best_target_idx)
    
    return total_movement


def get_tension_curve(row: pd.Series) -> list:
    """Get the tension curve (list of 4 tension values) for a progression."""
    return [
        get_tension(row['chord_1']),
        get_tension(row['chord_2']),
        get_tension(row['chord_3']),
        get_tension(row['chord_4'])
    ]


def soft_voice_leading_denominator(voice_distance: float) -> float:
    """
    Soft denominator for tension rate calculation.
    Uses logarithmic scaling to avoid punishing larger voice movements too hard.
    
    Formula: (log2(voice_distance) + 1) * 4
    
    Result: Average denominator is around 1, so rates are more intuitive.
    """
    import math
    if voice_distance < 1e-5:
        return 1.0
    return (math.log2(voice_distance) + 1) / 4


def get_rate_curve(row: pd.Series) -> dict:
    """
    Get the rate-of-change curve for a progression.
    
    Returns a dict with:
    - 'curve': [rate1, rate2, rate3] where rate_i = ΔTension / soft_denominator(ΔVoiceLeading)
    - 'tensions': raw tension values for each chord (tension = distance from home)
    - 'tension_curve': cumulative tension adjusted by soft voice leading [T₁, T₁+rate₁, T₁+rate₁+rate₂, ...]
    - 'slopes': raw tension derivatives [ΔT₁₂, ΔT₂₃, ΔT₃₄]
    - 'voice_costs': voice leading costs between chords
    - 'rates': rate of change values (same as curve)
    """
    chords = [row['chord_1'], row['chord_2'], row['chord_3'], row['chord_4']]
    tensions = [get_tension(c) for c in chords]  # tension = distance from home
    notes = [get_chord_notes(c) for c in chords]
    
    voice_costs = []
    rates = []
    slopes = []
    
    for i in range(1, 4):
        d_tension = tensions[i] - tensions[i-1]
        d_voice = voice_leading_cost(notes[i-1], notes[i])
        
        # Use soft denominator: log2(voice_distance + 1) + 1
        soft_denom = soft_voice_leading_denominator(d_voice)
        
        rate = d_tension / soft_denom
        voice_costs.append(d_voice)
        rates.append(rate)
        slopes.append(d_tension)
    
    # Build tension curve: cumulative sum of rates starting from T₁
    tension_curve = [tensions[0]]
    for rate in rates:
        tension_curve.append(tension_curve[-1] + rate)
    
    return {
        'curve': rates,  # Only the 3 rate values
        'tensions': tensions,  # Raw tension values (distance from home)
        'tension_curve': tension_curve,  # Cumulative tension with soft VL denominator
        'slopes': slopes,  # Raw tension derivatives
        'voice_costs': voice_costs,
        'rates': rates,
    }


def main():
    # Load the data
    results_dir = Path(__file__).parent.parent / "results"
    csv_path = results_dir / "rock_4bar_progressions_roman.csv"
    
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Get top 5 progressions by count
    top_5 = df.head(5).copy()
    
    print("\n📊 Top 5 Chord Progressions:")
    print("-" * 60)
    for i, row in top_5.iterrows():
        print(f"{i+1}. {row['progression']} (count: {row['count']}, {row['percentage']:.2f}%)")
    
    # Calculate rate-of-change curves
    rate_curves = []
    for _, row in top_5.iterrows():
        rate_data = get_rate_curve(row)
        rate_curves.append({
            'progression': row['progression'],
            'count': row['count'],
            'rate_curve': rate_data['curve'],
            'tensions': rate_data['tensions'],
            'tension_curve': rate_data['tension_curve'],
            'slopes': rate_data['slopes'],
            'voice_costs': rate_data['voice_costs'],
            'rates': rate_data['rates'],
            'avg_rate': np.mean([abs(r) for r in rate_data['rates']]),
            'rate_range': max(rate_data['rates']) - min(rate_data['rates']),
        })
    
    # Create the visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Calculate line widths based on frequency count (scale from 1 to 6)
    counts = [d['count'] for d in rate_curves]
    max_count, min_count = max(counts), min(counts)
    if max_count > min_count:
        line_widths = [1 + 5 * (c - min_count) / (max_count - min_count) for c in counts]
    else:
        line_widths = [3] * len(counts)
    
    # Plot 1: Rate-of-change curves (using soft denominator)
    ax1 = axes[0, 0]
    x_positions_rate = [1, 2, 3]
    x_labels_rate = ['Rate₁₂', 'Rate₂₃', 'Rate₃₄']
    
    for i, data in enumerate(rate_curves):
        label = f"{data['progression']} ({data['count']})"
        ax1.plot(x_positions_rate, data['rate_curve'], 
                marker='o', linewidth=line_widths[i], markersize=6 + line_widths[i],
                color=colors[i], label=label, alpha=0.8)
    
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Transition', fontsize=12)
    ax1.set_ylabel('Rate (ΔTension / soft(ΔVoice))', fontsize=12)
    ax1.set_title('Rate-of-Change Curves: ΔT / log₂(ΔV+1)+1', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_positions_rate)
    ax1.set_xticklabels(x_labels_rate)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Tension Curves (with soft voice leading denominator)
    ax2 = axes[0, 1]
    x_positions_tension = [1, 2, 3, 4]
    
    for i, data in enumerate(rate_curves):
        label = f"{data['progression']}"
        ax2.plot(x_positions_tension, data['tension_curve'], 
                marker='s', linewidth=line_widths[i], markersize=6 + line_widths[i],
                color=colors[i], label=label, alpha=0.8)
    
    ax2.set_xlabel('Bar Position', fontsize=12)
    ax2.set_ylabel('Tension (cumulative ΔT/soft(ΔV))', fontsize=12)
    ax2.set_title('Tension Curves (T₁ + Σ rates)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_positions_tension)
    ax2.set_xticklabels(['Chord 1', 'Chord 2', 'Chord 3', 'Chord 4'])
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 3: Tension Derivative (ΔTension between consecutive chords)
    ax3 = axes[1, 0]
    
    progressions = [d['progression'] for d in rate_curves]
    x = np.arange(len(progressions))
    width = 0.25
    
    # Calculate tension slopes (derivatives) for each progression
    slopes_1 = [d['tensions'][1] - d['tensions'][0] for d in rate_curves]
    slopes_2 = [d['tensions'][2] - d['tensions'][1] for d in rate_curves]
    slopes_3 = [d['tensions'][3] - d['tensions'][2] for d in rate_curves]
    
    bars1 = ax3.bar(x - width, slopes_1, width, label='ΔT 1→2', color='steelblue', alpha=0.8)
    bars2 = ax3.bar(x, slopes_2, width, label='ΔT 2→3', color='coral', alpha=0.8)
    bars3 = ax3.bar(x + width, slopes_3, width, label='ΔT 3→4', color='seagreen', alpha=0.8)
    
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Progression', fontsize=12)
    ax3.set_ylabel('ΔTension (Slope)', fontsize=12)
    ax3.set_title('Tension Derivative (ΔTension per Transition)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"#{i+1}" for i in range(5)], fontsize=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Voice leading costs
    ax4 = axes[1, 1]
    
    vc_1 = [d['voice_costs'][0] for d in rate_curves]
    vc_2 = [d['voice_costs'][1] for d in rate_curves]
    vc_3 = [d['voice_costs'][2] for d in rate_curves]
    
    bars1 = ax4.bar(x - width, vc_1, width, label='Voice 1→2', color='purple', alpha=0.8)
    bars2 = ax4.bar(x, vc_2, width, label='Voice 2→3', color='orange', alpha=0.8)
    bars3 = ax4.bar(x + width, vc_3, width, label='Voice 3→4', color='teal', alpha=0.8)
    
    ax4.set_xlabel('Progression', fontsize=12)
    ax4.set_ylabel('Voice Leading Cost (semitones)', fontsize=12)
    ax4.set_title('Voice Leading Costs by Transition', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"#{i+1}" for i in range(5)], fontsize=10)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = results_dir / "rate_curves_top10_rock.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Chart saved to: {output_path}")
    
    # Also show the plot
    plt.show()
    
    # Print detailed rate analysis
    print("\n📈 Detailed Rate-of-Change Analysis:")
    print("-" * 130)
    print(f"{'#':<3} {'Progression':<25} {'T₁':<7} {'ΔT₁₂':<8} {'ΔT₂₃':<8} {'ΔT₃₄':<8} {'Rate₁₂':<9} {'Rate₂₃':<9} {'Rate₃₄':<9} {'VC₁₂':<6} {'VC₂₃':<6} {'VC₃₄':<6}")
    print("-" * 130)
    
    for i, data in enumerate(rate_curves):
        print(f"{i+1:<3} {data['progression']:<25} "
              f"{data['tensions'][0]:<7.3f} "
              f"{data['slopes'][0]:+.3f}  "
              f"{data['slopes'][1]:+.3f}  "
              f"{data['slopes'][2]:+.3f}  "
              f"{data['rates'][0]:+.4f}  "
              f"{data['rates'][1]:+.4f}  "
              f"{data['rates'][2]:+.4f}  "
              f"{data['voice_costs'][0]:<6.1f} "
              f"{data['voice_costs'][1]:<6.1f} "
              f"{data['voice_costs'][2]:<6.1f}")
    
    # Additional insights
    print("\n🔍 Insights (Rate-of-Change Analysis):")
    
    # Find progression with highest average absolute rate
    max_avg_rate = max(rate_curves, key=lambda x: x['avg_rate'])
    print(f"  • Most dynamic (highest avg |rate|): {max_avg_rate['progression']} ({max_avg_rate['avg_rate']:.4f})")
    
    # Find progression with lowest average absolute rate
    min_avg_rate = min(rate_curves, key=lambda x: x['avg_rate'])
    print(f"  • Most stable (lowest avg |rate|): {min_avg_rate['progression']} ({min_avg_rate['avg_rate']:.4f})")
    
    # Find progression with largest rate swing
    max_range = max(rate_curves, key=lambda x: x['rate_range'])
    print(f"  • Largest rate swing: {max_range['progression']} ({max_range['rate_range']:.4f})")
    
    # Find progressions with consistent positive or negative rates
    print("\n  Rate patterns:")
    for data in rate_curves:
        rates = data['rates']
        if all(r > 0 for r in rates):
            print(f"    • {data['progression']}: consistently increasing tension")
        elif all(r < 0 for r in rates):
            print(f"    • {data['progression']}: consistently decreasing tension")
        elif rates[0] > 0 and rates[1] > 0 and rates[2] < 0:
            print(f"    • {data['progression']}: build-up then resolution (↑↑↓)")
        elif rates[0] > 0 and rates[1] < 0 and rates[2] > 0:
            print(f"    • {data['progression']}: wave pattern (↑↓↑)")
        elif rates[0] < 0 and rates[1] > 0 and rates[2] < 0:
            print(f"    • {data['progression']}: inverse wave (↓↑↓)")
    
    # Print rate curve for use in optimizer
    print("\n📋 Curves for Optimizer (copy-paste ready):")
    print("-" * 80)
    print("  Tension Curves [T₁, T₂, T₃, T₄] (with soft VL denominator):")
    for i, data in enumerate(rate_curves):
        t = data['tension_curve']
        curve_str = f"[{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}, {t[3]:.4f}]"
        print(f"    #{i+1} {data['progression']}: {curve_str}")
    
    print("\n  Raw Distance from Home [D₁, D₂, D₃, D₄]:")
    for i, data in enumerate(rate_curves):
        t = data['tensions']
        curve_str = f"[{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}, {t[3]:.4f}]"
        print(f"    #{i+1} {data['progression']}: {curve_str}")
    
    print("\n  Tension Slopes [ΔT₁₂, ΔT₂₃, ΔT₃₄] (raw):")
    for i, data in enumerate(rate_curves):
        s = data['slopes']
        curve_str = f"[{s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f}]"
        print(f"    #{i+1} {data['progression']}: {curve_str}")
    
    print("\n  Rate Curves [Rate₁₂, Rate₂₃, Rate₃₄] (with soft denominator):")
    for i, data in enumerate(rate_curves):
        curve = data['rate_curve']
        curve_str = f"[{curve[0]:.4f}, {curve[1]:.4f}, {curve[2]:.4f}]"
        print(f"    #{i+1} {data['progression']}: {curve_str}")


if __name__ == "__main__":
    main()
