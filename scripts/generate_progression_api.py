#!/usr/bin/env python3
"""
API wrapper for chord_progression_from_palette.py
Outputs JSON for the web API to consume.
"""
import os
import sys
import json
import argparse
import random
import math
import io
from contextlib import redirect_stdout, redirect_stderr

# Suppress all print output during imports
_devnull = io.StringIO()
with redirect_stdout(_devnull), redirect_stderr(_devnull):
    sys.path.insert(0, os.path.dirname(__file__))
    from chord_progression_from_palette import (
        analyze_chord_palette,
        build_chord_database_from_palette,
        brute_force_optimize,
        freq_to_note_name,
        voice_leading_cost_freq,
        soft_voice_leading_denominator,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate progression from palette and output JSON")
    parser.add_argument("--frequencies", type=float, nargs="+", required=True,
                        help="Input frequencies in Hz")
    parser.add_argument("--home-idx", type=int, default=0,
                        help="Index of home note (default: 0)")
    parser.add_argument("--curve", type=float, nargs=4, default=[0.0, 0.2, 0.65, -0.1],
                        help="Target tension curve (4 values)")
    parser.add_argument("--temperature", type=float, default=0.01,
                        help="Randomness factor (default: 0.01)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    
    frequencies = args.frequencies
    # Use frequency values as labels (continuous frequency space)
    freq_labels = [f"{f:.1f}" for f in frequencies]
    home_idx = args.home_idx % len(frequencies)
    
    # Suppress print output
    f = io.StringIO()
    with redirect_stdout(f):
        analysis = analyze_chord_palette(frequencies, freq_labels, home_idx)
        chords = build_chord_database_from_palette(analysis)
        best_path, best_score = brute_force_optimize(
            chords=chords,
            target_curve=args.curve,
            temperature=args.temperature,
        )
    
    # Compute rate matching details
    target_slopes = [args.curve[i] - args.curve[i-1] for i in range(1, len(args.curve))]
    
    progression = []
    for i, chord in enumerate(best_path):
        chord_info = {
            "frequencies": chord["frequencies"],
            "tension": chord["tension"],
        }
        
        progression.append(chord_info)
    
    output = {
        "chords": progression,
        "totalCost": best_score,
        "curve": args.curve,
        "temperature": args.temperature,
    }
    
    print(json.dumps(output))


if __name__ == "__main__":
    main()
