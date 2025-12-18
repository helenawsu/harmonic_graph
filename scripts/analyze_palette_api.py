#!/usr/bin/env python3
"""
API wrapper for chord_progression_setup.py
Outputs JSON for the web API to consume.
"""
import os
import sys
import json
import argparse
import io
from contextlib import redirect_stdout, redirect_stderr

# Suppress all print output during imports
_devnull = io.StringIO()
with redirect_stdout(_devnull), redirect_stderr(_devnull):
    sys.path.insert(0, os.path.dirname(__file__))
    from chord_progression_setup import (
        analyze_chord_palette,
        freq_to_note_name,
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze chord palette and output JSON")
    parser.add_argument("--frequencies", type=float, nargs="+", required=True,
                        help="Input frequencies in Hz")
    parser.add_argument("--home-idx", type=int, default=0,
                        help="Index of home note (default: 0)")
    
    args = parser.parse_args()
    
    frequencies = args.frequencies
    # Use frequency values as names (no note name conversion)
    freq_labels = [f"{f:.1f}" for f in frequencies]
    home_idx = args.home_idx % len(frequencies)
    
    # Suppress print output from analyze_chord_palette
    f = io.StringIO()
    with redirect_stdout(f):
        analysis = analyze_chord_palette(frequencies, freq_labels, home_idx)
    
    # Get raw RQA values and normalize to 0-1 range within this palette
    # Higher RQA = more consonant with home = lower tension
    rqa_values = [c["rqa_with_home"] for c in analysis["all_chords"]]
    min_rqa = min(rqa_values)
    max_rqa = max(rqa_values)
    rqa_range = max_rqa - min_rqa if max_rqa > min_rqa else 1.0
    
    # Format output for JSON
    output = {
        "home_freq": analysis["home_freq"],
        "frequencies": frequencies,
        "chords": []
    }
    
    for chord in analysis["all_chords"]:
        # Normalize tension: highest RQA = 0 tension, lowest RQA = 1 tension
        normalized_rqa = (chord["rqa_with_home"] - min_rqa) / rqa_range
        tension = 1.0 - normalized_rqa  # Invert so high RQA = low tension
        
        output["chords"].append({
            "frequencies": chord["frequencies"],
            "tension": tension,
        })
    
    print(json.dumps(output))


if __name__ == "__main__":
    main()
