"""
Generate a bidirectional Farey sequence grid with both ratios and their inverses.
This allows lookup in both [0, 1] and [1, 2) ranges without needing to invert.
"""

import pickle
from fractions import Fraction
import json


def generate_bidirectional_farey_grid(max_denominator=100, min_val=0.0, max_val=1.0):
    """
    Generate Farey sequence and include both n:d and d:n for each fraction.
    
    Returns: Sorted list of Fraction objects covering [0, 2) space
    """
    ratios = set()
    
    # Generate Farey sequence for [0, 1]
    for d in range(1, max_denominator + 1):
        for n in range(int(min_val * d), int(max_val * d) + 1):
            if n == 0:  # Skip 0:d (equivalent to 0)
                continue
            f = Fraction(n, d)
            if min_val <= f <= max_val:
                ratios.add(f)
                
                # Also add the inverse (n:d becomes d:n, which is in [1, ∞))
                # But we only add inverses that stay in reasonable range for intervals
                inv_f = Fraction(d, n)
                # Only add if inverse is < 2 (one octave)
                if inv_f < 2.0:
                    ratios.add(inv_f)
    
    return sorted(list(ratios))


def main():
    print("Generating bidirectional Farey sequence grid...")
    
    # Generate grid
    grid = generate_bidirectional_farey_grid(max_denominator=100)
    
    print(f"Generated {len(grid)} ratios")
    
    # Verify coverage
    grid_floats = [float(f) for f in grid]
    
    # Check max gap in [0, 1]
    max_gap = 0
    for i in range(1, len(grid_floats)):
        gap = grid_floats[i] - grid_floats[i-1]
        max_gap = max(max_gap, gap)
    
    print(f"Coverage: {grid_floats[0]:.6f} to {grid_floats[-1]:.6f}")
    print(f"Max gap: {max_gap:.6f} ({max_gap*100:.2f}%)")
    print(f"Min gap: {min(grid_floats[i] - grid_floats[i-1] for i in range(1, len(grid_floats))):.6f}")
    
    # Save to pickle
    pkl_path = "/Users/helena/Desktop/harmonic_graph/results/farey_sequence_grid_bidirectional.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(grid, f)
    print(f"\n✓ Saved to {pkl_path}")
    
    # Save to CSV
    csv_path = "/Users/helena/Desktop/harmonic_graph/results/farey_sequence_grid_bidirectional.csv"
    with open(csv_path, "w") as f:
        f.write("fraction,decimal,numerator,denominator\n")
        for frac in grid:
            f.write(f"{frac.numerator}:{frac.denominator},{float(frac):.10f},{frac.numerator},{frac.denominator}\n")
    print(f"✓ Saved to {csv_path}")
    
    # Save to JSON with stats
    json_path = "/Users/helena/Desktop/harmonic_graph/results/farey_sequence_grid_bidirectional.json"
    with open(json_path, "w") as f:
        json.dump({
            "description": "Bidirectional Farey sequence grid with ratios and inverses",
            "max_denominator": 100,
            "total_ratios": len(grid),
            "range": [float(grid[0]), float(grid[-1])],
            "max_gap": max_gap,
            "ratios": [f"{frac.numerator}:{frac.denominator}" for frac in grid]
        }, f, indent=2)
    print(f"✓ Saved to {json_path}")
    
    # Show sample
    print("\nSample ratios:")
    for i in range(0, min(20, len(grid))):
        print(f"  {grid[i]}: {float(grid[i]):.6f}")
    print(f"  ...")
    for i in range(max(len(grid)-20, 0), len(grid)):
        print(f"  {grid[i]}: {float(grid[i]):.6f}")


if __name__ == "__main__":
    main()
