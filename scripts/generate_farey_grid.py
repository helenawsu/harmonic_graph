#!/usr/bin/env python3
"""
Generate a Farey Sequence-based rational grid where no two adjacent values 
are further apart than 0.01.

A Farey Sequence of order n is the set of all completely reduced fractions 
between 0 and 1 with denominators less than or equal to n.

For max step size ≤ 0.01, we need n ≥ 100.
"""

from fractions import Fraction
import json
from pathlib import Path


def generate_rational_grid(max_denominator, min_val=0.0, max_val=1.0):
    """
    Generate a set of rational numbers (as Fractions) with no two adjacent
    values further apart than 1/max_denominator.
    
    Args:
        max_denominator: Maximum denominator for Farey Sequence
        min_val: Minimum value (default 0.0)
        max_val: Maximum value (default 1.0)
    
    Returns:
        List of sorted Fractions
    """
    ratios = set()
    
    # Iterate through all possible denominators up to max_denominator
    for d in range(1, max_denominator + 1):
        # Calculate numerators that keep the fraction within our range
        start_n = int(min_val * d)
        end_n = int(max_val * d) + 1
        
        for n in range(start_n, end_n):
            f = Fraction(n, d)
            if min_val <= f <= max_val:
                ratios.add(f)
    
    # Sort the unique fractions
    sorted_ratios = sorted(list(ratios))
    return sorted_ratios


def main():
    """Generate Farey grid and save results to file."""
    
    # Generate the grid with order 100
    max_denominator = 100
    grid = generate_rational_grid(max_denominator)
    
    # Verify the step size
    step_sizes = [float(grid[i+1] - grid[i]) for i in range(len(grid)-1)]
    max_step = max(step_sizes)
    min_step = min(step_sizes)
    avg_step = sum(step_sizes) / len(step_sizes)
    
    # Prepare output data
    output_data = {
        "farey_order": max_denominator,
        "total_points": len(grid),
        "min_value": float(grid[0]),
        "max_value": float(grid[-1]),
        "step_sizes": {
            "maximum": max_step,
            "minimum": min_step,
            "average": avg_step
        },
        "first_20_ratios": [
            {"fraction": str(f), "decimal": float(f)} 
            for f in grid[:20]
        ],
        "last_20_ratios": [
            {"fraction": str(f), "decimal": float(f)} 
            for f in grid[-20:]
        ]
    }
    
    # Print summary to console
    print("=" * 70)
    print("FAREY SEQUENCE RATIONAL GRID GENERATION")
    print("=" * 70)
    print(f"Farey Sequence Order (n): {max_denominator}")
    print(f"Total points in grid: {len(grid)}")
    print(f"Range: [{float(grid[0]):.6f}, {float(grid[-1]):.6f}]")
    print()
    print("Step Size Statistics:")
    print(f"  Maximum step: {max_step:.6f}")
    print(f"  Minimum step: {min_step:.6f}")
    print(f"  Average step: {avg_step:.6f}")
    print()
    print("Verification: Maximum step ≤ 0.01?", "✓ YES" if max_step <= 0.01 else "✗ NO")
    print()
    print("First 20 ratios:")
    for f in grid[:20]:
        print(f"  {str(f):>5} = {float(f):.8f}")
    print()
    print("Last 20 ratios:")
    for f in grid[-20:]:
        print(f"  {str(f):>5} = {float(f):.8f}")
    print()
    
    # Save to JSON file
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    json_output_path = results_dir / "farey_sequence_grid.json"
    with open(json_output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"✓ JSON output saved to: {json_output_path}")
    
    # Save all ratios as CSV for easy import
    csv_output_path = results_dir / "farey_sequence_grid.csv"
    with open(csv_output_path, 'w') as f:
        f.write("fraction,decimal\n")
        for frac in grid:
            f.write(f"{frac},{float(frac):.10f}\n")
    print(f"✓ CSV output saved to: {csv_output_path}")
    
    # Save all ratios as plain text (one per line)
    txt_output_path = results_dir / "farey_sequence_grid.txt"
    with open(txt_output_path, 'w') as f:
        f.write("Farey Sequence of Order 100\n")
        f.write("Rational ratios from 0 to 1, max step size 0.01\n")
        f.write("=" * 50 + "\n\n")
        for i, frac in enumerate(grid, 1):
            f.write(f"{i:4d}. {str(frac):>5} = {float(frac):.10f}\n")
    print(f"✓ Text output saved to: {txt_output_path}")
    
    # Save raw fractions as Python pickle for future use
    import pickle
    pickle_output_path = results_dir / "farey_sequence_grid.pkl"
    with open(pickle_output_path, 'wb') as f:
        pickle.dump(grid, f)
    print(f"✓ Pickle output saved to: {pickle_output_path}")
    
    print()
    print("=" * 70)
    print(f"All outputs saved to: {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
