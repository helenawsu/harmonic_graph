#!/usr/bin/env python3
"""
Ratio Snapping Module

Provides functionality to snap frequency ratios to just intonation using
the Farey sequence grid for efficient and accurate ratio matching.

Key functions:
- load_farey_grid(): Load pre-computed Farey sequence grid
- snap_to_just_intonation(): Snap a ratio to nearest just intonation ratio
"""

import pickle
from pathlib import Path
from fractions import Fraction
import bisect


# =============================================================================
# Load Farey Sequence Grid for Ratio Snapping
# =============================================================================

def load_farey_grid():
    """
    Load the pre-computed bidirectional Farey sequence grid for fast snapping.
    This grid includes both ratios n:d and their inverses d:n, covering [0.01, 1.98].
    The grid is ordered, so binary search can quickly find the nearest ratio.
    
    Returns:
        List of Fraction objects sorted by value, or None if file not found
    """
    pkl_path = Path(__file__).parent.parent / "results" / "farey_sequence_grid_bidirectional.pkl"
    
    if pkl_path.exists():
        try:
            with open(pkl_path, 'rb') as f:
                grid = pickle.load(f)
            print(f"✓ Loaded bidirectional Farey sequence grid with {len(grid)} ratios from {pkl_path.name}")
            return grid
        except Exception as e:
            print(f"⚠ Failed to load Farey grid: {e}")
            return None
    else:
        print(f"⚠ Farey grid not found at {pkl_path}")
        return None


# Load the grid once at module startup
FAREY_GRID = load_farey_grid()

# Tolerance for matching ratios (e.g., 1.5 vs 1.501 should still match 3:2)
RATIO_TOLERANCE = 0.01  # 1% tolerance


def snap_to_just_intonation(ratio: float, tolerance: float = RATIO_TOLERANCE) -> float:
    """
    Snap a frequency ratio to the best just intonation ratio within tolerance.
    Uses the Farey sequence grid (order 100) with efficient binary search.
    
    Searches ALL ratios within tolerance range and selects the one with the
    lowest numerator + denominator score (most stable/simple).
    
    The Farey grid provides 3,045 rational ratios with guaranteed max gap of 0.01.
    
    Args:
        ratio: The frequency ratio to snap (should be >= 1.0)
        tolerance: Maximum deviation allowed (default 10%)
    
    Returns:
        The snapped just intonation ratio, or original if no match
    """
    if FAREY_GRID is None:
        raise RuntimeError("Farey grid not loaded - cannot snap ratios")
    
    # Octave-reduce to 1.0-2.0 range
    octave_reduced = ratio
    octaves = 0
    while octave_reduced >= 2.0:
        octave_reduced /= 2.0
        octaves += 1
    while octave_reduced < 1.0:
        octave_reduced *= 2.0
        octaves -= 1
    
    # Convert Fractions in grid to float for comparison
    grid_floats = [float(f) for f in FAREY_GRID]
    
    # Find ALL ratios within tolerance range
    tolerance_lower = octave_reduced / (1.0 + tolerance)
    tolerance_upper = octave_reduced * (1.0 + tolerance)
    
    # Binary search for range boundaries
    idx_lower = bisect.bisect_left(grid_floats, tolerance_lower)
    idx_upper = bisect.bisect_right(grid_floats, tolerance_upper)
    
    # Collect all candidates within tolerance range
    candidates_with_scores = []
    
    for i in range(idx_lower, idx_upper):
        candidate_frac = FAREY_GRID[i]
        candidate_float = grid_floats[i]
        
        deviation = abs(octave_reduced - candidate_float) / candidate_float
        
        if deviation <= tolerance:
            # Score = numerator + denominator (lower = more stable)
            score = candidate_frac.numerator + candidate_frac.denominator
            candidates_with_scores.append((score, candidate_float, candidate_frac))
    
    # Select the best candidate (lowest score = most stable)
    if candidates_with_scores:
        candidates_with_scores.sort(key=lambda x: x[0])
        best_score, best_match, best_frac = candidates_with_scores[0]
        return best_match * (2.0 ** octaves)
    
    return ratio  # No match, return original


def ratio_stability_score(freq1: float, freq2: float, tolerance: float = RATIO_TOLERANCE) -> tuple:
    """
    Calculate how "stable" the ratio between two frequencies is.
    
    Uses the bidirectional Farey sequence grid (covering 0.01 to 1.98) to find 
    the best matching rational ratio within tolerance.
    
    Octave-reduces the ratio first, then searches the grid directly.
    
    Returns:
        Tuple of (stability_score, ratio_name)
        Lower score = simpler/more stable ratio
    """
    if freq1 <= 0 or freq2 <= 0:
        return (float('inf'), "invalid")
    
    if FAREY_GRID is None:
        raise RuntimeError("Farey grid not loaded - cannot score ratios")
    
    # Ensure freq1 <= freq2 for consistent ratio calculation
    if freq1 > freq2:
        freq1, freq2 = freq2, freq1
    
    actual_ratio = freq2 / freq1
    
    # Octave-reduce: bring ratio into 1.0 - 2.0 range
    while actual_ratio >= 2.0:
        actual_ratio /= 2.0
    while actual_ratio < 1.0:
        actual_ratio *= 2.0
    
    # Now search the grid which covers [0.01, 1.98] and includes inverses
    # so we can find the ratio directly
    grid_floats = [float(f) for f in FAREY_GRID]
    
    # Find ALL ratios within tolerance range
    tolerance_lower = actual_ratio / (1.0 + tolerance)
    tolerance_upper = actual_ratio * (1.0 + tolerance)
    
    # Binary search for range boundaries
    idx_lower = bisect.bisect_left(grid_floats, tolerance_lower)
    idx_upper = bisect.bisect_right(grid_floats, tolerance_upper)
    
    # Collect all candidates within tolerance range
    candidates_with_scores = []
    
    for i in range(idx_lower, idx_upper):
        candidate_frac = FAREY_GRID[i]
        candidate_float = grid_floats[i]
        
        # Skip 1:1 ratio (unison) - we need different notes
        if candidate_frac.numerator == candidate_frac.denominator:
            continue
        
        deviation = abs(actual_ratio - candidate_float) / candidate_float
        
        if deviation <= tolerance:
            # Calculate stability score as sum of numerator + denominator
            # Lower = more stable (3:2 = 5, 4:3 = 7, 5:4 = 9, 17:18 = 35, etc.)
            stability_score = candidate_frac.numerator + candidate_frac.denominator
            ratio_name = f"{candidate_frac.numerator}:{candidate_frac.denominator}"
            
            # Tuple: (stability_score, deviation, ratio_name)
            # Sort by SIMPLICITY FIRST (lower score = simpler), then deviation as tiebreaker
            candidates_with_scores.append((stability_score, deviation, ratio_name))
    
    # Select the best candidate (lowest score = most simple, then lowest deviation for tie-breaking)
    if candidates_with_scores:
        candidates_with_scores.sort(key=lambda x: (x[0], x[1]))
        best_score, deviation, best_name = candidates_with_scores[0]
        return (float(best_score), best_name)
    
    return (float('inf'), "out of range")
