# Chord Palette Analysis with RQA Distance from Home

## Overview

The `chord_progression_arbitrary.py` script now implements a **chord palette analyzer** that:

1. **Accepts arbitrary frequency inputs** (up to 12 notes)
2. **Identifies best triads** for each root using ratio stability + RQA scoring
3. **Selects a home note** (default: lowest frequency)
4. **Calculates distance from home** using RQA with CHORD_ROOT_BOOST
5. **Ranks all chords** by consonance with home note

## Key Features

### 1. Ratio Snapping (`ratio_snapping.py`)

Separated into dedicated module for clean architecture:
- `snap_to_just_intonation()`: Maps frequency ratios to Farey sequence grid (3,045 rational ratios)
- `ratio_stability_score()`: Scores intervals by simplicity (3:2 better than 17:18)
- Avoids spurious high-denominator fractions from 12-TET equal temperament

### 2. Chord Root Identification

**Key Innovation**: The true chord root is identified using `calculate_bass_stability_weight()`:

```
Bass stability weight = 1.0  → Root position (bass note IS the root)
Bass stability weight = 0.7  → Strong harmonic (e.g., 3rd harmonic)
Bass stability weight = 0.5  → Weak harmonic (e.g., 5th harmonic)
```

**Examples**:
- **C-E-G**: C has weight 1.0 → C is the root
- **E-G-C** (1st inversion): C still has weight 1.0 → C is the root
- **F-C-A**: C has weight 1.0 → C is the root (F major in 1st inversion, not F-C-A chord)

### 3. Distance from Home with CHORD_ROOT_BOOST

When computing RQA distance from home:

1. **Identify the true chord root** using bass stability weights
2. **Boost the chord root amplitude** by `CHORD_ROOT_BOOST = 10.0`
3. **Compute RQA** of home note + boosted chord signal
4. **Calculate distance** as `1.0 - (normalized_recurrence)`

**Result**:
- **C-E-G with C as home**: C root gets boosted → high consonance (distance ≈ 0.979)
- **F-C-A with C as home**: C root (true root!) gets boosted → still consonant (distance ≈ 0.985)
- Chords without C in root position get lower boost → higher distance values

## Output

### Console Output
- Chord palette analysis table showing top 20 chords
- Sorted by distance from home (closest = most consonant)

### CSV Export
- `results/chord_palette_<home_note>_analysis.csv`
- Columns: Rank, Chord Name, Root Note, Chord Type, RQA Recurrence, Bass Weight, RQA with Home, Distance from Home, Consonance %

## Example Results (C3 as Home)

| Rank | Chord | Root | Type | Distance | Consonance |
|------|-------|------|------|----------|-----------|
| 4 | C3-E3-G3 | C3 | Major | 0.979 | 2.1% |
| 5 | C3-D#3-G3 | C3 | Minor | 0.983 | 1.7% |
| 9 | F3-C3-A3 | F3 | Major | 0.985 | 1.5% |

**Note**: C-E-G ranks #4 despite being the "perfect" major triad, because other chords with C as root also get boosted. The order reflects RQA consonance with home, not traditional music theory ranking.

## Usage

### Basic Usage
```python
from chord_progression_arbitrary import analyze_chord_palette

# Define frequencies and names
frequencies = [130.81, 138.59, 146.83, ...]  # C3 to B3
note_names = ["C3", "C#3", "D3", ...]

# Analyze with C3 as home (index 0)
analysis = analyze_chord_palette(frequencies, note_names, home_note_idx=0)

# Export results
export_chord_palette_to_csv(analysis)
```

### Custom Home Note
```python
# Use G3 (index 7) as home instead
analysis = analyze_chord_palette(frequencies, note_names, home_note_idx=7)
```

## RQA Parameters

- **Duration**: 0.3 seconds (optimized for speed)
- **Window Size**: 480 samples
- **Shift**: 48 samples
- **Embedding Dimension**: 5
- **Time Delay**: 3
- **Epsilon Factor**: 0.1 (relative to avg pairwise distance)
- **CHORD_ROOT_BOOST**: 10.0 (chord root amplitude multiplier)

## Files Modified

- `chord_progression_arbitrary.py`: Main analyzer
  - Added `identify_chord_root()` function
  - Added `compute_distance_from_home_with_boost()` function
  - Added `analyze_chord_palette()` function
  - Added `export_chord_palette_to_csv()` function
  - Added `CHORD_ROOT_BOOST` constant

- `ratio_snapping.py`: New module for ratio snapping
  - `load_farey_grid()`: Load pre-computed grid
  - `snap_to_just_intonation()`: Snap ratios to just intonation
  - `ratio_stability_score()`: Score interval stability

## Performance

- **Runtime**: ~5-10 minutes for 12-note chromatic scale
- **Memory**: ~100-200 MB
- **Optimization**: 0.3 second RQA duration per chord (vs. 1.0 in rqa_distance_from_home.py)

## See Also

- `rqa_distance_from_home.py`: Original distance-from-home implementation
- `chord_progression_optimizer.py`: Earlier chord optimization approach
- `rqa_all_roots.py`: RQA analysis framework
