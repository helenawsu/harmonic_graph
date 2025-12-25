/**
 * RQA (Recurrence Quantification Analysis) for Chord Analysis
 * 
 * TypeScript port of the Python chord_progression_setup.py
 * Computes chord consonance and distance from home using signal analysis.
 */

// =============================================================================
// Farey Grid for Just Intonation Snapping
// =============================================================================

interface Fraction {
    numerator: number;
    denominator: number;
    value: number;
}

function gcd(a: number, b: number): number {
    a = Math.abs(Math.round(a));
    b = Math.abs(Math.round(b));
    while (b) {
        const t = b;
        b = a % b;
        a = t;
    }
    return a;
}

function generateFareySequence(order: number): Fraction[] {
    const fracs = new Map<string, Fraction>();
    
    for (let d = 1; d <= order; d++) {
        for (let n = 0; n <= d; n++) {
            const g = gcd(n, d);
            const num = n / g;
            const den = d / g;
            const key = `${num}/${den}`;
            if (!fracs.has(key)) {
                fracs.set(key, {
                    numerator: num,
                    denominator: den,
                    value: num / den
                });
            }
        }
    }
    
    return Array.from(fracs.values()).sort((a, b) => a.value - b.value);
}

function generateBidirectionalFareyGrid(order: number = 50): Fraction[] {
    const farey = generateFareySequence(order);
    const grid = new Map<string, Fraction>();
    
    for (const frac of farey) {
        if (frac.numerator > 0 && frac.denominator > 0) {
            const key1 = `${frac.numerator}/${frac.denominator}`;
            grid.set(key1, frac);
            
            // Add inverse
            const invKey = `${frac.denominator}/${frac.numerator}`;
            if (!grid.has(invKey)) {
                grid.set(invKey, {
                    numerator: frac.denominator,
                    denominator: frac.numerator,
                    value: frac.denominator / frac.numerator
                });
            }
        }
    }
    
    return Array.from(grid.values()).sort((a, b) => a.value - b.value);
}

// Generate grid at module load
const FAREY_GRID = generateBidirectionalFareyGrid(50);
const RATIO_TOLERANCE = 0.01;

// =============================================================================
// RQA Parameters - Matched to Python chord_progression_setup.py
// =============================================================================

const BASELINE_FREQ = 400.0;
const BASELINE_SR = 8000;
const SAMPLES_PER_CYCLE = BASELINE_SR / BASELINE_FREQ; // = 20

// Parameters from Python (matched exactly)
const RQA_DURATION = 0.3;   // Same as Python
const RQA_WINDOW = 480;     // Same as Python
const RQA_SHIFT = 48;       // Same as Python
const RQA_EMB_DIM = 5;      // Same as Python
const RQA_DELAY = 3;        // Same as Python
const RQA_EPS_FACTOR = 0.1; // Same as Python
const CHORD_ROOT_BOOST = 10.0;

function getSrForFreq(rootFreq: number): number {
    return Math.floor(SAMPLES_PER_CYCLE * rootFreq);
}

// =============================================================================
// Ratio Snapping Functions
// =============================================================================

export function snapToJustIntonation(ratio: number, tolerance: number = RATIO_TOLERANCE): number {
    if (FAREY_GRID.length === 0) return ratio;
    
    // Octave-reduce to 1.0-2.0 range
    let octaveReduced = ratio;
    let octaves = 0;
    while (octaveReduced >= 2.0) {
        octaveReduced /= 2.0;
        octaves += 1;
    }
    while (octaveReduced < 1.0) {
        octaveReduced *= 2.0;
        octaves -= 1;
    }
    
    // Find best match in grid
    let bestMatch = octaveReduced;
    let bestScore = Infinity;
    
    for (const frac of FAREY_GRID) {
        const deviation = Math.abs(octaveReduced - frac.value);
        
        if (deviation / frac.value <= tolerance) {
            const score = frac.numerator + frac.denominator;
            if (score < bestScore) {
                bestScore = score;
                bestMatch = frac.value;
            }
        }
    }
    
    return bestMatch * Math.pow(2.0, octaves);
}

export function ratioStabilityScore(freq1: number, freq2: number): { score: number; name: string } {
    if (freq1 <= 0 || freq2 <= 0) {
        return { score: Infinity, name: "invalid" };
    }
    
    if (freq1 > freq2) {
        [freq1, freq2] = [freq2, freq1];
    }
    
    let actualRatio = freq2 / freq1;
    
    // Octave-reduce
    while (actualRatio >= 2.0) actualRatio /= 2.0;
    while (actualRatio < 1.0) actualRatio *= 2.0;
    
    let bestScore = Infinity;
    let bestName = "out of range";
    
    for (const frac of FAREY_GRID) {
        const deviation = Math.abs(actualRatio - frac.value);
        
        if (deviation / frac.value <= RATIO_TOLERANCE) {
            if (frac.numerator === frac.denominator) continue;
            const score = frac.numerator + frac.denominator;
            if (score < bestScore) {
                bestScore = score;
                bestName = `${frac.numerator}:${frac.denominator}`;
            }
        }
    }
    
    return { score: bestScore, name: bestName };
}

// =============================================================================
// RQA Functions - OPTIMIZED with typed arrays
// =============================================================================

function percentRecurrenceFast(x: Float64Array): number {
    const N = x.length;
    const embDim = RQA_EMB_DIM;
    const delay = RQA_DELAY;
    const L = N - (embDim - 1) * delay;
    if (L < 2) return 0.0;
    
    // Compute distances and stats in single pass
    let sumDist = 0;
    let count = 0;
    const numPairs = (L * (L - 1)) / 2;
    
    // First pass: compute mean distance
    for (let i = 0; i < L; i++) {
        for (let j = i + 1; j < L; j++) {
            let dist2 = 0;
            for (let k = 0; k < embDim; k++) {
                const diff = x[i + k * delay] - x[j + k * delay];
                dist2 += diff * diff;
            }
            sumDist += Math.sqrt(dist2);
            count++;
        }
    }
    
    if (count === 0) return 0.0;
    const avgDist = sumDist / count;
    const eps = RQA_EPS_FACTOR * avgDist;
    const eps2 = eps * eps;
    
    // Second pass: count recurrences (using squared distance to avoid sqrt)
    let recCount = 0;
    for (let i = 0; i < L; i++) {
        for (let j = i + 1; j < L; j++) {
            let dist2 = 0;
            for (let k = 0; k < embDim; k++) {
                const diff = x[i + k * delay] - x[j + k * delay];
                dist2 += diff * diff;
            }
            if (dist2 <= eps2) recCount++;
        }
    }
    
    return recCount / numPairs;
}

function lcm(a: number, b: number): number {
    return Math.abs(a * b) / gcd(a, b);
}

export function calculateBassStabilityWeight(rootFreq: number, allFreqs: number[]): number {
    if (allFreqs.length < 2) return 1.0;
    
    const rawRatios = allFreqs.map(f => f / rootFreq);
    const snappedRatios = rawRatios.map(r => snapToJustIntonation(r));
    
    // Convert to fractions and find LCM of denominators
    let lcmVal = 1;
    for (const r of snappedRatios) {
        // Approximate as fraction with limit denominator
        const frac = toFraction(r, 100);
        lcmVal = lcm(lcmVal, frac.denominator);
    }
    
    const bassHarmonicIndex = lcmVal;
    
    if (bassHarmonicIndex <= 0) return 0.5;
    
    const logVal = Math.log2(bassHarmonicIndex);
    
    // Check if power of 2
    if (Math.abs(logVal - Math.round(logVal)) < 0.001) {
        return 1.0;
    }
    
    const penalty = 0.2 * Math.log2(bassHarmonicIndex);
    return Math.max(0.4, 1.0 - penalty);
}

function toFraction(value: number, maxDenominator: number): { numerator: number; denominator: number } {
    // Simple continued fraction approximation
    let bestNum = 1;
    let bestDen = 1;
    let bestError = Math.abs(value - 1);
    
    for (let den = 1; den <= maxDenominator; den++) {
        const num = Math.round(value * den);
        const error = Math.abs(value - num / den);
        if (error < bestError) {
            bestError = error;
            bestNum = num;
            bestDen = den;
        }
    }
    
    return { numerator: bestNum, denominator: bestDen };
}

export function computeTriadRqa(rootFreq: number, allFrequencies: number[]): { rqaScore: number; bassWeight: number } {
    if (allFrequencies.length < 2) {
        return { rqaScore: 0.0, bassWeight: 0.0 };
    }
    
    const sr = getSrForFreq(rootFreq);
    
    const rawRatios = allFrequencies.map(f => f / rootFreq);
    const ratios = rawRatios.map(r => snapToJustIntonation(r));
    
    // Generate signal using typed array
    const numSamples = Math.floor(RQA_DURATION * sr);
    const sig = new Float64Array(numSamples);
    
    for (let i = 0; i < numSamples; i++) {
        const t = i / sr;
        for (const ratio of ratios) {
            const f = rootFreq * ratio;
            sig[i] += Math.sin(2.0 * Math.PI * f * t);
        }
    }
    
    // Normalize
    let maxVal = 0;
    for (let i = 0; i < numSamples; i++) {
        const abs = Math.abs(sig[i]);
        if (abs > maxVal) maxVal = abs;
    }
    if (maxVal > 0) {
        for (let i = 0; i < numSamples; i++) {
            sig[i] /= maxVal;
        }
    }
    
    // Sliding window RQA (matching Python)
    const results: number[] = [];
    let idx = 0;
    while (idx + RQA_WINDOW <= numSamples) {
        const window = sig.slice(idx, idx + RQA_WINDOW);
        const pr = percentRecurrenceFast(window);
        results.push(pr);
        idx += RQA_SHIFT;
    }
    
    const rqaScore = results.length > 0 
        ? results.reduce((a, b) => a + b, 0) / results.length 
        : 0.0;
    
    const bassWeight = calculateBassStabilityWeight(rootFreq, allFrequencies);
    
    return { rqaScore, bassWeight };
}

export function identifyChordRoot(chordFrequencies: number[]): number {
    if (chordFrequencies.length === 0) return 0.0;
    if (chordFrequencies.length === 1) return chordFrequencies[0];
    
    let bestRoot = chordFrequencies[0];
    let bestWeight = -1.0;
    
    for (const candidateRoot of chordFrequencies) {
        const weight = calculateBassStabilityWeight(candidateRoot, chordFrequencies);
        if (weight > bestWeight) {
            bestWeight = weight;
            bestRoot = candidateRoot;
        }
    }
    
    return bestRoot;
}

export function computeDistanceFromHomeWithBoost(
    homeFreq: number,
    chordRootFreq: number,
    chordFrequencies: number[]
): { rqaWithHome: number; distance: number } {
    const { rqaScore: homeOnlyRqa } = computeTriadRqa(homeFreq, [homeFreq]);
    
    const sr = getSrForFreq(homeFreq);
    
    const allFreqsForSignal = [homeFreq, ...chordFrequencies];
    const rawRatios = allFreqsForSignal.map(f => f / homeFreq);
    const snappedRatios = rawRatios.map(r => snapToJustIntonation(r));
    
    const numSamples = Math.floor(RQA_DURATION * sr);
    const sig = new Float64Array(numSamples);
    
    for (let i = 0; i < numSamples; i++) {
        const t = i / sr;
        
        // Home note - BOOSTED by 10x for harmonic grounding
        const homeRatio = snappedRatios[0];
        sig[i] += CHORD_ROOT_BOOST * Math.sin(2.0 * Math.PI * homeFreq * homeRatio * t);
        
        // Chord frequencies at normal amplitude
        for (let j = 0; j < chordFrequencies.length; j++) {
            const freq = chordFrequencies[j];
            const ratio = snappedRatios[j + 1];
            const f = homeFreq * ratio;
            
            sig[i] += Math.sin(2.0 * Math.PI * f * t);
        }
    }
    
    // Normalize
    let maxVal = 0;
    for (let i = 0; i < numSamples; i++) {
        const abs = Math.abs(sig[i]);
        if (abs > maxVal) maxVal = abs;
    }
    if (maxVal > 0) {
        for (let i = 0; i < numSamples; i++) {
            sig[i] /= maxVal;
        }
    }
    
    // Sliding window RQA (matching Python)
    const results: number[] = [];
    let idx = 0;
    while (idx + RQA_WINDOW <= numSamples) {
        const window = sig.slice(idx, idx + RQA_WINDOW);
        const pr = percentRecurrenceFast(window);
        results.push(pr);
        idx += RQA_SHIFT;
    }
    
    const chordWithHomeRqa = results.length > 0 
        ? results.reduce((a, b) => a + b, 0) / results.length 
        : 0.0;
    
    const normalizedRecurrence = homeOnlyRqa > 0.0001 
        ? chordWithHomeRqa / homeOnlyRqa 
        : chordWithHomeRqa;
    
    const distance = 1.0 / (1.0 + normalizedRecurrence);
    
    return { rqaWithHome: chordWithHomeRqa, distance };
}

export function computeDistanceFromHome(
    homeFreq: number,
    chordFrequencies: number[]
): { rqaWithHome: number; distance: number } {
    const chordRoot = identifyChordRoot(chordFrequencies);
    return computeDistanceFromHomeWithBoost(homeFreq, chordRoot, chordFrequencies);
}

// =============================================================================
// Chord Database Builder - Matching Python's find_best_triads_per_root exactly
// =============================================================================

export interface ChordInfo {
    frequencies: number[];
    rqaWithHome: number;
    rqaRecurrence: number;
    bassWeight: number;
    tension: number;
}

interface IntervalInfo {
    from: number;
    to: number;
    score: number;
    name: string;
}

interface TriadCandidate {
    rootFreq: number;
    otherFreqs: number[];
    otherIndices: number[];
    rqaScore: number;
    bassWeight: number;
    combinedScore: number;
    intervals: IntervalInfo[];
}

function combinations<T>(arr: T[], k: number): T[][] {
    if (k === 0) return [[]];
    if (arr.length === 0) return [];
    
    const [first, ...rest] = arr;
    const withFirst = combinations(rest, k - 1).map(combo => [first, ...combo]);
    const withoutFirst = combinations(rest, k);
    
    return [...withFirst, ...withoutFirst];
}

/**
 * Calculate stability score for a triad with a FIXED ROOT.
 * Returns intervals array with [root-note1, root-note2, note1-note2]
 */
function triadStabilityScoreRooted(rootFreq: number, otherFreqs: number[]): IntervalInfo[] {
    const sortedOthers = [...otherFreqs].sort((a, b) => a - b);
    
    const intervals: IntervalInfo[] = [];
    
    // Interval 1: Root to lower note
    const { score: score1, name: name1 } = ratioStabilityScore(rootFreq, sortedOthers[0]);
    intervals.push({ from: rootFreq, to: sortedOthers[0], score: score1, name: name1 });
    
    // Interval 2: Root to upper note
    const { score: score2, name: name2 } = ratioStabilityScore(rootFreq, sortedOthers[1]);
    intervals.push({ from: rootFreq, to: sortedOthers[1], score: score2, name: name2 });
    
    // Interval 3: Between the two other notes
    const { score: score3, name: name3 } = ratioStabilityScore(sortedOthers[0], sortedOthers[1]);
    intervals.push({ from: sortedOthers[0], to: sortedOthers[1], score: score3, name: name3 });
    
    return intervals;
}

/**
 * For each root note, find the best 2 triads (major and minor equivalent).
 * 
 * STAGE 1: Find most stable triad (major equivalent) by RQA + bass weight
 * STAGE 2: Find minor equivalent by:
 *   - Identify least stable note in the major triad
 *   - Replace with nearest neighbor (higher or lower)
 *   - Select the one with lower RQA as minor
 */
export function findBestTriadsPerRoot(
    frequencies: number[],
    homeIdx: number = 0
): ChordInfo[] {
    const n = frequencies.length;
    const homeFreq = frequencies[homeIdx];
    const allChords: ChordInfo[] = [];
    
    for (let rootIdx = 0; rootIdx < n; rootIdx++) {
        const rootFreq = frequencies[rootIdx];
        
        // Get all other notes (excluding root)
        const otherNotes: { idx: number; freq: number }[] = [];
        for (let i = 0; i < n; i++) {
            if (i !== rootIdx) {
                otherNotes.push({ idx: i, freq: frequencies[i] });
            }
        }
        
        // STAGE 1: Find all candidates and score them
        const triadCandidates: TriadCandidate[] = [];
        
        for (const combo of combinations(otherNotes, 2)) {
            const otherIndices = combo.map(c => c.idx);
            const otherFreqs = combo.map(c => c.freq);
            
            const intervals = triadStabilityScoreRooted(rootFreq, otherFreqs);
            const allFreqs = [rootFreq, ...otherFreqs.sort((a, b) => a - b)];
            
            const { rqaScore, bassWeight } = computeTriadRqa(rootFreq, allFreqs);
            
            // Combined score: higher = more stable (for finding major equivalent)
            const bonusFactor = 0.05;
            const combinedScore = rqaScore + bassWeight * bonusFactor;
            
            triadCandidates.push({
                rootFreq,
                otherFreqs: otherFreqs.sort((a, b) => a - b),
                otherIndices,
                rqaScore,
                bassWeight,
                combinedScore,
                intervals,
            });
        }
        
        if (triadCandidates.length === 0) continue;
        
        // Sort to find most stable (highest score)
        triadCandidates.sort((a, b) => b.combinedScore - a.combinedScore);
        const majorTriad = triadCandidates[0];
        
        // Add major triad to results
        const majorAllFreqs = [majorTriad.rootFreq, ...majorTriad.otherFreqs];
        const { rqaWithHome: majorRqaWithHome } = computeDistanceFromHome(homeFreq, majorAllFreqs);
        allChords.push({
            frequencies: majorAllFreqs,
            rqaWithHome: majorRqaWithHome,
            rqaRecurrence: majorTriad.rqaScore,
            bassWeight: majorTriad.bassWeight,
            tension: 0,
        });
        
        // STAGE 2: Find minor equivalent by replacing least stable note
        const intervals = majorTriad.intervals;
        
        // Find the interval with highest stability score (least stable = highest score)
        let leastStableIdx = 0;
        let maxScore = intervals[0].score;
        for (let i = 1; i < intervals.length; i++) {
            if (intervals[i].score > maxScore) {
                maxScore = intervals[i].score;
                leastStableIdx = i;
            }
        }
        
        const leastStableInterval = intervals[leastStableIdx];
        const freqA = leastStableInterval.from;
        const freqB = leastStableInterval.to;
        
        // Find which note to replace (not the root if interval involves root)
        // intervals are: [root-note1, root-note2, note1-note2]
        let freqToReplace: number;
        let idxToReplace: number;
        
        // Helper to find index in otherIndices for a given frequency
        const findOtherIdx = (freq: number): number => {
            for (let j = 0; j < majorTriad.otherFreqs.length; j++) {
                if (Math.abs(majorTriad.otherFreqs[j] - freq) < 0.1) {
                    return majorTriad.otherIndices[j];
                }
            }
            return majorTriad.otherIndices[0]; // Fallback
        };
        
        if (Math.abs(freqA - rootFreq) < 0.1) {
            // freq_a is root, so replace freq_b
            freqToReplace = freqB;
            idxToReplace = findOtherIdx(freqB);
        } else if (Math.abs(freqB - rootFreq) < 0.1) {
            // freq_b is root, so replace freq_a
            freqToReplace = freqA;
            idxToReplace = findOtherIdx(freqA);
        } else {
            // Interval is between the two other notes, replace the first one (freqA)
            freqToReplace = freqA;
            idxToReplace = findOtherIdx(freqA);
        }
        
        // Find nearest neighbors (higher and lower in the original frequency list)
        const neighborIndices: number[] = [];
        
        // Lower neighbor
        const lowerCandidates = frequencies
            .map((f, i) => ({ f, i }))
            .filter(x => x.i !== rootIdx && x.f < freqToReplace);
        if (lowerCandidates.length > 0) {
            const closest = lowerCandidates.reduce((a, b) => a.f > b.f ? a : b);
            neighborIndices.push(closest.i);
        }
        
        // Higher neighbor
        const higherCandidates = frequencies
            .map((f, i) => ({ f, i }))
            .filter(x => x.i !== rootIdx && x.f > freqToReplace);
        if (higherCandidates.length > 0) {
            const closest = higherCandidates.reduce((a, b) => a.f < b.f ? a : b);
            neighborIndices.push(closest.i);
        }
        
        // Create minor candidates by replacing with neighbors
        const minorCandidates: { allFreqs: number[]; rqaScore: number; bassWeight: number }[] = [];
        
        for (const neighborIdx of neighborIndices) {
            const neighborFreq = frequencies[neighborIdx];
            
            // Create new chord with neighbor note replacing the least stable
            const newOtherFreqs = [...majorTriad.otherFreqs];
            const newOtherIndices = [...majorTriad.otherIndices];
            
            // Find and replace the old note
            for (let j = 0; j < newOtherIndices.length; j++) {
                if (newOtherIndices[j] === idxToReplace) {
                    newOtherFreqs[j] = neighborFreq;
                    newOtherIndices[j] = neighborIdx;
                    break;
                }
            }
            
            // Sort the other notes
            const sortedOtherFreqs = [...newOtherFreqs].sort((a, b) => a - b);
            const minorAllFreqs = [rootFreq, ...sortedOtherFreqs];
            
            const { rqaScore, bassWeight } = computeTriadRqa(rootFreq, minorAllFreqs);
            
            minorCandidates.push({
                allFreqs: minorAllFreqs,
                rqaScore,
                bassWeight,
            });
        }
        
        // Select best minor (lowest RQA = more stable for minor)
        if (minorCandidates.length > 0) {
            minorCandidates.sort((a, b) => a.rqaScore - b.rqaScore);
            const minorTriad = minorCandidates[0];
            
            const { rqaWithHome: minorRqaWithHome } = computeDistanceFromHome(homeFreq, minorTriad.allFreqs);
            allChords.push({
                frequencies: minorTriad.allFreqs,
                rqaWithHome: minorRqaWithHome,
                rqaRecurrence: minorTriad.rqaScore,
                bassWeight: minorTriad.bassWeight,
                tension: 0,
            });
        }
    }
    
    return allChords;
}

export function buildChordDatabase(frequencies: number[], homeIdx: number = 0): ChordInfo[] {
    // Use the limited chord palette (2 per root) matching Python exactly
    const allChords = findBestTriadsPerRoot(frequencies, homeIdx);
    
    if (allChords.length === 0) return [];
    
    const rqaValues = allChords.map(c => c.rqaWithHome);
    const minRqa = Math.min(...rqaValues);
    const maxRqa = Math.max(...rqaValues);
    const rqaRange = maxRqa > minRqa ? maxRqa - minRqa : 1.0;
    
    for (const chord of allChords) {
        const normalizedRqa = (chord.rqaWithHome - minRqa) / rqaRange;
        chord.tension = 1.0 - normalizedRqa; // Invert: high RQA = low tension
    }
    
    return allChords;
}

// =============================================================================
// Voice Leading
// =============================================================================

function freqToContinuousSemitone(freq: number): number {
    if (freq <= 0) return 0;
    return 69 + 12 * Math.log2(freq / 440.0);
}

export function voiceLeadingCostFreq(chord1Freqs: number[], chord2Freqs: number[]): number {
    const pitch1 = chord1Freqs.map(freqToContinuousSemitone);
    const pitch2 = chord2Freqs.map(freqToContinuousSemitone);
    
    const pc1 = pitch1.map(p => p % 12.0);
    const pc2 = pitch2.map(p => p % 12.0);
    
    let totalMovement = 0.0;
    const remainingTargets = [...pc2];
    
    for (const p1 of pc1) {
        let bestDist = 100.0;
        let bestTargetIdx = -1;
        
        for (let i = 0; i < remainingTargets.length; i++) {
            const p2 = remainingTargets[i];
            const diff = Math.abs(p1 - p2);
            const dist = Math.min(diff, 12.0 - diff);
            
            if (dist < bestDist) {
                bestDist = dist;
                bestTargetIdx = i;
            }
        }
        
        totalMovement += bestDist;
        
        if (bestTargetIdx !== -1) {
            remainingTargets.splice(bestTargetIdx, 1);
        }
    }
    
    return totalMovement;
}

export function softVoiceLeadingDenominator(voiceDistance: number): number {
    if (voiceDistance < 1e-5) return 1.0;
    return Math.log2(voiceDistance + 1) + 1;
}

// =============================================================================
// Brute Force Optimizer
// =============================================================================

function chordsAreIdentical(chord1: ChordInfo, chord2: ChordInfo): boolean {
    const freq1 = [...chord1.frequencies].sort((a, b) => a - b);
    const freq2 = [...chord2.frequencies].sort((a, b) => a - b);
    
    if (freq1.length !== freq2.length) return false;
    
    const tolerance = 0.5;
    for (let i = 0; i < freq1.length; i++) {
        if (Math.abs(freq1[i] - freq2[i]) > tolerance) return false;
    }
    
    return true;
}

export function bruteForceOptimize(
    chords: ChordInfo[],
    targetCurve: number[],
    temperature: number = 0.0
): { bestPath: ChordInfo[] | null; bestScore: number } {
    if (targetCurve.length !== 4) {
        throw new Error("Target curve must have exactly 4 tension values");
    }
    
    if (chords.length < 4) {
        throw new Error(`Need at least 4 chords, got ${chords.length}`);
    }
    
    const targetSlopes = [
        targetCurve[1] - targetCurve[0],
        targetCurve[2] - targetCurve[1],
        targetCurve[3] - targetCurve[2],
    ];
    
    let bestPath: ChordInfo[] | null = null;
    let bestScore = Infinity;
    const allCandidates: { path: ChordInfo[]; cost: number; jitteredScore: number }[] = [];
    
    const nChords = chords.length;
    
    for (let c1 = 0; c1 < nChords; c1++) {
        for (let c2 = 0; c2 < nChords; c2++) {
            if (c2 === c1 || chordsAreIdentical(chords[c1], chords[c2])) continue;
            
            for (let c3 = 0; c3 < nChords; c3++) {
                if (c3 === c2) continue;
                if (chordsAreIdentical(chords[c1], chords[c3]) || 
                    chordsAreIdentical(chords[c2], chords[c3])) continue;
                
                for (let c4 = 0; c4 < nChords; c4++) {
                    if (c4 === c3) continue;
                    if (chordsAreIdentical(chords[c1], chords[c4]) ||
                        chordsAreIdentical(chords[c2], chords[c4]) ||
                        chordsAreIdentical(chords[c3], chords[c4])) continue;
                    
                    const path = [chords[c1], chords[c2], chords[c3], chords[c4]];
                    let pathCost = 0.0;
                    
                    for (let i = 1; i < path.length; i++) {
                        const targetSlope = targetSlopes[i - 1];
                        const dTension = path[i].tension - path[i - 1].tension;
                        
                        const freq1 = path[i - 1].frequencies;
                        const freq2 = path[i].frequencies;
                        const dVoice = voiceLeadingCostFreq(freq1, freq2);
                        
                        const softDenom = softVoiceLeadingDenominator(dVoice);
                        const actualRate = dTension / softDenom;
                        
                        pathCost += Math.abs(actualRate - targetSlope);
                    }
                    
                    if (temperature > 0) {
                        // Box-Muller for Gaussian
                        const u1 = Math.random();
                        const u2 = Math.random();
                        const jitter = temperature * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
                        const jitteredScore = pathCost + jitter;
                        allCandidates.push({ path, cost: pathCost, jitteredScore });
                    } else {
                        if (pathCost < bestScore) {
                            bestScore = pathCost;
                            bestPath = path;
                        }
                    }
                }
            }
        }
    }
    
    if (temperature > 0 && allCandidates.length > 0) {
        allCandidates.sort((a, b) => a.jitteredScore - b.jitteredScore);
        bestPath = allCandidates[0].path;
        bestScore = allCandidates[0].cost;
    }
    
    return { bestPath, bestScore };
}
