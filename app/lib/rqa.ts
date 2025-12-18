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
// RQA Parameters - ULTRA OPTIMIZED for browser speed
// =============================================================================

const BASELINE_FREQ = 400.0;
const BASELINE_SR = 8000;
const SAMPLES_PER_CYCLE = BASELINE_SR / BASELINE_FREQ; // = 20

// Heavily reduced parameters for fast browser computation
const RQA_DURATION = 0.05;  // Very short - just enough for pattern detection
const RQA_WINDOW = 100;     // Much smaller window
const RQA_SHIFT = 100;      // Single window (no overlap)
const RQA_EMB_DIM = 3;      // Minimal embedding dimension
const RQA_DELAY = 2;
const RQA_EPS_FACTOR = 0.15;
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
        
        if (deviation / Math.max(frac.value, 0.001) <= tolerance) {
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
    let bestName = "unknown";
    
    for (const frac of FAREY_GRID) {
        const deviation = Math.abs(actualRatio - frac.value);
        
        if (deviation / Math.max(frac.value, 0.001) <= RATIO_TOLERANCE) {
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
    
    // Single window RQA (fast mode)
    const rqaScore = percentRecurrenceFast(sig);
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
        
        // Home note
        const homeRatio = snappedRatios[0];
        sig[i] += Math.sin(2.0 * Math.PI * homeFreq * homeRatio * t);
        
        // Chord frequencies with boost for chord root
        for (let j = 0; j < chordFrequencies.length; j++) {
            const freq = chordFrequencies[j];
            const ratio = snappedRatios[j + 1];
            const f = homeFreq * ratio;
            
            const amplitude = Math.abs(freq - chordRootFreq) < 0.1 ? CHORD_ROOT_BOOST : 1.0;
            sig[i] += amplitude * Math.sin(2.0 * Math.PI * f * t);
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
    
    // Single window RQA (fast mode)
    const chordWithHomeRqa = percentRecurrenceFast(sig);
    
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
// Chord Database Builder - Limited to 2 chords per root (major/minor equivalent)
// =============================================================================

export interface ChordInfo {
    frequencies: number[];
    rqaWithHome: number;
    rqaRecurrence: number;
    bassWeight: number;
    tension: number;
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
 * For each root note, find the best 2 triads (major and minor equivalent).
 * This limits the chord palette to 2 * n chords instead of n * C(n-1,2).
 */
export function findBestTriadsPerRoot(
    frequencies: number[],
    homeIdx: number = 0,
    topN: number = 2
): ChordInfo[] {
    const n = frequencies.length;
    const homeFreq = frequencies[homeIdx];
    const allChords: ChordInfo[] = [];
    
    for (let rootIdx = 0; rootIdx < n; rootIdx++) {
        const rootFreq = frequencies[rootIdx];
        const otherIndices = Array.from({ length: n }, (_, i) => i).filter(i => i !== rootIdx);
        
        // Generate all triads for this root and score them
        const triadsForRoot: { chord: ChordInfo; combinedScore: number }[] = [];
        
        for (const combo of combinations(otherIndices, 2)) {
            const otherFreqs = combo.map(i => frequencies[i]).sort((a, b) => a - b);
            const allFreqs = [rootFreq, ...otherFreqs];
            
            const { rqaScore, bassWeight } = computeTriadRqa(rootFreq, allFreqs);
            const { rqaWithHome } = computeDistanceFromHome(homeFreq, allFreqs);
            
            // Combined score: higher = more stable (for finding major equivalent)
            const bonusFactor = 0.05;
            const combinedScore = rqaScore + bassWeight * bonusFactor;
            
            triadsForRoot.push({
                chord: {
                    frequencies: allFreqs,
                    rqaWithHome,
                    rqaRecurrence: rqaScore,
                    bassWeight,
                    tension: 0,
                },
                combinedScore,
            });
        }
        
        if (triadsForRoot.length === 0) continue;
        
        // Sort by combined score (highest first = most stable = "major")
        triadsForRoot.sort((a, b) => b.combinedScore - a.combinedScore);
        
        // Take top N chords per root (default 2: major + minor equivalent)
        const selectedChords = triadsForRoot.slice(0, topN).map(t => t.chord);
        allChords.push(...selectedChords);
    }
    
    return allChords;
}

export function buildChordDatabase(frequencies: number[], homeIdx: number = 0): ChordInfo[] {
    // Use the limited chord palette (2 per root) instead of all combinations
    const allChords = findBestTriadsPerRoot(frequencies, homeIdx, 2);
    
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
