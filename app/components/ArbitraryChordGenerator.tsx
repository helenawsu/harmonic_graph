"use client";
import React, { useState, useCallback, useRef } from "react";
import TensionCurveEditor from "./TensionCurveEditor";
import { playFrequencies, resumeAudioIfNeeded } from "../tonnetz/audio";
import { voiceLeadingCostFreq, softVoiceLeadingDenominator } from "../lib/rqa";

interface ChordInfo {
  frequencies: number[];
  tension: number;
}

interface PaletteChord {
  frequencies: number[];
  tension: number;
}

interface PaletteAnalysis {
  home_freq: number;
  frequencies: number[];
  chords: PaletteChord[];
}

interface ProgressionResult {
  chords: ChordInfo[];
  totalCost: number;
  curve: number[];
  temperature: number;
}

// Frequency range constants
const MIN_FREQ = 150;  // ~D3
const MAX_FREQ = 300;  // ~D4

// Generate evenly spaced frequencies in log space
function generateEvenlySpacedFrequencies(numNotes: number): number[] {
  const frequencies: number[] = [];
  if (numNotes === 1) return [MIN_FREQ];
  
  for (let i = 0; i < numNotes; i++) {
    const ratio = Math.pow(MAX_FREQ / MIN_FREQ, i / (numNotes - 1));
    frequencies.push(MIN_FREQ * ratio);
  }
  return frequencies;
}

// Frequency Editor Component - 1D horizontal line where position = frequency
function FrequencyEditor({
  frequencies,
  onChange,
  rootIdx,
  onRootChange,
}: {
  frequencies: number[];
  onChange: (freqs: number[]) => void;
  rootIdx: number;
  onRootChange: (idx: number) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [draggingIndex, setDraggingIndex] = useState<number | null>(null);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editValue, setEditValue] = useState("");

  const TRACK_HEIGHT = 60;
  const NODE_SIZE = 16;
  const PADDING = 20;

  // Convert frequency to X position (log scale)
  const freqToX = (freq: number, width: number) => {
    const usableWidth = width - 2 * PADDING;
    const logMin = Math.log(MIN_FREQ);
    const logMax = Math.log(MAX_FREQ);
    const logFreq = Math.log(Math.max(MIN_FREQ, Math.min(MAX_FREQ, freq)));
    const normalized = (logFreq - logMin) / (logMax - logMin);
    return PADDING + normalized * usableWidth;
  };

  // Convert X position to frequency (log scale)
  const xToFreq = (x: number, width: number) => {
    const usableWidth = width - 2 * PADDING;
    const normalized = (x - PADDING) / usableWidth;
    const logMin = Math.log(MIN_FREQ);
    const logMax = Math.log(MAX_FREQ);
    const logFreq = logMin + normalized * (logMax - logMin);
    return Math.exp(logFreq);
  };

  const handleFrequencyChange = (index: number, newFreq: number) => {
    const clamped = Math.max(MIN_FREQ, Math.min(MAX_FREQ, newFreq));
    const newFreqs = [...frequencies];
    newFreqs[index] = clamped;
    onChange(newFreqs);
  };

  const handleDragStart = (e: React.MouseEvent | React.TouchEvent, index: number) => {
    e.preventDefault();
    e.stopPropagation();
    setDraggingIndex(index);
  };

  const handleDragMove = (e: React.MouseEvent | React.TouchEvent) => {
    if (draggingIndex === null || !containerRef.current) return;
    
    const rect = containerRef.current.getBoundingClientRect();
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
    const x = clientX - rect.left;
    const newFreq = xToFreq(x, rect.width);
    handleFrequencyChange(draggingIndex, newFreq);
  };

  const handleDragEnd = () => {
    setDraggingIndex(null);
  };

  const handleDoubleClick = (e: React.MouseEvent, index: number) => {
    e.stopPropagation();
    setEditingIndex(index);
    setEditValue(frequencies[index].toFixed(1));
  };

  const handleEditSubmit = (index: number) => {
    const newFreq = parseFloat(editValue);
    if (!isNaN(newFreq)) {
      handleFrequencyChange(index, newFreq);
    }
    setEditingIndex(null);
  };

  const handleClick = (e: React.MouseEvent, index: number) => {
    e.stopPropagation();
    onRootChange(index);
  };

  const [containerWidth, setContainerWidth] = useState(600);
  
  // Update width on mount and resize
  React.useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        setContainerWidth(containerRef.current.offsetWidth);
      }
    };
    updateWidth();
    window.addEventListener('resize', updateWidth);
    return () => window.removeEventListener('resize', updateWidth);
  }, []);

  return (
    <div style={{ width: "100%" }}>
      {/* Track container */}
      <div 
        ref={containerRef}
        className="freq-editor-container"
        style={{ 
          position: "relative",
          height: TRACK_HEIGHT,
          width: "100%",
          background: "#1a1a1a",
          borderRadius: 4,
          cursor: draggingIndex !== null ? "ew-resize" : "default",
        }}
        onMouseMove={handleDragMove}
        onMouseUp={handleDragEnd}
        onMouseLeave={handleDragEnd}
        onTouchMove={handleDragMove}
        onTouchEnd={handleDragEnd}
      >
        {/* Horizontal line */}
        <div style={{
          position: "absolute",
          top: "50%",
          left: PADDING,
          right: PADDING,
          height: 2,
          background: "#333",
          transform: "translateY(-50%)",
        }} />
        
        {/* Nodes */}
        {frequencies.map((freq, i) => {
          const x = freqToX(freq, containerWidth);
          return (
            <div
              key={i}
              style={{
                position: "absolute",
                left: x,
                top: "50%",
                transform: "translate(-50%, -50%)",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                cursor: draggingIndex === i ? "ew-resize" : "grab",
                userSelect: "none",
                zIndex: draggingIndex === i || editingIndex === i ? 10 : 1,
              }}
              onMouseDown={(e) => editingIndex !== i && handleDragStart(e, i)}
              onTouchStart={(e) => editingIndex !== i && handleDragStart(e, i)}
              onDoubleClick={(e) => handleDoubleClick(e, i)}
              onClick={(e) => handleClick(e, i)}
            >
              <div
                style={{
                  width: NODE_SIZE,
                  height: NODE_SIZE,
                  borderRadius: "50%",
                  backgroundColor: "#000",
                  border: `2px solid ${i === rootIdx ? "#fff" : "#888"}`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 8,
                  fontWeight: "bold",
                  color: "#fff",
                  transition: "transform 0.05s",
                  transform: draggingIndex === i ? "scale(1.3)" : "scale(1)",
                }}
              >
                {i + 1}
              </div>
              {/* Frequency label below node */}
              {editingIndex === i ? (
                <input
                  type="number"
                  value={editValue}
                  onChange={(e) => setEditValue(e.target.value)}
                  onBlur={() => handleEditSubmit(i)}
                  onKeyDown={(e) => e.key === "Enter" && handleEditSubmit(i)}
                  onClick={(e) => e.stopPropagation()}
                  onMouseDown={(e) => e.stopPropagation()}
                  autoFocus
                  style={{
                    width: 40,
                    fontSize: 9,
                    textAlign: "center",
                    background: "#222",
                    border: "1px solid #fff",
                    borderRadius: 2,
                    color: "#fff",
                    padding: "1px",
                    marginTop: 2,
                  }}
                />
              ) : (
                <span style={{ fontSize: 8, color: i === rootIdx ? "#fff" : "#666", marginTop: 2 }}>
                  {freq.toFixed(0)}
                </span>
              )}
            </div>
          );
        })}
      </div>
      
      {/* Help text */}
      <div style={{ fontSize: 9, color: "#aaaaaa", textAlign: "center", marginTop: 4 }}>
        drag ← lower | higher → | click = root | dbl-click = edit
      </div>
    </div>
  );
}

// Format frequencies for display
function formatFreqs(freqs: number[]): string {
  return freqs.map(f => f.toFixed(1)).join(" - ");
}

export default function ArbitraryChordGenerator() {
  const [numNotes, setNumNotes] = useState(7);
  const [frequencies, setFrequencies] = useState<number[]>(() => generateEvenlySpacedFrequencies(7));
  const [rootIdx, setRootIdx] = useState(0);
  const [curve, setCurve] = useState<number[]>([0.0, 0.2, 0.65, -0.1]);
  const [temperature, setTemperature] = useState(0.01);
  
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isAnalyzeHovered, setIsAnalyzeHovered] = useState(false);
  const [isGenerateHovered, setIsGenerateHovered] = useState(false);
  
  const [palette, setPalette] = useState<PaletteAnalysis | null>(null);
  const [result, setResult] = useState<ProgressionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentChordIndex, setCurrentChordIndex] = useState<number | null>(null);
  
  const playingRef = useRef(false);

  const handleNumNotesChange = useCallback((newNum: number) => {
    setNumNotes(newNum);
    setFrequencies(generateEvenlySpacedFrequencies(newNum));
    setRootIdx(0);
    setPalette(null);
    setResult(null);
  }, []);

  const handleFrequenciesChange = useCallback((newFreqs: number[]) => {
    setFrequencies(newFreqs);
    setPalette(null);
    setResult(null);
  }, []);

  const handleRootChange = useCallback((idx: number) => {
    setRootIdx(idx);
    setPalette(null);
    setResult(null);
  }, []);

  const analyzePalette = useCallback(async () => {
    setIsAnalyzing(true);
    setError(null);
    setPalette(null);
    setResult(null);

    try {
      const response = await fetch("/api/analyze-palette", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frequencies, homeIdx: rootIdx }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to analyze palette");
      }

      const data: PaletteAnalysis = await response.json();
      setPalette(data);
    } catch (err) {
      setError(String(err));
    } finally {
      setIsAnalyzing(false);
    }
  }, [frequencies, rootIdx]);

  const generateProgression = useCallback(async () => {
    setIsGenerating(true);
    setError(null);

    try {
      const response = await fetch("/api/generate-from-palette", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          frequencies,
          homeIdx: rootIdx,
          curve,
          temperature,
          seed: Math.floor(Math.random() * 10000),
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to generate progression");
      }

      const data: ProgressionResult = await response.json();
      setResult(data);
      await playProgression(data.chords);
    } catch (err) {
      setError(String(err));
    } finally {
      setIsGenerating(false);
    }
  }, [frequencies, rootIdx, curve, temperature]);

  const playProgression = useCallback(async (chords: ChordInfo[]) => {
    if (playingRef.current) return;

    await resumeAudioIfNeeded();
    setIsPlaying(true);
    playingRef.current = true;

    const chordDuration = 1000;

    for (let i = 0; i < chords.length; i++) {
      if (!playingRef.current) break;
      setCurrentChordIndex(i);
      playFrequencies(chords[i].frequencies, { duration: chordDuration / 1000 });
      await new Promise(resolve => setTimeout(resolve, chordDuration));
    }

    setCurrentChordIndex(null);
    setIsPlaying(false);
    playingRef.current = false;
  }, []);

  const stopPlayback = useCallback(() => {
    playingRef.current = false;
    setIsPlaying(false);
    setCurrentChordIndex(null);
  }, []);

  const handlePlayAgain = useCallback(async () => {
    if (result) await playProgression(result.chords);
  }, [result, playProgression]);

  return (
    <div style={{ padding: 20, fontFamily: "monospace", maxWidth: 700 }}>
      <h1>Continuous Frequency Chord Progression Generator</h1>
      <p style={{ marginBottom: 12 }}>
        <a
          href="https://helenasu-blog.vercel.app/posts/music-proj0"
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: "#888", fontSize: 12, textDecoration: "underline" }}
        >
          Blog for details
        </a>
      </p>

      {/* Number of notes */}
      <div style={{ marginBottom: 20 }}>
        <label>
          Number of notes:{" "}
          <select value={numNotes} onChange={(e) => handleNumNotesChange(parseInt(e.target.value))}>
            {[5, 6, 7, 8, 9, 10, 11, 12].map(n => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </label>
        <button onClick={() => setFrequencies(generateEvenlySpacedFrequencies(numNotes))} style={{ marginLeft: 10 }}>
          Reset to Even
        </button>
      </div>

      {/* Frequency visualizer */}
      <div style={{ marginBottom: 20 }}>
        <h3>Frequencies</h3>
        <FrequencyEditor
          frequencies={frequencies}
          onChange={handleFrequenciesChange}
          rootIdx={rootIdx}
          onRootChange={handleRootChange}
        />
        {/* Manual inputs */}
        
      </div>

      {/* Analyze */}
      <div style={{ marginBottom: 20 }}>
        <button 
          onClick={analyzePalette} 
          disabled={isAnalyzing} 
          style={{ 
            cursor: "pointer",
            border: isAnalyzeHovered ? "1px solid white" : "none",
          }}
          onMouseEnter={() => setIsAnalyzeHovered(true)}
          onMouseLeave={() => setIsAnalyzeHovered(false)}
        >
          {isAnalyzing ? "Analyzing..." : "1. Analyze Chord Palette"}
        </button>
      </div>

      {/* Palette */}
      {palette && (
        <div style={{ marginBottom: 20 }}>
          <h3>Chord Palette ({palette.chords.length} triads)</h3>
          <div style={{ maxHeight: 150, overflowY: "auto", background: "#111", padding: 10, fontSize: 11 }}>
            <table style={{ width: "100%" }}>
              <thead>
                <tr style={{ color: "#888" }}>
                  <th style={{ textAlign: "left" }}>Frequencies (Hz)</th>
                  <th>Tension</th>
                </tr>
              </thead>
              <tbody>
                {palette.chords.slice(0, 20).map((chord, i) => (
                  <tr key={i} style={{ color: "#ccc" }}>
                    <td>{formatFreqs(chord.frequencies)}</td>
                    <td style={{ textAlign: "center" }}>{chord.tension.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tension curve */}
      {palette && (
        <div style={{ marginBottom: 20 }}>
          <h3>2. Tension Curve</h3>
          <TensionCurveEditor values={curve} onChange={setCurve} minValue={-0.15} maxValue={1} />
          <div style={{ fontSize: 11, color: "#888", marginTop: 5 }}>
            [{curve.map(v => v.toFixed(2)).join(", ")}]
          </div>
        </div>
      )}

      {/* Temperature */}
      {palette && (
        <div style={{ marginBottom: 20 }}>
          <label style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 10 }}>
            Temperature: {temperature.toFixed(3)}
            <input
              type="range"
              min="0"
              max="0.1"
              step="0.001"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              style={{
                width: 150,
                accentColor: "#fff",
                background: "#000",
              } as React.CSSProperties}
            />
          </label>
        </div>
      )}

      {/* Generate */}
      {palette && (
        <div style={{ marginBottom: 20 }}>
          <button 
            onClick={generateProgression} 
            disabled={isGenerating || isPlaying} 
            style={{ 
              cursor: "pointer",
              border: isGenerateHovered ? "1px solid white" : "none",
            }}
            onMouseEnter={() => setIsGenerateHovered(true)}
            onMouseLeave={() => setIsGenerateHovered(false)}
          >
            {isGenerating ? "Generating..." : "3. Generate & Play"}
          </button>
          {result && !isPlaying && (
            <button onClick={handlePlayAgain} style={{ marginLeft: 10 }}>Play Again</button>
          )}
          {isPlaying && (
            <button onClick={stopPlayback} style={{ marginLeft: 10 }}>Stop</button>
          )}
        </div>
      )}

      {/* Error */}
      {error && <div style={{ color: "red", marginBottom: 20 }}>{error}</div>}

      {/* Result */}
      {result && (
        <div>
          <h3>Progression</h3>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "flex-start" }}>
            {result.chords.map((chord, i) => (
              <div key={i}>
                <div
                  style={{
                    padding: 10,
                    background: currentChordIndex === i ? "#335" : "#222",
                    border: "1px solid #444",
                    minWidth: 120,
                  }}
                >
                  <div style={{ fontSize: 11, color: "#aaa" }}>
                    {chord.frequencies.map(f => f.toFixed(1)).join(", ")}
                  </div>
                  <div style={{ fontSize: 12, marginTop: 5 }}>T: {chord.tension.toFixed(3)}</div>
                </div>
                {/* Transition metrics */}
                {i < result.chords.length - 1 && (
                  (() => {
                    const nextChord = result.chords[i + 1];
                    const deltaTension = nextChord.tension - chord.tension;
                    const voiceLeadingCost = voiceLeadingCostFreq(chord.frequencies, nextChord.frequencies);
                    const softDenom = softVoiceLeadingDenominator(voiceLeadingCost);
                    const normalizedDeltaTension = deltaTension / softDenom;
                    
                    return (
                      <div
                        style={{
                          fontSize: 10,
                          color: "#666",
                          marginTop: 5,
                          paddingTop: 5,
                          borderTop: "1px solid #444",
                          textAlign: "center",
                        }}
                      >
                        <div>ΔT: {normalizedDeltaTension > 0 ? "+" : ""}{normalizedDeltaTension.toFixed(3)}</div>
                        <div>VL: {voiceLeadingCost.toFixed(2)}</div>
                      </div>
                    );
                  })()
                )}
              </div>
            ))}
          </div>
          <div style={{ fontSize: 11, color: "#888", marginTop: 10 }}>
            Deviation from path: {result.totalCost.toFixed(4)}
          </div>
        </div>
      )}
    </div>
  );
}
