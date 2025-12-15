"use client";
import React, { useState, useCallback, useRef } from "react";
import TensionCurveEditor from "./TensionCurveEditor";
import { playNotesFromPitchClasses, resumeAudioIfNeeded } from "../tonnetz/audio";

interface Chord {
  roman: string;
  name: string;
  tension: number;
  notes: number[];
  type: string;
}

interface ProgressionResult {
  chords: Chord[];
  totalCost: number | null;
  curve: number[];
  temperature: number;
}

export default function ChordProgressionGenerator() {
  const [curve, setCurve] = useState<number[]>([0.0, 0.2, 0.65, -0.1]);
  const [temperature, setTemperature] = useState(0.03);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [result, setResult] = useState<ProgressionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentChordIndex, setCurrentChordIndex] = useState<number | null>(null);
  const playingRef = useRef(false);

  const generateProgression = useCallback(async () => {
    setIsGenerating(true);
    setError(null);

    try {
      const response = await fetch("/api/generate-progression", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
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
  }, [curve, temperature]);

  const playProgression = useCallback(async (chords: Chord[]) => {
    if (playingRef.current) return;

    await resumeAudioIfNeeded();
    setIsPlaying(true);
    playingRef.current = true;

    const chordDuration = 1000;

    for (let i = 0; i < chords.length; i++) {
      if (!playingRef.current) break;
      setCurrentChordIndex(i);
      const chord = chords[i];
      const pitchClasses = chord.notes.map(n => n % 12);
      playNotesFromPitchClasses(pitchClasses, { duration: chordDuration / 1000 });
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
    if (result) {
      await playProgression(result.chords);
    }
  }, [result, playProgression]);

  return (
    <div style={{ padding: 20, fontFamily: "sans-serif" }}>
      <h1>Chord Progression Generator</h1>

      <div style={{ marginBottom: 20 }}>
        <h3>Tension Curve</h3>
        <TensionCurveEditor
          values={curve}
          onChange={setCurve}
          minValue={-0.15}
          maxValue={1}
        />
        <div style={{ fontSize: 12, color: "#666", marginTop: 5 }}>
          Values: [{curve.map(v => v.toFixed(2)).join(", ")}]
        </div>
      </div>

      <div style={{ marginBottom: 20 }}>
        <h3>Temperature: {temperature.toFixed(3)}</h3>
        <input
          type="range"
          min="0"
          max="0.2"
          step="0.001"
          value={temperature}
          onChange={(e) => setTemperature(parseFloat(e.target.value))}
          style={{ width: 300 }}
        />
      </div>

      <div style={{ marginBottom: 20 }}>
        <button
          onClick={generateProgression}
          disabled={isGenerating || isPlaying}
          style={{ padding: "10px 20px", marginRight: 10 }}
        >
          {isGenerating ? "Generating..." : "Generate & Play"}
        </button>

        {result && !isPlaying && (
          <button onClick={handlePlayAgain} style={{ padding: "10px 20px", marginRight: 10 }}>
            Play Again
          </button>
        )}

        {isPlaying && (
          <button onClick={stopPlayback} style={{ padding: "10px 20px" }}>
            Stop
          </button>
        )}
      </div>

      {error && (
        <div style={{ color: "red", marginBottom: 20 }}>{error}</div>
      )}

      {result && (
        <div>
          <h3>Generated Progression</h3>
          <div style={{ display: "flex", gap: 20, marginBottom: 20 }}>
            {result.chords.map((chord, index) => (
              <div
                key={index}
                style={{
                  padding: 15,
                  border: "1px solid #ccc",
                  background: currentChordIndex === index ? "#ddf" : "#fff",
                  textAlign: "center",
                }}
              >
                <div style={{ fontSize: 24, fontWeight: "bold" }}>{chord.roman}</div>
                <div style={{ fontSize: 14, color: "#666" }}>{chord.name}</div>
                <div style={{ fontSize: 12, color: "#999" }}>T: {chord.tension.toFixed(2)}</div>
              </div>
            ))}
          </div>
          <div style={{ fontSize: 12, color: "#666" }}>
            Cost: {result.totalCost?.toFixed(3) ?? "N/A"}
          </div>
        </div>
      )}
    </div>
  );
}
