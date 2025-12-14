"use client";
import React, { useState, useCallback } from "react";
import { playNotesFromPitchClasses, resumeAudioIfNeeded } from "./audio";

// Chord progression definitions (semitones from root for major chords)
interface ChordDef {
    name: string;
    root: number; // pitch class of root (0 = C)
    notes: number[]; // all pitch classes in the chord
}

// Helper to build a major triad from a root pitch class
function majorTriad(root: number): number[] {
    return [root % 12, (root + 4) % 12, (root + 7) % 12];
}

// Helper to build a minor triad from a root pitch class
function minorTriad(root: number): number[] {
    return [root % 12, (root + 3) % 12, (root + 7) % 12];
}

// Chord progressions available (assuming C as root/I)
// Scale degrees: I=C(0), ii=Dm(2), iii=Em(4), IV=F(5), V=G(7), vi=Am(9), vii°=Bdim(11)
const CHORD_PROGRESSIONS: { name: string; chords: ChordDef[] }[] = [
    {
        name: "I - IV - V - I",
        chords: [
            { name: "I (C)", root: 0, notes: majorTriad(0) },
            { name: "IV (F)", root: 5, notes: majorTriad(5) },
            { name: "V (G)", root: 7, notes: majorTriad(7) },
            { name: "I (C)", root: 0, notes: majorTriad(0) },
        ],
    },
    {
        name: "I - IV - vi - V",
        chords: [
            { name: "I (C)", root: 0, notes: majorTriad(0) },
            { name: "IV (F)", root: 5, notes: majorTriad(5) },
            { name: "vi (Am)", root: 9, notes: minorTriad(9) },
            { name: "V (G)", root: 7, notes: majorTriad(7) },
        ],
    },
    {
        name: "I - IV - I - V",
        chords: [
            { name: "I (C)", root: 0, notes: majorTriad(0) },
            { name: "IV (F)", root: 5, notes: majorTriad(5) },
            { name: "I (C)", root: 0, notes: majorTriad(0) },
            { name: "V (G)", root: 7, notes: majorTriad(7) },
        ],
    },
    {
        name: "I - V - IV - V",
        chords: [
            { name: "I (C)", root: 0, notes: majorTriad(0) },
            { name: "V (G)", root: 7, notes: majorTriad(7) },
            { name: "IV (F)", root: 5, notes: majorTriad(5) },
            { name: "V (G)", root: 7, notes: majorTriad(7) },
        ],
    },
    {
        name: "I - vi - IV - V",
        chords: [
            { name: "I (C)", root: 0, notes: majorTriad(0) },
            { name: "vi (Am)", root: 9, notes: minorTriad(9) },
            { name: "IV (F)", root: 5, notes: majorTriad(5) },
            { name: "V (G)", root: 7, notes: majorTriad(7) },
        ],
    },
    {
        name: "ii - V - I",
        chords: [
            { name: "ii (Dm)", root: 2, notes: minorTriad(2) },
            { name: "V (G)", root: 7, notes: majorTriad(7) },
            { name: "I (C)", root: 0, notes: majorTriad(0) },
        ],
    },
];

// Helper to get chord key (sorted pitch classes as string)
export function getChordKey(pcs: number[]): string {
    return [...pcs].sort((a, b) => a - b).join(',');
}

interface ChordProgressionPanelProps {
    onChordChange: (chordKey: string | null) => void;
}

export default function ChordProgressionPanel({ onChordChange }: ChordProgressionPanelProps) {
    const [selectedProgression, setSelectedProgression] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentChordIndex, setCurrentChordIndex] = useState<number | null>(null);

    // Play chord progression
    const playProgression = useCallback(async () => {
        if (isPlaying) return;

        await resumeAudioIfNeeded();
        setIsPlaying(true);

        const progression = CHORD_PROGRESSIONS[selectedProgression];
        const chordDuration = 1000; // 1 second per chord

        for (let i = 0; i < progression.chords.length; i++) {
            const chord = progression.chords[i];
            setCurrentChordIndex(i);
            onChordChange(getChordKey(chord.notes));

            // Play the chord
            playNotesFromPitchClasses(chord.notes, { duration: chordDuration / 1000 });

            // Wait for chord duration
            await new Promise(resolve => setTimeout(resolve, chordDuration));
        }

        // Reset state after progression finishes
        setCurrentChordIndex(null);
        onChordChange(null);
        setIsPlaying(false);
    }, [isPlaying, selectedProgression, onChordChange]);

    const currentProgression = CHORD_PROGRESSIONS[selectedProgression];

    return (
        <div style={{
            position: 'fixed',
            right: 12,
            top: 12,
            background: 'rgba(0,0,0,0.75)',
            color: 'white',
            padding: '12px 16px',
            borderRadius: 10,
            fontSize: 13,
            zIndex: 1000,
            minWidth: 200,
            backdropFilter: 'blur(8px)',
        }}>
            <div style={{ fontWeight: 600, marginBottom: 10, fontSize: 14 }}>
                🎵 Chord Progression
            </div>

            <div style={{ marginBottom: 12 }}>
                <label style={{ fontSize: 11, color: 'rgba(255,255,255,0.7)', display: 'block', marginBottom: 4 }}>
                    Root: C (fixed)
                </label>
                <select
                    value={selectedProgression}
                    onChange={(e) => setSelectedProgression(Number(e.target.value))}
                    disabled={isPlaying}
                    style={{
                        width: '100%',
                        padding: '6px 8px',
                        borderRadius: 6,
                        border: 'none',
                        background: 'rgba(255,255,255,0.15)',
                        color: 'white',
                        fontSize: 13,
                        cursor: isPlaying ? 'not-allowed' : 'pointer',
                    }}
                >
                    {CHORD_PROGRESSIONS.map((prog, idx) => (
                        <option key={idx} value={idx} style={{ background: '#333' }}>
                            {prog.name}
                        </option>
                    ))}
                </select>
            </div>

            {/* Chord indicators */}
            <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
                {currentProgression.chords.map((chord, idx) => (
                    <div
                        key={idx}
                        style={{
                            flex: 1,
                            padding: '8px 4px',
                            borderRadius: 6,
                            background: currentChordIndex === idx
                                ? 'rgba(99, 102, 241, 0.8)'
                                : 'rgba(255,255,255,0.1)',
                            textAlign: 'center',
                            fontSize: 11,
                            fontWeight: currentChordIndex === idx ? 600 : 400,
                            transition: 'all 0.15s ease',
                            transform: currentChordIndex === idx ? 'scale(1.05)' : 'scale(1)',
                        }}
                    >
                        {chord.name}
                    </div>
                ))}
            </div>

            <button
                onClick={playProgression}
                disabled={isPlaying}
                style={{
                    width: '100%',
                    padding: '10px 16px',
                    borderRadius: 8,
                    border: 'none',
                    background: isPlaying
                        ? 'rgba(99, 102, 241, 0.4)'
                        : 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                    color: 'white',
                    fontSize: 14,
                    fontWeight: 600,
                    cursor: isPlaying ? 'not-allowed' : 'pointer',
                    transition: 'all 0.2s ease',
                }}
            >
                {isPlaying ? '▶ Playing...' : '▶ Play'}
            </button>
        </div>
    );
}
