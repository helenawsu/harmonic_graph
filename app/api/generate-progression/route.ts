import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();
        const { curve, temperature, seed } = body;

        // Validate inputs
        if (!Array.isArray(curve) || curve.length !== 4) {
            return NextResponse.json(
                { error: 'Curve must be an array of 4 numbers' },
                { status: 400 }
            );
        }

        // Build command arguments
        const args = [
            path.join(process.cwd(), 'scripts', 'chord_progression_optimizer.py'),
            '--curve', ...curve.map(String),
            '--temperature', String(temperature ?? 0.03),
            '--no-audio', // We'll play audio in the browser
        ];

        if (seed !== undefined) {
            args.push('--seed', String(seed));
        }

        // Run the Python script
        const result = await new Promise<{ stdout: string; stderr: string }>((resolve, reject) => {
            const pythonPath = path.join(process.cwd(), '.venv', 'bin', 'python');
            const proc = spawn(pythonPath, args, {
                cwd: process.cwd(),
            });

            let stdout = '';
            let stderr = '';

            proc.stdout.on('data', (data) => {
                stdout += data.toString();
            });

            proc.stderr.on('data', (data) => {
                stderr += data.toString();
            });

            proc.on('close', (code) => {
                if (code === 0) {
                    resolve({ stdout, stderr });
                } else {
                    reject(new Error(`Process exited with code ${code}: ${stderr}`));
                }
            });

            proc.on('error', (err) => {
                reject(err);
            });
        });

        // Parse the output to extract chord progression
        // Look for the Roman numerals and chord names in the output
        const output = result.stdout;

        // Extract chord progression using regex
        const romanMatch = output.match(/Progression \(Roman Numerals\):\s*(.+)/);
        const chordNamesMatch = output.match(/Progression \(Chord Names\):\s*(.+)/);

        // Extract detailed chord info from "Step" lines
        const stepMatches = output.matchAll(/Step \d+:\s+(\S+)\s+\((\S+)\s*\)\s*\|\s*Tension:\s*([\d.]+)/g);
        const chords: { roman: string; name: string; tension: number }[] = [];

        for (const match of stepMatches) {
            chords.push({
                roman: match[1],
                name: match[2],
                tension: parseFloat(match[3]),
            });
        }

        // If we didn't get 4 chords from steps, try parsing from the progression lines
        if (chords.length < 4 && romanMatch && chordNamesMatch) {
            const romans = romanMatch[1].split(' - ').map(s => s.trim());
            const names = chordNamesMatch[1].split(' - ').map(s => s.trim());

            for (let i = 0; i < Math.min(romans.length, names.length); i++) {
                if (!chords[i]) {
                    chords.push({
                        roman: romans[i],
                        name: names[i],
                        tension: 0,
                    });
                }
            }
        }

        // Extract total rate cost
        const costMatch = output.match(/Total rate cost:\s*([\d.]+)/);
        const totalCost = costMatch ? parseFloat(costMatch[1]) : null;

        // Parse chord names to get MIDI notes
        const NOTE_TO_SEMITONE: Record<string, number> = {
            'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
            'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11,
        };

        const chordsWithNotes = chords.map(chord => {
            // Parse chord name like "C_Maj" or "A_min"
            const [noteName, typeAbbrev] = chord.name.split('_');
            const rootSemitone = NOTE_TO_SEMITONE[noteName] ?? 0;
            const isMajor = typeAbbrev === 'Maj';

            // Build MIDI notes (base octave C4 = 48)
            const base = 48 + rootSemitone;
            const notes = isMajor
                ? [base, base + 4, base + 7]  // Major: root, M3, P5
                : [base, base + 3, base + 7]; // Minor: root, m3, P5

            return {
                ...chord,
                notes,
                type: isMajor ? 'Major' : 'minor',
            };
        });

        return NextResponse.json({
            chords: chordsWithNotes,
            totalCost,
            curve,
            temperature,
        });

    } catch (error) {
        console.error('Error generating progression:', error);
        return NextResponse.json(
            { error: String(error) },
            { status: 500 }
        );
    }
}
