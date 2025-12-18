import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();
        const { frequencies, homeIdx, curve, temperature, seed } = body;

        // Validate inputs
        if (!Array.isArray(frequencies) || frequencies.length < 5 || frequencies.length > 12) {
            return NextResponse.json(
                { error: 'Frequencies must be an array of 5-12 numbers' },
                { status: 400 }
            );
        }

        if (!Array.isArray(curve) || curve.length !== 4) {
            return NextResponse.json(
                { error: 'Curve must be an array of 4 numbers' },
                { status: 400 }
            );
        }

        const pythonPath = path.join(process.cwd(), '.venv', 'bin', 'python');
        const scriptPath = path.join(process.cwd(), 'scripts', 'generate_progression_api.py');

        const args = [
            scriptPath,
            '--frequencies', ...frequencies.map(String),
            '--home-idx', String(homeIdx ?? 0),
            '--curve', ...curve.map(String),
            '--temperature', String(temperature ?? 0.01),
        ];

        if (seed !== undefined) {
            args.push('--seed', String(seed));
        }

        const result = await new Promise<{ stdout: string; stderr: string }>((resolve, reject) => {
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

        // Parse JSON output
        const progression = JSON.parse(result.stdout);

        return NextResponse.json(progression);

    } catch (error) {
        console.error('Error generating progression:', error);
        return NextResponse.json(
            { error: String(error) },
            { status: 500 }
        );
    }
}
