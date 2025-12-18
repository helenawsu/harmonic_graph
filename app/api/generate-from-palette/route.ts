import { NextRequest, NextResponse } from 'next/server';
import { buildChordDatabase, bruteForceOptimize } from '@/app/lib/rqa';

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

        const validHomeIdx = (homeIdx ?? 0) % frequencies.length;
        const validTemperature = temperature ?? 0.01;

        // Set seed if provided (JavaScript doesn't have built-in seeded random, but we can ignore for now)
        // For production, consider using a seeded PRNG library

        // Build chord database with RQA
        const chords = buildChordDatabase(frequencies, validHomeIdx);

        if (chords.length < 4) {
            return NextResponse.json(
                { error: `Not enough chords generated: ${chords.length}` },
                { status: 400 }
            );
        }

        // Optimize
        const { bestPath, bestScore } = bruteForceOptimize(
            chords,
            curve,
            validTemperature
        );

        if (!bestPath) {
            return NextResponse.json(
                { error: 'Optimization failed' },
                { status: 500 }
            );
        }

        // Format output
        const output = {
            chords: bestPath.map(chord => ({
                frequencies: chord.frequencies,
                tension: chord.tension,
            })),
            totalCost: bestScore,
            curve: curve,
            temperature: validTemperature,
        };

        return NextResponse.json(output);

    } catch (error) {
        console.error('Error generating progression:', error);
        return NextResponse.json(
            { error: String(error) },
            { status: 500 }
        );
    }
}
