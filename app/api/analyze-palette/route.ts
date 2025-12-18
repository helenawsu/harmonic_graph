import { NextRequest, NextResponse } from 'next/server';
import { buildChordDatabase } from '@/app/lib/rqa';

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();
        const { frequencies, homeIdx } = body;

        // Validate inputs
        if (!Array.isArray(frequencies) || frequencies.length < 5 || frequencies.length > 12) {
            return NextResponse.json(
                { error: 'Frequencies must be an array of 5-12 numbers' },
                { status: 400 }
            );
        }

        const validHomeIdx = (homeIdx ?? 0) % frequencies.length;

        // Build chord database with RQA
        const chords = buildChordDatabase(frequencies, validHomeIdx);

        // Format output
        const output = {
            home_freq: frequencies[validHomeIdx],
            frequencies: frequencies,
            chords: chords.map(chord => ({
                frequencies: chord.frequencies,
                tension: chord.tension,
            }))
        };

        return NextResponse.json(output);

    } catch (error) {
        console.error('Error analyzing palette:', error);
        return NextResponse.json(
            { error: String(error) },
            { status: 500 }
        );
    }
}
