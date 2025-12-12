// Light-weight WebAudio helper to play simple chords (sine + ADSR)
let audioCtx: AudioContext | undefined;

function getAudioCtx(): AudioContext {
    if (!audioCtx) {
        const win = window as unknown as { webkitAudioContext?: typeof AudioContext };
        const Ctor = (window.AudioContext ?? win.webkitAudioContext) as unknown as { new(): AudioContext };
        audioCtx = new Ctor();
    }
    return audioCtx!;
}

export function midiToFreq(m: number) {
    return 440 * Math.pow(2, (m - 69) / 12);
}

export function playChordFromPitchClass(pitchClass: number, opts?: { duration?: number; detune?: number }) {
    const dur = opts?.duration ?? 1.6;
    const ctx = getAudioCtx();
    const now = ctx.currentTime;

    // pick a comfortable octave for the root (C4 = midi 60)
    const rootMidi = 60 + ((pitchClass % 12 + 12) % 12);
    const thirdMidi = rootMidi + 4; // major third
    const fifthMidi = rootMidi + 7; // perfect fifth

    const nodes: { osc: OscillatorNode; gain: GainNode }[] = [];
    const freqs = [rootMidi, thirdMidi, fifthMidi].map(midiToFreq);

    for (let i = 0; i < freqs.length; i++) {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.value = freqs[i];
        if (opts?.detune) osc.detune.value = opts.detune;
        gain.gain.setValueAtTime(0.0001, now);
        // simple ADSR
        gain.gain.exponentialRampToValueAtTime(0.12 / (i + 1) + 0.02, now + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.01, now + dur - 0.05);
        osc.connect(gain).connect(ctx.destination);
        osc.start(now);
        osc.stop(now + dur + 0.1);
        nodes.push({ osc, gain });
    }
}

// Play arbitrary pitch classes (array) with the same ADSR as chords
export function playNotesFromPitchClasses(pcs: number[], opts?: { duration?: number; detune?: number }) {
    const dur = opts?.duration ?? 1.6;
    const ctx = getAudioCtx();
    const now = ctx.currentTime;

    const nodes: { osc: OscillatorNode; gain: GainNode }[] = [];

    const freqs = pcs.map(pc => {
        const midi = 60 + ((pc % 12 + 12) % 12);
        return midiToFreq(midi);
    });

    for (let i = 0; i < freqs.length; i++) {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.value = freqs[i];
        if (opts?.detune) osc.detune.value = opts.detune;
        gain.gain.setValueAtTime(0.0001, now);
        // simple ADSR
        gain.gain.exponentialRampToValueAtTime(0.12 / (i + 1) + 0.02, now + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.01, now + dur - 0.05);
        osc.connect(gain).connect(ctx.destination);
        osc.start(now);
        osc.stop(now + dur + 0.1);
        nodes.push({ osc, gain });
    }
}

export function playIntervalFromPitchClasses(pc1: number, pc2: number, opts?: { duration?: number; detune?: number }) {
    return playNotesFromPitchClasses([pc1, pc2], opts);
}

export function playNoteFromPitchClass(pc: number, opts?: { duration?: number; detune?: number }) {
    return playNotesFromPitchClasses([pc], opts);
}

export function resumeAudioIfNeeded() {
    const ctx = getAudioCtx();
    if (ctx.state === 'suspended') return ctx.resume();
    return Promise.resolve();
}
