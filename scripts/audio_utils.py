#!/usr/bin/env python3
"""
Audio Utility Functions

Functions for generating and saving audio from chord progressions.
"""
import os
import math
import numpy as np
import wave
import subprocess
from typing import List, Dict


def midi_to_freq(midi_note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def generate_chord_audio(
    notes: List[int],
    duration: float = 1.0,
    sample_rate: int = 44100,
    harmonics: int = 4,
    fade_duration: float = 0.05,
) -> np.ndarray:
    """
    Generate audio for a chord.
    
    Args:
        notes: List of MIDI note numbers
        duration: Duration in seconds
        sample_rate: Audio sample rate
        harmonics: Number of harmonics per note (for richer sound)
        fade_duration: Fade in/out duration in seconds
    
    Returns:
        Audio samples as numpy array
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = np.zeros_like(t)
    
    for midi_note in notes:
        freq = midi_to_freq(midi_note)
        # Add harmonics for richer sound
        for h in range(1, harmonics + 1):
            amplitude = 1.0 / h  # Decreasing amplitude for higher harmonics
            audio += amplitude * np.sin(2.0 * math.pi * freq * h * t)
    
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    # Apply fade in/out to avoid clicks
    fade_samples = int(fade_duration * sample_rate)
    if fade_samples > 0 and fade_samples < len(audio) // 2:
        # Fade in
        fade_in = np.linspace(0, 1, fade_samples)
        audio[:fade_samples] *= fade_in
        # Fade out
        fade_out = np.linspace(1, 0, fade_samples)
        audio[-fade_samples:] *= fade_out
    
    return audio


def generate_progression_audio(
    path: List[Dict],
    chord_duration: float = 1.0,
    sample_rate: int = 44100,
) -> np.ndarray:
    """
    Generate audio for the entire chord progression.
    
    Args:
        path: List of chord dictionaries (must have 'notes' key)
        chord_duration: Duration of each chord in seconds
        sample_rate: Audio sample rate
    
    Returns:
        Audio samples as numpy array
    """
    audio_segments = []
    
    for chord in path:
        segment = generate_chord_audio(
            notes=chord["notes"],
            duration=chord_duration,
            sample_rate=sample_rate,
        )
        audio_segments.append(segment)
    
    # Concatenate all segments
    full_audio = np.concatenate(audio_segments)
    
    # Final normalization
    if np.max(np.abs(full_audio)) > 0:
        full_audio = full_audio / np.max(np.abs(full_audio)) * 0.9
    
    return full_audio


def save_wav(audio: np.ndarray, filepath: str, sample_rate: int = 44100):
    """Save audio as WAV file."""
    # Convert to 16-bit PCM
    audio_int = np.int16(audio * 32767)
    
    with wave.open(filepath, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes = 16 bits
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int.tobytes())


def save_mp3(audio: np.ndarray, filepath: str, sample_rate: int = 44100):
    """
    Save audio as MP3 file.
    Uses ffmpeg if available, otherwise falls back to WAV.
    """
    # First save as WAV
    wav_path = filepath.replace('.mp3', '.wav')
    save_wav(audio, wav_path, sample_rate)
    
    # Try to convert to MP3 using ffmpeg
    try:
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', wav_path, '-acodec', 'libmp3lame', '-q:a', '2', filepath],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            # Remove the temporary WAV file
            os.remove(wav_path)
            print(f"Audio saved to: {filepath}")
            return
        else:
            print(f"ffmpeg conversion failed, keeping WAV file: {wav_path}")
    except FileNotFoundError:
        print(f"ffmpeg not found, saving as WAV instead: {wav_path}")
    except Exception as e:
        print(f"Error converting to MP3: {e}")
        print(f"Audio saved as WAV: {wav_path}")


def save_audio(audio: np.ndarray, filepath: str, sample_rate: int = 44100):
    """
    Save audio to file. Automatically chooses format based on extension.
    
    Args:
        audio: Audio samples as numpy array
        filepath: Output file path (.wav or .mp3)
        sample_rate: Audio sample rate
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    
    if filepath.endswith('.mp3'):
        save_mp3(audio, filepath, sample_rate)
    else:
        save_wav(audio, filepath, sample_rate)
        print(f"Audio saved to: {filepath}")
