#!/usr/bin/env python3
"""
Generate test audio fixtures for AEC testing.

Creates:
- farend.wav: Reference signal (system audio being played)
- nearend_echo_only.wav: Pure echo scenario (mic picks up only the speaker)
- nearend_doubletalk.wav: Echo + near-end speech (mic picks up speaker + user talking)

All files are 16kHz mono float32 WAV format.
"""

import subprocess
import struct
import sys
from pathlib import Path

SAMPLE_RATE = 16000
ECHO_DELAY_MS = 50  # Typical room echo delay
ECHO_ATTENUATION = 0.4  # Echo is quieter than source


def read_wav_float32(path: Path) -> list[float]:
    """Read float32 WAV file, return samples."""
    with open(path, "rb") as f:
        data = f.read()

    # Find data chunk
    offset = 12
    while offset < len(data) - 8:
        chunk_id = data[offset:offset + 4].decode("ascii", errors="ignore")
        chunk_size = struct.unpack("<I", data[offset + 4:offset + 8])[0]
        if chunk_id == "data":
            offset += 8
            break
        offset += 8 + chunk_size

    # Read float samples
    samples = []
    while offset + 4 <= len(data):
        sample = struct.unpack("<f", data[offset:offset + 4])[0]
        samples.append(sample)
        offset += 4

    return samples


def write_wav_float32(path: Path, samples: list[float], sample_rate: int = 16000):
    """Write float32 WAV file."""
    num_samples = len(samples)
    data_size = num_samples * 4
    file_size = data_size + 36

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", file_size))
        f.write(b"WAVE")

        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # chunk size
        f.write(struct.pack("<H", 3))   # audio format (3 = IEEE float)
        f.write(struct.pack("<H", 1))   # num channels
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * 4))  # byte rate
        f.write(struct.pack("<H", 4))   # block align
        f.write(struct.pack("<H", 32))  # bits per sample

        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        for sample in samples:
            f.write(struct.pack("<f", sample))


def generate_tts(text: str, voice: str, output_path: Path):
    """Generate TTS audio using macOS say command."""
    aiff_path = output_path.with_suffix(".aiff")

    # Generate with say
    subprocess.run(
        ["say", "-v", voice, "-o", str(aiff_path), text],
        check=True
    )

    # Convert to 16kHz mono float32 WAV
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(aiff_path),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_f32le",
         str(output_path)],
        check=True,
        capture_output=True
    )

    # Clean up AIFF
    aiff_path.unlink()


def create_echo(farend: list[float], delay_ms: float, attenuation: float) -> list[float]:
    """Create echo signal from far-end."""
    delay_samples = int(delay_ms * SAMPLE_RATE / 1000)
    echo = [0.0] * len(farend)

    for i in range(delay_samples, len(farend)):
        echo[i] = farend[i - delay_samples] * attenuation

    return echo


def mix_signals(sig1: list[float], sig2: list[float]) -> list[float]:
    """Mix two signals, padding shorter one with zeros."""
    length = max(len(sig1), len(sig2))
    result = [0.0] * length

    for i in range(len(sig1)):
        result[i] += sig1[i]
    for i in range(len(sig2)):
        result[i] += sig2[i]

    # Normalize if clipping
    max_val = max(abs(s) for s in result)
    if max_val > 0.95:
        result = [s / max_val * 0.95 for s in result]

    return result


def main():
    script_dir = Path(__file__).parent
    fixtures_dir = script_dir.parent / "Tests" / "DTLNAecCoreMLTests" / "Fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path("/tmp/aec_fixtures")
    tmp_dir.mkdir(exist_ok=True)

    print("Generating TTS audio...")

    # Far-end: what's playing through speakers
    farend_text = "The quick brown fox jumps over the lazy dog. This is a test of acoustic echo cancellation."
    farend_tmp = tmp_dir / "farend.wav"
    generate_tts(farend_text, "Samantha", farend_tmp)
    farend_samples = read_wav_float32(farend_tmp)
    print(f"  Far-end: {len(farend_samples)} samples ({len(farend_samples)/SAMPLE_RATE:.2f}s)")

    # Near-end speech: user talking (different voice)
    nearend_text = "One two three four five. Hello, can you hear me?"
    nearend_tmp = tmp_dir / "nearend_speech.wav"
    generate_tts(nearend_text, "Daniel", nearend_tmp)
    nearend_speech = read_wav_float32(nearend_tmp)
    print(f"  Near-end speech: {len(nearend_speech)} samples ({len(nearend_speech)/SAMPLE_RATE:.2f}s)")

    print("\nCreating test scenarios...")

    # Create echo from far-end
    echo = create_echo(farend_samples, ECHO_DELAY_MS, ECHO_ATTENUATION)

    # Scenario 1: Pure echo (no near-end talker)
    # This is the simplest case - should show maximum echo reduction
    nearend_echo_only = echo.copy()

    # Scenario 2: Double-talk (echo + near-end speech)
    # Pad near-end speech to start after 1 second
    speech_offset = SAMPLE_RATE  # 1 second
    padded_speech = [0.0] * speech_offset + nearend_speech
    nearend_doubletalk = mix_signals(echo, padded_speech)

    # Ensure all files are same length as far-end for easy testing
    max_len = len(farend_samples)
    nearend_echo_only = (nearend_echo_only + [0.0] * max_len)[:max_len]
    nearend_doubletalk = (nearend_doubletalk + [0.0] * max_len)[:max_len]

    print(f"  Echo delay: {ECHO_DELAY_MS}ms, attenuation: {ECHO_ATTENUATION}")

    # Write output files
    print("\nWriting fixtures...")

    write_wav_float32(fixtures_dir / "farend.wav", farend_samples)
    print(f"  {fixtures_dir / 'farend.wav'}")

    write_wav_float32(fixtures_dir / "nearend_echo_only.wav", nearend_echo_only)
    print(f"  {fixtures_dir / 'nearend_echo_only.wav'}")

    write_wav_float32(fixtures_dir / "nearend_doubletalk.wav", nearend_doubletalk)
    print(f"  {fixtures_dir / 'nearend_doubletalk.wav'}")

    # Clean up
    farend_tmp.unlink()
    nearend_tmp.unlink()
    tmp_dir.rmdir()

    print("\nDone! Test fixtures created.")
    print(f"\nTo test manually:")
    print(f"  swift run FileProcessor \\")
    print(f"    --mic {fixtures_dir}/nearend_echo_only.wav \\")
    print(f"    --ref {fixtures_dir}/farend.wav \\")
    print(f"    --output /tmp/cleaned.wav")


if __name__ == "__main__":
    main()
