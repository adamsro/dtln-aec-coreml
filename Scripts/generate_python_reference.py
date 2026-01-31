#!/usr/bin/env python3
"""
Generate Python reference output for DTLN-aec using the 128-unit TFLite models.

This script processes audio through the original 128-unit TFLite models to create
a reference output that can be compared against the CoreML implementation.

Model source: https://github.com/breizhn/DTLN-aec
Model files: dtln_aec_128_1.tflite, dtln_aec_128_2.tflite
"""
import struct
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
TFLITE_DIR = PROJECT_ROOT / "Resources" / "DTLN_AEC"
SAMPLES_DIR = PROJECT_ROOT / "Samples" / "aec_challenge"

# Model parameters (128-unit model)
BLOCK_LEN = 512
BLOCK_SHIFT = 128
FFT_BINS = 257
NUM_LAYERS = 2
NUM_UNITS = 128


def read_wav(path: Path) -> np.ndarray:
    """Read WAV file, supporting both Int16 and Float32 formats."""
    with open(path, "rb") as f:
        data = f.read()

    audio_format = None
    bits_per_sample = None
    data_offset = 0
    data_size = 0

    offset = 12
    while offset < len(data) - 8:
        chunk_id = data[offset:offset + 4].decode("ascii", errors="ignore")
        chunk_size = struct.unpack("<I", data[offset + 4:offset + 8])[0]

        if chunk_id == "fmt ":
            audio_format = struct.unpack("<H", data[offset + 8:offset + 10])[0]
            bits_per_sample = struct.unpack("<H", data[offset + 22:offset + 24])[0]
        elif chunk_id == "data":
            data_offset = offset + 8
            data_size = chunk_size
            break
        offset += 8 + chunk_size

    if data_offset == 0:
        raise ValueError("No data chunk found")

    if audio_format == 1 and bits_per_sample == 16:
        # PCM Int16
        samples = np.frombuffer(
            data[data_offset:data_offset + data_size],
            dtype=np.int16
        ).astype(np.float32) / 32768.0
    elif audio_format == 3 and bits_per_sample == 32:
        # IEEE Float32
        samples = np.frombuffer(
            data[data_offset:data_offset + data_size],
            dtype=np.float32
        )
    else:
        raise ValueError(f"Unsupported format: {audio_format}, {bits_per_sample}-bit")

    return samples


def write_wav_float32(path: Path, samples: np.ndarray, sample_rate: int = 16000):
    """Write float32 WAV file."""
    samples = samples.astype(np.float32)
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
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 3))  # IEEE float
        f.write(struct.pack("<H", 1))  # mono
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * 4))
        f.write(struct.pack("<H", 4))  # block align
        f.write(struct.pack("<H", 32))  # bits per sample

        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(samples.tobytes())


def compute_rms(samples: np.ndarray) -> float:
    """Compute RMS of samples."""
    return np.sqrt(np.mean(samples ** 2))


def compute_reduction_db(input_rms: float, output_rms: float) -> float:
    """Compute reduction in dB."""
    if output_rms <= 0:
        return float('inf')
    return 20 * np.log10(input_rms / output_rms)


class DTLNAecProcessor:
    """Process audio using DTLN-aec TFLite models."""

    def __init__(self, model_dir: Path, num_units: int = 128):
        import tensorflow as tf

        self.num_units = num_units
        self.num_layers = NUM_LAYERS
        self.block_len = BLOCK_LEN
        self.block_shift = BLOCK_SHIFT
        self.fft_bins = FFT_BINS

        # Load TFLite models
        model1_path = model_dir / f"dtln_aec_{num_units}_1.tflite"
        model2_path = model_dir / f"dtln_aec_{num_units}_2.tflite"

        print(f"Loading models from:")
        print(f"  Part 1: {model1_path}")
        print(f"  Part 2: {model2_path}")

        self.interpreter1 = tf.lite.Interpreter(model_path=str(model1_path))
        self.interpreter1.allocate_tensors()

        self.interpreter2 = tf.lite.Interpreter(model_path=str(model2_path))
        self.interpreter2.allocate_tensors()

        # Get input/output details
        self._setup_io()

        # Initialize states
        self.reset()

    def _setup_io(self):
        """Setup input/output indices for both models."""
        # Model 1 inputs
        input_details = self.interpreter1.get_input_details()
        for d in input_details:
            name = d['name']
            shape = tuple(d['shape'])
            if shape == (1, 1, self.fft_bins) and 'input_3' in name:
                self.m1_mic_idx = d['index']
            elif shape == (1, 1, self.fft_bins) and 'input_4' in name:
                self.m1_lpb_idx = d['index']
            elif shape == (1, self.num_layers, self.num_units, 2):
                self.m1_states_idx = d['index']

        # Model 1 outputs
        output_details = self.interpreter1.get_output_details()
        for d in output_details:
            if 'Identity' in d['name'] and tuple(d['shape']) == (1, 1, self.fft_bins):
                self.m1_mask_idx = d['index']
            elif tuple(d['shape']) == (1, self.num_layers, self.num_units, 2):
                self.m1_states_out_idx = d['index']

        # Model 2 inputs
        input_details = self.interpreter2.get_input_details()
        for d in input_details:
            name = d['name']
            shape = tuple(d['shape'])
            if shape == (1, 1, self.block_len) and 'input_6' in name:
                self.m2_est_idx = d['index']
            elif shape == (1, 1, self.block_len) and 'input_7' in name:
                self.m2_lpb_idx = d['index']
            elif shape == (1, self.num_layers, self.num_units, 2):
                self.m2_states_idx = d['index']

        # Model 2 outputs
        output_details = self.interpreter2.get_output_details()
        for d in output_details:
            if 'Identity' in d['name'] and tuple(d['shape']) == (1, 1, self.block_len):
                self.m2_out_idx = d['index']
            elif tuple(d['shape']) == (1, self.num_layers, self.num_units, 2):
                self.m2_states_out_idx = d['index']

    def reset(self):
        """Reset LSTM states and buffers."""
        self.states1 = np.zeros(
            (1, self.num_layers, self.num_units, 2), dtype=np.float32
        )
        self.states2 = np.zeros(
            (1, self.num_layers, self.num_units, 2), dtype=np.float32
        )
        self.in_buffer = np.zeros(self.block_len, dtype=np.float32)
        self.lpb_buffer = np.zeros(self.block_len, dtype=np.float32)
        self.out_buffer = np.zeros(self.block_len, dtype=np.float32)

    def process(self, mic: np.ndarray, lpb: np.ndarray) -> np.ndarray:
        """
        Process audio streams through DTLN-aec.

        Args:
            mic: Microphone signal (with echo)
            lpb: Far-end/loopback reference signal

        Returns:
            Echo-cancelled output signal
        """
        # Ensure same length
        length = min(len(mic), len(lpb))
        mic = mic[:length]
        lpb = lpb[:length]

        output = np.zeros(length, dtype=np.float32)
        num_blocks = (length - self.block_len) // self.block_shift + 1

        for block_idx in range(num_blocks):
            start = block_idx * self.block_shift
            end = start + self.block_shift

            # Shift buffers
            self.in_buffer[:-self.block_shift] = self.in_buffer[self.block_shift:]
            self.in_buffer[-self.block_shift:] = mic[start:end]

            self.lpb_buffer[:-self.block_shift] = self.lpb_buffer[self.block_shift:]
            self.lpb_buffer[-self.block_shift:] = lpb[start:end]

            # FFT
            mic_fft = np.fft.rfft(self.in_buffer)
            lpb_fft = np.fft.rfft(self.lpb_buffer)

            mic_mag = np.abs(mic_fft).astype(np.float32)
            mic_phase = np.angle(mic_fft)
            lpb_mag = np.abs(lpb_fft).astype(np.float32)

            # Part 1: Frequency domain processing
            self.interpreter1.set_tensor(
                self.m1_mic_idx,
                mic_mag.reshape(1, 1, self.fft_bins)
            )
            self.interpreter1.set_tensor(
                self.m1_lpb_idx,
                lpb_mag.reshape(1, 1, self.fft_bins)
            )
            self.interpreter1.set_tensor(self.m1_states_idx, self.states1)
            self.interpreter1.invoke()

            mask1 = self.interpreter1.get_tensor(self.m1_mask_idx).flatten()
            self.states1 = self.interpreter1.get_tensor(self.m1_states_out_idx)

            # Apply mask and IFFT
            estimated_fft = mic_fft * mask1
            estimated = np.fft.irfft(estimated_fft).astype(np.float32)

            # Part 2: Time domain processing
            self.interpreter2.set_tensor(
                self.m2_est_idx,
                estimated.reshape(1, 1, self.block_len)
            )
            self.interpreter2.set_tensor(
                self.m2_lpb_idx,
                self.lpb_buffer.reshape(1, 1, self.block_len)
            )
            self.interpreter2.set_tensor(self.m2_states_idx, self.states2)
            self.interpreter2.invoke()

            out_block = self.interpreter2.get_tensor(self.m2_out_idx).flatten()
            self.states2 = self.interpreter2.get_tensor(self.m2_states_out_idx)

            # Overlap-add
            self.out_buffer[:-self.block_shift] = self.out_buffer[self.block_shift:]
            self.out_buffer[-self.block_shift:] = 0
            self.out_buffer += out_block

            output[start:end] = self.out_buffer[:self.block_shift]

        return output


def main():
    print("=" * 60)
    print("DTLN-aec Python Reference Generator")
    print("=" * 60)
    print(f"\nModel: 128-unit DTLN-aec")
    print(f"Source: https://github.com/breizhn/DTLN-aec")
    print(f"Model files: dtln_aec_128_1.tflite, dtln_aec_128_2.tflite")

    # Load input files
    mic_path = SAMPLES_DIR / "farend_singletalk_mic.wav"
    lpb_path = SAMPLES_DIR / "farend_singletalk_lpb.wav"

    print(f"\nInput files:")
    print(f"  Mic: {mic_path}")
    print(f"  LPB: {lpb_path}")

    mic = read_wav(mic_path)
    lpb = read_wav(lpb_path)

    print(f"\nAudio info:")
    print(f"  Mic samples: {len(mic)}, RMS: {compute_rms(mic):.6f}")
    print(f"  LPB samples: {len(lpb)}, RMS: {compute_rms(lpb):.6f}")

    # Process through TFLite
    print("\n" + "-" * 40)
    processor = DTLNAecProcessor(TFLITE_DIR, num_units=128)
    output = processor.process(mic, lpb)

    output_rms = compute_rms(output)
    reduction_db = compute_reduction_db(compute_rms(mic), output_rms)

    print(f"\nOutput info:")
    print(f"  Samples: {len(output)}")
    print(f"  RMS: {output_rms:.6f}")
    print(f"  Reduction: {reduction_db:.1f} dB")

    # Save output
    output_path = SAMPLES_DIR / "farend_singletalk_processed_python.wav"
    write_wav_float32(output_path, output)

    print(f"\nSaved to: {output_path}")

    # Write metadata file
    metadata_path = output_path.with_suffix('.txt')
    with open(metadata_path, 'w') as f:
        f.write("DTLN-aec Python Reference Output\n")
        f.write("=" * 40 + "\n\n")
        f.write("Model: 128-unit DTLN-aec\n")
        f.write("Model source: https://github.com/breizhn/DTLN-aec\n")
        f.write("Model files:\n")
        f.write("  - dtln_aec_128_1.tflite\n")
        f.write("  - dtln_aec_128_2.tflite\n\n")
        f.write(f"Input mic: farend_singletalk_mic.wav\n")
        f.write(f"Input lpb: farend_singletalk_lpb.wav\n\n")
        f.write(f"Output samples: {len(output)}\n")
        f.write(f"Output RMS: {output_rms:.6f}\n")
        f.write(f"Echo reduction: {reduction_db:.1f} dB\n")

    print(f"Metadata saved to: {metadata_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
