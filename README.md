# DTLN-aec CoreML

Neural acoustic echo cancellation for Apple platforms using CoreML.

This package provides a Swift wrapper for [DTLN-aec](https://github.com/breizhn/DTLN-aec), a dual-signal transformation LSTM network that placed **3rd in the Microsoft AEC Challenge 2021**.

## Features

- Real-time echo cancellation on Apple Silicon (~0.8ms per 8ms frame on M1)
- Three model sizes: 128, 256, or 512 LSTM units
- Pure Swift implementation with Accelerate framework for FFT
- iOS 16+ and macOS 13+ support

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/anthropics/dtln-aec-coreml.git", from: "1.0.0")
]
```

Or in Xcode: File → Add Package Dependencies → Enter the repository URL.

## Usage

```swift
import DTLNAecCoreML

// Initialize processor (use .small for best latency, .large for best quality)
let processor = DTLNAecEchoProcessor(modelSize: .small)

// Load CoreML models (do this once at startup)
try processor.loadModels()

// During audio processing:
// 1. Feed far-end (system/speaker) audio as reference
processor.feedFarEnd(systemAudioSamples)  // [Float] at 16kHz

// 2. Process near-end (microphone) audio
let cleanAudio = processor.processNearEnd(microphoneSamples)  // Returns echo-cancelled audio

// Reset when starting a new session
processor.resetStates()
```

## Model Sizes

| Model | Units | Parameters | Size | Latency (M1) | Use Case |
|-------|-------|------------|------|--------------|----------|
| `.small` | 128 | 1.8M | 3.6 MB | 0.76ms | Production (recommended) |
| `.medium` | 256 | 3.9M | 8.0 MB | 0.93ms | Higher quality |
| `.large` | 512 | 10.4M | 20.3 MB | 1.43ms | Maximum quality |

All models run well within real-time requirements (8ms per frame).

## Audio Requirements

- **Sample rate:** 16,000 Hz
- **Channels:** Mono
- **Format:** Float32

If your audio is at a different sample rate, resample before processing.

## How It Works

DTLN-aec uses a two-part architecture:

1. **Part 1 (Frequency Domain):** Takes magnitude spectra of mic and loopback signals, generates a frequency mask using LSTM layers, removes echo in frequency domain

2. **Part 2 (Time Domain):** Refines the output using learned time-domain representations with Conv1D encoders and LSTM layers

Both parts maintain LSTM state across frames to capture temporal context.

## Benchmarks

Tested on Apple M1 (8-core CPU, 8-core Neural Engine):

```
| Model | Avg Frame | RT Ratio | Status |
|-------|-----------|----------|--------|
| 128   | 0.76ms    | 0.09x    | ✅     |
| 256   | 0.93ms    | 0.12x    | ✅     |
| 512   | 1.43ms    | 0.18x    | ✅     |

Real-time requirement: <8ms per frame
```

## Converting Your Own Models

The included models are converted from the original TFLite weights. To convert different model sizes:

```bash
# Download TFLite models from DTLN-aec repo
python3 Scripts/convert_dtln_aec_to_coreml.py --download --size 256

# Convert to CoreML (requires TensorFlow 2.12+)
pip install tensorflow coremltools
python3 Scripts/convert_dtln_aec_to_coreml.py --convert --size 256
```

## Credits

- **DTLN-aec:** [Nils L. Westhausen](https://github.com/breizhn/DTLN-aec) - Original TensorFlow implementation
- **Microsoft AEC Challenge 2021:** Competition where DTLN-aec placed 3rd
- **CoreML Conversion:** This package

## License

MIT License - see LICENSE file.

The original DTLN-aec model weights are provided under MIT License by Nils L. Westhausen.
