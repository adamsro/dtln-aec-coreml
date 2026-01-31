# DTLN-aec CoreML

> **Note:** This repository is still being dogfooded and has not been thoroughly tested.

Neural acoustic echo cancellation for Apple platforms using CoreML.

This package provides a Swift wrapper for [DTLN-aec](https://github.com/breizhn/DTLN-aec), a dual-signal transformation LSTM network that placed **3rd in the Microsoft AEC Challenge 2021**.

## Features

- Real-time echo cancellation on Apple Silicon (~0.4ms per 8ms frame on M1)
- Three model sizes: 128 units (~7 MB), 256 units (~15 MB), 512 units (~40 MB)
- **Separate model packages** - only bundle the model size you need
- Modern Swift API with async/await support
- Configurable compute units (CPU, GPU, Neural Engine)
- iOS 16+ and macOS 13+ support

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/anthropics/dtln-aec-coreml.git", from: "1.0.0")
],
targets: [
    .target(
        name: "YourApp",
        dependencies: [
            "DTLNAecCoreML",   // Core processing code
            "DTLNAec128",      // Small model (~7 MB) - pick one or more
            // "DTLNAec256",   // Medium model (~15 MB)
            // "DTLNAec512",   // Large model (~40 MB)
        ]
    )
]
```

Or in Xcode: File → Add Package Dependencies → Enter the repository URL, then select which products to include.

## Quick Start

```swift
import DTLNAecCoreML
import DTLNAec128  // Import the model package you need

// Initialize processor
let processor = DTLNAecEchoProcessor(modelSize: .small)

// Load CoreML models from the model package bundle
try processor.loadModels(from: DTLNAec128.bundle)

// During audio processing:
processor.feedFarEnd(systemAudioSamples)  // [Float] at 16kHz
let cleanAudio = processor.processNearEnd(microphoneSamples)

// Reset when starting a new session
processor.resetStates()
```

### Async Model Loading

```swift
import DTLNAecCoreML
import DTLNAec512

let processor = DTLNAecEchoProcessor(modelSize: .large)
try await processor.loadModelsAsync(from: DTLNAec512.bundle)  // Non-blocking
```

### Configuration Options

```swift
import DTLNAecCoreML
import DTLNAec512

var config = DTLNAecConfig()
config.modelSize = .large                    // Best quality
config.computeUnits = .cpuAndNeuralEngine   // Use Neural Engine
config.enablePerformanceTracking = true

let processor = DTLNAecEchoProcessor(config: config)
try processor.loadModels(from: DTLNAec512.bundle)
```

## Model Sizes

| Model | Units | Parameters | Bundle Size | Latency (M1) | Use Case |
|-------|-------|------------|-------------|--------------|----------|
| `.small` | 128 | 1.8M | ~7 MB | 0.4ms | Low latency, good quality |
| `.medium` | 256 | 3.9M | ~15 MB | 0.6ms | Balanced |
| `.large` | 512 | 10.4M | ~40 MB | 1.0ms | Best quality |

All models run well within real-time requirements (8ms per frame).

### Understanding Latency

- **Processing time** (shown above): How long the neural network takes to compute output for one 8ms frame. Since all models process in <2ms, they run comfortably faster than real-time.
- **Algorithmic latency**: The inherent delay from buffering required by the STFT algorithm. DTLN-aec uses 512-sample frames at 16kHz, so the algorithmic latency is ~32ms (one STFT window).

The processing time adds negligibly on top of the algorithmic latency.

**Import the corresponding model package:**
- `DTLNAec128` for `.small`
- `DTLNAec256` for `.medium`
- `DTLNAec512` for `.large`

### Model Selection Guide

Each model has different convergence characteristics due to LSTM state initialization:

| Model | Convergence Time | Steady-State Suppression | Best For |
|-------|------------------|--------------------------|----------|
| `.small` (128) | ~1.0s | 49 dB | General use, balanced |
| `.medium` (256) | ~0.3s | 50 dB | Fast convergence needed |
| `.large` (512) | ~0.9s | 53 dB | Long audio, best quality |

**Key insight:** Larger models have more LSTM hidden states that need time to stabilize. The 256-unit model hits a sweet spot with the fastest convergence, while 512-unit achieves the best steady-state suppression but takes longer to reach it.

**Recommendations:**
- For **short audio clips** (<2s): Use `.medium` (256) for fastest convergence
- For **long audio/calls**: Use `.large` (512) for best overall suppression
- For **low latency + small bundle**: Use `.small` (128) as a balanced choice

## Audio Requirements

- **Sample rate:** 16,000 Hz
- **Channels:** Mono
- **Format:** Float32

If your audio is at a different sample rate, resample before processing.

## Documentation

- [Getting Started](Documentation/GettingStarted.md) - Installation and basic usage
- [Audio Requirements](Documentation/AudioRequirements.md) - Sample rates, formats, buffering
- [API Reference](Documentation/API.md) - Complete API documentation
- [Benchmarking](Documentation/Benchmarking.md) - Measure performance
- [Model Conversion](Documentation/ModelConversion.md) - Convert custom models

## Benchmarking

Run the included benchmark to measure performance on your hardware:

```bash
swift run dtln-benchmark        # Basic benchmark
swift run dtln-benchmark -n 1000  # More iterations
swift run dtln-benchmark --json   # JSON output for CI
```

Sample output on Apple M1:

```
| Model | Params | Load    | Avg     | P99     | RT Ratio | Status |
|-------|--------|---------|---------|---------|----------|--------|
| 128   | 1.8M   |   448ms |  0.36ms |  1.58ms |   0.04x  | ✅     |
| 256   | 3.9M   |   509ms |  0.59ms |  2.79ms |   0.07x  | ✅     |
| 512   | 10.4M  |   604ms |  1.01ms |  4.45ms |   0.13x  | ✅     |
```

## Testing Echo Cancellation

### Unit Tests

Run the synthetic AEC quality tests:

```bash
swift test --filter AECQualityTests
```

### Real-World Test

Test with actual speaker-to-microphone echo on your Mac:

```bash
# 1. Record: plays audio through speakers while recording from mic
swift Scripts/record_aec_test.swift \
  Tests/DTLNAecCoreMLTests/Fixtures/farend.wav \
  /tmp/recorded_nearend.wav

# 2. Process: run echo cancellation
swift run FileProcessor \
  --mic /tmp/recorded_nearend.wav \
  --ref Tests/DTLNAecCoreMLTests/Fixtures/farend.wav \
  --output /tmp/cleaned.wav

# 3. Compare before/after
afplay /tmp/recorded_nearend.wav  # With echo
afplay /tmp/cleaned.wav           # Echo cancelled
```

### Process Your Own Files

```bash
swift run FileProcessor \
  --mic your_mic_recording.wav \
  --ref your_system_audio.wav \
  --output cleaned.wav \
  --model small  # or 'large'
```

Input files must be 16kHz mono WAV. Convert with ffmpeg if needed:

```bash
ffmpeg -i input.wav -ar 16000 -ac 1 -c:a pcm_f32le output.wav
```

## How It Works

DTLN-aec uses a two-part architecture:

1. **Part 1 (Frequency Domain):** Takes magnitude spectra of mic and loopback signals, generates a frequency mask using LSTM layers

2. **Part 2 (Time Domain):** Refines the output using learned time-domain representations with Conv1D encoders and LSTM layers

Both parts maintain LSTM state across frames to capture temporal context.

## Credits

- **DTLN-aec:** [Nils L. Westhausen](https://github.com/breizhn/DTLN-aec) - Original TensorFlow implementation
- **Microsoft AEC Challenge 2021:** Competition where DTLN-aec placed 3rd

## License

MIT License - see [LICENSE](LICENSE) file.

The original DTLN-aec model weights are provided under MIT License by Nils L. Westhausen. See [ThirdPartyLicenses/](ThirdPartyLicenses/) for details.
