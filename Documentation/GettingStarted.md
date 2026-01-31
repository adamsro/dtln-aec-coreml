# Getting Started with DTLN-aec CoreML

This guide will help you integrate neural echo cancellation into your iOS or macOS app.

## Installation

### Swift Package Manager

Add DTLNAecCoreML to your project using Xcode:

1. Open your project in Xcode
2. Go to **File → Add Package Dependencies**
3. Enter the repository URL: `https://github.com/anthropics/dtln-aec-coreml.git`
4. Select the version and click **Add Package**

Or add it to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/anthropics/dtln-aec-coreml.git", from: "0.4.0-beta")
]
```

## Quick Start

### Basic Usage

```swift
import DTLNAecCoreML

// 1. Create the processor (256-unit model recommended for most apps)
let processor = DTLNAecEchoProcessor(modelSize: .medium)

// 2. Load the models (do this once at startup)
try processor.loadModels()

// 3. In your audio processing callback:
// Feed the far-end (speaker) audio as reference
processor.feedFarEnd(speakerSamples)

// Process the near-end (microphone) audio
let cleanAudio = processor.processNearEnd(microphoneSamples)
```

### Using Configuration

For more control, use `DTLNAecConfig`:

```swift
var config = DTLNAecConfig()
config.modelSize = .large  // Best quality
config.computeUnits = .cpuAndNeuralEngine  // Use Neural Engine when available
config.enablePerformanceTracking = true

let processor = DTLNAecEchoProcessor(config: config)
```

### Async Model Loading

Load models without blocking the main thread:

```swift
Task {
    let processor = DTLNAecEchoProcessor(modelSize: .medium)
    try await processor.loadModelsAsync()
    // Models are ready
}
```

## Choosing a Model Size

| Model | Quality | Latency | Use Case |
|-------|---------|---------|----------|
| `.small` (128 units) | Good | ~0.4ms | Minimal bundle size |
| `.medium` (256 units) | Great | ~0.6ms | **Recommended for most apps** |
| `.large` (512 units) | Best | ~1.0ms | Best quality for long audio |

All models run well within the 8ms real-time budget. The 256-unit model is recommended for most applications due to its fast convergence (~0.3s) and excellent quality.

## What's Next?

- [Audio Requirements](AudioRequirements.md) - Sample rate, format, and buffering
- [API Reference](API.md) - Complete API documentation
- [Benchmarking](Benchmarking.md) - Measure performance on your device
