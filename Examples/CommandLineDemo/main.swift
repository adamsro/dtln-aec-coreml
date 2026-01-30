// CommandLineDemo/main.swift
// Simple demonstration of DTLN-aec echo cancellation
//
// This example shows basic usage with synthetic audio data.
// For real applications, you would use AVAudioEngine or Core Audio.
//
// Usage:
//   swift run CommandLineDemo
//
// To build as a standalone executable in this repo:
//   1. Add to Package.swift:
//      .executableTarget(name: "CommandLineDemo", dependencies: ["DTLNAecCoreML"], path: "Examples/CommandLineDemo")
//   2. Run: swift run CommandLineDemo

import DTLNAecCoreML
import Foundation

/// Generate a simple sine wave tone
func generateTone(frequency: Float, duration: Float, sampleRate: Float = 16000) -> [Float] {
  let numSamples = Int(duration * sampleRate)
  var samples = [Float](repeating: 0, count: numSamples)

  for i in 0..<numSamples {
    let t = Float(i) / sampleRate
    samples[i] = sin(2 * .pi * frequency * t)
  }

  return samples
}

/// Mix two signals with given levels
func mix(_ a: [Float], _ b: [Float], levelA: Float = 1.0, levelB: Float = 1.0) -> [Float] {
  let count = min(a.count, b.count)
  var result = [Float](repeating: 0, count: count)

  for i in 0..<count {
    result[i] = a[i] * levelA + b[i] * levelB
  }

  return result
}

/// Calculate RMS level in dB
func rmsLevel(_ samples: [Float]) -> Float {
  guard !samples.isEmpty else { return -Float.infinity }

  let sumSquares = samples.reduce(0) { $0 + $1 * $1 }
  let rms = sqrt(sumSquares / Float(samples.count))

  return 20 * log10(max(rms, 1e-10))
}

// MARK: - Main

print("DTLN-aec CoreML Demo")
print("====================")
print()

// Create processor with small model for fast loading
print("Creating processor with 128-unit model...")
let processor = DTLNAecEchoProcessor(modelSize: .small)

// Load models
print("Loading CoreML models...")
do {
  let startTime = Date()
  try processor.loadModels()
  let loadTime = Date().timeIntervalSince(startTime) * 1000
  print("Models loaded in \(String(format: "%.0f", loadTime))ms")
} catch {
  print("Error loading models: \(error)")
  exit(1)
}

print()

// Generate test signals
let duration: Float = 1.0  // 1 second
let sampleRate: Float = 16000

print("Generating test signals...")

// Far-end: 440Hz tone (like music or speech from speaker)
let farEnd = generateTone(frequency: 440, duration: duration)

// Near-end: 880Hz tone (like local speech)
let nearEnd = generateTone(frequency: 880, duration: duration, sampleRate: sampleRate)

// Simulated mic input: near-end + attenuated far-end echo
let echoLevel: Float = 0.3  // 30% echo
let micInput = mix(nearEnd, farEnd, levelA: 1.0, levelB: echoLevel)

print("  Far-end (speaker): 440Hz tone, \(farEnd.count) samples")
print("  Near-end (voice): 880Hz tone, \(nearEnd.count) samples")
print("  Mic input: near-end + \(Int(echoLevel * 100))% echo")
print()

// Process audio
print("Processing echo cancellation...")
let startTime = Date()

// Feed the far-end reference
processor.feedFarEnd(farEnd)

// Process near-end in chunks (simulating real-time audio callback)
let chunkSize = 128  // 8ms at 16kHz
var processedSamples: [Float] = []

for start in stride(from: 0, to: micInput.count, by: chunkSize) {
  let end = min(start + chunkSize, micInput.count)
  let chunk = Array(micInput[start..<end])
  let processed = processor.processNearEnd(chunk)
  processedSamples.append(contentsOf: processed)
}

let processingTime = Date().timeIntervalSince(startTime) * 1000
let audioDurationMs = duration * 1000

print("  Processed \(processedSamples.count) samples in \(String(format: "%.1f", processingTime))ms")
print("  Real-time factor: \(String(format: "%.2f", processingTime / Double(audioDurationMs)))x")
print("  Average frame time: \(String(format: "%.2f", processor.averageFrameTimeMs))ms")
print()

// Analyze results
print("Signal Analysis:")
print("  Input mic level:  \(String(format: "%.1f", rmsLevel(micInput))) dB")
print("  Output level:     \(String(format: "%.1f", rmsLevel(processedSamples))) dB")
print("  Near-end only:    \(String(format: "%.1f", rmsLevel(nearEnd))) dB")
print()

// Calculate echo reduction (simplified)
// In a real scenario, you'd use ERLE or similar metrics
let inputEchoLevel = rmsLevel(mix(farEnd, nearEnd, levelA: echoLevel, levelB: 0))
let outputEchoEstimate = rmsLevel(processedSamples) - rmsLevel(nearEnd)
print("  Estimated echo reduction: ~\(String(format: "%.1f", max(0, inputEchoLevel - outputEchoEstimate))) dB")
print()

// Show config options
print("Configuration Options:")
print("  Model size: \(processor.modelSize) (\(processor.numUnits) units)")
print("  Compute units: cpuAndNeuralEngine (default)")
print()

// Demonstrate async loading
print("Async loading example:")
print("  Use 'try await processor.loadModelsAsync()' for non-blocking load")
print()

print("Demo complete!")
