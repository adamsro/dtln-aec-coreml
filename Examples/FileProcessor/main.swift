// FileProcessor/main.swift
// Process WAV files with DTLN-aec echo cancellation
//
// This example demonstrates file-based audio processing.
// It reads 16kHz mono WAV files and outputs echo-cancelled audio.
//
// Usage:
//   swift run FileProcessor --mic input_mic.wav --ref input_ref.wav --output output.wav
//
// To build as a standalone executable:
//   1. Add to Package.swift:
//      .executableTarget(name: "FileProcessor", dependencies: ["DTLNAecCoreML"], path: "Examples/FileProcessor")
//   2. Run: swift run FileProcessor --help

import DTLNAecCoreML
import Foundation

// MARK: - Simple WAV Reader/Writer

/// Minimal WAV file header
struct WAVHeader {
  var riffID: UInt32 = 0x4646_4952  // "RIFF"
  var fileSize: UInt32 = 0
  var waveID: UInt32 = 0x4556_4157  // "WAVE"
  var fmtID: UInt32 = 0x2074_6D66  // "fmt "
  var fmtSize: UInt32 = 16
  var audioFormat: UInt16 = 3  // IEEE float
  var numChannels: UInt16 = 1
  var sampleRate: UInt32 = 16000
  var byteRate: UInt32 = 64000  // sampleRate * channels * bitsPerSample/8
  var blockAlign: UInt16 = 4  // channels * bitsPerSample/8
  var bitsPerSample: UInt16 = 32
  var dataID: UInt32 = 0x6174_6164  // "data"
  var dataSize: UInt32 = 0
}

func readWAVFile(_ url: URL) throws -> [Float] {
  let data = try Data(contentsOf: url)
  guard data.count > 44 else {
    throw NSError(domain: "WAV", code: 1, userInfo: [NSLocalizedDescriptionKey: "File too small"])
  }

  // Find data chunk
  var offset = 12  // Skip RIFF header
  while offset < data.count - 8 {
    let chunkID = String(bytes: data[offset..<offset + 4], encoding: .ascii) ?? ""
    let chunkSize = data.withUnsafeBytes { ptr in
      ptr.load(fromByteOffset: offset + 4, as: UInt32.self)
    }

    if chunkID == "data" {
      offset += 8
      break
    }
    offset += 8 + Int(chunkSize)
  }

  // Read float samples
  let sampleCount = (data.count - offset) / 4
  var samples = [Float](repeating: 0, count: sampleCount)

  data.withUnsafeBytes { ptr in
    for i in 0..<sampleCount {
      samples[i] = ptr.load(fromByteOffset: offset + i * 4, as: Float.self)
    }
  }

  return samples
}

func writeWAVFile(_ url: URL, samples: [Float], sampleRate: UInt32 = 16000) throws {
  var header = WAVHeader()
  header.sampleRate = sampleRate
  header.byteRate = sampleRate * 4
  header.dataSize = UInt32(samples.count * 4)
  header.fileSize = header.dataSize + 36

  var data = Data()

  // Write header
  withUnsafeBytes(of: header.riffID) { data.append(contentsOf: $0) }
  withUnsafeBytes(of: header.fileSize) { data.append(contentsOf: $0) }
  withUnsafeBytes(of: header.waveID) { data.append(contentsOf: $0) }
  withUnsafeBytes(of: header.fmtID) { data.append(contentsOf: $0) }
  withUnsafeBytes(of: header.fmtSize) { data.append(contentsOf: $0) }
  withUnsafeBytes(of: header.audioFormat) { data.append(contentsOf: $0) }
  withUnsafeBytes(of: header.numChannels) { data.append(contentsOf: $0) }
  withUnsafeBytes(of: header.sampleRate) { data.append(contentsOf: $0) }
  withUnsafeBytes(of: header.byteRate) { data.append(contentsOf: $0) }
  withUnsafeBytes(of: header.blockAlign) { data.append(contentsOf: $0) }
  withUnsafeBytes(of: header.bitsPerSample) { data.append(contentsOf: $0) }
  withUnsafeBytes(of: header.dataID) { data.append(contentsOf: $0) }
  withUnsafeBytes(of: header.dataSize) { data.append(contentsOf: $0) }

  // Write samples
  for sample in samples {
    withUnsafeBytes(of: sample) { data.append(contentsOf: $0) }
  }

  try data.write(to: url)
}

// MARK: - Command Line Parsing

func printUsage() {
  print(
    """
    DTLN-aec File Processor

    Usage:
      FileProcessor --mic <mic.wav> --ref <reference.wav> --output <output.wav> [options]

    Required:
      --mic <file>      Microphone input WAV file (16kHz mono float32)
      --ref <file>      Reference/loopback WAV file (16kHz mono float32)
      --output <file>   Output WAV file path

    Options:
      --model <size>    Model size: small, large (default)
      --help            Show this help message

    Example:
      FileProcessor --mic mic_with_echo.wav --ref speaker_playback.wav --output cleaned.wav

    Note: Input files must be 16kHz mono. For other formats, convert first:
      ffmpeg -i input.wav -ar 16000 -ac 1 -f wav -acodec pcm_f32le output.wav
    """)
}

// MARK: - Main

var micPath: String?
var refPath: String?
var outputPath: String?
var modelSize: DTLNAecModelSize = .large

// Parse arguments
var args = CommandLine.arguments.dropFirst()
while let arg = args.popFirst() {
  switch arg {
  case "--mic":
    micPath = args.popFirst()
  case "--ref":
    refPath = args.popFirst()
  case "--output":
    outputPath = args.popFirst()
  case "--model":
    if let size = args.popFirst() {
      switch size.lowercased() {
      case "small", "128": modelSize = .small
      case "large", "512": modelSize = .large
      default:
        print("Unknown model size: \(size). Use 'small' or 'large'.")
        exit(1)
      }
    }
  case "--help", "-h":
    printUsage()
    exit(0)
  default:
    print("Unknown argument: \(arg)")
    printUsage()
    exit(1)
  }
}

// Validate arguments
guard let micPath = micPath, let refPath = refPath, let outputPath = outputPath else {
  print("Error: Missing required arguments.\n")
  printUsage()
  exit(1)
}

print("DTLN-aec File Processor")
print("=======================")
print()

// Load files
print("Loading input files...")
let micSamples: [Float]
let refSamples: [Float]

do {
  micSamples = try readWAVFile(URL(fileURLWithPath: micPath))
  print("  Mic: \(micPath) (\(micSamples.count) samples, \(String(format: "%.2f", Float(micSamples.count) / 16000))s)")

  refSamples = try readWAVFile(URL(fileURLWithPath: refPath))
  print("  Ref: \(refPath) (\(refSamples.count) samples, \(String(format: "%.2f", Float(refSamples.count) / 16000))s)")
} catch {
  print("Error reading files: \(error)")
  exit(1)
}

// Ensure same length
let minLength = min(micSamples.count, refSamples.count)
let mic = Array(micSamples.prefix(minLength))
let ref = Array(refSamples.prefix(minLength))

print()

// Create and load processor
print("Loading DTLN-aec model (\(modelSize.units) units)...")
let processor = DTLNAecEchoProcessor(modelSize: modelSize)

do {
  try processor.loadModels()
  print("  Model loaded successfully")
} catch {
  print("Error loading model: \(error)")
  exit(1)
}

print()

// Process audio
print("Processing...")
let startTime = Date()

// Process in chunks - feed far-end and near-end in sync
let chunkSize = 128
var output: [Float] = []

for start in stride(from: 0, to: mic.count, by: chunkSize) {
  let end = min(start + chunkSize, mic.count)

  // Feed far-end chunk (reference signal)
  let refChunk = Array(ref[start..<end])
  processor.feedFarEnd(refChunk)

  // Process near-end chunk (microphone signal)
  let micChunk = Array(mic[start..<end])
  let processed = processor.processNearEnd(micChunk)
  output.append(contentsOf: processed)
}

let processingTime = Date().timeIntervalSince(startTime)
let audioDuration = Double(mic.count) / 16000.0

print("  Processed \(output.count) samples in \(String(format: "%.2f", processingTime))s")
print("  Real-time factor: \(String(format: "%.2f", processingTime / audioDuration))x")
print("  Average frame time: \(String(format: "%.2f", processor.averageFrameTimeMs))ms")
print()

// Write output
print("Writing output...")
do {
  try writeWAVFile(URL(fileURLWithPath: outputPath), samples: output)
  print("  Saved: \(outputPath)")
} catch {
  print("Error writing output: \(error)")
  exit(1)
}

print()
print("Done!")
