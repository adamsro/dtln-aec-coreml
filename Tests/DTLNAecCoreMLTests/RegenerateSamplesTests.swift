// RegenerateSamplesTests.swift
// Test to regenerate CoreML processed sample files

import Foundation
import XCTest

@testable import DTLNAecCoreML

/// Regenerates the CoreML processed sample WAV files
final class RegenerateSamplesTests: XCTestCase {

  /// Regenerate all CoreML sample files
  func testRegenerateSamples() throws {
    let samplesDir = RegressionTestUtils.samplesDir()
    let micFile = samplesDir.appendingPathComponent("farend_singletalk_mic.wav")
    let lpbFile = samplesDir.appendingPathComponent("farend_singletalk_lpb.wav")

    guard FileManager.default.fileExists(atPath: micFile.path),
      FileManager.default.fileExists(atPath: lpbFile.path)
    else {
      throw XCTSkip("Input audio files not found")
    }

    let micSamples = try RegressionTestUtils.readWAVFile(micFile)
    let lpbSamples = try RegressionTestUtils.readWAVFile(lpbFile)

    print("\n=== Regenerating CoreML Sample Files ===")
    print("Input: \(micFile.lastPathComponent)")
    print("Samples: \(micSamples.count)")

    for modelSize in DTLNAecModelSize.allCases {
      print("\nProcessing with \(modelSize.units)-unit model...")

      let processor = DTLNAecEchoProcessor(modelSize: modelSize)
      try processor.loadModelsFromPackage()

      // Process in streaming chunks (128 samples = 8ms)
      let chunkSize = 128
      var output: [Float] = []

      for i in stride(from: 0, to: min(micSamples.count, lpbSamples.count), by: chunkSize) {
        let endIdx = min(i + chunkSize, micSamples.count, lpbSamples.count)
        let micChunk = Array(micSamples[i..<endIdx])
        let lpbChunk = Array(lpbSamples[i..<endIdx])

        processor.feedFarEnd(lpbChunk)
        let outChunk = processor.processNearEnd(micChunk)
        output.append(contentsOf: outChunk)
      }

      // Compute metrics
      let micRMS = RegressionTestUtils.computeRMS(micSamples)
      let outRMS = RegressionTestUtils.computeRMS(output)
      let reductionDB = RegressionTestUtils.computeReductionDB(micRMS, outRMS)

      print("  Output samples: \(output.count)")
      print("  Output RMS: \(String(format: "%.6f", outRMS))")
      print("  Reduction: \(String(format: "%.1f", reductionDB)) dB")

      // Write output WAV file
      let outputFile = samplesDir.appendingPathComponent(
        "farend_singletalk_processed_coreml_\(modelSize.units).wav")
      try writeWAVFile(samples: output, to: outputFile)
      print("  Saved: \(outputFile.lastPathComponent)")
    }

    print("\n=== Sample Regeneration Complete ===")
  }

  /// Regenerate realworld CoreML sample files
  func testRegenerateRealworldSamples() throws {
    let samplesDir = RegressionTestUtils.samplesDir()
      .deletingLastPathComponent()
      .appendingPathComponent("realworld")
    let micFile = samplesDir.appendingPathComponent("test_mic.wav")
    let lpbFile = samplesDir.appendingPathComponent("test_lpb.wav")

    guard FileManager.default.fileExists(atPath: micFile.path),
      FileManager.default.fileExists(atPath: lpbFile.path)
    else {
      throw XCTSkip("Realworld input audio files not found")
    }

    let micSamples = try RegressionTestUtils.readWAVFile(micFile)
    let lpbSamples = try RegressionTestUtils.readWAVFile(lpbFile)

    print("\n=== Regenerating Realworld CoreML Sample Files ===")
    print("Input: \(micFile.lastPathComponent)")
    print("Samples: \(micSamples.count)")

    for modelSize in DTLNAecModelSize.allCases {
      print("\nProcessing with \(modelSize.units)-unit model...")

      let processor = DTLNAecEchoProcessor(modelSize: modelSize)
      try processor.loadModelsFromPackage()

      // Process in streaming chunks (128 samples = 8ms)
      let chunkSize = 128
      var output: [Float] = []

      for i in stride(from: 0, to: min(micSamples.count, lpbSamples.count), by: chunkSize) {
        let endIdx = min(i + chunkSize, micSamples.count, lpbSamples.count)
        let micChunk = Array(micSamples[i..<endIdx])
        let lpbChunk = Array(lpbSamples[i..<endIdx])

        processor.feedFarEnd(lpbChunk)
        let outChunk = processor.processNearEnd(micChunk)
        output.append(contentsOf: outChunk)
      }

      // Compute metrics
      let micRMS = RegressionTestUtils.computeRMS(micSamples)
      let outRMS = RegressionTestUtils.computeRMS(output)
      let reductionDB = RegressionTestUtils.computeReductionDB(micRMS, outRMS)

      print("  Output samples: \(output.count)")
      print("  Output RMS: \(String(format: "%.6f", outRMS))")
      print("  Reduction: \(String(format: "%.1f", reductionDB)) dB")

      // Write output WAV file
      let outputFile = samplesDir.appendingPathComponent(
        "test_processed_coreml_\(modelSize.units).wav")
      try writeWAVFile(samples: output, to: outputFile)
      print("  Saved: \(outputFile.lastPathComponent)")
    }

    print("\n=== Realworld Sample Regeneration Complete ===")
  }

  /// Write samples to a 16-bit mono WAV file at 16kHz
  private func writeWAVFile(samples: [Float], to url: URL) throws {
    let sampleRate: UInt32 = 16000
    let numChannels: UInt16 = 1
    let bitsPerSample: UInt16 = 16
    let byteRate = sampleRate * UInt32(numChannels) * UInt32(bitsPerSample / 8)
    let blockAlign = numChannels * (bitsPerSample / 8)

    // Convert float samples to Int16
    var int16Samples = [Int16](repeating: 0, count: samples.count)
    for i in 0..<samples.count {
      let clamped = max(-1.0, min(1.0, samples[i]))
      int16Samples[i] = Int16(clamped * 32767)
    }

    let dataSize = UInt32(samples.count * 2)
    let fileSize = 36 + dataSize

    var data = Data()

    // RIFF header
    data.append(contentsOf: "RIFF".utf8)
    data.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
    data.append(contentsOf: "WAVE".utf8)

    // fmt chunk
    data.append(contentsOf: "fmt ".utf8)
    data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })  // chunk size
    data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })  // PCM format
    data.append(contentsOf: withUnsafeBytes(of: numChannels.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: sampleRate.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian) { Array($0) })

    // data chunk
    data.append(contentsOf: "data".utf8)
    data.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })

    // Sample data
    for sample in int16Samples {
      data.append(contentsOf: withUnsafeBytes(of: sample.littleEndian) { Array($0) })
    }

    try data.write(to: url)
  }
}
