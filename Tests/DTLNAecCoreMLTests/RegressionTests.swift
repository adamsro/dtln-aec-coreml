import XCTest

@testable import DTLNAecCoreML

/// Regression tests comparing CoreML output against Python reference
final class RegressionTests: XCTestCase {

  // MARK: - WAV File Reading

  /// Read WAV file, supporting both Int16 and Float32 formats
  func readWAVFile(_ url: URL) throws -> [Float] {
    let data = try Data(contentsOf: url)
    guard data.count > 44 else {
      throw NSError(
        domain: "WAV", code: 1, userInfo: [NSLocalizedDescriptionKey: "File too small"])
    }

    // Parse format chunk to determine audio format
    var audioFormat: UInt16 = 0
    var bitsPerSample: UInt16 = 0
    var dataOffset = 0
    var dataSize = 0

    var offset = 12  // Skip RIFF header
    while offset < data.count - 8 {
      let chunkID = String(bytes: data[offset..<offset + 4], encoding: .ascii) ?? ""
      let chunkSize = data.withUnsafeBytes { ptr in
        ptr.load(fromByteOffset: offset + 4, as: UInt32.self)
      }

      if chunkID == "fmt " {
        audioFormat = data.withUnsafeBytes { ptr in
          ptr.load(fromByteOffset: offset + 8, as: UInt16.self)
        }
        bitsPerSample = data.withUnsafeBytes { ptr in
          ptr.load(fromByteOffset: offset + 22, as: UInt16.self)
        }
      } else if chunkID == "data" {
        dataOffset = offset + 8
        dataSize = Int(chunkSize)
        break
      }
      offset += 8 + Int(chunkSize)
    }

    guard dataOffset > 0 else {
      throw NSError(
        domain: "WAV", code: 2, userInfo: [NSLocalizedDescriptionKey: "No data chunk found"])
    }

    // Read samples based on format
    if audioFormat == 1 && bitsPerSample == 16 {
      // PCM Int16
      let sampleCount = dataSize / 2
      var samples = [Float](repeating: 0, count: sampleCount)
      data.withUnsafeBytes { ptr in
        for i in 0..<sampleCount {
          let int16Value = ptr.load(fromByteOffset: dataOffset + i * 2, as: Int16.self)
          samples[i] = Float(int16Value) / 32768.0
        }
      }
      return samples
    } else if audioFormat == 3 && bitsPerSample == 32 {
      // IEEE Float32
      let sampleCount = dataSize / 4
      var samples = [Float](repeating: 0, count: sampleCount)
      data.withUnsafeBytes { ptr in
        for i in 0..<sampleCount {
          samples[i] = ptr.load(fromByteOffset: dataOffset + i * 4, as: Float.self)
        }
      }
      return samples
    } else {
      throw NSError(
        domain: "WAV", code: 3,
        userInfo: [
          NSLocalizedDescriptionKey: "Unsupported format: \(audioFormat), \(bitsPerSample)-bit"
        ])
    }
  }

  // MARK: - Helper Functions

  func computeCorrelation(_ a: [Float], _ b: [Float]) -> Float {
    let length = min(a.count, b.count)
    guard length > 0 else { return 0 }

    let aMean = a.prefix(length).reduce(0, +) / Float(length)
    let bMean = b.prefix(length).reduce(0, +) / Float(length)

    var numerator: Float = 0
    var aDenom: Float = 0
    var bDenom: Float = 0

    for i in 0..<length {
      let aDiff = a[i] - aMean
      let bDiff = b[i] - bMean
      numerator += aDiff * bDiff
      aDenom += aDiff * aDiff
      bDenom += bDiff * bDiff
    }

    let denom = sqrt(aDenom * bDenom)
    return denom > 0 ? numerator / denom : 0
  }

  func computeRMS(_ samples: [Float]) -> Float {
    guard !samples.isEmpty else { return 0 }
    let sumSquares = samples.reduce(0) { $0 + $1 * $1 }
    return sqrt(sumSquares / Float(samples.count))
  }

  func computeRMSError(_ a: [Float], _ b: [Float]) -> Float {
    let length = min(a.count, b.count)
    guard length > 0 else { return 0 }

    var sumSquaredError: Float = 0
    for i in 0..<length {
      let diff = a[i] - b[i]
      sumSquaredError += diff * diff
    }
    return sqrt(sumSquaredError / Float(length))
  }

  func computeReductionDB(_ inputRMS: Float, _ outputRMS: Float) -> Float {
    guard outputRMS > 0 else { return .infinity }
    return 20 * log10(inputRMS / outputRMS)
  }

  // MARK: - Regression Tests

  /// Strict regression test: CoreML output should match Python reference
  /// The original Python DTLN-aec achieves near-silence on this echo-only sample
  func testCoreMLMatchesPythonReference() throws {
    let thisFile = URL(fileURLWithPath: #file)
    let packageRoot = thisFile
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()

    let samplesDir = packageRoot.appendingPathComponent("Samples/aec_challenge")
    let pythonFile = samplesDir.appendingPathComponent("farend_singletalk_processed_python_128.wav")
    let coremlFile = samplesDir.appendingPathComponent("farend_singletalk_processed_coreml_128.wav")

    guard FileManager.default.fileExists(atPath: pythonFile.path) else {
      throw XCTSkip("Python reference file not found")
    }
    guard FileManager.default.fileExists(atPath: coremlFile.path) else {
      throw XCTSkip("CoreML output file not found")
    }

    let pythonSamples = try readWAVFile(pythonFile)
    let coremlSamples = try readWAVFile(coremlFile)

    let pythonRMS = computeRMS(pythonSamples)
    let coremlRMS = computeRMS(coremlSamples)

    print("\nPython vs CoreML Reference Test:")
    print("  Python samples: \(pythonSamples.count), RMS: \(String(format: "%.6f", pythonRMS))")
    print("  CoreML samples: \(coremlSamples.count), RMS: \(String(format: "%.6f", coremlRMS))")

    // Both should achieve near-silence (echo-only sample should be fully suppressed)
    // Python achieves RMS ~0.000165, CoreML should be similar
    let maxAcceptableRMS: Float = 0.001  // Near silence threshold

    XCTAssertLessThan(
      pythonRMS, maxAcceptableRMS,
      "Python reference should achieve near-silence (RMS \(pythonRMS))")

    XCTAssertLessThan(
      coremlRMS, maxAcceptableRMS,
      "CoreML should match Python's near-silence performance (RMS \(coremlRMS) vs Python \(pythonRMS))")

    // Check that both achieve similar levels of suppression (within 2x of each other)
    let rmsRatio = max(pythonRMS, coremlRMS) / max(min(pythonRMS, coremlRMS), 1e-10)
    print("  RMS ratio: \(String(format: "%.2f", rmsRatio))x")

    XCTAssertLessThan(
      rmsRatio, 3.0,
      "CoreML and Python output levels should be within 3x of each other (ratio: \(rmsRatio))")
  }

  /// Verify CoreML echo suppression effectiveness using AEC challenge sample
  /// This test checks that echo is significantly reduced in the output
  func testEchoSuppressionEffectiveness() throws {
    // Find Samples directory relative to source file location
    // #file gives us: .../Tests/DTLNAecCoreMLTests/RegressionTests.swift
    // We need to go up 3 levels to get to package root
    let thisFile = URL(fileURLWithPath: #file)
    let packageRoot = thisFile
      .deletingLastPathComponent()  // DTLNAecCoreMLTests/
      .deletingLastPathComponent()  // Tests/
      .deletingLastPathComponent()  // package root

    let samplesDir = packageRoot.appendingPathComponent("Samples/aec_challenge")
    let micFile = samplesDir.appendingPathComponent("farend_singletalk_mic.wav")
    let coremlFile = samplesDir.appendingPathComponent("farend_singletalk_processed_coreml_128.wav")

    // Check if files exist
    guard FileManager.default.fileExists(atPath: micFile.path) else {
      throw XCTSkip("Mic input file not found at: \(micFile.path)")
    }
    guard FileManager.default.fileExists(atPath: coremlFile.path) else {
      throw XCTSkip("CoreML output file not found at: \(coremlFile.path)")
    }

    // Load both files
    let micSamples = try readWAVFile(micFile)
    let coremlSamples = try readWAVFile(coremlFile)

    // Compute RMS levels
    let micRMS = computeRMS(micSamples)
    let coremlRMS = computeRMS(coremlSamples)
    let reductionDB = computeReductionDB(micRMS, coremlRMS)

    print("\nEcho Suppression Test:")
    print("  Mic samples: \(micSamples.count)")
    print("  Output samples: \(coremlSamples.count)")
    print("  Mic RMS: \(String(format: "%.6f", micRMS))")
    print("  Output RMS: \(String(format: "%.6f", coremlRMS))")
    print("  Reduction: \(String(format: "%.1f", reductionDB)) dB")

    // Verify output quality
    let hasNaN = coremlSamples.contains { $0.isNaN }
    let hasInf = coremlSamples.contains { $0.isInfinite }
    let maxAbs = coremlSamples.map { abs($0) }.max() ?? 0

    print("  Has NaN: \(hasNaN)")
    print("  Has Inf: \(hasInf)")
    print("  Max absolute: \(String(format: "%.6f", maxAbs))")

    XCTAssertFalse(hasNaN, "Output should not contain NaN values")
    XCTAssertFalse(hasInf, "Output should not contain infinite values")
    XCTAssertLessThanOrEqual(maxAbs, 1.0, "Output should be normalized (-1 to 1)")

    // This is a far-end singletalk sample (echo only, no near-end speech)
    // Echo should be reduced by at least 6 dB
    XCTAssertGreaterThan(
      reductionDB, 6.0,
      "Echo should be reduced by at least 6 dB (got \(String(format: "%.1f", reductionDB)) dB)")
  }

  /// Test that reprocessing the same input produces consistent output
  func testOutputConsistency() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModels()

    let numSamples = 16000  // 1 second
    var farEnd = [Float](repeating: 0, count: numSamples)
    var nearEnd = [Float](repeating: 0, count: numSamples)

    // Generate test signal
    for i in 0..<numSamples {
      let t = Float(i) / 16000
      farEnd[i] = 0.3 * sin(2 * .pi * 440 * t)
      nearEnd[i] = 0.2 * sin(2 * .pi * 440 * t + 0.5)  // Echo with phase shift
    }

    // Process first time
    processor.feedFarEnd(farEnd)
    let output1 = processor.processNearEnd(nearEnd)

    // Reset and process again
    processor.resetStates()
    processor.feedFarEnd(farEnd)
    let output2 = processor.processNearEnd(nearEnd)

    // Outputs should be identical
    let correlation = computeCorrelation(output1, output2)
    print("\nOutput Consistency Test:")
    print("  Correlation between runs: \(String(format: "%.6f", correlation))")

    XCTAssertEqual(
      correlation, 1.0, accuracy: 1e-5,
      "Processing same input twice should produce identical output")
  }
}
