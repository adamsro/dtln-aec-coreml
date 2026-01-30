// DTLNAecEchoProcessor.swift
// DTLN-aec Neural Echo Cancellation for CoreML
//
// Two-part neural network for acoustic echo cancellation:
// Part 1: Frequency domain - generates mask from mic + loopback magnitudes
// Part 2: Time domain - refines output using encoded features
//
// Model specs:
// - Sample rate: 16kHz mono
// - Block size: 512 samples (32ms)
// - Block shift: 128 samples (8ms)
// - FFT bins: 257
// - LSTM states: [1, 2, units, 2] (2 layers, N units, h/c states)
//
// Available model sizes:
// - 128 units: 1.8M params, ~0.8ms/frame on M1
// - 256 units: 3.9M params, ~0.9ms/frame on M1
// - 512 units: 10.4M params, ~1.4ms/frame on M1

import Accelerate
import CoreML
import Foundation
import os.log

private let logger = Logger(subsystem: "DTLNAecCoreML", category: "EchoProcessor")

/// Available DTLN-aec model sizes.
/// Larger models have better quality but higher latency.
public enum DTLNAecModelSize: Int, CaseIterable, Sendable {
  case small = 128  // 1.8M params, ~0.8ms/frame on M1
  case medium = 256  // 3.9M params, ~0.9ms/frame on M1
  case large = 512  // 10.4M params, ~1.4ms/frame on M1

  public var units: Int { rawValue }

  var modelNamePrefix: String { "DTLN_AEC_\(rawValue)" }

  /// Estimated model file size (both parts combined) in MB
  public var estimatedSizeMB: Double {
    switch self {
    case .small: return 3.6
    case .medium: return 8.0
    case .large: return 20.3
    }
  }
}

/// Neural echo cancellation using DTLN-aec architecture.
/// Processes 8ms frames (128 samples) with overlap-add output.
///
/// ## Usage
/// ```swift
/// let processor = DTLNAecEchoProcessor(modelSize: .small)
/// try processor.loadModels()
///
/// // During audio processing:
/// processor.feedFarEnd(systemAudioSamples)  // 16kHz Float array
/// let cleanAudio = processor.processNearEnd(microphoneSamples)
/// ```
///
/// ## Thread Safety
/// NOT thread-safe. Call from a single thread or serial queue.
public final class DTLNAecEchoProcessor {

  // MARK: - Constants

  /// Audio sample rate (must be 16kHz)
  public static let sampleRate: Double = 16_000

  /// FFT/IFFT block size in samples (512 = 32ms at 16kHz)
  public static let blockLen = 512

  /// Frame shift in samples (128 = 8ms at 16kHz)
  public static let blockShift = 128

  /// Number of FFT bins (blockLen/2 + 1)
  public static let fftBins = 257

  /// Number of LSTM layers
  static let numLayers = 2

  // MARK: - Model Configuration

  /// The model size being used
  public let modelSize: DTLNAecModelSize

  /// Number of LSTM units per layer (from model size)
  public var numUnits: Int { modelSize.units }

  // MARK: - CoreML Models

  private var modelPart1: MLModel?
  private var modelPart2: MLModel?

  // MARK: - LSTM States (persist across frames)

  private var states1: MLMultiArray?
  private var states2: MLMultiArray?

  // MARK: - Audio Buffers

  private var micBuffer: [Float] = []
  private var loopbackBuffer: [Float] = []
  private var outputBuffer: [Float]

  // MARK: - FFT Setup (Accelerate vDSP)

  private var fftSetup: vDSP.FFT<DSPSplitComplex>?
  private var window: [Float]
  private var fftRealBuffer: [Float]
  private var fftImagBuffer: [Float]

  // MARK: - Preallocated MLMultiArrays

  private var micMagArray: MLMultiArray?
  private var lpbMagArray: MLMultiArray?
  private var estimatedFrameArray: MLMultiArray?
  private var lpbTimeArray: MLMultiArray?

  // MARK: - Statistics

  private var framesProcessed: Int = 0
  private var totalProcessingTimeMs: Double = 0

  // MARK: - Initialization

  /// Whether models are loaded and ready for processing
  public var isInitialized: Bool {
    modelPart1 != nil && modelPart2 != nil
  }

  /// Initialize with specified model size.
  /// - Parameter modelSize: The DTLN-aec model variant to use (default: .small = 128 units)
  public init(modelSize: DTLNAecModelSize = .small) {
    self.modelSize = modelSize
    outputBuffer = [Float](repeating: 0, count: Self.blockLen)
    window = [Float](repeating: 0, count: Self.blockLen)
    vDSP_hann_window(&window, vDSP_Length(Self.blockLen), Int32(vDSP_HANN_NORM))
    fftRealBuffer = [Float](repeating: 0, count: Self.blockLen)
    fftImagBuffer = [Float](repeating: 0, count: Self.blockLen)
    fftSetup = vDSP.FFT(log2n: 9, radix: .radix2, ofType: DSPSplitComplex.self)
  }

  /// Load CoreML models from bundle.
  /// Call this before processing audio.
  /// - Throws: `DTLNAecError.modelNotFound` if models are not in the bundle
  public func loadModels() throws {
    let startTime = Date()

    let part1Name = "\(modelSize.modelNamePrefix)_Part1"
    let part2Name = "\(modelSize.modelNamePrefix)_Part2"

    guard let part1URL = try findAndCompileModel(name: part1Name) else {
      throw DTLNAecError.modelNotFound(part1Name)
    }

    guard let part2URL = try findAndCompileModel(name: part2Name) else {
      throw DTLNAecError.modelNotFound(part2Name)
    }

    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndNeuralEngine

    modelPart1 = try MLModel(contentsOf: part1URL, configuration: config)
    modelPart2 = try MLModel(contentsOf: part2URL, configuration: config)

    try initializeStates()
    try preallocateArrays()

    let loadTimeMs = Date().timeIntervalSince(startTime) * 1000
    logger.info(
      "DTLN-aec \(self.modelSize.units)-unit models loaded in \(String(format: "%.1f", loadTimeMs))ms"
    )
  }

  private func findAndCompileModel(name: String) throws -> URL? {
    // Check for pre-compiled model
    if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
      return url
    }

    // Check module bundle (SPM)
    if let url = Bundle.module.url(forResource: name, withExtension: "mlmodelc") {
      return url
    }
    if let url = Bundle.module.url(forResource: name, withExtension: "mlpackage") {
      logger.info("Compiling \(name).mlpackage...")
      return try MLModel.compileModel(at: url)
    }

    // Check main bundle for mlpackage
    if let url = Bundle.main.url(forResource: name, withExtension: "mlpackage") {
      logger.info("Compiling \(name).mlpackage...")
      return try MLModel.compileModel(at: url)
    }

    return nil
  }

  private func initializeStates() throws {
    let stateShape = [1, Self.numLayers, numUnits, 2] as [NSNumber]
    states1 = try MLMultiArray(shape: stateShape, dataType: .float32)
    states2 = try MLMultiArray(shape: stateShape, dataType: .float32)
    resetStates()
  }

  private func preallocateArrays() throws {
    let magShape = [1, 1, Self.fftBins] as [NSNumber]
    micMagArray = try MLMultiArray(shape: magShape, dataType: .float32)
    lpbMagArray = try MLMultiArray(shape: magShape, dataType: .float32)

    let timeShape = [1, 1, Self.blockLen] as [NSNumber]
    estimatedFrameArray = try MLMultiArray(shape: timeShape, dataType: .float32)
    lpbTimeArray = try MLMultiArray(shape: timeShape, dataType: .float32)
  }

  /// Reset LSTM states to zeros (call when starting new recording)
  public func resetStates() {
    guard let states1, let states2 else { return }

    let count = states1.count
    let ptr1 = states1.dataPointer.assumingMemoryBound(to: Float.self)
    let ptr2 = states2.dataPointer.assumingMemoryBound(to: Float.self)

    for i in 0..<count {
      ptr1[i] = 0
      ptr2[i] = 0
    }

    micBuffer.removeAll(keepingCapacity: true)
    loopbackBuffer.removeAll(keepingCapacity: true)
    outputBuffer = [Float](repeating: 0, count: Self.blockLen)
    framesProcessed = 0
    totalProcessingTimeMs = 0
  }

  // MARK: - Public API

  /// Feed far-end (loopback/system audio) samples.
  /// Call this BEFORE processNearEnd for proper echo reference.
  /// - Parameter samples: Audio samples at 16kHz, Float format
  public func feedFarEnd(_ samples: [Float]) {
    loopbackBuffer.append(contentsOf: samples)

    let maxSamples = 8000  // ~500ms buffer
    if loopbackBuffer.count > maxSamples {
      loopbackBuffer.removeFirst(loopbackBuffer.count - maxSamples)
    }
  }

  /// Process near-end (microphone) samples and return echo-cancelled output.
  /// Returns the same number of samples as input (with processing delay).
  /// - Parameter samples: Microphone audio at 16kHz, Float format
  /// - Returns: Echo-cancelled audio samples
  public func processNearEnd(_ samples: [Float]) -> [Float] {
    guard isInitialized else {
      logger.warning("DTLN-aec not initialized, passing through")
      return samples
    }

    micBuffer.append(contentsOf: samples)
    var outputSamples: [Float] = []

    while micBuffer.count >= Self.blockLen && loopbackBuffer.count >= Self.blockLen {
      let frameStart = Date()

      let micFrame = Array(micBuffer.prefix(Self.blockLen))
      let loopbackFrame = Array(loopbackBuffer.prefix(Self.blockLen))

      if let processed = processFrame(mic: micFrame, loopback: loopbackFrame) {
        let frameOutput = overlapAdd(processed)
        outputSamples.append(contentsOf: frameOutput)
      } else {
        let frameOutput = overlapAdd(micFrame)
        outputSamples.append(contentsOf: frameOutput)
      }

      micBuffer.removeFirst(Self.blockShift)
      loopbackBuffer.removeFirst(Self.blockShift)

      framesProcessed += 1
      totalProcessingTimeMs += Date().timeIntervalSince(frameStart) * 1000
    }

    return outputSamples
  }

  /// Get average frame processing time in milliseconds
  public var averageFrameTimeMs: Double {
    guard framesProcessed > 0 else { return 0 }
    return totalProcessingTimeMs / Double(framesProcessed)
  }

  // MARK: - Frame Processing

  private func processFrame(mic: [Float], loopback: [Float]) -> [Float]? {
    guard let modelPart1, let modelPart2,
      let states1, let states2,
      let micMagArray, let lpbMagArray,
      let estimatedFrameArray, let lpbTimeArray
    else { return nil }

    // Part 1: Frequency Domain
    let (micMag, micPhase) = computeMagnitudeAndPhase(mic)
    let (lpbMag, _) = computeMagnitudeAndPhase(loopback)

    copyToMLArray(micMag, to: micMagArray)
    copyToMLArray(lpbMag, to: lpbMagArray)

    let part1Input: [String: Any] = [
      "mic_magnitude": micMagArray,
      "lpb_magnitude": lpbMagArray,
      "states_in": states1,
    ]

    guard let part1Provider = try? MLDictionaryFeatureProvider(dictionary: part1Input),
      let part1Output = try? modelPart1.prediction(from: part1Provider),
      let mask = part1Output.featureValue(for: "Identity")?.multiArrayValue,
      let newStates1 = part1Output.featureValue(for: "Identity_1")?.multiArrayValue
    else { return nil }

    copyStates(from: newStates1, to: states1)
    let estimatedFrame = applyMaskAndIFFT(micSamples: mic, micPhase: micPhase, mask: mask)

    // Part 2: Time Domain
    copyToMLArray(estimatedFrame, to: estimatedFrameArray)
    copyToMLArray(loopback, to: lpbTimeArray)

    let part2Input: [String: Any] = [
      "estimated_frame": estimatedFrameArray,
      "lpb_time": lpbTimeArray,
      "states_in": states2,
    ]

    guard let part2Provider = try? MLDictionaryFeatureProvider(dictionary: part2Input),
      let part2Output = try? modelPart2.prediction(from: part2Provider),
      let output = part2Output.featureValue(for: "Identity")?.multiArrayValue,
      let newStates2 = part2Output.featureValue(for: "Identity_1")?.multiArrayValue
    else { return nil }

    copyStates(from: newStates2, to: states2)
    return extractFromMLArray(output, count: Self.blockLen)
  }

  // MARK: - FFT Helpers

  private func computeMagnitudeAndPhase(_ samples: [Float]) -> (magnitude: [Float], phase: [Float])
  {
    guard let fftSetup else {
      return (
        [Float](repeating: 0, count: Self.fftBins), [Float](repeating: 0, count: Self.fftBins)
      )
    }

    fftRealBuffer = samples
    fftImagBuffer = [Float](repeating: 0, count: Self.blockLen)

    var magnitude = [Float](repeating: 0, count: Self.fftBins)
    var phase = [Float](repeating: 0, count: Self.fftBins)

    fftRealBuffer.withUnsafeMutableBufferPointer { realPtr in
      fftImagBuffer.withUnsafeMutableBufferPointer { imagPtr in
        var splitComplex = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)
        fftSetup.forward(input: splitComplex, output: &splitComplex)
        vDSP_zvabs(&splitComplex, 1, &magnitude, 1, vDSP_Length(Self.fftBins))
        for i in 0..<Self.fftBins {
          phase[i] = atan2(imagPtr[i], realPtr[i])
        }
      }
    }

    vDSP.multiply(0.5, magnitude, result: &magnitude)
    return (magnitude, phase)
  }

  private func applyMaskAndIFFT(micSamples: [Float], micPhase: [Float], mask: MLMultiArray)
    -> [Float]
  {
    guard let fftSetup else { return micSamples }

    fftRealBuffer = micSamples
    fftImagBuffer = [Float](repeating: 0, count: Self.blockLen)
    var output = [Float](repeating: 0, count: Self.blockLen)

    fftRealBuffer.withUnsafeMutableBufferPointer { realPtr in
      fftImagBuffer.withUnsafeMutableBufferPointer { imagPtr in
        var splitComplex = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)
        fftSetup.forward(input: splitComplex, output: &splitComplex)

        for i in 0..<Self.fftBins {
          let m: Float
          if mask.dataType == .float16 {
            let maskPtr = mask.dataPointer.assumingMemoryBound(to: Float16.self)
            m = Float(maskPtr[i])
          } else {
            let maskPtr = mask.dataPointer.assumingMemoryBound(to: Float.self)
            m = maskPtr[i]
          }
          realPtr[i] *= m
          imagPtr[i] *= m
        }

        for i in Self.fftBins..<Self.blockLen {
          let mirrorIdx = Self.blockLen - i
          realPtr[i] = realPtr[mirrorIdx]
          imagPtr[i] = -imagPtr[mirrorIdx]
        }

        fftSetup.inverse(input: splitComplex, output: &splitComplex)

        for i in 0..<Self.blockLen {
          output[i] = realPtr[i]
        }
      }
    }

    vDSP.multiply(2.0 / Float(Self.blockLen), output, result: &output)
    return output
  }

  // MARK: - Overlap-Add

  private func overlapAdd(_ frame: [Float]) -> [Float] {
    vDSP.add(outputBuffer, frame, result: &outputBuffer)
    let output = Array(outputBuffer.prefix(Self.blockShift))

    for i in 0..<(Self.blockLen - Self.blockShift) {
      outputBuffer[i] = outputBuffer[i + Self.blockShift]
    }
    for i in (Self.blockLen - Self.blockShift)..<Self.blockLen {
      outputBuffer[i] = 0
    }

    return output
  }

  // MARK: - MLMultiArray Helpers

  private func copyToMLArray(_ source: [Float], to array: MLMultiArray) {
    if array.dataType == .float16 {
      let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
      for i in 0..<min(source.count, array.count) {
        ptr[i] = Float16(source[i])
      }
    } else {
      let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
      for i in 0..<min(source.count, array.count) {
        ptr[i] = source[i]
      }
    }
  }

  private func extractFromMLArray(_ array: MLMultiArray, count: Int) -> [Float] {
    var result = [Float](repeating: 0, count: count)
    if array.dataType == .float16 {
      let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
      for i in 0..<min(count, array.count) {
        result[i] = Float(ptr[i])
      }
    } else {
      let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
      for i in 0..<min(count, array.count) {
        result[i] = ptr[i]
      }
    }
    return result
  }

  private func copyStates(from source: MLMultiArray, to dest: MLMultiArray) {
    let count = min(source.count, dest.count)
    if source.dataType == .float16 {
      let srcPtr = source.dataPointer.assumingMemoryBound(to: Float16.self)
      let dstPtr = dest.dataPointer.assumingMemoryBound(to: Float.self)
      for i in 0..<count {
        dstPtr[i] = Float(srcPtr[i])
      }
    } else {
      let srcPtr = source.dataPointer.assumingMemoryBound(to: Float.self)
      let dstPtr = dest.dataPointer.assumingMemoryBound(to: Float.self)
      for i in 0..<count {
        dstPtr[i] = srcPtr[i]
      }
    }
  }
}

// MARK: - Errors

/// Errors that can occur during DTLN-aec processing
public enum DTLNAecError: Error, LocalizedError {
  case modelNotFound(String)
  case initializationFailed(String)
  case inferenceFailed(String)

  public var errorDescription: String? {
    switch self {
    case .modelNotFound(let name):
      return
        "DTLN-aec model not found: \(name). Ensure the mlpackage files are included in your bundle."
    case .initializationFailed(let reason):
      return "DTLN-aec initialization failed: \(reason)"
    case .inferenceFailed(let reason):
      return "DTLN-aec inference failed: \(reason)"
    }
  }
}
