import CoreML
import XCTest

@testable import DTLNAecCoreML

final class DTLNAecTests: XCTestCase {

  // MARK: - Model Size Tests

  func testModelSizeConstants() {
    XCTAssertEqual(DTLNAecModelSize.small.units, 128)
    XCTAssertEqual(DTLNAecModelSize.medium.units, 256)
    XCTAssertEqual(DTLNAecModelSize.large.units, 512)
  }

  func testModelSizeEstimatedSize() {
    XCTAssertEqual(DTLNAecModelSize.small.estimatedSizeMB, 3.6)
    XCTAssertEqual(DTLNAecModelSize.medium.estimatedSizeMB, 8.0)
    XCTAssertEqual(DTLNAecModelSize.large.estimatedSizeMB, 20.3)
  }

  func testModelSizeCaseIterable() {
    let allSizes = DTLNAecModelSize.allCases
    XCTAssertEqual(allSizes.count, 3)
    XCTAssertTrue(allSizes.contains(.small))
    XCTAssertTrue(allSizes.contains(.medium))
    XCTAssertTrue(allSizes.contains(.large))
  }

  // MARK: - Configuration Tests

  func testDefaultConfig() {
    let config = DTLNAecConfig()
    XCTAssertEqual(config.modelSize, .small)
    XCTAssertTrue(config.enablePerformanceTracking)
  }

  func testCustomConfig() {
    var config = DTLNAecConfig()
    config.modelSize = .large
    config.enablePerformanceTracking = false

    XCTAssertEqual(config.modelSize, .large)
    XCTAssertFalse(config.enablePerformanceTracking)
  }

  func testConfigInitWithParameters() {
    let config = DTLNAecConfig(
      modelSize: .large,
      computeUnits: .cpuOnly,
      enablePerformanceTracking: false
    )

    XCTAssertEqual(config.modelSize, .large)
    XCTAssertEqual(config.computeUnits, .cpuOnly)
    XCTAssertFalse(config.enablePerformanceTracking)
  }

  // MARK: - Processor Initialization Tests

  func testProcessorInitialization() {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    XCTAssertFalse(processor.isInitialized)
    XCTAssertEqual(processor.modelSize, .small)
    XCTAssertEqual(processor.numUnits, 128)
  }

  func testProcessorInitializationWithConfig() {
    let config = DTLNAecConfig(modelSize: .large)
    let processor = DTLNAecEchoProcessor(config: config)

    XCTAssertFalse(processor.isInitialized)
    XCTAssertEqual(processor.modelSize, .large)
    XCTAssertEqual(processor.numUnits, 512)
    XCTAssertEqual(processor.config.modelSize, .large)
  }

  func testAudioConstants() {
    XCTAssertEqual(DTLNAecEchoProcessor.sampleRate, 16_000)
    XCTAssertEqual(DTLNAecEchoProcessor.blockLen, 512)
    XCTAssertEqual(DTLNAecEchoProcessor.blockShift, 128)
    XCTAssertEqual(DTLNAecEchoProcessor.fftBins, 257)
  }

  // MARK: - Model Loading Tests

  func testModelLoadingSmall() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()
    XCTAssertTrue(processor.isInitialized)
  }

  func testModelLoadingLarge() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .large)
    try processor.loadModelsFromPackage()
    XCTAssertTrue(processor.isInitialized)
  }

  // MARK: - Processing Tests

  func testProcessingWithSyntheticData() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Generate synthetic test data
    let numSamples = 16000  // 1 second at 16kHz
    var micSamples = [Float](repeating: 0, count: numSamples)
    var loopbackSamples = [Float](repeating: 0, count: numSamples)

    for i in 0..<numSamples {
      let t = Float(i) / 16000.0
      micSamples[i] = sin(2 * .pi * 440 * t) * 0.5
      loopbackSamples[i] = sin(2 * .pi * 880 * t) * 0.3
    }

    // Feed far-end first
    processor.feedFarEnd(loopbackSamples)

    // Process in chunks
    let chunkSize = 128
    var outputSamples: [Float] = []

    for start in stride(from: 0, to: numSamples, by: chunkSize) {
      let end = min(start + chunkSize, numSamples)
      let chunk = Array(micSamples[start..<end])
      let processed = processor.processNearEnd(chunk)
      outputSamples.append(contentsOf: processed)
    }

    // Should have produced some output
    XCTAssertGreaterThan(outputSamples.count, 0)
    XCTAssertGreaterThan(processor.averageFrameTimeMs, 0)

    print("Processed \(outputSamples.count) samples")
    print("Average frame time: \(String(format: "%.2f", processor.averageFrameTimeMs))ms")
  }

  // MARK: - Edge Case Tests

  func testEmptyInput() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Feed empty arrays
    processor.feedFarEnd([])
    let output = processor.processNearEnd([])

    // Should return empty (no samples to process)
    XCTAssertEqual(output.count, 0)
  }

  func testSmallInput() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Feed very small input (less than one block)
    let smallInput = [Float](repeating: 0.5, count: 64)
    processor.feedFarEnd(smallInput)
    let output = processor.processNearEnd(smallInput)

    // Not enough samples for processing yet
    XCTAssertEqual(output.count, 0)
  }

  func testMinimumBlockSize() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Feed one block shift (128 samples) - the minimum processable chunk
    let blockShift = DTLNAecEchoProcessor.blockShift
    let input = [Float](repeating: 0.5, count: blockShift)

    processor.feedFarEnd(input)
    let output = processor.processNearEnd(input)

    // Should produce output for one frame (128 samples)
    XCTAssertGreaterThan(output.count, 0)
    XCTAssertEqual(output.count, blockShift)
  }

  func testLongInput() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Process 10 seconds of audio
    let numSamples = 160000  // 10 seconds at 16kHz
    let longInput = [Float](repeating: 0.1, count: numSamples)

    processor.feedFarEnd(longInput)
    let output = processor.processNearEnd(longInput)

    // Should handle large inputs
    XCTAssertGreaterThan(output.count, 0)
    print("Processed \(numSamples) samples, got \(output.count) output samples")
  }

  // MARK: - State Reset Tests

  func testResetStates() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Process some data
    let input = [Float](repeating: 0.5, count: 1024)
    processor.feedFarEnd(input)
    _ = processor.processNearEnd(input)

    XCTAssertGreaterThan(processor.averageFrameTimeMs, 0)

    // Reset states
    processor.resetStates()

    // Average time should be reset
    XCTAssertEqual(processor.averageFrameTimeMs, 0)

    // Processor should still be initialized
    XCTAssertTrue(processor.isInitialized)
  }

  func testResetDoesNotAffectModel() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Reset before processing
    processor.resetStates()

    // Should still be able to process
    let input = [Float](repeating: 0.5, count: 512)
    processor.feedFarEnd(input)
    let output = processor.processNearEnd(input)

    XCTAssertGreaterThan(output.count, 0)
  }

  // MARK: - Flush Tests

  func testFlushReturnsBufferedSamples() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Process some data to fill buffers
    let input = [Float](repeating: 0.3, count: 200)
    processor.feedFarEnd(input)
    let output = processor.processNearEnd(input)

    // 200 samples -> 128 processed, 72 pending
    XCTAssertEqual(output.count, 128)

    // Before flush, should have 72 pending + 384 overlap tail = 456
    XCTAssertEqual(processor.pendingSampleCount, 72 + 384)

    // Flush should return all remaining samples
    let flushed = processor.flush()

    // Should get: 128 (from processing padded pending) + 384 (overlap tail) = 512
    XCTAssertEqual(flushed.count, 512)

    // After flush, should have 0 pending + 384 tail (but outputBuffer is zeroed)
    XCTAssertEqual(processor.pendingSampleCount, 384)
  }

  func testFlushWithNoPendingSamples() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Process exactly blockShift samples so no pending remain
    let input = [Float](repeating: 0.3, count: 128)
    processor.feedFarEnd(input)
    let output = processor.processNearEnd(input)

    XCTAssertEqual(output.count, 128)
    XCTAssertEqual(processor.pendingSampleCount, 384)  // Only overlap tail

    // Flush should return just the overlap-add tail
    let flushed = processor.flush()
    XCTAssertEqual(flushed.count, 384)
  }

  func testFlushWithoutProcessing() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Just feed some samples without triggering processing
    let input = [Float](repeating: 0.3, count: 64)
    processor.feedFarEnd(input)
    _ = processor.processNearEnd(input)

    // Should have 64 pending samples
    XCTAssertEqual(processor.pendingSampleCount, 64 + 384)

    let flushed = processor.flush()

    // 128 from padded pending frame + 384 overlap tail
    XCTAssertEqual(flushed.count, 512)
  }

  func testFlushPreservesLSTMStates() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Generate a distinctive signal pattern
    var input = [Float](repeating: 0, count: 512)
    for i in 0..<512 {
      input[i] = Float(sin(Double(i) * 0.1)) * 0.5
    }

    // Process to warm up LSTM states
    processor.feedFarEnd(input)
    let _ = processor.processNearEnd(input)

    // Flush (should preserve LSTM states)
    _ = processor.flush()

    // Process more data after flush - LSTM states should be preserved
    // (the output would be different if states were reset)
    processor.feedFarEnd(input)
    let outputAfterFlush = processor.processNearEnd(input)

    // Create a fresh processor and compare
    let freshProcessor = DTLNAecEchoProcessor(modelSize: .small)
    try freshProcessor.loadModelsFromPackage()
    freshProcessor.feedFarEnd(input)
    let freshOutput = freshProcessor.processNearEnd(input)

    // Outputs should differ because LSTM states were preserved after flush
    // vs fresh states in the new processor
    var differenceFound = false
    for i in 0..<min(outputAfterFlush.count, freshOutput.count) {
      if abs(outputAfterFlush[i] - freshOutput[i]) > 0.001 {
        differenceFound = true
        break
      }
    }
    XCTAssertTrue(differenceFound, "LSTM states should be preserved after flush, causing different output than fresh processor")
  }

  func testFlushClearsPendingBuffers() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Add pending samples
    let input = [Float](repeating: 0.3, count: 64)
    processor.feedFarEnd(input)
    _ = processor.processNearEnd(input)

    XCTAssertEqual(processor.pendingSampleCount, 64 + 384)

    // Flush
    _ = processor.flush()

    // Pending should be cleared (only overlap tail remaining, which is zeroed)
    XCTAssertEqual(processor.pendingSampleCount, 384)

    // Calling flush again should only return zeros from zeroed outputBuffer
    let secondFlush = processor.flush()
    XCTAssertEqual(secondFlush.count, 384)

    // Should be all zeros
    let nonZeroCount = secondFlush.filter { abs($0) > 0.0001 }.count
    XCTAssertEqual(nonZeroCount, 0, "Second flush should return zeros from cleared buffer")
  }

  func testFlushUninitializedProcessor() {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    // Don't load models

    // When uninitialized, processNearEnd passes through directly without buffering
    let input = [Float](repeating: 0.5, count: 64)
    processor.feedFarEnd(input)
    let output = processor.processNearEnd(input)

    // Passthrough mode: input returned directly
    XCTAssertEqual(output.count, 64)
    XCTAssertEqual(output, input)

    // Flush on uninitialized processor returns empty (nothing buffered)
    let flushed = processor.flush()
    XCTAssertEqual(flushed.count, 0)
  }

  func testPendingSampleCount() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Initially should have just overlap tail
    XCTAssertEqual(processor.pendingSampleCount, 384)

    // Add 50 samples (not enough to process)
    let input50 = [Float](repeating: 0.3, count: 50)
    processor.feedFarEnd(input50)
    _ = processor.processNearEnd(input50)
    XCTAssertEqual(processor.pendingSampleCount, 50 + 384)

    // Add 100 more samples (150 total pending, should process one frame)
    let input100 = [Float](repeating: 0.3, count: 100)
    processor.feedFarEnd(input100)
    _ = processor.processNearEnd(input100)
    // 150 - 128 = 22 pending
    XCTAssertEqual(processor.pendingSampleCount, 22 + 384)
  }

  // MARK: - Performance Tracking Tests

  func testPerformanceTrackingEnabled() throws {
    var config = DTLNAecConfig()
    config.enablePerformanceTracking = true
    let processor = DTLNAecEchoProcessor(config: config)
    try processor.loadModelsFromPackage()

    // Process some data
    let input = [Float](repeating: 0.5, count: 1024)
    processor.feedFarEnd(input)
    _ = processor.processNearEnd(input)

    // Should have tracked time
    XCTAssertGreaterThan(processor.averageFrameTimeMs, 0)
  }

  func testPerformanceTrackingDisabled() throws {
    var config = DTLNAecConfig()
    config.enablePerformanceTracking = false
    let processor = DTLNAecEchoProcessor(config: config)
    try processor.loadModelsFromPackage()

    // Process some data
    let input = [Float](repeating: 0.5, count: 1024)
    processor.feedFarEnd(input)
    _ = processor.processNearEnd(input)

    // Should not have tracked time
    XCTAssertEqual(processor.averageFrameTimeMs, 0)
  }

  // MARK: - Passthrough Test (uninitialized)

  func testUninitializedPassthrough() {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    // Don't load models

    let input = [Float](repeating: 0.5, count: 256)
    let output = processor.processNearEnd(input)

    // Should pass through the input when not initialized
    XCTAssertEqual(output.count, input.count)
    XCTAssertEqual(output, input)
  }

  // MARK: - Multiple Model Sizes

  func testMultipleProcessors() throws {
    // Create processors for different sizes
    let smallProcessor = DTLNAecEchoProcessor(modelSize: .small)
    let largeProcessor = DTLNAecEchoProcessor(modelSize: .large)

    try smallProcessor.loadModelsFromPackage()
    try largeProcessor.loadModelsFromPackage()

    XCTAssertTrue(smallProcessor.isInitialized)
    XCTAssertTrue(largeProcessor.isInitialized)
    XCTAssertEqual(smallProcessor.numUnits, 128)
    XCTAssertEqual(largeProcessor.numUnits, 512)
  }

  /// Test that 512-unit model works after float16 state fix
  func testLargeModelCPUOnly() throws {
    let config = DTLNAecConfig(modelSize: .large, computeUnits: .cpuOnly)
    let processor = DTLNAecEchoProcessor(config: config)
    try processor.loadModelsFromPackage()

    let numSamples = 16000  // 1 second
    var farEnd = [Float](repeating: 0, count: numSamples)
    var nearEnd = [Float](repeating: 0, count: numSamples)

    for i in 0..<numSamples {
      let t = Float(i) / 16000.0
      farEnd[i] = 0.3 * sin(2 * .pi * 440 * t)
      nearEnd[i] = 0.2 * sin(2 * .pi * 440 * t)
    }

    // Process in chunks
    let chunkSize = 128
    var output: [Float] = []
    var firstNaN: Int? = nil

    for (idx, start) in stride(from: 0, to: numSamples, by: chunkSize).enumerated() {
      let end = min(start + chunkSize, numSamples)
      processor.feedFarEnd(Array(farEnd[start..<end]))
      let processed = processor.processNearEnd(Array(nearEnd[start..<end]))

      if firstNaN == nil && processed.contains(where: { $0.isNaN }) {
        firstNaN = idx
      }
      output.append(contentsOf: processed)
    }

    print("Large model (CPU-only): \(output.count) samples")
    let hasNaN = output.contains { $0.isNaN }
    print("Has NaN: \(hasNaN), first at frame: \(firstNaN ?? -1)")
    XCTAssertFalse(hasNaN, "Large model should not produce NaN with float16 states")
  }

  // MARK: - Async Loading Tests

  @available(macOS 10.15, iOS 13.0, *)
  func testAsyncModelLoading() async throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    XCTAssertFalse(processor.isInitialized)

    try await processor.loadModelsFromPackageAsync()

    XCTAssertTrue(processor.isInitialized)
  }

  // MARK: - Error Tests

  func testModelNotFoundError() {
    // Test error description
    let error = DTLNAecError.modelNotFound("TestModel")
    XCTAssertNotNil(error.errorDescription)
    XCTAssertTrue(error.errorDescription?.contains("TestModel") ?? false)
  }

  func testInitializationFailedError() {
    let error = DTLNAecError.initializationFailed("Test reason")
    XCTAssertNotNil(error.errorDescription)
    XCTAssertTrue(error.errorDescription?.contains("Test reason") ?? false)
  }

  func testInferenceFailedError() {
    let error = DTLNAecError.inferenceFailed("Test reason")
    XCTAssertNotNil(error.errorDescription)
    XCTAssertTrue(error.errorDescription?.contains("Test reason") ?? false)
  }

  // MARK: - Sendable Conformance Tests

  func testModelSizeSendable() {
    // Verify DTLNAecModelSize is Sendable by using it across threads
    let size: DTLNAecModelSize = .small
    let expectation = XCTestExpectation(description: "Sendable test")

    Task.detached {
      // This should compile without warnings because DTLNAecModelSize is Sendable
      let localSize = size
      XCTAssertEqual(localSize.units, 128)
      expectation.fulfill()
    }

    wait(for: [expectation], timeout: 1.0)
  }

  func testConfigSendable() {
    let config = DTLNAecConfig(modelSize: .large)
    let expectation = XCTestExpectation(description: "Config Sendable test")

    Task.detached {
      let localConfig = config
      XCTAssertEqual(localConfig.modelSize, .large)
      expectation.fulfill()
    }

    wait(for: [expectation], timeout: 1.0)
  }
}
