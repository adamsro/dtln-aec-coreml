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
    try processor.loadModels()
    XCTAssertTrue(processor.isInitialized)
  }

  func testModelLoadingLarge() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .large)
    try processor.loadModels()
    XCTAssertTrue(processor.isInitialized)
  }

  // MARK: - Processing Tests

  func testProcessingWithSyntheticData() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModels()

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
    try processor.loadModels()

    // Feed empty arrays
    processor.feedFarEnd([])
    let output = processor.processNearEnd([])

    // Should return empty (no samples to process)
    XCTAssertEqual(output.count, 0)
  }

  func testSmallInput() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModels()

    // Feed very small input (less than one block)
    let smallInput = [Float](repeating: 0.5, count: 64)
    processor.feedFarEnd(smallInput)
    let output = processor.processNearEnd(smallInput)

    // Not enough samples for processing yet
    XCTAssertEqual(output.count, 0)
  }

  func testMinimumBlockSize() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModels()

    // Feed exactly one block's worth of samples
    let blockLen = DTLNAecEchoProcessor.blockLen
    let input = [Float](repeating: 0.5, count: blockLen)

    processor.feedFarEnd(input)
    let output = processor.processNearEnd(input)

    // Should produce output for one frame
    XCTAssertGreaterThan(output.count, 0)
    XCTAssertEqual(output.count, DTLNAecEchoProcessor.blockShift)
  }

  func testLongInput() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModels()

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
    try processor.loadModels()

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
    try processor.loadModels()

    // Reset before processing
    processor.resetStates()

    // Should still be able to process
    let input = [Float](repeating: 0.5, count: 512)
    processor.feedFarEnd(input)
    let output = processor.processNearEnd(input)

    XCTAssertGreaterThan(output.count, 0)
  }

  // MARK: - Performance Tracking Tests

  func testPerformanceTrackingEnabled() throws {
    var config = DTLNAecConfig()
    config.enablePerformanceTracking = true
    let processor = DTLNAecEchoProcessor(config: config)
    try processor.loadModels()

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
    try processor.loadModels()

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

    try smallProcessor.loadModels()
    try largeProcessor.loadModels()

    XCTAssertTrue(smallProcessor.isInitialized)
    XCTAssertTrue(largeProcessor.isInitialized)
    XCTAssertEqual(smallProcessor.numUnits, 128)
    XCTAssertEqual(largeProcessor.numUnits, 512)
  }

  // MARK: - Async Loading Tests

  @available(macOS 10.15, iOS 13.0, *)
  func testAsyncModelLoading() async throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    XCTAssertFalse(processor.isInitialized)

    try await processor.loadModelsAsync()

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
