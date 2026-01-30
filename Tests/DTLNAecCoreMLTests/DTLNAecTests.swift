import XCTest

@testable import DTLNAecCoreML

final class DTLNAecTests: XCTestCase {

  func testModelSizeConstants() {
    XCTAssertEqual(DTLNAecModelSize.small.units, 128)
    XCTAssertEqual(DTLNAecModelSize.medium.units, 256)
    XCTAssertEqual(DTLNAecModelSize.large.units, 512)
  }

  func testProcessorInitialization() {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    XCTAssertFalse(processor.isInitialized)
    XCTAssertEqual(processor.modelSize, .small)
    XCTAssertEqual(processor.numUnits, 128)
  }

  func testAudioConstants() {
    XCTAssertEqual(DTLNAecEchoProcessor.sampleRate, 16_000)
    XCTAssertEqual(DTLNAecEchoProcessor.blockLen, 512)
    XCTAssertEqual(DTLNAecEchoProcessor.blockShift, 128)
    XCTAssertEqual(DTLNAecEchoProcessor.fftBins, 257)
  }

  func testModelLoading() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModels()
    XCTAssertTrue(processor.isInitialized)
  }

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
}
