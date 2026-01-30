// DTLN-aec CoreML Benchmark
// Measures latency of echo cancellation models on Apple Silicon

import DTLNAecCoreML
import Foundation

@main
struct Benchmark {
  static func main() async {
    print("═══════════════════════════════════════════════════════")
    print("DTLN-aec CoreML Benchmark")
    print("═══════════════════════════════════════════════════════")
    print("")
    print("Real-time requirement: <8ms per frame (128 samples at 16kHz)")
    print("")

    // Parse command line arguments
    let args = CommandLine.arguments
    let iterations = parseIterations(args) ?? 125  // Default: 1 second of audio
    let warmupFrames = 10

    // Generate synthetic test data
    let blockLen = 512
    let blockShift = 128
    let totalSamples = iterations * blockShift + blockLen

    var micSamples = [Float](repeating: 0, count: totalSamples)
    var loopbackSamples = [Float](repeating: 0, count: totalSamples)

    // Fill with synthetic audio (sine waves)
    for i in 0..<totalSamples {
      let t = Float(i) / 16000.0
      micSamples[i] = sin(2 * .pi * 440 * t) * 0.5  // 440 Hz
      loopbackSamples[i] = sin(2 * .pi * 880 * t) * 0.3  // 880 Hz (simulated echo)
    }

    var results: [BenchmarkResult] = []

    for modelSize in DTLNAecModelSize.allCases {
      print("─────────────────────────────────────────────────────")
      print("Testing \(modelSize.units)-unit model (\(modelSize.estimatedSizeMB) MB)")
      print("─────────────────────────────────────────────────────")

      let processor = DTLNAecEchoProcessor(modelSize: modelSize)

      // Measure model loading time
      let loadStart = Date()
      do {
        try processor.loadModels()
      } catch {
        print("❌ Failed to load model: \(error)")
        print("   Make sure the model files are in the bundle.")
        continue
      }
      let loadTimeMs = Date().timeIntervalSince(loadStart) * 1000
      print("Model load time: \(String(format: "%.1f", loadTimeMs))ms")

      // Warm up (first frames may be slower due to Neural Engine init)
      processor.feedFarEnd(Array(loopbackSamples.prefix(blockLen * warmupFrames)))
      for i in 0..<warmupFrames {
        let start = i * blockShift
        _ = processor.processNearEnd(Array(micSamples[start..<(start + blockShift)]))
      }

      // Reset for actual benchmark
      processor.resetStates()

      // Benchmark
      var frameTimes: [Double] = []
      processor.feedFarEnd(loopbackSamples)

      var offset = 0
      while offset + blockShift <= totalSamples - blockLen {
        let frameStart = Date()
        let chunk = Array(micSamples[offset..<(offset + blockShift)])
        _ = processor.processNearEnd(chunk)
        let frameTimeMs = Date().timeIntervalSince(frameStart) * 1000
        frameTimes.append(frameTimeMs)
        offset += blockShift
      }

      // Calculate statistics
      let avgFrameMs = frameTimes.reduce(0, +) / Double(frameTimes.count)
      let minFrameMs = frameTimes.min() ?? 0
      let maxFrameMs = frameTimes.max() ?? 0
      let realTimeRatio = avgFrameMs / 8.0

      // Percentiles
      let sorted = frameTimes.sorted()
      let p50 = sorted[sorted.count / 2]
      let p95 = sorted[Int(Double(sorted.count) * 0.95)]
      let p99 = sorted[Int(Double(sorted.count) * 0.99)]

      results.append(
        BenchmarkResult(
          size: modelSize,
          loadTimeMs: loadTimeMs,
          avgFrameMs: avgFrameMs,
          minFrameMs: minFrameMs,
          maxFrameMs: maxFrameMs,
          p50Ms: p50,
          p95Ms: p95,
          p99Ms: p99,
          realTimeRatio: realTimeRatio
        ))

      let status = realTimeRatio < 1.0 ? "✅ REAL-TIME" : "❌ TOO SLOW"
      print("Frames processed: \(frameTimes.count)")
      print("Avg frame time:   \(String(format: "%.2f", avgFrameMs))ms")
      print("Min frame time:   \(String(format: "%.2f", minFrameMs))ms")
      print("Max frame time:   \(String(format: "%.2f", maxFrameMs))ms")
      print(
        "P50/P95/P99:      \(String(format: "%.2f", p50))/\(String(format: "%.2f", p95))/\(String(format: "%.2f", p99))ms"
      )
      print("Real-time ratio:  \(String(format: "%.2f", realTimeRatio))x (\(status))")
      print("")
    }

    // Print summary
    print("═══════════════════════════════════════════════════════")
    print("BENCHMARK SUMMARY")
    print("═══════════════════════════════════════════════════════")
    print("")
    print("| Model | Params | Load    | Avg     | P99     | RT Ratio | Status |")
    print("|-------|--------|---------|---------|---------|----------|--------|")

    for result in results {
      let params: String
      switch result.size {
      case .small: params = "1.8M "
      case .medium: params = "3.9M "
      case .large: params = "10.4M"
      }
      let status = result.realTimeRatio < 1.0 ? "✅" : "❌"
      print(
        "| \(String(format: "%3d", result.size.units))   | \(params) | \(String(format: "%5.0f", result.loadTimeMs))ms | \(String(format: "%5.2f", result.avgFrameMs))ms | \(String(format: "%5.2f", result.p99Ms))ms | \(String(format: "%6.2f", result.realTimeRatio))x   | \(status)     |"
      )
    }

    print("")
    print("Real-time requirement: <8ms per frame")
    print("RT Ratio <1.0 = real-time capable")

    // JSON output option
    if args.contains("--json") {
      print("")
      print("═══════════════════════════════════════════════════════")
      print("JSON OUTPUT")
      print("═══════════════════════════════════════════════════════")
      printJSON(results)
    }
  }

  static func parseIterations(_ args: [String]) -> Int? {
    if let idx = args.firstIndex(of: "--iterations"), idx + 1 < args.count {
      return Int(args[idx + 1])
    }
    if let idx = args.firstIndex(of: "-n"), idx + 1 < args.count {
      return Int(args[idx + 1])
    }
    return nil
  }

  static func printJSON(_ results: [BenchmarkResult]) {
    print("[")
    for (i, result) in results.enumerated() {
      let comma = i < results.count - 1 ? "," : ""
      print(
        """
          {
            "model_units": \(result.size.units),
            "load_time_ms": \(String(format: "%.2f", result.loadTimeMs)),
            "avg_frame_ms": \(String(format: "%.3f", result.avgFrameMs)),
            "min_frame_ms": \(String(format: "%.3f", result.minFrameMs)),
            "max_frame_ms": \(String(format: "%.3f", result.maxFrameMs)),
            "p50_ms": \(String(format: "%.3f", result.p50Ms)),
            "p95_ms": \(String(format: "%.3f", result.p95Ms)),
            "p99_ms": \(String(format: "%.3f", result.p99Ms)),
            "realtime_ratio": \(String(format: "%.3f", result.realTimeRatio)),
            "is_realtime": \(result.realTimeRatio < 1.0)
          }\(comma)
        """)
    }
    print("]")
  }
}

struct BenchmarkResult {
  let size: DTLNAecModelSize
  let loadTimeMs: Double
  let avgFrameMs: Double
  let minFrameMs: Double
  let maxFrameMs: Double
  let p50Ms: Double
  let p95Ms: Double
  let p99Ms: Double
  let realTimeRatio: Double
}
