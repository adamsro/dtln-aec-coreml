import DTLNAec128
import DTLNAec256
import DTLNAec512
import DTLNAecCoreML
import Foundation

/// Get the bundle containing models for the given size.
func modelBundle(for size: DTLNAecModelSize) -> Bundle {
  switch size {
  case .small: return DTLNAec128.bundle
  case .medium: return DTLNAec256.bundle
  case .large: return DTLNAec512.bundle
  }
}

extension DTLNAecEchoProcessor {
  /// Convenience method to load models using the correct bundle for the processor's model size.
  func loadModelsFromPackage() throws {
    try loadModels(from: modelBundle(for: modelSize))
  }

  /// Async convenience method to load models using the correct bundle for the processor's model size.
  @available(macOS 10.15, iOS 13.0, *)
  func loadModelsFromPackageAsync() async throws {
    try await loadModelsAsync(from: modelBundle(for: modelSize))
  }
}
