import Foundation

/// Provides 512-unit (large) DTLN-aec model resources (~40 MB).
///
/// Usage:
/// ```swift
/// import DTLNAecCoreML
/// import DTLNAec512
///
/// let processor = DTLNAecEchoProcessor(modelSize: .large)
/// try processor.loadModels(from: DTLNAec512.bundle)
/// ```
public enum DTLNAec512 {
  /// The bundle containing the 512-unit model files.
  public static let bundle: Bundle = .module
}
