import Foundation

/// Provides 256-unit (medium) DTLN-aec model resources (~15 MB).
///
/// Usage:
/// ```swift
/// import DTLNAecCoreML
/// import DTLNAec256
///
/// let processor = DTLNAecEchoProcessor(modelSize: .medium)
/// try processor.loadModels(from: DTLNAec256.bundle)
/// ```
public enum DTLNAec256 {
  /// The bundle containing the 256-unit model files.
  public static let bundle: Bundle = .module
}
