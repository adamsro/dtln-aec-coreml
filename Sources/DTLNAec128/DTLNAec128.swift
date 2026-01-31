import Foundation

/// Provides 128-unit (small) DTLN-aec model resources (~7 MB).
///
/// Usage:
/// ```swift
/// import DTLNAecCoreML
/// import DTLNAec128
///
/// let processor = DTLNAecEchoProcessor(modelSize: .small)
/// try processor.loadModels(from: DTLNAec128.bundle)
/// ```
public enum DTLNAec128 {
  /// The bundle containing the 128-unit model files.
  public static let bundle: Bundle = .module
}
