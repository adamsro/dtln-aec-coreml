Pod::Spec.new do |s|
  s.name             = 'DTLNAecCoreML'
  s.version          = '1.0.0'
  s.summary          = 'Neural acoustic echo cancellation for Apple platforms using CoreML'

  s.description      = <<-DESC
    A Swift wrapper for DTLN-aec (Dual-Signal Transformation LSTM Network),
    providing real-time echo cancellation on iOS and macOS using CoreML.

    Features:
    - Real-time echo cancellation (~0.8ms per 8ms frame on M1)
    - Two model sizes: 128 units (small) and 512 units (large)
    - Modern Swift API with async/await support
    - Configurable compute units (CPU, GPU, Neural Engine)
  DESC

  s.homepage         = 'https://github.com/anthropics/dtln-aec-coreml'
  s.license          = { :type => 'MIT', :file => 'LICENSE' }
  s.author           = { 'Anthropic' => 'support@anthropic.com' }
  s.source           = { :git => 'https://github.com/anthropics/dtln-aec-coreml.git', :tag => s.version.to_s }

  s.ios.deployment_target = '16.0'
  s.osx.deployment_target = '13.0'

  s.swift_version = '5.9'

  s.source_files = 'Sources/DTLNAecCoreML/**/*.swift'

  s.resources = [
    'Sources/DTLNAecCoreML/Resources/*.mlpackage'
  ]

  s.resource_bundles = {
    'DTLNAecCoreML' => [
      'Sources/DTLNAecCoreML/Resources/DTLN_AEC_128_Part1.mlpackage',
      'Sources/DTLNAecCoreML/Resources/DTLN_AEC_128_Part2.mlpackage',
      'Sources/DTLNAecCoreML/Resources/DTLN_AEC_256_Part1.mlpackage',
      'Sources/DTLNAecCoreML/Resources/DTLN_AEC_256_Part2.mlpackage',
      'Sources/DTLNAecCoreML/Resources/DTLN_AEC_512_Part1.mlpackage',
      'Sources/DTLNAecCoreML/Resources/DTLN_AEC_512_Part2.mlpackage'
    ]
  }

  s.frameworks = 'CoreML', 'Accelerate'

  s.test_spec 'Tests' do |test_spec|
    test_spec.source_files = 'Tests/DTLNAecCoreMLTests/**/*.swift'
  end
end
