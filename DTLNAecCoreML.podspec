Pod::Spec.new do |s|
  s.name             = 'DTLNAecCoreML'
  s.version          = '1.0.0'
  s.summary          = 'Neural acoustic echo cancellation for Apple platforms using CoreML'

  s.description      = <<-DESC
    A Swift wrapper for DTLN-aec (Dual-Signal Transformation LSTM Network),
    providing real-time echo cancellation on iOS and macOS using CoreML.

    Features:
    - Real-time echo cancellation (~0.8ms per 8ms frame on M1)
    - Three model sizes: 128 units (~7 MB), 256 units (~15 MB), 512 units (~40 MB)
    - Modern Swift API with async/await support
    - Configurable compute units (CPU, GPU, Neural Engine)
    - Separate subspecs for each model size to minimize app bundle size
  DESC

  s.homepage         = 'https://github.com/anthropics/dtln-aec-coreml'
  s.license          = { :type => 'MIT', :file => 'LICENSE' }
  s.author           = { 'Anthropic' => 'support@anthropic.com' }
  s.source           = { :git => 'https://github.com/anthropics/dtln-aec-coreml.git', :tag => s.version.to_s }

  s.ios.deployment_target = '16.0'
  s.osx.deployment_target = '13.0'

  s.swift_version = '5.9'
  s.frameworks = 'CoreML', 'Accelerate'

  # Core subspec - processing code without models
  s.subspec 'Core' do |core|
    core.source_files = 'Sources/DTLNAecCoreML/**/*.swift'
  end

  # 128-unit model (~7 MB)
  s.subspec 'Model128' do |m|
    m.dependency 'DTLNAecCoreML/Core'
    m.source_files = 'Sources/DTLNAec128/**/*.swift'
    m.resource_bundles = {
      'DTLNAec128' => [
        'Sources/DTLNAec128/Resources/DTLN_AEC_128_Part1.mlpackage',
        'Sources/DTLNAec128/Resources/DTLN_AEC_128_Part2.mlpackage'
      ]
    }
  end

  # 256-unit model (~15 MB)
  s.subspec 'Model256' do |m|
    m.dependency 'DTLNAecCoreML/Core'
    m.source_files = 'Sources/DTLNAec256/**/*.swift'
    m.resource_bundles = {
      'DTLNAec256' => [
        'Sources/DTLNAec256/Resources/DTLN_AEC_256_Part1.mlpackage',
        'Sources/DTLNAec256/Resources/DTLN_AEC_256_Part2.mlpackage'
      ]
    }
  end

  # 512-unit model (~40 MB)
  s.subspec 'Model512' do |m|
    m.dependency 'DTLNAecCoreML/Core'
    m.source_files = 'Sources/DTLNAec512/**/*.swift'
    m.resource_bundles = {
      'DTLNAec512' => [
        'Sources/DTLNAec512/Resources/DTLN_AEC_512_Part1.mlpackage',
        'Sources/DTLNAec512/Resources/DTLN_AEC_512_Part2.mlpackage'
      ]
    }
  end

  # Default to the small model for minimal bundle size
  s.default_subspecs = 'Core', 'Model128'

  s.test_spec 'Tests' do |test_spec|
    test_spec.source_files = 'Tests/DTLNAecCoreMLTests/**/*.swift'
    test_spec.dependency 'DTLNAecCoreML/Model128'
    test_spec.dependency 'DTLNAecCoreML/Model256'
    test_spec.dependency 'DTLNAecCoreML/Model512'
  end
end
