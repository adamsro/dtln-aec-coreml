// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "DTLNAecCoreML",
  platforms: [
    .macOS(.v13),
    .iOS(.v16),
  ],
  products: [
    // Core library (no models - import a model package below)
    .library(
      name: "DTLNAecCoreML",
      targets: ["DTLNAecCoreML"]
    ),
    // Model packages - import only what you need
    .library(
      name: "DTLNAec128",
      targets: ["DTLNAec128"]
    ),
    .library(
      name: "DTLNAec256",
      targets: ["DTLNAec256"]
    ),
    .library(
      name: "DTLNAec512",
      targets: ["DTLNAec512"]
    ),
    .executable(
      name: "dtln-benchmark",
      targets: ["DTLNBenchmark"]
    ),
  ],
  targets: [
    // Core library - processing code only, no models
    .target(
      name: "DTLNAecCoreML"
    ),
    // 128-unit model (~7 MB)
    .target(
      name: "DTLNAec128",
      dependencies: ["DTLNAecCoreML"],
      resources: [
        .copy("Resources/DTLN_AEC_128_Part1.mlpackage"),
        .copy("Resources/DTLN_AEC_128_Part2.mlpackage"),
      ]
    ),
    // 256-unit model (~15 MB)
    .target(
      name: "DTLNAec256",
      dependencies: ["DTLNAecCoreML"],
      resources: [
        .copy("Resources/DTLN_AEC_256_Part1.mlpackage"),
        .copy("Resources/DTLN_AEC_256_Part2.mlpackage"),
      ]
    ),
    // 512-unit model (~40 MB)
    .target(
      name: "DTLNAec512",
      dependencies: ["DTLNAecCoreML"],
      resources: [
        .copy("Resources/DTLN_AEC_512_Part1.mlpackage"),
        .copy("Resources/DTLN_AEC_512_Part2.mlpackage"),
      ]
    ),
    .executableTarget(
      name: "DTLNBenchmark",
      dependencies: ["DTLNAecCoreML", "DTLNAec128", "DTLNAec256", "DTLNAec512"]
    ),
    .executableTarget(
      name: "FileProcessor",
      dependencies: ["DTLNAecCoreML", "DTLNAec128", "DTLNAec256", "DTLNAec512"],
      path: "Examples/FileProcessor"
    ),
    .testTarget(
      name: "DTLNAecCoreMLTests",
      dependencies: ["DTLNAecCoreML", "DTLNAec128", "DTLNAec256", "DTLNAec512"]
    ),
  ]
)
