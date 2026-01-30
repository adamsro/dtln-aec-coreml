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
    .library(
      name: "DTLNAecCoreML",
      targets: ["DTLNAecCoreML"]
    ),
    .executable(
      name: "dtln-benchmark",
      targets: ["DTLNBenchmark"]
    ),
  ],
  targets: [
    .target(
      name: "DTLNAecCoreML",
      resources: [
        .copy("Resources/DTLN_AEC_128_Part1.mlpackage"),
        .copy("Resources/DTLN_AEC_128_Part2.mlpackage"),
        .copy("Resources/DTLN_AEC_512_Part1.mlpackage"),
        .copy("Resources/DTLN_AEC_512_Part2.mlpackage"),
      ]
    ),
    .executableTarget(
      name: "DTLNBenchmark",
      dependencies: ["DTLNAecCoreML"]
    ),
    .testTarget(
      name: "DTLNAecCoreMLTests",
      dependencies: ["DTLNAecCoreML"]
    ),
  ]
)
