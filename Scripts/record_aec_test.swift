#!/usr/bin/env swift
//
// record_aec_test.swift
// Records a real AEC test by playing audio through speakers while recording from microphone.
//
// Usage:
//   swift Scripts/record_aec_test.swift <farend.wav> <output_nearend.wav>
//
// Example:
//   swift Scripts/record_aec_test.swift Tests/DTLNAecCoreMLTests/Fixtures/farend.wav /tmp/recorded_nearend.wav
//
// The script will:
//   1. Play farend.wav through your speakers
//   2. Record from your microphone at the same time
//   3. Save the recording as output_nearend.wav
//
// You can then test AEC with:
//   swift run FileProcessor --mic /tmp/recorded_nearend.wav --ref Fixtures/farend.wav --output /tmp/cleaned.wav

import AVFoundation
import Foundation

// MARK: - WAV File Handling

struct WAVHeader {
    var riffID: UInt32 = 0x4646_4952  // "RIFF"
    var fileSize: UInt32 = 0
    var waveID: UInt32 = 0x4556_4157  // "WAVE"
    var fmtID: UInt32 = 0x2074_6D66   // "fmt "
    var fmtSize: UInt32 = 16
    var audioFormat: UInt16 = 3       // IEEE float
    var numChannels: UInt16 = 1
    var sampleRate: UInt32 = 16000
    var byteRate: UInt32 = 64000
    var blockAlign: UInt16 = 4
    var bitsPerSample: UInt16 = 32
    var dataID: UInt32 = 0x6174_6164  // "data"
    var dataSize: UInt32 = 0
}

func writeWAV(samples: [Float], to url: URL, sampleRate: UInt32 = 16000) throws {
    var header = WAVHeader()
    header.sampleRate = sampleRate
    header.byteRate = sampleRate * 4
    header.dataSize = UInt32(samples.count * 4)
    header.fileSize = header.dataSize + 36

    var data = Data()

    withUnsafeBytes(of: header.riffID) { data.append(contentsOf: $0) }
    withUnsafeBytes(of: header.fileSize) { data.append(contentsOf: $0) }
    withUnsafeBytes(of: header.waveID) { data.append(contentsOf: $0) }
    withUnsafeBytes(of: header.fmtID) { data.append(contentsOf: $0) }
    withUnsafeBytes(of: header.fmtSize) { data.append(contentsOf: $0) }
    withUnsafeBytes(of: header.audioFormat) { data.append(contentsOf: $0) }
    withUnsafeBytes(of: header.numChannels) { data.append(contentsOf: $0) }
    withUnsafeBytes(of: header.sampleRate) { data.append(contentsOf: $0) }
    withUnsafeBytes(of: header.byteRate) { data.append(contentsOf: $0) }
    withUnsafeBytes(of: header.blockAlign) { data.append(contentsOf: $0) }
    withUnsafeBytes(of: header.bitsPerSample) { data.append(contentsOf: $0) }
    withUnsafeBytes(of: header.dataID) { data.append(contentsOf: $0) }
    withUnsafeBytes(of: header.dataSize) { data.append(contentsOf: $0) }

    for sample in samples {
        withUnsafeBytes(of: sample) { data.append(contentsOf: $0) }
    }

    try data.write(to: url)
}

// MARK: - Audio Recorder

class AudioRecorder: NSObject {
    private var audioEngine: AVAudioEngine!
    private var recordedSamples: [Float] = []
    private var isRecording = false
    private let targetSampleRate: Double = 16000

    func startRecording() throws {
        audioEngine = AVAudioEngine()
        let inputNode = audioEngine.inputNode

        let inputFormat = inputNode.outputFormat(forBus: 0)
        print("  Microphone format: \(inputFormat.sampleRate)Hz, \(inputFormat.channelCount) channels")

        // Install tap to capture audio
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: inputFormat) { [weak self] buffer, _ in
            guard let self = self, self.isRecording else { return }

            let channelData = buffer.floatChannelData![0]
            let frameCount = Int(buffer.frameLength)

            // Simple downsampling if needed
            let ratio = inputFormat.sampleRate / self.targetSampleRate
            if ratio > 1 {
                let step = Int(ratio)
                for i in stride(from: 0, to: frameCount, by: step) {
                    self.recordedSamples.append(channelData[i])
                }
            } else {
                for i in 0..<frameCount {
                    self.recordedSamples.append(channelData[i])
                }
            }
        }

        try audioEngine.start()
        isRecording = true
    }

    func stopRecording() -> [Float] {
        isRecording = false
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
        return recordedSamples
    }
}

// MARK: - Main

func main() {
    let args = CommandLine.arguments

    guard args.count >= 3 else {
        print("""
            AEC Test Recorder

            Records from microphone while playing audio through speakers.

            Usage: swift \(args[0]) <farend.wav> <output_nearend.wav>

            Example:
              swift Scripts/record_aec_test.swift \\
                Tests/DTLNAecCoreMLTests/Fixtures/farend.wav \\
                /tmp/recorded_nearend.wav

            Then test AEC:
              # Add FileProcessor to Package.swift first, then:
              swift run FileProcessor \\
                --mic /tmp/recorded_nearend.wav \\
                --ref Tests/DTLNAecCoreMLTests/Fixtures/farend.wav \\
                --output /tmp/cleaned.wav
            """)
        exit(1)
    }

    let farendPath = args[1]
    let outputPath = args[2]

    let farendURL = URL(fileURLWithPath: farendPath)
    let outputURL = URL(fileURLWithPath: outputPath)

    guard FileManager.default.fileExists(atPath: farendPath) else {
        print("Error: Far-end file not found: \(farendPath)")
        exit(1)
    }

    print("AEC Test Recorder")
    print("=================")
    print("Far-end file: \(farendPath)")
    print("Output file: \(outputPath)")
    print()

    // Set up audio player
    var audioPlayer: AVAudioPlayer!
    do {
        audioPlayer = try AVAudioPlayer(contentsOf: farendURL)
        audioPlayer.prepareToPlay()
        print("Audio duration: \(String(format: "%.2f", audioPlayer.duration))s")
    } catch {
        print("Error loading audio file: \(error)")
        exit(1)
    }

    // Set up recorder
    let recorder = AudioRecorder()

    print()
    print("Starting in 2 seconds...")
    print("  - Audio will play through speakers")
    print("  - Microphone will record simultaneously")
    print()
    Thread.sleep(forTimeInterval: 2.0)

    // Start recording
    do {
        try recorder.startRecording()
        print("Recording started...")
    } catch {
        print("Error starting recorder: \(error)")
        exit(1)
    }

    // Small delay to ensure recording is ready
    Thread.sleep(forTimeInterval: 0.1)

    // Play audio
    print("Playing audio...")
    audioPlayer.play()

    // Wait for playback to finish
    while audioPlayer.isPlaying {
        Thread.sleep(forTimeInterval: 0.1)
    }

    // Small buffer after playback
    Thread.sleep(forTimeInterval: 0.2)

    // Stop recording
    let samples = recorder.stopRecording()
    print("Recording stopped.")
    print()

    // Save recording
    print("Saving \(samples.count) samples (\(String(format: "%.2f", Double(samples.count) / 16000))s)...")
    do {
        try writeWAV(samples: samples, to: outputURL)
        print("Saved to: \(outputPath)")
    } catch {
        print("Error saving recording: \(error)")
        exit(1)
    }

    print()
    print("Done! You can now test AEC with the recorded audio.")
}

main()
