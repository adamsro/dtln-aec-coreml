# Audio Samples

Test audio files for evaluating DTLN-aec echo cancellation quality.

## Directory Structure

### `aec_challenge/`

Samples from the Microsoft AEC Challenge dataset.

| File | Description |
|------|-------------|
| `farend_singletalk_lpb.wav` | Far-end (loopback) reference signal |
| `farend_singletalk_mic.wav` | Microphone recording with echo |
| `farend_singletalk_processed_python_128.wav` | Processed by 128-unit TFLite model (49.6 dB reduction) |
| `farend_singletalk_processed_python_128.txt` | Metadata documenting Python reference provenance |
| `farend_singletalk_processed_coreml_128.wav` | Processed by CoreML 128-unit model (47.5 dB reduction) |
| `farend_singletalk_processed_coreml_512.wav` | Processed by CoreML 512-unit model (43.8 dB reduction) |
| `farend_singletalk_realworld_mic.wav` | Real speaker-to-mic recording using lpb |
| `farend_singletalk_realworld_processed_coreml_128.wav` | Realworld processed by CoreML 128-unit model |
| `farend_singletalk_realworld_processed_coreml_512.wav` | Realworld processed by CoreML 512-unit model |

### `realworld/`

Real-world recordings made by playing audio through speakers and recording with microphone.

| File | Description |
|------|-------------|
| `test_lpb.wav` | Reference signal played through speakers |
| `test_mic.wav` | Microphone recording (contains echo) |
| `test_processed_python_128.wav` | Processed by 128-unit TFLite model (9.2 dB reduction) |
| `test_processed_python_128.txt` | Metadata documenting Python reference provenance |
| `test_processed_coreml_128.wav` | Processed by CoreML 128-unit model |
| `test_processed_coreml_512.wav` | Processed by CoreML 512-unit model |

## Usage

### Compare outputs

```bash
# Listen to Python reference vs CoreML outputs (AEC challenge sample)
afplay Samples/aec_challenge/farend_singletalk_processed_python_128.wav  # Python 128-unit
afplay Samples/aec_challenge/farend_singletalk_processed_coreml_128.wav  # CoreML 128-unit
afplay Samples/aec_challenge/farend_singletalk_processed_coreml_512.wav  # CoreML 512-unit

# Compare real-world recordings
afplay Samples/realworld/test_mic.wav                   # Original with echo
afplay Samples/realworld/test_processed_coreml_128.wav  # CoreML 128-unit
afplay Samples/realworld/test_processed_coreml_512.wav  # CoreML 512-unit
```

### Process your own files

```bash
swift run FileProcessor \
  --mic your_mic.wav \
  --ref your_reference.wav \
  --output cleaned.wav \
  --model large
```

## Audio Format

All files are:
- Sample rate: 16 kHz
- Channels: Mono
- Format: PCM (Int16 or Float32)
