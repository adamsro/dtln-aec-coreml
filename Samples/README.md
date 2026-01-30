# Audio Samples

Test audio files for evaluating DTLN-aec echo cancellation quality.

## Directory Structure

### `aec_challenge/`

Samples from the Microsoft AEC Challenge dataset.

| File | Description |
|------|-------------|
| `farend_singletalk_lpb.wav` | Far-end (loopback) reference signal |
| `farend_singletalk_mic.wav` | Microphone recording with echo |
| `farend_singletalk_processed_python.wav` | Processed by original Python DTLN-aec |
| `farend_singletalk_processed_coreml.wav` | Processed by this CoreML implementation |
| `farend_singletalk_realworld_mic.wav` | Real speaker-to-mic recording using lpb |
| `farend_singletalk_realworld_processed.wav` | Processed realworld recording |

### `realworld/`

Real-world recordings made by playing audio through speakers and recording with microphone.

| File | Description |
|------|-------------|
| `test_lpb.wav` | Reference signal played through speakers |
| `test_mic.wav` | Microphone recording (contains echo) |
| `test_processed.wav` | Echo-cancelled output |

## Usage

### Compare Python vs CoreML output

```bash
# Listen to both
afplay Samples/aec_challenge/farend_singletalk_processed_python.wav
afplay Samples/aec_challenge/farend_singletalk_processed_coreml.wav
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
