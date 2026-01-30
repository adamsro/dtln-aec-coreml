# Model Conversion Guide

This guide explains how to convert DTLN-aec TFLite models to CoreML format.

## Bundled Models

The package includes pre-converted models for:

- **128 units** (small) - 3.6 MB, best latency
- **512 units** (large) - 20.3 MB, best quality

These cover most use cases. Only follow this guide if you need a custom model size.

## Converting Custom Models

### Prerequisites

```bash
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
# Note: TensorFlow 2.12 is recommended for best compatibility with coremltools
pip install tensorflow==2.12.0 coremltools
```

### Step 1: Download TFLite Models

```bash
python Scripts/convert_dtln_aec_to_coreml.py --download --size 256
```

This downloads the original TFLite models from the [DTLN-aec repository](https://github.com/breizhn/DTLN-aec).

### Step 2: Analyze Model Structure (Optional)

```bash
python Scripts/convert_dtln_aec_to_coreml.py --analyze --size 256
```

This shows the model architecture and tensor shapes.

### Step 3: Convert to CoreML

```bash
python Scripts/convert_dtln_aec_to_coreml.py --convert --size 256
```

The converted models are saved to `Sources/DTLNAecCoreML/Resources/`.

### Step 4: Update Package.swift

Add the new model resources:

```swift
.target(
    name: "DTLNAecCoreML",
    resources: [
        .copy("Resources/DTLN_AEC_128_Part1.mlpackage"),
        .copy("Resources/DTLN_AEC_128_Part2.mlpackage"),
        .copy("Resources/DTLN_AEC_256_Part1.mlpackage"),  // Add new
        .copy("Resources/DTLN_AEC_256_Part2.mlpackage"),  // Add new
        .copy("Resources/DTLN_AEC_512_Part1.mlpackage"),
        .copy("Resources/DTLN_AEC_512_Part2.mlpackage"),
    ]
)
```

## Conversion Architecture

The DTLN-aec model uses a two-part architecture:

### Part 1: Frequency Domain

```
Input: mic_magnitude [1,1,257], lpb_magnitude [1,1,257], states [1,2,units,2]
  │
  ├── Log transform + InstantLayerNorm (mic)
  ├── Log transform + InstantLayerNorm (lpb)
  │
  ├── Concatenate → [1,1,514]
  │
  ├── LSTM Layer 1 (units) with state
  ├── LSTM Layer 2 (units) with state
  │
  └── Dense (257) + Sigmoid → Frequency mask

Output: mask [1,1,257], states_out [1,2,units,2]
```

### Part 2: Time Domain

```
Input: estimated_frame [1,1,512], lpb_time [1,1,512], states [1,2,units,2]
  │
  ├── Conv1D encoder (estimated) + InstantLayerNorm
  ├── Conv1D encoder (lpb) + InstantLayerNorm
  │
  ├── Concatenate → [1,1,1024]
  │
  ├── LSTM Layer 1 (units) with state
  ├── LSTM Layer 2 (units) with state
  │
  ├── Dense (512) + Sigmoid → Time mask
  ├── Multiply with encoded estimated
  │
  └── Conv1D decoder → Output frame

Output: output [1,1,512], states_out [1,2,units,2]
```

## Troubleshooting

### "Weight loading failed"

The conversion script extracts weights from TFLite by shape matching. If the TFLite model structure changes, the shape detection may fail. Check that you're using the official DTLN-aec models.

### "CoreML conversion failed"

Ensure you're using compatible versions:
- TensorFlow 2.12.x (recommended)
- coremltools 7.x or later

### Model produces incorrect output

Verify weights were loaded successfully. The script warns if random weights are used.

## References

- [DTLN-aec Paper](https://ieeexplore.ieee.org/document/9414945)
- [Original DTLN-aec Repository](https://github.com/breizhn/DTLN-aec)
- [CoreML Tools Documentation](https://coremltools.readme.io/)
