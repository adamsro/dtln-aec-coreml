#!/usr/bin/env python3
"""
DTLN-aec to CoreML Conversion Script

Converts DTLN-aec TFLite models to CoreML format.

Architecture (reverse-engineered from TFLite analysis):

Model 1 (Frequency Domain):
- Inputs: mic_mag [1,1,257], lpb_mag [1,1,257], states [1,2,units,2]
- InstantLayerNorm on log(mic_mag + 1e-7)
- InstantLayerNorm on log(lpb_mag + 1e-7)
- Concatenate → [1,1,514]
- LSTM layer 1 (units): kernel [4*units,514], recurrent [4*units,units], bias [4*units]
- LSTM layer 2 (units): kernel [4*units,units], recurrent [4*units,units], bias [4*units]
- Dense [257,units] + bias [257] → sigmoid → mask [1,1,257]
- Output: mask, states_out

Model 2 (Time Domain):
- Inputs: estimated [1,1,512], lpb_time [1,1,512], states [1,2,units,2]
- Conv1D encoder (512 → 512) on estimated
- InstantLayerNorm on encoded
- Conv1D encoder (512 → 512) on lpb_time
- InstantLayerNorm on encoded lpb
- Concatenate → [1,1,1024]
- LSTM layer 1 (units): kernel [4*units,1024], recurrent [4*units,units], bias [4*units]
- LSTM layer 2 (units): kernel [4*units,units], recurrent [4*units,units], bias [4*units]
- Dense [512,units] + bias [512] → sigmoid → mask
- Multiply with encoded estimated
- Conv1D decoder (512 → 512) → output [1,1,512]
- Output: output, states_out

Usage:
    # Download TFLite models first
    python Scripts/convert_dtln_aec_to_coreml.py --download --size 256

    # Then convert to CoreML
    source .venv/bin/activate
    pip install tensorflow coremltools
    python Scripts/convert_dtln_aec_to_coreml.py --convert --size 256
"""

import argparse
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TFLITE_DIR = PROJECT_ROOT / "Resources" / "DTLN_AEC"  # Source TFLite models
OUTPUT_DIR = PROJECT_ROOT / "mimicscribe" / "Resources"  # CoreML output


def download_tflite_models(model_size: str):
    """Download TFLite models from DTLN-aec GitHub repo."""
    TFLITE_DIR.mkdir(parents=True, exist_ok=True)

    base_url = "https://github.com/breizhn/DTLN-aec/raw/main/pretrained_models"

    for part in [1, 2]:
        filename = f"dtln_aec_{model_size}_{part}.tflite"
        url = f"{base_url}/{filename}"
        dest = TFLITE_DIR / filename

        if dest.exists():
            print(f"✅ {filename} already exists")
            continue

        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"✅ Downloaded: {dest}")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            return False

    return True


def build_and_convert_models(model_size: str = "128"):
    """Build Keras models matching DTLN-aec architecture and convert to CoreML."""

    try:
        import tensorflow as tf
        import numpy as np
        import coremltools as ct
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, LSTM, Multiply, Conv1D, \
            Activation, Layer, Concatenate, Lambda
    except ImportError as e:
        print(f"Error: Missing dependency: {e}")
        print("Run: pip install tensorflow coremltools")
        sys.exit(1)

    print(f"Converting DTLN-aec {model_size} to CoreML...")

    num_units = int(model_size)
    num_layers = 2
    block_len = 512
    fft_bins = 257
    encoder_size = 512  # Based on TFLite analysis

    # Custom InstantLayerNormalization (matching DTLN)
    class InstantLayerNormalization(Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.epsilon = 1e-7

        def build(self, input_shape):
            self.gamma = self.add_weight(
                shape=input_shape[-1:],
                initializer='ones',
                trainable=True,
                name='gamma'
            )
            self.beta = self.add_weight(
                shape=input_shape[-1:],
                initializer='zeros',
                trainable=True,
                name='beta'
            )

        def call(self, inputs):
            mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
            variance = tf.reduce_mean(tf.square(inputs - mean), axis=-1, keepdims=True)
            std = tf.sqrt(variance + self.epsilon)
            outputs = (inputs - mean) / std
            return outputs * self.gamma + self.beta

    # =============================
    # Build Model 1 (Frequency Domain)
    # =============================
    print("\n=== Building Model 1 (Frequency Domain) ===")

    mic_mag = Input(shape=(1, fft_bins), name='mic_magnitude')
    lpb_mag = Input(shape=(1, fft_bins), name='lpb_magnitude')
    states_in_1 = Input(shape=(num_layers, num_units, 2), name='states_in')

    # Log magnitude + InstantLayerNorm (matching DTLN-aec)
    mic_log = Lambda(lambda x: tf.math.log(x + 1e-7), name='mic_log')(mic_mag)
    mic_norm = InstantLayerNormalization(name='mic_norm')(mic_log)

    lpb_log = Lambda(lambda x: tf.math.log(x + 1e-7), name='lpb_log')(lpb_mag)
    lpb_norm = InstantLayerNormalization(name='lpb_norm')(lpb_log)

    # Concatenate normalized features
    concat_1 = Concatenate(axis=-1, name='concat_features')([mic_norm, lpb_norm])  # [1,1,514]

    # LSTM layers with explicit state handling
    x = concat_1
    states_h = []
    states_c = []

    for i in range(num_layers):
        h_in = Lambda(lambda s, idx=i: s[:, idx, :, 0], name=f'h_in_{i}')(states_in_1)
        c_in = Lambda(lambda s, idx=i: s[:, idx, :, 1], name=f'c_in_{i}')(states_in_1)

        x, h_out, c_out = LSTM(
            num_units,
            return_sequences=True,
            return_state=True,
            name=f'lstm_1_{i}'
        )(x, initial_state=[h_in, c_in])

        states_h.append(h_out)
        states_c.append(c_out)

    # Dense + sigmoid for mask
    mask = Dense(fft_bins, name='dense_mask_1')(x)
    mask = Activation('sigmoid', name='mask_1')(mask)

    # Pack output states
    states_h_stack = Lambda(lambda h: tf.stack(h, axis=1), name='stack_h_1')(states_h)
    states_c_stack = Lambda(lambda c: tf.stack(c, axis=1), name='stack_c_1')(states_c)
    states_out_1 = Lambda(
        lambda sc: tf.stack(sc, axis=-1),
        name='states_out_1'
    )([states_h_stack, states_c_stack])

    model_1 = Model(
        inputs=[mic_mag, states_in_1, lpb_mag],
        outputs=[mask, states_out_1],
        name='DTLN_AEC_Part1'
    )
    print(model_1.summary())

    # =============================
    # Build Model 2 (Time Domain)
    # =============================
    print("\n=== Building Model 2 (Time Domain) ===")

    estimated_frame = Input(shape=(1, block_len), name='estimated_frame')
    lpb_time = Input(shape=(1, block_len), name='lpb_time')
    states_in_2 = Input(shape=(num_layers, num_units, 2), name='states_in')

    # Encode estimated frame
    encoded_est = Conv1D(encoder_size, 1, use_bias=False, name='encoder_est')(estimated_frame)
    encoded_est_norm = InstantLayerNormalization(name='enc_est_norm')(encoded_est)

    # Encode loopback
    encoded_lpb = Conv1D(encoder_size, 1, use_bias=False, name='encoder_lpb')(lpb_time)
    encoded_lpb_norm = InstantLayerNormalization(name='enc_lpb_norm')(encoded_lpb)

    # Concatenate
    concat_2 = Concatenate(axis=-1, name='concat_encoded')([encoded_est_norm, encoded_lpb_norm])  # [1,1,1024]

    # LSTM layers
    x2 = concat_2
    states_h_2 = []
    states_c_2 = []

    for i in range(num_layers):
        h_in = Lambda(lambda s, idx=i: s[:, idx, :, 0], name=f'h_in_2_{i}')(states_in_2)
        c_in = Lambda(lambda s, idx=i: s[:, idx, :, 1], name=f'c_in_2_{i}')(states_in_2)

        x2, h_out, c_out = LSTM(
            num_units,
            return_sequences=True,
            return_state=True,
            name=f'lstm_2_{i}'
        )(x2, initial_state=[h_in, c_in])

        states_h_2.append(h_out)
        states_c_2.append(c_out)

    # Dense + sigmoid for mask
    mask_2 = Dense(encoder_size, name='dense_mask_2')(x2)
    mask_2 = Activation('sigmoid', name='mask_2')(mask_2)

    # Apply mask to encoded (not normalized) features
    masked = Multiply(name='apply_mask')([encoded_est, mask_2])

    # Decode back to time domain
    decoded = Conv1D(block_len, 1, use_bias=False, name='decoder')(masked)

    # Pack output states
    states_h_stack_2 = Lambda(lambda h: tf.stack(h, axis=1), name='stack_h_2')(states_h_2)
    states_c_stack_2 = Lambda(lambda c: tf.stack(c, axis=1), name='stack_c_2')(states_c_2)
    states_out_2 = Lambda(
        lambda sc: tf.stack(sc, axis=-1),
        name='states_out_2'
    )([states_h_stack_2, states_c_stack_2])

    model_2 = Model(
        inputs=[estimated_frame, states_in_2, lpb_time],
        outputs=[decoded, states_out_2],
        name='DTLN_AEC_Part2'
    )
    print(model_2.summary())

    # =============================
    # Load weights from TFLite
    # =============================
    print("\n=== Loading weights from TFLite ===")

    weights_loaded = load_tflite_weights(model_1, model_2, model_size)

    if not weights_loaded:
        print("\n⚠️  Using random weights - model will not produce correct results!")
        print("Weight transfer from TFLite requires matching layer names.")

    # =============================
    # Convert to CoreML directly from Keras model
    # =============================
    print("\n=== Converting to CoreML ===")
    sys.stdout.flush()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        print("Converting Part 1 to CoreML...")
        sys.stdout.flush()

        # Let coremltools auto-detect input names to avoid mismatches
        mlmodel_1 = ct.convert(
            model_1,
            minimum_deployment_target=ct.target.macOS13,
            convert_to='mlprogram',
        )

        output_path_1 = OUTPUT_DIR / f"DTLN_AEC_{model_size}_Part1.mlpackage"
        mlmodel_1.save(str(output_path_1))
        print(f"✅ Saved: {output_path_1}")
        sys.stdout.flush()

        print("\nConverting Part 2 to CoreML...")
        sys.stdout.flush()

        mlmodel_2 = ct.convert(
            model_2,
            minimum_deployment_target=ct.target.macOS13,
            convert_to='mlprogram',
        )

        output_path_2 = OUTPUT_DIR / f"DTLN_AEC_{model_size}_Part2.mlpackage"
        mlmodel_2.save(str(output_path_2))
        print(f"✅ Saved: {output_path_2}")

        print("\n✅ CoreML conversion complete!")
        print(f"\nTo use this model, update DTLNEchoProcessor.swift:")
        print(f"  1. Change numUnits from 128 to {model_size}")
        print(f"  2. Update model bundle names to DTLN_AEC_{model_size}_Part1/Part2")
        print(f"  3. Add the .mlpackage files to Package.swift resources")

    except Exception as e:
        print(f"\n❌ CoreML conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def load_tflite_weights(model_1, model_2, model_size: str):
    """Load weights from TFLite models into Keras models using shape-based detection."""
    import tensorflow as tf
    import numpy as np

    num_units = int(model_size)
    lstm_gates = 4 * num_units  # i, f, c, o gates

    # Load TFLite interpreters
    tflite_1 = TFLITE_DIR / f"dtln_aec_{model_size}_1.tflite"
    tflite_2 = TFLITE_DIR / f"dtln_aec_{model_size}_2.tflite"

    if not tflite_1.exists() or not tflite_2.exists():
        print(f"❌ TFLite models not found. Run: python {__file__} --download --size {model_size}")
        return False

    interpreter_1 = tf.lite.Interpreter(model_path=str(tflite_1))
    interpreter_1.allocate_tensors()

    interpreter_2 = tf.lite.Interpreter(model_path=str(tflite_2))
    interpreter_2.allocate_tensors()

    def find_tensors_by_shape(interpreter, target_shape):
        """Find all tensors matching a specific shape."""
        matches = []
        for detail in interpreter.get_tensor_details():
            try:
                tensor = interpreter.get_tensor(detail['index'])
                if tuple(tensor.shape) == tuple(target_shape):
                    matches.append((tensor, detail['index'], detail['name']))
            except ValueError:
                pass
        return matches

    def find_tensor_by_shape(interpreter, target_shape, name_hint=""):
        """Find first tensor matching shape, optionally filtered by name hint."""
        matches = find_tensors_by_shape(interpreter, target_shape)
        if name_hint:
            for tensor, idx, name in matches:
                if name_hint.lower() in name.lower():
                    return tensor, idx, name
        return matches[0] if matches else (None, None, None)

    print(f"\n=== Loading weights for {model_size}-unit model ===")

    try:
        # =====================
        # MODEL 1 WEIGHTS
        # =====================
        print("Part 1 weights:")

        # InstantLayerNorm gamma/beta - shape (257,) for fft_bins
        norm_params = find_tensors_by_shape(interpreter_1, (257,))
        if len(norm_params) >= 4:
            # First 2 are mic_norm (gamma, beta), next 2 are lpb_norm
            mic_norm_gamma, _, _ = norm_params[0]
            mic_norm_beta, _, _ = norm_params[1]
            lpb_norm_gamma, _, _ = norm_params[2]
            lpb_norm_beta, _, _ = norm_params[3]
            print(f"  mic_norm: gamma={mic_norm_gamma.shape}, beta={mic_norm_beta.shape}")
            print(f"  lpb_norm: gamma={lpb_norm_gamma.shape}, beta={lpb_norm_beta.shape}")
        else:
            print(f"  ⚠️  Could not find all LayerNorm params, found {len(norm_params)}")
            return False

        # LSTM 1 (first LSTM) - input size 514 (257+257 concatenated)
        # kernel: [4*units, 514], recurrent: [4*units, units], bias: [4*units]
        lstm1_kernel, _, _ = find_tensor_by_shape(interpreter_1, (lstm_gates, 514))
        lstm1_recurrent, _, _ = find_tensor_by_shape(interpreter_1, (lstm_gates, num_units))
        lstm1_bias_matches = find_tensors_by_shape(interpreter_1, (lstm_gates,))
        lstm1_bias = lstm1_bias_matches[0][0] if lstm1_bias_matches else None

        if lstm1_kernel is None:
            print(f"  ❌ Could not find LSTM1 kernel with shape ({lstm_gates}, 514)")
            return False
        print(f"  lstm_1_0: kernel={lstm1_kernel.shape}, recurrent={lstm1_recurrent.shape}, bias={lstm1_bias.shape}")

        # LSTM 2 (second LSTM) - input size = units
        # kernel: [4*units, units], recurrent: [4*units, units], bias: [4*units]
        lstm2_kernels = find_tensors_by_shape(interpreter_1, (lstm_gates, num_units))
        # Filter out the one we already used as lstm1_recurrent
        lstm2_kernel = None
        lstm2_recurrent = None
        for tensor, idx, name in lstm2_kernels:
            if not np.array_equal(tensor, lstm1_recurrent):
                if lstm2_kernel is None:
                    lstm2_kernel = tensor
                elif lstm2_recurrent is None:
                    lstm2_recurrent = tensor
        lstm2_bias = lstm1_bias_matches[1][0] if len(lstm1_bias_matches) > 1 else None

        if lstm2_kernel is None or lstm2_recurrent is None:
            print(f"  ❌ Could not find LSTM2 weights")
            return False
        print(f"  lstm_1_1: kernel={lstm2_kernel.shape}, recurrent={lstm2_recurrent.shape}, bias={lstm2_bias.shape}")

        # Dense mask layer - kernel: [257, units], bias: [257]
        dense1_kernel, _, _ = find_tensor_by_shape(interpreter_1, (257, num_units))
        dense1_bias_matches = find_tensors_by_shape(interpreter_1, (257,))
        # Find the one that's NOT a LayerNorm param (different values)
        dense1_bias = None
        for tensor, idx, name in dense1_bias_matches:
            # LayerNorm beta is typically near-zero initialized
            if np.abs(tensor).max() > 0.1:  # Dense bias has learned values
                dense1_bias = tensor
                break
        if dense1_bias is None and dense1_bias_matches:
            dense1_bias = dense1_bias_matches[-1][0]  # Fallback to last one

        print(f"  dense_mask_1: kernel={dense1_kernel.shape}, bias={dense1_bias.shape}")

        # Set weights for Model 1
        model_1.get_layer('mic_norm').set_weights([mic_norm_gamma, mic_norm_beta])
        model_1.get_layer('lpb_norm').set_weights([lpb_norm_gamma, lpb_norm_beta])

        # LSTM layers - Keras expects [kernel, recurrent_kernel, bias]
        # TFLite stores as [output_size, input_size], Keras expects [input_size, output_size]
        model_1.get_layer('lstm_1_0').set_weights([
            lstm1_kernel.T,
            lstm1_recurrent.T,
            lstm1_bias
        ])
        model_1.get_layer('lstm_1_1').set_weights([
            lstm2_kernel.T,
            lstm2_recurrent.T,
            lstm2_bias
        ])

        # Dense layer
        model_1.get_layer('dense_mask_1').set_weights([
            dense1_kernel.T,
            dense1_bias
        ])

        print("✅ Part 1 weights loaded")

        # =====================
        # MODEL 2 WEIGHTS
        # =====================
        print("\nPart 2 weights:")

        # Conv1D encoders - TFLite stores as (out, 1, 1, in) or (out, in)
        # Some models have both formats; prefer 2D (512, 512) which is more common in larger models
        conv_matches_4d = find_tensors_by_shape(interpreter_2, (512, 1, 1, 512))
        conv_matches_2d = find_tensors_by_shape(interpreter_2, (512, 512))

        # Use 2D matches if we have enough, otherwise try 4D
        if len(conv_matches_2d) >= 3:
            conv_matches = conv_matches_2d
        elif len(conv_matches_4d) >= 3:
            conv_matches = conv_matches_4d
        else:
            # Combine both if neither has enough alone
            conv_matches = conv_matches_4d + conv_matches_2d

        if len(conv_matches) >= 3:
            conv_est = np.squeeze(conv_matches[0][0])
            conv_lpb = np.squeeze(conv_matches[1][0])
            conv_decoder = np.squeeze(conv_matches[2][0])
            print(f"  encoder_est: {conv_est.shape}")
            print(f"  encoder_lpb: {conv_lpb.shape}")
            print(f"  decoder: {conv_decoder.shape}")
        else:
            print(f"  ❌ Could not find Conv1D weights, found {len(conv_matches)}")
            return False

        # InstantLayerNorm for encoded features - shape (512,)
        enc_norm_params = find_tensors_by_shape(interpreter_2, (512,))
        # Filter out bias tensors (Dense bias has larger values typically)
        norm_512 = [(t, i, n) for t, i, n in enc_norm_params if np.abs(t).mean() < 1.0]
        if len(norm_512) >= 4:
            enc_est_gamma = norm_512[0][0]
            enc_est_beta = norm_512[1][0]
            enc_lpb_gamma = norm_512[2][0]
            enc_lpb_beta = norm_512[3][0]
            print(f"  enc_est_norm: gamma={enc_est_gamma.shape}, beta={enc_est_beta.shape}")
            print(f"  enc_lpb_norm: gamma={enc_lpb_gamma.shape}, beta={enc_lpb_beta.shape}")
        else:
            print(f"  ⚠️  Could not find all encoder LayerNorm params")
            return False

        # LSTM 1 (first LSTM in Part 2) - input size 1024 (512+512 concatenated)
        lstm2_1_kernel, _, _ = find_tensor_by_shape(interpreter_2, (lstm_gates, 1024))
        lstm2_1_recurrent, _, _ = find_tensor_by_shape(interpreter_2, (lstm_gates, num_units))
        lstm2_bias_matches = find_tensors_by_shape(interpreter_2, (lstm_gates,))
        lstm2_1_bias = lstm2_bias_matches[0][0] if lstm2_bias_matches else None

        if lstm2_1_kernel is None:
            print(f"  ❌ Could not find Part2 LSTM1 kernel with shape ({lstm_gates}, 1024)")
            return False
        print(f"  lstm_2_0: kernel={lstm2_1_kernel.shape}, recurrent={lstm2_1_recurrent.shape}, bias={lstm2_1_bias.shape}")

        # LSTM 2 (second LSTM in Part 2) - input size = units
        lstm2_2_kernels = find_tensors_by_shape(interpreter_2, (lstm_gates, num_units))
        lstm2_2_kernel = None
        lstm2_2_recurrent = None
        for tensor, idx, name in lstm2_2_kernels:
            if not np.array_equal(tensor, lstm2_1_recurrent):
                if lstm2_2_kernel is None:
                    lstm2_2_kernel = tensor
                elif lstm2_2_recurrent is None:
                    lstm2_2_recurrent = tensor
        lstm2_2_bias = lstm2_bias_matches[1][0] if len(lstm2_bias_matches) > 1 else None

        print(f"  lstm_2_1: kernel={lstm2_2_kernel.shape}, recurrent={lstm2_2_recurrent.shape}, bias={lstm2_2_bias.shape}")

        # Dense mask layer - kernel: [512, units], bias: [512]
        dense2_kernel, _, _ = find_tensor_by_shape(interpreter_2, (512, num_units))
        # Find dense bias (not LayerNorm)
        dense2_bias = None
        for t, i, n in enc_norm_params:
            if np.abs(t).mean() > 0.5:  # Dense bias has larger learned values
                dense2_bias = t
                break
        if dense2_bias is None:
            # Fallback - find by exclusion
            all_512 = find_tensors_by_shape(interpreter_2, (512,))
            for t, i, n in all_512:
                if not any(np.array_equal(t, x[0]) for x in norm_512[:4]):
                    dense2_bias = t
                    break

        print(f"  dense_mask_2: kernel={dense2_kernel.shape}, bias={dense2_bias.shape if dense2_bias is not None else 'None'}")

        # Set weights for Model 2
        # Conv1D layers - Keras Conv1D expects (kernel_size, input_dim, filters) = (1, 512, 512)
        model_2.get_layer('encoder_est').set_weights([
            conv_est.T.reshape(1, 512, 512)
        ])
        model_2.get_layer('encoder_lpb').set_weights([
            conv_lpb.T.reshape(1, 512, 512)
        ])

        model_2.get_layer('enc_est_norm').set_weights([enc_est_gamma, enc_est_beta])
        model_2.get_layer('enc_lpb_norm').set_weights([enc_lpb_gamma, enc_lpb_beta])

        model_2.get_layer('lstm_2_0').set_weights([
            lstm2_1_kernel.T,
            lstm2_1_recurrent.T,
            lstm2_1_bias
        ])
        model_2.get_layer('lstm_2_1').set_weights([
            lstm2_2_kernel.T,
            lstm2_2_recurrent.T,
            lstm2_2_bias
        ])

        model_2.get_layer('dense_mask_2').set_weights([
            dense2_kernel.T,
            dense2_bias
        ])

        model_2.get_layer('decoder').set_weights([
            conv_decoder.T.reshape(1, 512, 512)
        ])

        print("✅ Part 2 weights loaded")
        return True

    except Exception as e:
        print(f"❌ Weight loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_tflite_model(model_size: str = "128"):
    """Analyze TFLite model structure."""
    import tensorflow as tf

    for part in [1, 2]:
        model_path = TFLITE_DIR / f"dtln_aec_{model_size}_{part}.tflite"
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            print(f"   Run: python {__file__} --download --size {model_size}")
            return

        print(f"\n{'='*60}")
        print(f"Part {part}: {model_path.name}")
        print('='*60)

        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("\nInputs:")
        for d in input_details:
            print(f"  [{d['index']}] {d['name']}: {d['shape']}")

        print("\nOutputs:")
        for d in output_details:
            print(f"  [{d['index']}] {d['name']}: {d['shape']}")

        # Find weight tensors
        print("\nWeight tensors (2D with LSTM-like shapes):")
        for detail in interpreter.get_tensor_details():
            try:
                tensor = interpreter.get_tensor(detail['index'])
                if len(tensor.shape) == 2 and tensor.shape[0] >= 128:
                    print(f"  [{detail['index']}] {detail['name'][:60]}: {tensor.shape}")
            except ValueError:
                pass

        print("\n1D tensors (bias/norm params):")
        for detail in interpreter.get_tensor_details():
            try:
                tensor = interpreter.get_tensor(detail['index'])
                if len(tensor.shape) == 1 and tensor.shape[0] >= 128:
                    print(f"  [{detail['index']}] {detail['name'][:60]}: {tensor.shape}")
            except ValueError:
                pass


def main():
    parser = argparse.ArgumentParser(description="DTLN-aec to CoreML Converter")
    parser.add_argument("--size", default="128", choices=["128", "256", "512"],
                        help="Model size (LSTM units): 128=1.8M params, 256=3.9M, 512=10.4M")
    parser.add_argument("--download", action="store_true", help="Download TFLite models from GitHub")
    parser.add_argument("--analyze", action="store_true", help="Analyze TFLite structure")
    parser.add_argument("--convert", action="store_true", help="Convert to CoreML")

    args = parser.parse_args()

    if args.download:
        download_tflite_models(args.size)
    elif args.analyze:
        analyze_tflite_model(args.size)
    elif args.convert:
        build_and_convert_models(args.size)
    else:
        parser.print_help()
        print("\nExamples:")
        print(f"  python {__file__} --download --size 256  # Download 256-unit TFLite models")
        print(f"  python {__file__} --analyze --size 256   # Analyze model structure")
        print(f"  python {__file__} --convert --size 256   # Convert to CoreML")


if __name__ == "__main__":
    main()
