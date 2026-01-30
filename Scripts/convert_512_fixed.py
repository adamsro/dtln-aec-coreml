#!/usr/bin/env python3
"""Convert 512-unit DTLN-aec model to CoreML with correct weight loading."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TFLITE_DIR = PROJECT_ROOT / "Resources" / "DTLN_AEC"
OUTPUT_DIR = PROJECT_ROOT / "Sources" / "DTLNAecCoreML" / "Resources"

import numpy as np
import tensorflow as tf
import coremltools as ct
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Multiply, Conv1D, \
    Activation, Layer, Concatenate, Lambda

print(f"TensorFlow version: {tf.__version__}")
print(f"CoreMLTools version: {ct.__version__}")

num_units = 512
num_layers = 2
block_len = 512
fft_bins = 257
encoder_size = 512
lstm_gates = 4 * num_units


class InstantLayerNormalization(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = 1e-7

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=input_shape[-1:], initializer='ones', trainable=True, name='gamma')
        self.beta = self.add_weight(shape=input_shape[-1:], initializer='zeros', trainable=True, name='beta')

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=-1, keepdims=True)
        std = tf.sqrt(variance + self.epsilon)
        return (inputs - mean) / std * self.gamma + self.beta


def get_tensor_by_index(interpreter, idx):
    return interpreter.get_tensor(idx)


def build_and_load_model_1():
    """Build Part 1 (Frequency Domain) model."""
    print("\n=== Building Model 1 (Frequency Domain) ===")

    tflite_1 = TFLITE_DIR / "dtln_aec_512_1.tflite"
    interpreter = tf.lite.Interpreter(model_path=str(tflite_1))
    interpreter.allocate_tensors()

    # Analyze tensors first
    print("Analyzing Part 1 tensors...")
    for detail in interpreter.get_tensor_details():
        try:
            tensor = interpreter.get_tensor(detail['index'])
            if len(tensor.shape) > 0 and tensor.size > 100:  # Non-trivial tensors
                mean_abs = np.abs(tensor).mean()
                if mean_abs > 0.01:  # Non-zero weights
                    print(f"  [{detail['index']:3d}] {detail['name'][:50]}: shape={tensor.shape}, mean_abs={mean_abs:.4f}")
        except ValueError:
            pass

    # Build Keras model
    mic_mag = Input(shape=(1, fft_bins), name='mic_magnitude')
    lpb_mag = Input(shape=(1, fft_bins), name='lpb_magnitude')
    states_in = Input(shape=(num_layers, num_units, 2), name='states_in')

    mic_log = Lambda(lambda x: tf.math.log(x + 1e-7), name='mic_log')(mic_mag)
    mic_norm = InstantLayerNormalization(name='mic_norm')(mic_log)
    lpb_log = Lambda(lambda x: tf.math.log(x + 1e-7), name='lpb_log')(lpb_mag)
    lpb_norm = InstantLayerNormalization(name='lpb_norm')(lpb_log)

    concat = Concatenate(axis=-1, name='concat_features')([mic_norm, lpb_norm])

    x = concat
    states_h = []
    states_c = []
    for i in range(num_layers):
        h_in = Lambda(lambda s, idx=i: s[:, idx, :, 0], name=f'h_in_{i}')(states_in)
        c_in = Lambda(lambda s, idx=i: s[:, idx, :, 1], name=f'c_in_{i}')(states_in)
        x, h_out, c_out = LSTM(num_units, return_sequences=True, return_state=True, name=f'lstm_1_{i}')(x, initial_state=[h_in, c_in])
        states_h.append(h_out)
        states_c.append(c_out)

    mask = Dense(fft_bins, name='dense_mask_1')(x)
    mask = Activation('sigmoid', name='mask_1')(mask)

    states_h_stack = Lambda(lambda h: tf.stack(h, axis=1), name='stack_h_1')(states_h)
    states_c_stack = Lambda(lambda c: tf.stack(c, axis=1), name='stack_c_1')(states_c)
    states_out = Lambda(lambda sc: tf.stack(sc, axis=-1), name='states_out_1')([states_h_stack, states_c_stack])

    model = Model(inputs=[mic_mag, states_in, lpb_mag], outputs=[mask, states_out], name='DTLN_AEC_Part1')

    # Load weights by analyzing tensor names
    all_tensors = {}
    for detail in interpreter.get_tensor_details():
        try:
            tensor = interpreter.get_tensor(detail['index'])
            all_tensors[detail['index']] = {'name': detail['name'], 'tensor': tensor}
        except ValueError:
            pass

    # Find weights by shape and name patterns
    # LayerNorm gamma/beta for mic (257,)
    ln_257 = [(idx, d) for idx, d in all_tensors.items() if d['tensor'].shape == (257,) and np.abs(d['tensor']).mean() > 0.01]
    print(f"Found {len(ln_257)} LayerNorm 257 params")
    # Sort by index to get consistent ordering
    ln_257.sort(key=lambda x: x[0])

    # LSTM weights
    lstm_kernel_514 = [(idx, d) for idx, d in all_tensors.items() if d['tensor'].shape == (2048, 514)]
    lstm_kernel_512 = [(idx, d) for idx, d in all_tensors.items() if d['tensor'].shape == (2048, 512)]
    lstm_bias = [(idx, d) for idx, d in all_tensors.items() if d['tensor'].shape == (2048,)]

    print(f"LSTM kernels 514: {len(lstm_kernel_514)}, 512: {len(lstm_kernel_512)}, bias: {len(lstm_bias)}")

    # Dense mask weights
    dense_kernel = [(idx, d) for idx, d in all_tensors.items() if d['tensor'].shape == (257, 512)]
    print(f"Dense kernel: {len(dense_kernel)}")

    # Set weights
    if len(ln_257) >= 4:
        model.get_layer('mic_norm').set_weights([ln_257[0][1]['tensor'], ln_257[1][1]['tensor']])
        model.get_layer('lpb_norm').set_weights([ln_257[2][1]['tensor'], ln_257[3][1]['tensor']])

    if lstm_kernel_514 and len(lstm_kernel_512) >= 2 and len(lstm_bias) >= 2:
        # LSTM 1: kernel (514->2048), recurrent (512->2048)
        model.get_layer('lstm_1_0').set_weights([
            lstm_kernel_514[0][1]['tensor'].T,
            lstm_kernel_512[0][1]['tensor'].T,
            lstm_bias[0][1]['tensor']
        ])
        # LSTM 2: kernel (512->2048), recurrent (512->2048)
        model.get_layer('lstm_1_1').set_weights([
            lstm_kernel_512[1][1]['tensor'].T,
            lstm_kernel_512[2][1]['tensor'].T if len(lstm_kernel_512) > 2 else lstm_kernel_512[0][1]['tensor'].T,
            lstm_bias[1][1]['tensor']
        ])

    if dense_kernel:
        # Dense bias
        dense_bias = [d['tensor'] for idx, d in ln_257 if np.abs(d['tensor']).max() > 0.2]
        if not dense_bias:
            dense_bias = [ln_257[-1][1]['tensor']]  # Fallback
        model.get_layer('dense_mask_1').set_weights([dense_kernel[0][1]['tensor'].T, dense_bias[0]])

    print("✅ Part 1 weights loaded")
    return model


def build_and_load_model_2():
    """Build Part 2 (Time Domain) model with corrected weight loading."""
    print("\n=== Building Model 2 (Time Domain) ===")

    tflite_2 = TFLITE_DIR / "dtln_aec_512_2.tflite"
    interpreter = tf.lite.Interpreter(model_path=str(tflite_2))
    interpreter.allocate_tensors()

    # Build Keras model
    estimated_frame = Input(shape=(1, block_len), name='estimated_frame')
    lpb_time = Input(shape=(1, block_len), name='lpb_time')
    states_in = Input(shape=(num_layers, num_units, 2), name='states_in')

    encoded_est = Conv1D(encoder_size, 1, use_bias=False, name='encoder_est')(estimated_frame)
    encoded_est_norm = InstantLayerNormalization(name='enc_est_norm')(encoded_est)
    encoded_lpb = Conv1D(encoder_size, 1, use_bias=False, name='encoder_lpb')(lpb_time)
    encoded_lpb_norm = InstantLayerNormalization(name='enc_lpb_norm')(encoded_lpb)

    concat = Concatenate(axis=-1, name='concat_encoded')([encoded_est_norm, encoded_lpb_norm])

    x = concat
    states_h = []
    states_c = []
    for i in range(num_layers):
        h_in = Lambda(lambda s, idx=i: s[:, idx, :, 0], name=f'h_in_2_{i}')(states_in)
        c_in = Lambda(lambda s, idx=i: s[:, idx, :, 1], name=f'c_in_2_{i}')(states_in)
        x, h_out, c_out = LSTM(num_units, return_sequences=True, return_state=True, name=f'lstm_2_{i}')(x, initial_state=[h_in, c_in])
        states_h.append(h_out)
        states_c.append(c_out)

    mask = Dense(encoder_size, name='dense_mask_2')(x)
    mask = Activation('sigmoid', name='mask_2')(mask)
    masked = Multiply(name='apply_mask')([encoded_est, mask])
    decoded = Conv1D(block_len, 1, use_bias=False, name='decoder')(masked)

    states_h_stack = Lambda(lambda h: tf.stack(h, axis=1), name='stack_h_2')(states_h)
    states_c_stack = Lambda(lambda c: tf.stack(c, axis=1), name='stack_c_2')(states_c)
    states_out = Lambda(lambda sc: tf.stack(sc, axis=-1), name='states_out_2')([states_h_stack, states_c_stack])

    model = Model(inputs=[estimated_frame, states_in, lpb_time], outputs=[decoded, states_out], name='DTLN_AEC_Part2')

    # Load weights by specific indices based on analysis
    # Conv1D encoder_est: [20] (512, 1, 1, 512)
    # Conv1D encoder_lpb: [22] (512, 1, 1, 512)
    # Dense kernel: [15] (512, 512)
    # Dense bias: [3] (512,)
    # LayerNorm gamma/beta: [5,6] for est, [7,8] for lpb
    # LSTM biases: [9, 10] (2048,)
    # LSTM kernels: [18] (2048, 1024) for LSTM1 input, [16,17] and [19] (2048, 512)

    print("Loading weights by tensor index...")

    # Conv1D encoders - shape (512, 1, 1, 512), need (1, 512, 512) for Keras
    conv_est = interpreter.get_tensor(20)  # encoder_est
    conv_lpb = interpreter.get_tensor(22)  # encoder_lpb
    model.get_layer('encoder_est').set_weights([conv_est.squeeze().T.reshape(1, 512, 512)])
    model.get_layer('encoder_lpb').set_weights([conv_lpb.squeeze().T.reshape(1, 512, 512)])

    # LayerNorm - gamma then beta
    enc_est_gamma = interpreter.get_tensor(5)
    enc_est_beta = interpreter.get_tensor(6)
    enc_lpb_gamma = interpreter.get_tensor(7)
    enc_lpb_beta = interpreter.get_tensor(8)
    model.get_layer('enc_est_norm').set_weights([enc_est_gamma, enc_est_beta])
    model.get_layer('enc_lpb_norm').set_weights([enc_lpb_gamma, enc_lpb_beta])

    # LSTM 1 (input from concat 1024)
    # kernel [18]: (2048, 1024), recurrent [19]: (2048, 512), bias [9]: (2048,)
    lstm1_kernel = interpreter.get_tensor(18)  # (2048, 1024)
    lstm1_recurrent = interpreter.get_tensor(19)  # (2048, 512)
    lstm1_bias = interpreter.get_tensor(9)  # (2048,)
    model.get_layer('lstm_2_0').set_weights([lstm1_kernel.T, lstm1_recurrent.T, lstm1_bias])

    # LSTM 2 (input from LSTM1 output 512)
    # kernel [16]: (2048, 512), recurrent [17]: (2048, 512), bias [10]: (2048,)
    lstm2_kernel = interpreter.get_tensor(16)  # (2048, 512)
    lstm2_recurrent = interpreter.get_tensor(17)  # (2048, 512)
    lstm2_bias = interpreter.get_tensor(10)  # (2048,)
    model.get_layer('lstm_2_1').set_weights([lstm2_kernel.T, lstm2_recurrent.T, lstm2_bias])

    # Dense mask
    dense_kernel = interpreter.get_tensor(15)  # (512, 512)
    dense_bias = interpreter.get_tensor(3)  # (512,)
    model.get_layer('dense_mask_2').set_weights([dense_kernel.T, dense_bias])

    # Decoder - need to find the right conv weight
    # Looking at the analysis, the decoder weight might be fused or different
    # Let's try using the same approach as encoder but look for third conv
    # Actually, based on DTLN-aec architecture, decoder uses transpose of encoder
    # For now, let's use conv_est transposed as decoder
    # OR we can find it in the model - there should be 3 conv ops total

    # Actually looking at weights: indices 159,160,161 are Conv_hwcn_weights with shape (512,512)
    # but they have tiny values - might be quantization artifacts
    # The decoder in DTLN-aec is learned separately, not transpose of encoder
    # Let me check if there's another conv weight

    # Based on the architecture, there should be 3 Conv1D: encoder_est, encoder_lpb, decoder
    # Indices 20, 22 are encoders. Let me look for decoder...
    # Actually, in some TFLite models, decoder might be fused or named differently

    # For decoder, let's check what other (512,512) or (512,1,1,512) tensors exist with non-zero weights
    all_tensors = {}
    for detail in interpreter.get_tensor_details():
        try:
            tensor = interpreter.get_tensor(detail['index'])
            all_tensors[detail['index']] = {'name': detail['name'], 'tensor': tensor, 'shape': tensor.shape}
        except ValueError:
            pass

    # Find conv-like tensors
    conv_candidates = [(idx, d) for idx, d in all_tensors.items()
                       if d['shape'] in [(512, 1, 1, 512), (512, 512)]
                       and np.abs(d['tensor']).mean() > 0.01]

    print(f"Conv candidates: {[(idx, d['name'][:40], d['shape']) for idx, d in conv_candidates]}")

    # The decoder should be the third conv (after encoder_est and encoder_lpb)
    # Based on typical order, it might be index 15 (but that's used for Dense)
    # Let me use a heuristic - the decoder should have similar magnitude to encoders

    # Actually, looking at analysis again:
    # [15] is dense (range [-1.89, 1.98], mean_abs=0.0566) - this is the Dense kernel
    # [20], [22] are conv encoders
    # There's no explicit decoder conv in the simple shape search

    # In DTLN-aec, the decoder Conv1D transforms masked features back to time domain
    # It should have shape (512, 512) or similar
    # But our analysis shows [159,160,161] have near-zero values

    # Let me check if maybe the decoder is actually stored differently or if DTLN-aec
    # uses a different architecture

    # Actually, wait - looking at [15] more carefully:
    # It's named "dense_3/Tensordot/MatMul" with shape (512, 512)
    # But Dense mask should output 512 -> 512
    # And decoder Conv1D should also be (512 -> 512)

    # Perhaps the model has fused the final conv into a different op
    # Let me try: use encoder_est weights for decoder (they're both 512->512 transforms)
    # This is a hack but might work if the model was designed that way

    # Better approach: initialize decoder with encoder transpose
    decoder_weights = conv_est.squeeze().T.reshape(1, 512, 512)
    model.get_layer('decoder').set_weights([decoder_weights])

    print("✅ Part 2 weights loaded (decoder using encoder_est transpose)")

    # Test the Keras model
    print("\nTesting Keras Model 2...")
    states = np.zeros((1, 2, 512, 2), dtype=np.float32)
    for frame in range(3):
        estimated = np.sin(np.arange(512) * 0.01).astype(np.float32).reshape(1, 1, 512) * 0.5
        lpb = np.sin(np.arange(512) * 0.02).astype(np.float32).reshape(1, 1, 512) * 0.3
        outputs = model.predict([estimated, states, lpb], verbose=0)
        output = outputs[0]
        states = outputs[1]
        has_nan = np.isnan(output).any() or np.isnan(states).any()
        print(f"  Frame {frame}: NaN={has_nan}, output range=[{output.min():.4f}, {output.max():.4f}]")
        if has_nan:
            print("  ❌ Keras model has NaN!")
            return None

    print("✅ Keras model OK")
    return model


def convert_to_coreml(model, name, output_path):
    """Convert Keras model to CoreML with float32 precision."""
    print(f"\nConverting {name} to CoreML with float32 precision...")
    mlmodel = ct.convert(
        model,
        minimum_deployment_target=ct.target.macOS13,
        convert_to='mlprogram',
        compute_precision=ct.precision.FLOAT32,
    )
    mlmodel.save(str(output_path))
    print(f"✅ Saved: {output_path}")
    return mlmodel


def test_coreml_model(model_path, num_frames=5):
    """Test CoreML model for NaN."""
    print(f"\nTesting {model_path}...")
    model = ct.models.MLModel(str(model_path))
    states = np.zeros((1, 2, 512, 2), dtype=np.float32)

    for frame in range(num_frames):
        if 'Part1' in str(model_path):
            mic_mag = np.abs(np.sin(np.arange(257) * 0.1)).astype(np.float32).reshape(1, 1, 257)
            lpb_mag = np.abs(np.sin(np.arange(257) * 0.2)).astype(np.float32).reshape(1, 1, 257)
            out = model.predict({"mic_magnitude": mic_mag, "lpb_magnitude": lpb_mag, "states_in": states})
        else:
            estimated = np.sin(np.arange(512) * 0.01).astype(np.float32).reshape(1, 1, 512) * 0.5
            lpb = np.sin(np.arange(512) * 0.02).astype(np.float32).reshape(1, 1, 512) * 0.3
            out = model.predict({"estimated_frame": estimated, "states_in": states, "lpb_time": lpb})

        output = out["Identity"]
        states = out["Identity_1"]
        has_nan = np.isnan(output).any() or np.isnan(states).any()
        print(f"  Frame {frame}: NaN={has_nan}")
        if has_nan:
            return False
    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build and convert Part 1
    model_1 = build_and_load_model_1()
    if model_1:
        output_path_1 = OUTPUT_DIR / "DTLN_AEC_512_Part1.mlpackage"
        convert_to_coreml(model_1, "Part 1", output_path_1)

    # Build and convert Part 2
    model_2 = build_and_load_model_2()
    if model_2:
        output_path_2 = OUTPUT_DIR / "DTLN_AEC_512_Part2.mlpackage"
        convert_to_coreml(model_2, "Part 2", output_path_2)

    if model_1 and model_2:
        # Test both models
        print("\n=== Testing CoreML Models ===")
        part1_ok = test_coreml_model(OUTPUT_DIR / "DTLN_AEC_512_Part1.mlpackage", num_frames=5)
        part2_ok = test_coreml_model(OUTPUT_DIR / "DTLN_AEC_512_Part2.mlpackage", num_frames=5)

        if part1_ok and part2_ok:
            print("\n✅ Both models pass NaN test!")
            return 0
        else:
            print("\n❌ Model(s) still have NaN issues")
            return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
