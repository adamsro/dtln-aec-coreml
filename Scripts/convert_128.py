#!/usr/bin/env python3
"""Convert 128-unit DTLN-aec model to CoreML with correct weight loading.

Based on convert_512.py but adapted for the 128-unit model architecture.

Tensor names were extracted from the TFLite model using tensor analysis.
"""
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
TFLITE_DIR = PROJECT_ROOT / "Resources" / "DTLN_AEC"
OUTPUT_DIR = PROJECT_ROOT / "Sources" / "DTLNAecCoreML" / "Resources"

import numpy as np
import tensorflow as tf
import coremltools as ct
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Multiply, Conv1D,
    Activation, Layer, Concatenate, Lambda
)

print(f"TensorFlow version: {tf.__version__}")
print(f"CoreMLTools version: {ct.__version__}")


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the 128-unit DTLN-aec model."""
    num_units: int = 128
    num_layers: int = 2
    block_len: int = 512
    fft_bins: int = 257
    encoder_size: int = 512  # Conv1D encoder output size

    @property
    def lstm_gates(self) -> int:
        return 4 * self.num_units


CONFIG = ModelConfig()


class TensorNotFoundError(Exception):
    pass


class TFLiteTensorFinder:
    """Helper for finding tensors in TFLite models by name pattern and shape."""

    def __init__(self, interpreter: tf.lite.Interpreter):
        self.interpreter = interpreter
        self._build_index()

    def _build_index(self):
        self.by_name: dict[str, list[dict]] = {}
        self.by_index: dict[int, dict] = {}

        for detail in self.interpreter.get_tensor_details():
            try:
                tensor = self.interpreter.get_tensor(detail['index'])
                entry = {
                    'index': detail['index'],
                    'name': detail['name'],
                    'tensor': tensor,
                    'shape': tuple(tensor.shape),
                }
                self.by_index[detail['index']] = entry

                name = detail['name']
                if name not in self.by_name:
                    self.by_name[name] = []
                self.by_name[name].append(entry)
            except ValueError:
                pass

    def find(
        self,
        name_pattern: str,
        expected_shape: Optional[tuple] = None,
        description: str = ""
    ) -> np.ndarray:
        matches = []
        for name, entries in self.by_name.items():
            if name_pattern in name:
                for entry in entries:
                    if expected_shape is None or entry['shape'] == expected_shape:
                        matches.append(entry)

        if not matches:
            shape_msg = f" with shape {expected_shape}" if expected_shape else ""
            desc_msg = f" ({description})" if description else ""
            raise TensorNotFoundError(
                f"No tensor matching '{name_pattern}'{shape_msg}{desc_msg}"
            )

        if len(matches) > 1:
            if expected_shape:
                exact = [m for m in matches if m['shape'] == expected_shape]
                if len(exact) == 1:
                    return exact[0]['tensor']
            names = [m['name'] for m in matches]
            print(f"  ⚠️  Multiple tensors match '{name_pattern}': {names}, using first")

        return matches[0]['tensor']

    def find_output(self, name_pattern: str) -> int:
        for name, entries in self.by_name.items():
            if name_pattern in name:
                return entries[0]['index']
        raise TensorNotFoundError(f"No output tensor matching '{name_pattern}'")


class InstantLayerNormalization(Layer):
    """Layer normalization without running statistics."""

    def __init__(self, epsilon: float = 1e-7, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

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
        return (inputs - mean) / std * self.gamma + self.beta


def build_and_load_model_1() -> Model:
    """Build Part 1 (Frequency Domain) model and load weights from TFLite."""
    print("\n=== Building Model 1 (Frequency Domain) - 128 units ===")

    tflite_path = TFLITE_DIR / "dtln_aec_128_1.tflite"
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    finder = TFLiteTensorFinder(interpreter)

    # Build Keras model architecture
    mic_mag = Input(shape=(1, CONFIG.fft_bins), name='mic_magnitude')
    lpb_mag = Input(shape=(1, CONFIG.fft_bins), name='lpb_magnitude')
    states_in = Input(shape=(CONFIG.num_layers, CONFIG.num_units, 2), name='states_in')

    mic_log = Lambda(lambda x: tf.math.log(x + 1e-7), name='mic_log')(mic_mag)
    mic_norm = InstantLayerNormalization(name='mic_norm')(mic_log)
    lpb_log = Lambda(lambda x: tf.math.log(x + 1e-7), name='lpb_log')(lpb_mag)
    lpb_norm = InstantLayerNormalization(name='lpb_norm')(lpb_log)

    concat = Concatenate(axis=-1, name='concat_features')([mic_norm, lpb_norm])

    x = concat
    states_h = []
    states_c = []
    for i in range(CONFIG.num_layers):
        h_in = Lambda(lambda s, idx=i: s[:, idx, :, 0], name=f'h_in_{i}')(states_in)
        c_in = Lambda(lambda s, idx=i: s[:, idx, :, 1], name=f'c_in_{i}')(states_in)
        x, h_out, c_out = LSTM(
            CONFIG.num_units,
            return_sequences=True,
            return_state=True,
            name=f'lstm_1_{i}'
        )(x, initial_state=[h_in, c_in])
        states_h.append(h_out)
        states_c.append(c_out)

    mask = Dense(CONFIG.fft_bins, name='dense_mask_1')(x)
    mask = Activation('sigmoid', name='mask_1')(mask)

    states_h_stack = Lambda(lambda h: tf.stack(h, axis=1), name='stack_h_1')(states_h)
    states_c_stack = Lambda(lambda c: tf.stack(c, axis=1), name='stack_c_1')(states_c)
    states_out = Lambda(
        lambda sc: tf.stack(sc, axis=-1),
        name='states_out_1'
    )([states_h_stack, states_c_stack])

    model = Model(
        inputs=[mic_mag, states_in, lpb_mag],
        outputs=[mask, states_out],
        name='DTLN_AEC_Part1'
    )

    # Load weights using name-based lookup (tensor names from 128-unit TFLite)
    print("Loading weights by tensor name...")

    # LayerNorm for mic input
    mic_norm_gamma = finder.find(
        "instant_layer_normalization_4/3527",
        expected_shape=(CONFIG.fft_bins,),
        description="mic norm gamma"
    )
    mic_norm_beta = finder.find(
        "instant_layer_normalization_4/3529",
        expected_shape=(CONFIG.fft_bins,),
        description="mic norm beta"
    )
    model.get_layer('mic_norm').set_weights([mic_norm_gamma, mic_norm_beta])

    # LayerNorm for lpb input
    lpb_norm_gamma = finder.find(
        "instant_layer_normalization_5/3554",
        expected_shape=(CONFIG.fft_bins,),
        description="lpb norm gamma"
    )
    lpb_norm_beta = finder.find(
        "instant_layer_normalization_5/3556",
        expected_shape=(CONFIG.fft_bins,),
        description="lpb norm beta"
    )
    model.get_layer('lpb_norm').set_weights([lpb_norm_gamma, lpb_norm_beta])

    # LSTM 1 - input size 514 (257*2 concatenated)
    lstm1_kernel = finder.find(
        "lstm_4/lstm_cell_4/StatefulPartitionedCall/MatMul",
        expected_shape=(CONFIG.lstm_gates, CONFIG.fft_bins * 2),
        description="LSTM 1 kernel"
    )
    lstm1_recurrent = finder.find(
        "lstm_4/lstm_cell_4/StatefulPartitionedCall/MatMul_1",
        expected_shape=(CONFIG.lstm_gates, CONFIG.num_units),
        description="LSTM 1 recurrent"
    )
    lstm1_bias = finder.find(
        "lstm_4/lstm_cell_4/3667",
        expected_shape=(CONFIG.lstm_gates,),
        description="LSTM 1 bias"
    )
    model.get_layer('lstm_1_0').set_weights([
        lstm1_kernel.T,
        lstm1_recurrent.T,
        lstm1_bias
    ])

    # LSTM 2
    lstm2_kernel = finder.find(
        "lstm_5/lstm_cell_5/MatMul",
        expected_shape=(CONFIG.lstm_gates, CONFIG.num_units),
        description="LSTM 2 kernel"
    )
    lstm2_recurrent = finder.find(
        "lstm_5/lstm_cell_5/MatMul_1",
        expected_shape=(CONFIG.lstm_gates, CONFIG.num_units),
        description="LSTM 2 recurrent"
    )
    lstm2_bias = finder.find(
        "lstm_5/lstm_cell_5/BiasAdd",
        expected_shape=(CONFIG.lstm_gates,),
        description="LSTM 2 bias"
    )
    model.get_layer('lstm_1_1').set_weights([
        lstm2_kernel.T,
        lstm2_recurrent.T,
        lstm2_bias
    ])

    # Dense mask layer
    dense_kernel = finder.find(
        "dense_2/Tensordot/MatMul",
        expected_shape=(CONFIG.fft_bins, CONFIG.num_units),
        description="dense mask kernel"
    )
    dense_bias = finder.find(
        "dense_2/BiasAdd",
        expected_shape=(CONFIG.fft_bins,),
        description="dense mask bias"
    )
    model.get_layer('dense_mask_1').set_weights([dense_kernel.T, dense_bias])

    print("✅ Part 1 weights loaded")

    # Verify
    _verify_model_1(model, interpreter, finder)

    return model


def _verify_model_1(model: Model, interpreter: tf.lite.Interpreter, finder: TFLiteTensorFinder):
    """Verify Part 1 Keras model output matches TFLite."""
    print("\nVerifying Keras model against TFLite...")

    np.random.seed(42)
    test_mic = np.abs(np.random.randn(1, 1, CONFIG.fft_bins).astype(np.float32) * 0.1)
    test_lpb = np.abs(np.random.randn(1, 1, CONFIG.fft_bins).astype(np.float32) * 0.1)
    test_states = np.zeros((1, CONFIG.num_layers, CONFIG.num_units, 2), dtype=np.float32)

    # TFLite inference
    input_details = interpreter.get_input_details()
    mic_idx = next(d['index'] for d in input_details if d['shape'].tolist() == [1, 1, 257] and 'input_3' in d['name'])
    lpb_idx = next(d['index'] for d in input_details if d['shape'].tolist() == [1, 1, 257] and 'input_4' in d['name'])
    states_idx = next(d['index'] for d in input_details if d['shape'].tolist() == [1, 2, 128, 2])

    interpreter.set_tensor(mic_idx, test_mic)
    interpreter.set_tensor(lpb_idx, test_lpb)
    interpreter.set_tensor(states_idx, test_states)
    interpreter.invoke()

    output_idx = finder.find_output("Identity")
    tflite_mask = interpreter.get_tensor(output_idx)

    # Keras inference
    keras_mask = model.predict([test_mic, test_states, test_lpb], verbose=0)[0]

    # Compare
    corr = np.corrcoef(tflite_mask.flatten(), keras_mask.flatten())[0, 1]
    diff = np.abs(tflite_mask - keras_mask).mean()

    print(f"  TFLite mask: range=[{tflite_mask.min():.4f}, {tflite_mask.max():.4f}], mean={tflite_mask.mean():.4f}")
    print(f"  Keras mask:  range=[{keras_mask.min():.4f}, {keras_mask.max():.4f}], mean={keras_mask.mean():.4f}")
    print(f"  Correlation: {corr:.6f}")
    print(f"  Mean absolute diff: {diff:.6f}")

    if corr > 0.99:
        print("✅ Keras model matches TFLite!")
    else:
        print(f"⚠️  Correlation only {corr:.4f} - weights may be incorrect")


def build_and_load_model_2() -> Model:
    """Build Part 2 (Time Domain) model and load weights from TFLite."""
    print("\n=== Building Model 2 (Time Domain) - 128 units ===")

    tflite_path = TFLITE_DIR / "dtln_aec_128_2.tflite"
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    finder = TFLiteTensorFinder(interpreter)

    # Build Keras model architecture
    estimated_frame = Input(shape=(1, CONFIG.block_len), name='estimated_frame')
    lpb_time = Input(shape=(1, CONFIG.block_len), name='lpb_time')
    states_in = Input(shape=(CONFIG.num_layers, CONFIG.num_units, 2), name='states_in')

    # Encoders (share weights in original model)
    encoded_est = Conv1D(
        CONFIG.encoder_size, 1,
        use_bias=False,
        name='encoder_est'
    )(estimated_frame)
    encoded_est_norm = InstantLayerNormalization(name='enc_est_norm')(encoded_est)

    encoded_lpb = Conv1D(
        CONFIG.encoder_size, 1,
        use_bias=False,
        name='encoder_lpb'
    )(lpb_time)
    encoded_lpb_norm = InstantLayerNormalization(name='enc_lpb_norm')(encoded_lpb)

    concat = Concatenate(axis=-1, name='concat_encoded')([encoded_est_norm, encoded_lpb_norm])

    x = concat
    states_h = []
    states_c = []
    for i in range(CONFIG.num_layers):
        h_in = Lambda(lambda s, idx=i: s[:, idx, :, 0], name=f'h_in_2_{i}')(states_in)
        c_in = Lambda(lambda s, idx=i: s[:, idx, :, 1], name=f'c_in_2_{i}')(states_in)
        x, h_out, c_out = LSTM(
            CONFIG.num_units,
            return_sequences=True,
            return_state=True,
            name=f'lstm_2_{i}'
        )(x, initial_state=[h_in, c_in])
        states_h.append(h_out)
        states_c.append(c_out)

    mask = Dense(CONFIG.encoder_size, name='dense_mask_2')(x)
    mask = Activation('sigmoid', name='mask_2')(mask)
    masked = Multiply(name='apply_mask')([encoded_est, mask])
    decoded = Conv1D(CONFIG.block_len, 1, use_bias=False, name='decoder')(masked)

    states_h_stack = Lambda(lambda h: tf.stack(h, axis=1), name='stack_h_2')(states_h)
    states_c_stack = Lambda(lambda c: tf.stack(c, axis=1), name='stack_c_2')(states_c)
    states_out = Lambda(
        lambda sc: tf.stack(sc, axis=-1),
        name='states_out_2'
    )([states_h_stack, states_c_stack])

    model = Model(
        inputs=[estimated_frame, states_in, lpb_time],
        outputs=[decoded, states_out],
        name='DTLN_AEC_Part2'
    )

    # Load weights
    print("Loading weights by tensor name...")

    # Encoder weights (shared)
    encoder_weights = finder.find(
        "conv1d_2/StatefulPartitionedCall",
        expected_shape=(CONFIG.encoder_size, 1, 1, CONFIG.encoder_size),
        description="encoder weights (shared)"
    )
    encoder_weights_keras = encoder_weights.squeeze().T.reshape(
        1, CONFIG.encoder_size, CONFIG.encoder_size
    )
    print(f"  Encoder weights: mean_abs={np.abs(encoder_weights).mean():.4f}")
    model.get_layer('encoder_est').set_weights([encoder_weights_keras])
    model.get_layer('encoder_lpb').set_weights([encoder_weights_keras])

    # Decoder weights
    decoder_weights = finder.find(
        "conv1d_3/conv1d1",
        expected_shape=(CONFIG.block_len, 1, 1, CONFIG.block_len),
        description="decoder weights"
    )
    decoder_weights_keras = decoder_weights.squeeze().T.reshape(
        1, CONFIG.block_len, CONFIG.block_len
    )
    print(f"  Decoder weights: mean_abs={np.abs(decoder_weights).mean():.4f}")
    model.get_layer('decoder').set_weights([decoder_weights_keras])

    # LayerNorm for encoded estimate
    enc_est_gamma = finder.find(
        "instant_layer_normalization_6/4426",
        expected_shape=(CONFIG.encoder_size,),
        description="enc_est norm gamma"
    )
    enc_est_beta = finder.find(
        "instant_layer_normalization_6/4428",
        expected_shape=(CONFIG.encoder_size,),
        description="enc_est norm beta"
    )
    model.get_layer('enc_est_norm').set_weights([enc_est_gamma, enc_est_beta])

    # LayerNorm for encoded lpb
    enc_lpb_gamma = finder.find(
        "instant_layer_normalization_7/4453",
        expected_shape=(CONFIG.encoder_size,),
        description="enc_lpb norm gamma"
    )
    enc_lpb_beta = finder.find(
        "instant_layer_normalization_7/4455",
        expected_shape=(CONFIG.encoder_size,),
        description="enc_lpb norm beta"
    )
    model.get_layer('enc_lpb_norm').set_weights([enc_lpb_gamma, enc_lpb_beta])

    # LSTM 1 - input size 1024 (512*2)
    lstm1_kernel = finder.find(
        "lstm_6/lstm_cell_6/StatefulPartitionedCall/MatMul",
        expected_shape=(CONFIG.lstm_gates, CONFIG.encoder_size * 2),
        description="LSTM 1 kernel"
    )
    lstm1_recurrent = finder.find(
        "lstm_6/lstm_cell_6/StatefulPartitionedCall/MatMul_1",
        expected_shape=(CONFIG.lstm_gates, CONFIG.num_units),
        description="LSTM 1 recurrent"
    )
    lstm1_bias = finder.find(
        "lstm_6/lstm_cell_6/4566",
        expected_shape=(CONFIG.lstm_gates,),
        description="LSTM 1 bias"
    )
    model.get_layer('lstm_2_0').set_weights([
        lstm1_kernel.T,
        lstm1_recurrent.T,
        lstm1_bias
    ])

    # LSTM 2
    lstm2_kernel = finder.find(
        "lstm_7/lstm_cell_7/MatMul",
        expected_shape=(CONFIG.lstm_gates, CONFIG.num_units),
        description="LSTM 2 kernel"
    )
    lstm2_recurrent = finder.find(
        "lstm_7/lstm_cell_7/MatMul_1",
        expected_shape=(CONFIG.lstm_gates, CONFIG.num_units),
        description="LSTM 2 recurrent"
    )
    lstm2_bias = finder.find(
        "lstm_7/lstm_cell_7/BiasAdd",
        expected_shape=(CONFIG.lstm_gates,),
        description="LSTM 2 bias"
    )
    model.get_layer('lstm_2_1').set_weights([
        lstm2_kernel.T,
        lstm2_recurrent.T,
        lstm2_bias
    ])

    # Dense mask layer
    dense_kernel = finder.find(
        "dense_3/Tensordot/MatMul",
        expected_shape=(CONFIG.encoder_size, CONFIG.num_units),
        description="dense mask kernel"
    )
    dense_bias = finder.find(
        "dense_3/BiasAdd",
        expected_shape=(CONFIG.encoder_size,),
        description="dense mask bias"
    )
    model.get_layer('dense_mask_2').set_weights([dense_kernel.T, dense_bias])

    print("✅ Part 2 weights loaded")

    # Verify
    _verify_model_2(model, interpreter, finder)

    return model


def _verify_model_2(model: Model, interpreter: tf.lite.Interpreter, finder: TFLiteTensorFinder):
    """Verify Part 2 Keras model output matches TFLite."""
    print("\nVerifying Keras model against TFLite...")

    np.random.seed(42)
    test_est = np.random.randn(1, 1, CONFIG.block_len).astype(np.float32) * 0.3
    test_lpb = np.random.randn(1, 1, CONFIG.block_len).astype(np.float32) * 0.3
    test_states = np.zeros((1, CONFIG.num_layers, CONFIG.num_units, 2), dtype=np.float32)

    # TFLite inference
    input_details = interpreter.get_input_details()
    est_idx = next(d['index'] for d in input_details if 'input_6' in d['name'])
    states_idx = next(d['index'] for d in input_details if 'input_8' in d['name'])
    lpb_idx = next(d['index'] for d in input_details if 'input_7' in d['name'])

    interpreter.set_tensor(est_idx, test_est)
    interpreter.set_tensor(states_idx, test_states)
    interpreter.set_tensor(lpb_idx, test_lpb)
    interpreter.invoke()

    output_idx = finder.find_output("Identity")
    tflite_out = interpreter.get_tensor(output_idx)

    # Keras inference
    keras_out = model.predict([test_est, test_states, test_lpb], verbose=0)[0]

    # Compare
    corr = np.corrcoef(tflite_out.flatten(), keras_out.flatten())[0, 1]
    diff = np.abs(tflite_out - keras_out).mean()

    print(f"  TFLite: range=[{tflite_out.min():.4f}, {tflite_out.max():.4f}], mean_abs={np.abs(tflite_out).mean():.4f}")
    print(f"  Keras:  range=[{keras_out.min():.4f}, {keras_out.max():.4f}], mean_abs={np.abs(keras_out).mean():.4f}")
    print(f"  Correlation: {corr:.6f}")
    print(f"  Mean absolute diff: {diff:.6f}")

    if corr > 0.99:
        print("✅ Keras model matches TFLite!")
    else:
        print(f"⚠️  Correlation only {corr:.4f} - weights may be incorrect")


def convert_to_coreml(model: Model, name: str, output_path: Path) -> ct.models.MLModel:
    """Convert Keras model to CoreML with float32 precision."""
    print(f"\nConverting {name} to CoreML with float32 precision...")

    mlmodel = ct.convert(
        model,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT32,
    )

    mlmodel.save(str(output_path))
    print(f"✅ Saved to {output_path}")

    return mlmodel


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build and convert Part 1
    model1 = build_and_load_model_1()
    convert_to_coreml(
        model1,
        "DTLN_AEC_128_Part1",
        OUTPUT_DIR / "DTLN_AEC_128_Part1.mlpackage"
    )

    # Build and convert Part 2
    model2 = build_and_load_model_2()
    convert_to_coreml(
        model2,
        "DTLN_AEC_128_Part2",
        OUTPUT_DIR / "DTLN_AEC_128_Part2.mlpackage"
    )

    print("\n" + "=" * 60)
    print("✅ Conversion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
