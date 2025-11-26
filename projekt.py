import os
import pickle
import tempfile
import time

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


DATA_PATH = os.path.join(os.path.dirname(__file__), "IMDB Dataset.csv")


def clean_text(text: str) -> str:
    """Basic cleaning: remove HTML, lower, remove extra whitespace."""
    if not isinstance(text, str):
        return ""
    # remove HTML tags
    text = BeautifulSoup(text, "lxml").get_text()
    # lower
    text = text.lower()
    # collapse whitespace
    text = " ".join(text.split())
    return text


def load_and_preprocess(max_vocab=20000, max_len=200):
    print("Loading dataset from:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    # Expect columns 'review' and 'sentiment'
    if 'review' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("CSV must contain 'review' and 'sentiment' columns")

    # clean
    print("Cleaning texts...")
    texts = df['review'].astype(str).apply(clean_text).tolist()
    labels = (df['sentiment'] == 'positive').astype(int).values

    # tokenize
    print(f"Tokenizing (vocab_size={max_vocab})...")
    tokenizer = Tokenizer(num_words=max_vocab, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # pad
    print(f"Padding sequences to length {max_len}...")
    X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    y = labels

    return X, y, tokenizer


def build_model(vocab_size=20000, embed_dim=128, max_len=200):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim),
        Conv1D(128, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def convert_to_tflite_dynamic(keras_model, save_path=None):
    """Convert a Keras model to TFLite with dynamic range quantization."""
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    if save_path:
        with open(save_path, 'wb') as f:
            f.write(tflite_model)
    return tflite_model


def evaluate_tflite(tflite_model_bytes, X_test, y_test, batch_size=256):
    """Evaluate a TFLite model (provided as bytes) on X_test/y_test.
    Returns (accuracy, inference_time_seconds).
    """
    interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_index = input_details['index']
    output_index = output_details['index']
    input_dtype = input_details['dtype']

    n = X_test.shape[0]
    preds = np.zeros(n, dtype=np.float32)

    t0 = time.perf_counter()
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        batch = X_test[start:end].astype(input_dtype)

        # Try to resize the input tensor to the current batch.
        try:
            interpreter.resize_tensor_input(input_index, batch.shape)
            interpreter.allocate_tensors()
            # refresh output index in case tensors were reallocated
            output_details = interpreter.get_output_details()[0]
            output_index = output_details['index']

            interpreter.set_tensor(input_index, batch)
            interpreter.invoke()
            out = interpreter.get_tensor(output_index)
            preds[start:end] = out.reshape(-1)
        except Exception:
            # Fall back to per-sample inference if batch resize isn't supported.
            for i in range(batch.shape[0]):
                single = batch[i:i+1]
                try:
                    interpreter.resize_tensor_input(input_index, single.shape)
                    interpreter.allocate_tensors()
                    output_details = interpreter.get_output_details()[0]
                    output_index = output_details['index']
                except Exception:
                    pass
                interpreter.set_tensor(input_index, single)
                interpreter.invoke()
                out = interpreter.get_tensor(output_index)
                preds[start + i] = out.reshape(-1)[0]

    t1 = time.perf_counter()
    preds_bin = (preds >= 0.5).astype(int)
    acc = (preds_bin == y_test).mean()
    return acc, (t1 - t0)


def sizeof_fmt(num, suffix='B'):
    for unit in ['','K','M','G','T']:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}P{suffix}"


def main():
    # hyperparams
    VOCAB_SIZE = 20000
    MAX_LEN = 200
    EMBED_DIM = 128
    TEST_SIZE = 0.2
    BATCH_SIZE = 128
    EPOCHS = 5

    X, y, tokenizer = load_and_preprocess(max_vocab=VOCAB_SIZE, max_len=MAX_LEN)

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42, stratify=y)

    print("Building model...")
    model = build_model(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, max_len=MAX_LEN)
    model.summary()

    # callbacks
    os.makedirs('artifacts', exist_ok=True)
    checkpoint_path = os.path.join('artifacts', 'best_model.h5')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
    ]

    print("Training...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    print("Evaluating on test set (Keras)...")
    # measure inference time with predict
    t0 = time.perf_counter()
    preds = model.predict(X_test, batch_size=BATCH_SIZE)
    t1 = time.perf_counter()
    preds_bin = (preds.reshape(-1) >= 0.5).astype(int)
    keras_acc = (preds_bin == y_test).mean()
    keras_time = t1 - t0
    print(f"Keras model test accuracy: {keras_acc:.4f}  inference time: {keras_time:.3f}s")

    # save tokenizer
    with open(os.path.join('artifacts', 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Saved tokenizer to ./artifacts/tokenizer.pkl")

    # prefer the saved best model if present
    keras_model = None
    if os.path.exists(checkpoint_path):
        try:
            print(f"Loading best model from {checkpoint_path} for quantization...")
            keras_model = tf.keras.models.load_model(checkpoint_path)
        except Exception as e:
            print(f"Could not load checkpoint model: {e}. Falling back to current model in memory.")

    if keras_model is None:
        keras_model = model

    # determine Keras model size
    if os.path.exists(checkpoint_path):
        keras_size = os.path.getsize(checkpoint_path)
        keras_size_path = checkpoint_path
    else:
        # save temporary model file to get size
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
        try:
            keras_model.save(tmp.name)
            keras_size = os.path.getsize(tmp.name)
            keras_size_path = tmp.name
        finally:
            tmp.close()

    # convert to TFLite with dynamic range quantization and evaluate
    tflite_path = os.path.join('artifacts', 'model_dynamic.tflite')
    try:
        print("Converting model to TFLite with dynamic range quantization...")
        tflite_bytes = convert_to_tflite_dynamic(keras_model, save_path=tflite_path)
        print(f"Saved TFLite model to {tflite_path}")

        tflite_size = os.path.getsize(tflite_path) if os.path.exists(tflite_path) else len(tflite_bytes)

        print("Evaluating quantized TFLite model on test set...")
        tflite_acc, tflite_time = evaluate_tflite(tflite_bytes, X_test, y_test, batch_size=256)
        print(f"TFLite dynamic quantized model accuracy: {tflite_acc:.4f}  inference time: {tflite_time:.3f}s")

        # summary
        print('\n=== Summary ===')
        print(f"Keras accuracy: {keras_acc:.4f}")
        print(f"Keras model path: {keras_size_path} size: {sizeof_fmt(keras_size)}")
        print(f"Keras inference time (test set): {keras_time:.3f}s")
        print('---')
        print(f"TFLite accuracy: {tflite_acc:.4f}")
        print(f"TFLite model path: {tflite_path} size: {sizeof_fmt(tflite_size)}")
        print(f"TFLite inference time (test set): {tflite_time:.3f}s")
    except Exception as e:
        print(f"Quantization or TFLite evaluation failed: {e}")
    finally:
        # clean up temporary keras file if we created one
        if 'tmp' in locals() and os.path.exists(tmp.name):
            try:
                os.remove(tmp.name)
            except Exception:
                pass


if __name__ == '__main__':
    main()
