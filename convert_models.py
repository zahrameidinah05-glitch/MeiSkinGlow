import os
import tensorflow as tf
import zipfile
import json
import tempfile
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SKIN_PATH = os.path.join(BASE_DIR, 'models', 'skin_type_model.keras')
MODEL_ACNE_PATH = os.path.join(BASE_DIR, 'models', 'acne_type_model.keras')

TFLITE_SKIN_PATH = os.path.join(BASE_DIR, 'models', 'skin_type_model.tflite')
TFLITE_ACNE_PATH = os.path.join(BASE_DIR, 'models', 'acne_type_model.tflite')

def load_custom_model(model_path):
    temp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(model_path, 'r') as z:
            # Extract weights
            weights_path = os.path.join(temp_dir, "model.weights.h5")
            with open(weights_path, "wb") as f:
                f.write(z.read("model.weights.h5"))
            
            # Load config
            config_bytes = z.read("config.json")
            config_dict = json.loads(config_bytes.decode('utf-8'))
            
            # Modify config to remove the Lambda layer
            layers = config_dict["config"]["layers"]
            lambda_idx = -1
            for idx, layer in enumerate(layers):
                if layer.get('class_name') == 'Lambda':
                    lambda_idx = idx
                    break
            
            if lambda_idx != -1:
                layers.pop(lambda_idx)
            
            # Re-create model from modified config using keras
            import keras
            model = keras.saving.deserialize_keras_object(config_dict)
            
            # Load weights
            model.load_weights(weights_path)
            return model
    finally:
        shutil.rmtree(temp_dir)

def convert_to_tflite(keras_model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    # Optional: converter.optimizations = [tf.lite.Optimize.DEFAULT] # to make it even smaller
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved {output_path}")

print("Loading skin model...")
skin_model = load_custom_model(MODEL_SKIN_PATH)
print("Converting skin model...")
convert_to_tflite(skin_model, TFLITE_SKIN_PATH)

print("Loading acne model...")
acne_model = load_custom_model(MODEL_ACNE_PATH)
print("Converting acne model...")
convert_to_tflite(acne_model, TFLITE_ACNE_PATH)

print("Done!")
