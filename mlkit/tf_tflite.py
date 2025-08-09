import tensorflow as tf

saved_model_dir = 'model_tf'
tflite_model_path = 'model.tflite'

# Convert the model for ML Kit compatibility
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]  # Only use built-in ops
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optional: apply optimizations

try:
    tflite_model = converter.convert()
    # Save the model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print("TFLite model successfully converted and saved for ML Kit compatibility.")
except Exception as e:
    print("TFLite conversion failed:", e)
