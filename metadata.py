import os
import sys
import tensorflow as tf

model_path = "not.tflite"

if not os.path.exists(model_path):
    print(f"❌ ERROR: File '{model_path}' not found.")
    sys.exit(1)
else:
    print(f"✅ Found model at: {model_path}")

# Proceed
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\n=== INPUT DETAILS ===")
for input_detail in input_details:
    print(f"Name: {input_detail['name']}")
    print(f"Shape: {input_detail['shape']}")
    print(f"Type: {input_detail['dtype']}\n")

print("=== OUTPUT DETAILS ===")
for output_detail in output_details:
    print(f"Name: {output_detail['name']}")
    print(f"Shape: {output_detail['shape']}")
    print(f"Type: {output_detail['dtype']}\n")
