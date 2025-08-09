import cv2
import cvzone
import math
import numpy as np
import tensorflow as tf
import os
import glob

# Utility functions
def clean_ball_pos(ball_pos, frame_count):
    if len(ball_pos) > 1:
        w1 = ball_pos[-2][2]
        h1 = ball_pos[-2][3]
        w2 = ball_pos[-1][2]
        h2 = ball_pos[-1][3]
        x1 = ball_pos[-2][0][0]
        y1 = ball_pos[-2][0][1]
        x2 = ball_pos[-1][0][0]
        y2 = ball_pos[-1][0][1]
        f1 = ball_pos[-2][1]
        f2 = ball_pos[-1][1]
        f_dif = f2 - f1
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        max_dist = 4 * math.sqrt((w1) ** 2 + (h1) ** 2)
        if (dist > max_dist) and (f_dif < 5):
            ball_pos.pop()
        elif (w2*1.4 < h2) or (h2*1.4 < w2):
            ball_pos.pop()
    
    # Remove old positions (more aggressive cleaning)
    max_age = 45  # Keep positions for 45 frames (1.5 seconds at 30fps)
    ball_pos = [pos for pos in ball_pos if frame_count - pos[1] <= max_age]
    
    return ball_pos

def clean_hoop_pos(hoop_pos):
    if len(hoop_pos) > 1:
        x1 = hoop_pos[-2][0][0]
        y1 = hoop_pos[-2][0][1]
        x2 = hoop_pos[-1][0][0]
        y2 = hoop_pos[-1][0][1]
        w1 = hoop_pos[-2][2]
        h1 = hoop_pos[-2][3]
        w2 = hoop_pos[-1][2]
        h2 = hoop_pos[-1][3]
        f1 = hoop_pos[-2][1]
        f2 = hoop_pos[-1][1]
        f_dif = f2-f1
        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        max_dist = 0.5 * math.sqrt(w1 ** 2 + h1 ** 2)
        if dist > max_dist and f_dif < 5:
            hoop_pos.pop()
        if (w2*1.3 < h2) or (h2*1.3 < w2):
            hoop_pos.pop()
    if len(hoop_pos) > 25:
        hoop_pos.pop(0)
    return hoop_pos

def in_hoop_region(center, hoop_pos):
    if len(hoop_pos) < 1:
        return False
    x = center[0]
    y = center[1]
    x1 = hoop_pos[-1][0][0] - 1 * hoop_pos[-1][2]
    x2 = hoop_pos[-1][0][0] + 1 * hoop_pos[-1][2]
    y1 = hoop_pos[-1][0][1] - 1 * hoop_pos[-1][3]
    y2 = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]
    if x1 < x < x2 and y1 < y < y2:
        return True
    return False

def preprocess_image(image, input_size=(640, 640)):
    """Preprocess image for TensorFlow Lite model"""
    # Resize image
    resized = cv2.resize(image, input_size)
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Normalize to 0-1
    normalized = rgb_image.astype(np.float32) / 255.0
    # Add batch dimension and transpose to CHW format
    input_tensor = np.transpose(normalized, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor

def postprocess_detections(outputs, original_shape, input_size=(640, 640), conf_threshold=0.1):
    """Post-process TensorFlow Lite model outputs"""
    detections = []
    
    # Get the output tensor
    if isinstance(outputs, list):
        output = outputs[0]
    else:
        output = outputs
    
    # Print debug information
    print(f"Output shape: {output.shape}")
    
    # Handle different output shapes
    if len(output.shape) == 3:
        output = output[0]  # Remove batch dimension
    
    # TensorFlow Lite YOLO typically outputs in format: [batch, 84, 8400] or [batch, 6, 8400]
    # where 84 = 4 (bbox) + 80 (classes) for COCO, but for custom models it's 4 + num_classes
    # We need to transpose to [8400, 84] or [8400, 6]
    if output.shape[0] < output.shape[1]:
        output = output.T
    
    print(f"Transposed output shape: {output.shape}")
    print(f"Processing {len(output)} detections")
    
    # Calculate scale factors
    scale_x = original_shape[1] / input_size[0]
    scale_y = original_shape[0] / input_size[1]
    
    # Process each detection
    for i, detection in enumerate(output):
        # Extract coordinates and confidence
        x_center, y_center, width, height = detection[:4]
        
        # For YOLO, class scores start from index 4
        class_scores = detection[4:]
        
        # Get class with highest confidence
        if len(class_scores) > 0:
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            # Debug first few detections
            if i < 5:
                print(f"Detection {i}: conf={confidence:.3f}, class={class_id}, bbox=({x_center:.1f}, {y_center:.1f}, {width:.1f}, {height:.1f})")
            
            if confidence > conf_threshold:
                # Convert center format to corner format
                x1 = int((x_center - width/2) * scale_x)
                y1 = int((y_center - height/2) * scale_y)
                x2 = int((x_center + width/2) * scale_x)
                y2 = int((y_center + height/2) * scale_y)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, original_shape[1]))
                y1 = max(0, min(y1, original_shape[0]))
                x2 = max(0, min(x2, original_shape[1]))
                y2 = max(0, min(y2, original_shape[0]))
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(confidence),
                    'class_id': int(class_id)
                })
    
    print(f"Found {len(detections)} valid detections")
    return detections

class ShotDetector:
    def __init__(self, input_path, output_path, model_path):
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path
        
        # Initialize TensorFlow Lite interpreter
        print("Loading TensorFlow Lite model...")
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output tensor details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape and size
        self.input_shape = self.input_details[0]['shape']
        # For CHW format: [batch, channels, height, width]
        self.input_size = (self.input_shape[3], self.input_shape[2])  # (width, height)
        
        print(f"Model input shape: {self.input_shape}")
        print(f"Model input size: {self.input_size}")
        
        self.class_names = ['Basketball', 'Basketball Hoop']
        self.cap = cv2.VideoCapture(input_path)
        
        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 30
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width == 0 or height == 0:
            width, height = 640, 384
        self.out_size = (width, height)
        self.out = cv2.VideoWriter(output_path, fourcc, fps, self.out_size)
        
        # Initialize tracking variables
        self.ball_pos = []
        self.hoop_pos = []
        self.frame_count = 0
        self.frame = None

    def run(self):
        print("Starting processing...")
        
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                break
            
            # Preprocess frame for TensorFlow Lite model
            input_tensor = preprocess_image(self.frame, self.input_size)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Post-process detections (only debug for first frame)
            if self.frame_count == 0:
                print(f"Frame shape: {self.frame.shape}")
                print(f"Input size: {self.input_size}")
                detections = postprocess_detections([output_data], self.frame.shape, self.input_size, conf_threshold=0.1)
            else:
                detections = postprocess_detections([output_data], self.frame.shape, self.input_size, conf_threshold=0.1)
            
            # Process detections
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                w, h = x2 - x1, y2 - y1
                conf = detection['confidence']
                cls = detection['class_id']
                
                if cls >= len(self.class_names):
                    continue
                    
                current_class = self.class_names[cls]
                center = (int(x1 + w / 2), int(y1 + h / 2))
                
                # Debug: Print detection info for first few frames
                if self.frame_count < 3:
                    print(f"Frame {self.frame_count}: {current_class} - conf: {conf:.3f}, bbox: ({x1},{y1},{x2},{y2})")
                
                if (conf > 0.3 or (in_hoop_region(center, self.hoop_pos) and conf > 0.15)) and current_class == "Basketball":
                    self.ball_pos.append((center, self.frame_count, w, h, conf))
                    cvzone.cornerRect(self.frame, (x1, y1, w, h))
                    
                if conf > 0.5 and current_class == "Basketball Hoop":
                    self.hoop_pos.append((center, self.frame_count, w, h, conf))
                    cvzone.cornerRect(self.frame, (x1, y1, w, h))
            
            self.clean_motion()
            self.frame_count += 1
            
            # Progress indicator
            if self.frame_count % 30 == 0:
                print(f"Processed {self.frame_count} frames")
            
            out_frame = cv2.resize(self.frame, self.out_size)
            self.out.write(out_frame)
            
        self.cap.release()
        self.out.release()
        print(f"Detection complete. Output saved to: {self.output_path}")

    def clean_motion(self):
        # Clean ball positions (remove old and invalid positions)
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        
        # Additional cleaning: remove positions older than 60 frames (2 seconds at 30fps)
        max_age = 60
        self.ball_pos = [pos for pos in self.ball_pos if self.frame_count - pos[1] <= max_age]
        
        # Draw ball trajectory with red dots
        for i in range(len(self.ball_pos)):
            # Make older dots more transparent/smaller
            age = self.frame_count - self.ball_pos[i][1]
            alpha = max(0.3, 1.0 - (age / max_age))
            radius = max(1, int(3 * alpha))
            cv2.circle(self.frame, self.ball_pos[i][0], radius, (0, 0, 255), 2)
        
        # Clean hoop positions
        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)

def find_files_in_current_directory():
    """Find model and video files in current directory"""
    current_dir = os.getcwd()
    
    # Find TensorFlow Lite model
    model_files = glob.glob(os.path.join(current_dir, "*.tflite"))
    if not model_files:
        raise FileNotFoundError("No TensorFlow Lite model file (.tflite) found in current directory")
    model_path = model_files[0]  # Use first .tflite file found
    
    # Find video files
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(current_dir, ext)))
    
    if not video_files:
        raise FileNotFoundError("No video file found in current directory")
    
    input_path = video_files[0]  # Use first video file found
    
    # Generate output path
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(current_dir, f"{base_name}_output.mp4")
    
    return input_path, output_path, model_path

def run_detection():
    try:
        input_path, output_path, model_path = find_files_in_current_directory()
        
        print(f"Found model: {os.path.basename(model_path)}")
        print(f"Found video: {os.path.basename(input_path)}")
        print(f"Output will be saved as: {os.path.basename(output_path)}")
        
        detector = ShotDetector(input_path, output_path, model_path)
        detector.run()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. A TensorFlow Lite model file (.tflite) in the current directory")
        print("2. A video file in the current directory")
        print("3. Required packages installed: opencv-python, tensorflow, cvzone, numpy")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_detection()