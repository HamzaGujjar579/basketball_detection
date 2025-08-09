import cv2
import numpy as np
import os
import glob
import tensorflow as tf

class BasketballDetector:
    def __init__(self, input_path, output_path, model_path):
        self.input_path = input_path
        self.output_path = output_path
        
        # Load model using the specified approach
        print("Loading TensorFlow Lite model...")
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input size
        self.input_shape = self.input_details[0]['shape']
        # For CHW format: [batch, channels, height, width]
        self.input_size = (self.input_shape[3], self.input_shape[2])  # (width, height)
        
        print(f"Model input shape: {self.input_shape}")
        print(f"Model input size: {self.input_size}")
        
        # Video setup
        self.cap = cv2.VideoCapture(input_path)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties - FPS: {fps}, Width: {width}, Height: {height}")
        
        # Output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Detection tracking
        self.ball_positions = []
        self.hoop_positions = []
        self.makes = 0
        self.attempts = 0
        self.frame_count = 0
        
    def preprocess_frame(self, frame):
        # Resize and normalize
        resized = cv2.resize(frame, self.input_size)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_frame.astype(np.float32) / 255.0
        
        # Convert from HWC to CHW format (channels first)
        chw_frame = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(chw_frame, axis=0)
    
    def detect_objects(self, frame):
        # Preprocess
        input_tensor = self.preprocess_frame(frame)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Process detections
        detections = []
        if len(output_data.shape) == 3:
            output_data = output_data[0]
        
        # Scale factors
        scale_x = frame.shape[1] / self.input_size[0]
        scale_y = frame.shape[0] / self.input_size[1]
        
        for detection in output_data:
            if len(detection) < 6:
                continue
                
            x_center, y_center, width, height = detection[:4]
            class_scores = detection[4:]
            
            if len(class_scores) > 0:
                class_id = np.argmax(class_scores)
                confidence = class_scores[class_id]
                
                if confidence > 0.3:
                    x1 = int((x_center - width/2) * scale_x)
                    y1 = int((y_center - height/2) * scale_y)
                    x2 = int((x_center + width/2) * scale_x)
                    y2 = int((y_center + height/2) * scale_y)
                    
                    # Clamp to frame bounds
                    x1 = max(0, min(x1, frame.shape[1]))
                    y1 = max(0, min(y1, frame.shape[0]))
                    x2 = max(0, min(x2, frame.shape[1]))
                    y2 = max(0, min(y2, frame.shape[0]))
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id
                    })
        
        return detections
    
    def process_frame(self, frame):
        detections = self.detect_objects(frame)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cls = det['class_id']
            
            # Draw bounding box
            if cls == 0:  # Basketball
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Ball {conf:.2f}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif cls == 1:  # Hoop
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'Hoop {conf:.2f}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Display score
        score_text = f"Score: {self.makes}/{self.attempts}"
        cv2.putText(frame, score_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        print("Starting processing...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Write frame
            self.out.write(processed_frame)
            
            self.frame_count += 1
            
            # Progress
            if self.frame_count % 30 == 0:
                print(f"Processed {self.frame_count} frames")
        
        # Cleanup
        self.cap.release()
        self.out.release()
        print(f"Output saved to: {self.output_path}")

def find_files():
    current_dir = os.getcwd()
    
    # Find model
    model_files = glob.glob("*.tflite")
    if not model_files:
        raise FileNotFoundError("No .tflite file found")
    
    # Find video
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(ext))
    
    if not video_files:
        raise FileNotFoundError("No video file found")
    
    input_path = video_files[0]
    output_path = f"{os.path.splitext(input_path)[0]}_output.mp4"
    model_path = model_files[0]
    
    return input_path, output_path, model_path

if __name__ == "__main__":
    try:
        input_path, output_path, model_path = find_files()
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Model: {model_path}")
        
        detector = BasketballDetector(input_path, output_path, model_path)
        detector.run()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()