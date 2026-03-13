import os
import cv2
import numpy as np
from ultralytics import YOLO

# Target vehicle classes in COCO dataset:
# 2: car, 3: motorcycle, 5: bus, 7: truck
TARGET_CLASSES = [2, 3, 5, 7]

def main():
    # Ensure all operations (like model downloads) stay in this script's folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory set to: {os.getcwd()}")

    # Load the YOLOv8 Nano model
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    
    # Open default webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    window_name = "Adaptive Headlight Presentation Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Starting video stream... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
            
        height, width, _ = frame.shape
        
        # Zone boundaries (dividing the width into 3 vertical zones)
        zone_width = width // 3
        
        # Flags
        is_car_detected = False
        is_glare_detected = False
        
        # LED states (Bright Yellow by default)
        led_color_bright = (0, 255, 255) # BGR format: Bright Yellow
        led_color_dim = (0, 100, 100)    # BGR format: Dark Yellow
        
        led_states = {
            'L': led_color_bright,
            'C': led_color_bright,
            'R': led_color_bright
        }
        
        # 1. Global Glare / Bright Headlight Detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply a threshold to find very bright areas (like the flashlight in the image)
        _, bright_mask = cv2.threshold(gray_frame, 220, 255, cv2.THRESH_BINARY)
        
        # Need to find contours of these bright spots
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100: # Threshold for area to avoid small noise spots
                is_glare_detected = True
                
                # Get bounding box of the bright spot
                x, y, w, h = cv2.boundingRect(cnt)
                center_x = x + w // 2
                
                # Dim the corresponding LED based on the center X position
                if center_x < zone_width:
                    led_states['L'] = led_color_dim
                elif center_x < 2 * zone_width:
                    led_states['C'] = led_color_dim
                else:
                    led_states['R'] = led_color_dim
                
                # Draw a rectangle around the glare source
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
                cv2.putText(frame, "BRIGHT LIGHT", (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # 2. Run YOLO detection on the frame (for cars that might not be glaring but are still in front)
        results = model(frame, stream=True, verbose=False)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in TARGET_CLASSES:
                    is_car_detected = True
                    
                    # Extract rounding bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Ensure coordinates are within image boundaries
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    
                    # Determine the center X of the bounding box
                    center_x = (x1 + x2) // 2
                    
                    # Dim the corresponding LED based on the center X position
                    if center_x < zone_width:
                        led_states['L'] = led_color_dim
                    elif center_x < 2 * zone_width:
                        led_states['C'] = led_color_dim
                    else:
                        led_states['R'] = led_color_dim
                        
                    # Extract the Region of Interest (the vehicle) and check brightness
                    if y2 > y1 and x2 > x1:
                        roi = frame[y1:y2, x1:x2]
                        # Convert ROI to grayscale
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        # Calculate the average brightness
                        avg_brightness = np.mean(gray_roi)
                        
                        # Check for glare just within this vehicle's bounding box to be extra safe
                        if avg_brightness > 180:
                            is_glare_detected = True
                            box_color = (0, 0, 255) # Red
                            text = f"GLARE DETECTED! (Bright: {avg_brightness:.1f})"
                        else:
                            box_color = (0, 255, 0) # Green
                            text = f"Car Safe (Bright: {avg_brightness:.1f})"
                            
                        # Draw the bounding box and text on the frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        cv2.putText(frame, text, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # Draw the overall system status text
        if is_glare_detected:
            status_text = "SIMULATOR: LOW BEAM + NARROW"
            status_color = (0, 0, 255) # Red
        elif is_car_detected:
            status_text = "SIMULATOR: WIDE BEAM"
            status_color = (0, 255, 0) # Green
        else:
            status_text = "SIMULATOR: HIGH BEAM (Clear Road)"
            status_color = (255, 255, 255) # White
            
        # Place text at the top left
        cv2.putText(frame, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)

        # Draw simulated 3x3 LED Matrix (like the STM32) at the bottom
        led_radius = 20
        # Start X,Y for the top-left of the matrix
        matrix_start_x = width // 2 - 60
        matrix_start_y = height - 160
        spacing = 60
        
        # Columns represent Left, Center, Right zones
        matrix_zones = ['L', 'C', 'R']
        
        # Draw a bounding box for the "headlight assembly"
        cv2.rectangle(frame, (matrix_start_x - 40, matrix_start_y - 40), 
                             (matrix_start_x + 2 * spacing + 40, matrix_start_y + 2 * spacing + 60), 
                             (20, 20, 20), -1)
        cv2.putText(frame, "3x3 MATRIX HEADLIGHT", (matrix_start_x - 40, matrix_start_y - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        for col_idx, zone in enumerate(matrix_zones):
            state = led_states[zone] 
            
            for row_idx in range(3):
                x = matrix_start_x + col_idx * spacing
                y = matrix_start_y + row_idx * spacing
                
                # Logic: If DIM, the top LED (row_idx = 0) is turned OFF
                if state == led_color_dim and row_idx == 0:
                    color = (50, 50, 50) # Dark Gray (OFF)
                else:
                    color = state
                    
                # Draw the LED
                cv2.circle(frame, (x, y), led_radius, color, -1)
                cv2.circle(frame, (x, y), led_radius, (0, 0, 0), 2)
                
        # Label the columns below the matrix
        for col_idx, zone in enumerate(matrix_zones):
            x = matrix_start_x + col_idx * spacing
            y = matrix_start_y + 2 * spacing + 40
            text_size = cv2.getTextSize(zone, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.putText(frame, zone, (x - text_size[0] // 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Show the live feed
        cv2.imshow(window_name, frame)
        
        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
