import os
import cv2
import numpy as np
from ultralytics import YOLO
import serial
import time

# Target vehicle classes in COCO dataset:
# 2: car, 3: motorcycle, 5: bus, 7: truck
TARGET_CLASSES = [2, 3, 5, 7]

def main():
    # Ensure all operations (like model downloads) stay in this script's folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory set to: {os.getcwd()}")

    # ---------------------------------------------------------
    # STM32 Serial Connection Setup
    # ---------------------------------------------------------
    # IMPORTANT: Change 'COM3' to the actual COM port of your STM32 Nucleo
    # You can find it in the Arduino IDE or Device Manager
    stm32_port = 'COM3' 
    baud_rate = 115200
    stm32 = None
    
    try:
        print(f"Attempting to connect to STM32 on {stm32_port}...")
        stm32 = serial.Serial(stm32_port, baud_rate, timeout=1)
        print("Successfully connected to STM32!")
        time.sleep(2) # Give the microcontroller a moment to initialize
    except serial.SerialException as e:
        print(f"WARNING: Could not connect to STM32 on {stm32_port}.")
        print("The simulation will still run, but physical LEDs won't update.")
        print(f"Error details: {e}\n")

    # Load the YOLOv8 Nano model
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    
    # Open default webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    window_name = "Adaptive Headlight Presentation Demo (STM32)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Starting video stream... Press 'q' to exit.")

    # Track previous states to avoid spamming the serial port
    prev_zone_states = {'L': None, 'C': None, 'R': None}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
            
        height, width, _ = frame.shape
        zone_width = width // 3
        
        is_car_detected = False
        is_glare_detected = False
        
        led_color_bright = (0, 255, 255) # BGR: Bright Yellow
        led_color_dim = (0, 100, 100)    # BGR: Dark Yellow
        
        # Reset states at the start of every frame
        led_states = {'L': led_color_bright, 'C': led_color_bright, 'R': led_color_bright}
        zone_states = {'L': "BRIGHT", 'C': "BRIGHT", 'R': "BRIGHT"}
        
        # 1. Global Glare / Bright Headlight Detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray_frame, 220, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                is_glare_detected = True
                x, y, w, h = cv2.boundingRect(cnt)
                center_x = x + w // 2
                
                # Dim the corresponding LED zone
                if center_x < zone_width:
                    led_states['L'] = led_color_dim
                    zone_states['L'] = "DIM"
                elif center_x < 2 * zone_width:
                    led_states['C'] = led_color_dim
                    zone_states['C'] = "DIM"
                else:
                    led_states['R'] = led_color_dim
                    zone_states['R'] = "DIM"
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
                cv2.putText(frame, "BRIGHT LIGHT", (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # 2. Run YOLO detection
        results = model(frame, stream=True, verbose=False)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in TARGET_CLASSES:
                    is_car_detected = True
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    
                    center_x = (x1 + x2) // 2
                    
                    # Dim the corresponding LED zone
                    if center_x < zone_width:
                        led_states['L'] = led_color_dim
                        zone_states['L'] = "DIM"
                    elif center_x < 2 * zone_width:
                        led_states['C'] = led_color_dim
                        zone_states['C'] = "DIM"
                    else:
                        led_states['R'] = led_color_dim
                        zone_states['R'] = "DIM"
                        
                    if y2 > y1 and x2 > x1:
                        roi = frame[y1:y2, x1:x2]
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        avg_brightness = np.mean(gray_roi)
                        
                        if avg_brightness > 180:
                            is_glare_detected = True
                            box_color = (0, 0, 255) # Red
                            text = f"GLARE DETECTED! (Bright: {avg_brightness:.1f})"
                        else:
                            box_color = (0, 255, 0) # Green
                            text = f"Car Safe (Bright: {avg_brightness:.1f})"
                            
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        cv2.putText(frame, text, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # ---------------------------------------------------------
        # SEND COMMANDS TO STM32
        # ---------------------------------------------------------
        for zone in ['L', 'C', 'R']:
            # Only send the command if the state has changed since last frame
            if zone_states[zone] != prev_zone_states[zone]:
                cmd = f"{zone}:{zone_states[zone]}\n"
                
                # Print to terminal for debugging
                print(f"Sending to STM32 -> {cmd.strip()}")
                
                if stm32:
                    try:
                        stm32.write(cmd.encode('utf-8'))
                    except Exception as e:
                        print(f"Serial write error: {e}")
                
                prev_zone_states[zone] = zone_states[zone]

        # Draw system status
        if is_glare_detected:
            status_text = "SIMULATOR: LOW BEAM + NARROW"
            status_color = (0, 0, 255) # Red
        elif is_car_detected:
            status_text = "SIMULATOR: WIDE BEAM"
            status_color = (0, 255, 0) # Green
        else:
            status_text = "SIMULATOR: HIGH BEAM (Clear Road)"
            status_color = (255, 255, 255) # White
            
        cv2.putText(frame, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)

        # Draw matrix simulation
        led_radius = 40
        led_y = height - 60
        led_positions = {
            'L': (width // 6, led_y),
            'C': (width // 2, led_y),
            'R': (5 * width // 6, led_y)
        }
        
        for label, pos in led_positions.items():
            cv2.circle(frame, pos, led_radius, led_states[label], -1)
            cv2.circle(frame, pos, led_radius, (0, 0, 0), 2)
            
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = pos[0] - text_size[0] // 2
            text_y = pos[1] + text_size[1] // 2
            
            cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    if stm32:
        stm32.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
