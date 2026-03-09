import os
import cv2
import numpy as np
from ultralytics import YOLO

# Target vehicle classes in COCO dataset:
# 2: car, 3: motorcycle, 5: bus, 7: truck
TARGET_CLASSES = [2, 3, 5, 7]

def main():
    # Ensure operations stay in this script's folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print("--- STARTING PRESENTATION MODE ---")
    print(f"Working directory: {os.getcwd()}")

    # Load YOLOv8 Nano model
    print("Loading AI Model...")
    model = YOLO('yolov8n.pt')
    
    # Open default webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set up presentation window
    window_name = "Adaptive Matrix Headlight System - PRESENTATION DEMO"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720) # Nice large default size

    print("System active! Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        height, width, _ = frame.shape
        zone_width = width // 3
        
        is_car_detected = False
        is_glare_detected = False
        
        # Colors (BGR format)
        COLOR_YELLOW_ON = (0, 255, 255)
        COLOR_YELLOW_DIM = (0, 100, 100)
        COLOR_GREY_OFF = (60, 60, 60)
        
        # Matrix states: default to High Beams (all bright)
        # 3x3 array [row][col] where 0,0 is Top-Left and 2,2 is Bottom-Right
        matrix_states = [
            [COLOR_YELLOW_ON, COLOR_YELLOW_ON, COLOR_YELLOW_ON], # Top Row
            [COLOR_YELLOW_ON, COLOR_YELLOW_ON, COLOR_YELLOW_ON], # Middle Row
            [COLOR_YELLOW_ON, COLOR_YELLOW_ON, COLOR_YELLOW_ON]  # Bottom Row
        ]
        
        # Determine 3 vertical zones (Columns)
        zone_width = width // 3
        # Determine 3 horizontal zones (Rows)
        zone_height = height // 3
        
        # Helper function to check if two rectangles intersect
        def rects_intersect(x1, y1, w1, h1, x2, y2, w2, h2):
            return not (x1 > x2+w2 or x1+w1 < x2 or y1 > y2+h2 or y1+h1 < y2)

        # 1. Glare Detection (Bright Lights)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray_frame, 220, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                is_glare_detected = True
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Check EVERY zone to see if the glare box touches it
                for r_idx in range(3):
                    for c_idx in range(3):
                        zone_x = c_idx * zone_width
                        zone_y = r_idx * zone_height
                        
                        if rects_intersect(x, y, w, h, zone_x, zone_y, zone_width, zone_height):
                            matrix_states[r_idx][c_idx] = COLOR_GREY_OFF
                
                # Draw box around glare source
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
                cv2.putText(frame, "GLARE AREA", (x, max(y - 10, 20)), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 165, 255), 2)

        # 2. YOLO Vehicle Detection
        results = model(frame, stream=True, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in TARGET_CLASSES:
                    is_car_detected = True
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    car_w = x2 - x1
                    car_h = y2 - y1
                        
                    # Brightness check inside bounding box
                    if y2 > y1 and x2 > x1:
                        avg_brightness = np.mean(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY))
                        if avg_brightness > 180:
                            is_glare_detected = True
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, "VEHICLE+GLARE", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)
                            
                            # Dim ALL zones touching this glaring vehicle
                            for r_idx in range(3):
                                for c_idx in range(3):
                                    zone_x = c_idx * zone_width
                                    zone_y = r_idx * zone_height
                                    if rects_intersect(x1, y1, car_w, car_h, zone_x, zone_y, zone_width, zone_height):
                                        matrix_states[r_idx][c_idx] = COLOR_GREY_OFF
                            
                        else:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, "VEHICLE", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)
                            
                            # Soft-dim ALL zones touching this safe vehicle
                            for r_idx in range(3):
                                for c_idx in range(3):
                                    zone_x = c_idx * zone_width
                                    zone_y = r_idx * zone_height
                                    if rects_intersect(x1, y1, car_w, car_h, zone_x, zone_y, zone_width, zone_height):
                                        if matrix_states[r_idx][c_idx] != COLOR_GREY_OFF: 
                                            matrix_states[r_idx][c_idx] = COLOR_YELLOW_DIM

        # ---------------------------------------------------------
        # DRAW THE 3x3 VIRTUAL MATRIX HEADLIGHT
        # ---------------------------------------------------------
        # Top Header Status
        status_bg_color = (0, 100, 0) if is_car_detected else (50, 50, 50)
        status_bg_color = (0, 0, 150) if is_glare_detected else status_bg_color
        
        cv2.rectangle(frame, (0, 0), (width, 50), status_bg_color, -1)
        if is_glare_detected:
            title_text = "SYSTEM ACTIVE: PINPOINT DIMMING (Glare Prevented)"
        elif is_car_detected:
            title_text = "SYSTEM ACTIVE: PINPOINT DIMMING (Vehicle Detected)"
        else:
            title_text = "SYSTEM ACTIVE: HIGH BEAMS ON (Clear Road)"
            
        cv2.putText(frame, title_text, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        # Draw the Matrix Box at the bottom center
        led_radius = 25
        spacing = 70
        
        # Center the matrix
        matrix_w = 2 * spacing
        matrix_start_x = (width - matrix_w) // 2
        matrix_start_y = height - 260
        
        # Draw dark assembly box
        cv2.rectangle(frame, (matrix_start_x - 40, matrix_start_y - 40), 
                             (matrix_start_x + matrix_w + 40, matrix_start_y + 2 * spacing + 60), 
                             (30, 30, 30), -1)
        # White border
        cv2.rectangle(frame, (matrix_start_x - 40, matrix_start_y - 40), 
                             (matrix_start_x + matrix_w + 40, matrix_start_y + 2 * spacing + 60), 
                             (150, 150, 150), 2)
                             
        cv2.putText(frame, "VIRTUAL 3x3 MATRIX", (matrix_start_x - 30, matrix_start_y - 15), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 1)

        matrix_zones = ['LEFT', 'CENTER', 'RIGHT']
        
        # Draw the LEDs!
        for col_idx in range(3):
            for row_idx in range(3):
                x = matrix_start_x + col_idx * spacing
                y = matrix_start_y + row_idx * spacing
                
                # Fetch color from our new 9-LED matrix array
                current_led_color = matrix_states[row_idx][col_idx]
                    
                # Draw LED (filled) then the outline
                cv2.circle(frame, (x, y), led_radius, current_led_color, -1)
                cv2.circle(frame, (x, y), led_radius, (0, 0, 0), 2)
                
                # Add a little "glow" effect if it is ON
                if current_led_color != COLOR_GREY_OFF:
                    cv2.circle(frame, (x, y), led_radius + 4, current_led_color, 1)
                
        # Draw zone labels under the matrix
        for col_idx, zone in enumerate(["LEFT", "CENTER", "RIGHT"]):
            x = matrix_start_x + col_idx * spacing
            y = matrix_start_y + 2 * spacing + 45
            text_size = cv2.getTextSize(zone, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
            
            # Highlight label if ANY led in this column is dimmed or off
            is_active = any(matrix_states[r][col_idx] != COLOR_YELLOW_ON for r in range(3))
            label_color = (0, 255, 255) if is_active else (150, 150, 150)
            
            cv2.putText(frame, zone, (x - text_size[0] // 2, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, label_color, 1)

        # Show presentation frame
        cv2.imshow(window_name, frame)
        
        # 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
