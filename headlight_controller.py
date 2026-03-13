import cv2
import serial
import time

# Step 4 Setup: Initialize serial communication (adjust port/baudrate as needed for your STM32)
# "COM3" is a placeholder for Windows. Change this to the correct port for your STM32.
serial_port = 'COM3'
baud_rate = 115200

try:
    ser = serial.Serial(serial_port, baud_rate, timeout=1)
    time.sleep(2) # Give the serial connection a moment to initialize
    print(f"Successfully connected to STM32 on {serial_port}")
except Exception as e:
    print(f"Warning: Could not open serial port {serial_port}. Running in simulation mode without sending actual UART commands. Error: {e}")
    ser = None

# Initialize the webcam
try:
    cap = cv2.VideoCapture(0) # 0 is typically the default built-in webcam
    print("Webcam initialized successfully. Press 'q' to quit the window.")
except Exception as e:
    print(f"Error initializing webcam: {e}")
    exit()

# Thresholds for the "Anti-Gravity" Math decision logic
# You will likely need to tweak these numbers based on real-world testing with your specific camera.
SMALL_AREA_MAX = 5000   # Below this area, it's considered "Far away"
MEDIUM_AREA_MAX = 20000 # Below this area, it's considered "Getting closer"

# Variable to keep track of the last sent command to prevent spamming the serial port
last_command = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam")
        break

    # -------------------------------------------------------------------------
    # Step 1: Put on the "Digital Sunglasses" (Thresholding)
    # -------------------------------------------------------------------------
    # Convert the frame to grayscale. We only care about brightness, not color.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a high threshold. Anything dimmer than 200 (out of 255) becomes pitch black (0).
    # Anything brighter than 200 becomes bright white (255).
    # This filters out the road, trees, etc., leaving only bright headlights.
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)


    # -------------------------------------------------------------------------
    # Step 2: Draw the Box (Contour Mapping)
    # -------------------------------------------------------------------------
    # Find the edges of the white blobs (the "connect-the-dots" phase)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # We want to focus on the *largest* bright spot on the screen
    largest_contour = None
    max_area = 0
    bounding_box = None

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # ---------------------------------------------------------------------
        # Step 3: The "Anti-Gravity" Math (Distance Estimation)
        # ---------------------------------------------------------------------
        # Simple math: Width * Height = Area
        area = w * h
        
        if area > max_area:
            max_area = area
            bounding_box = (x, y, w, h)
            largest_contour = contour

    # -------------------------------------------------------------------------
    # Step 4: Send the Text Message (UART Serial)
    # -------------------------------------------------------------------------
    if bounding_box is not None:
        x, y, w, h = bounding_box
        
        # Draw the green bounding box on the original frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Determine the command based on the area size
        command = ""
        if max_area < SMALL_AREA_MAX:
            # Box = Small (Far away)
            command = "DIM_HIGH_BEAM"
        elif max_area < MEDIUM_AREA_MAX:
            # Box = Medium (Getting closer)
            command = "TURN_OFF_HIGH_DIM_MID"
        else:
            # Box = Huge (Right in our face)
            command = "LOW_BEAM_ONLY"

        # Display the math and command on the screen
        text = f"Area: {max_area} | Cmd: {command}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Send the command to the STM32 via USB/UART if the command changed
        # We only send if it changed to avoid flooding the UART buffer with identical messages
        if command != last_command:
            print(f"Sending UART Command: {command} (Area: {max_area})")
            if ser:
                # Encode text to bytes before sending
                ser.write((command + "\n").encode('utf-8'))
            last_command = command

    else:
        # If no bright lights are detected, you might want a default state (e.g., full high beams)
        # For safety we just reset last_command if needed
        # last_command = ""
        pass

    # Show the video windows (useful for debugging and setup)
    cv2.imshow("Main Camera - Target Tracking", frame)
    cv2.imshow("Digital Sunglasses View", thresh)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up gracefully
cap.release()
if ser:
    ser.close()
cv2.destroyAllWindows()
print("Program terminated successfully.")
