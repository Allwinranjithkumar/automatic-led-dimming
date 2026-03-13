// Matrix pins mapping based on your diagram
int rowPins[3] = {7, 6, 5};   // Row 0: Top, Row 1: Middle, Row 2: Bottom
int colPins[3] = {4, 3, 2};   // Col 0: Left, Col 1: Center, Col 2: Right

// Brightness state for each column
// 2 = BRIGHT (High Beam - all 3 LEDs ON)
// 1 = DIM (Low Beam - Top LED OFF, Mid/Bot ON)
int colState[3] = {2, 2, 2}; // Left, Center, Right

void setup() {
  Serial.begin(115200); // Fast baud rate for smooth communication with Python
  
  // Initialize pins
  for (int i = 0; i < 3; i++) {
    pinMode(rowPins[i], OUTPUT);
    pinMode(colPins[i], OUTPUT);
    digitalWrite(rowPins[i], LOW);  // All rows OFF (Anodes)
    digitalWrite(colPins[i], HIGH); // All cols OFF (Cathodes)
  }
}

void loop() {
  // 1. Read commands from Python script
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim(); // Remove whitespace/newlines
    
    if (cmd == "L:DIM") colState[0] = 1;
    else if (cmd == "L:BRIGHT") colState[0] = 2;
    else if (cmd == "C:DIM") colState[1] = 1;
    else if (cmd == "C:BRIGHT") colState[1] = 2;
    else if (cmd == "R:DIM") colState[2] = 1;
    else if (cmd == "R:BRIGHT") colState[2] = 2;
  }

  // 2. Matrix multiplexing (Scanning columns rapidly)
  for (int col = 0; col < 3; col++) {
    
    // Step A: Turn off all columns to prevent ghosting
    for (int c = 0; c < 3; c++) digitalWrite(colPins[c], HIGH);
    
    // Step B: Set the correct rows for the current column
    for (int row = 0; row < 3; row++) {
      bool isOn = false;
      
      if (colState[col] == 2) {
        // High Beam: Turn on all LEDs in this column
        isOn = true;
      } 
      else if (colState[col] == 1) {
        // Low Beam (Dim): Turn OFF top LED (Row 0), turn ON Mid (Row 1) & Bot (Row 2)
        if (row > 0) isOn = true;
      }
      
      digitalWrite(rowPins[row], isOn ? HIGH : LOW);
    }
    
    // Step C: Turn ON the current column
    digitalWrite(colPins[col], LOW);
    
    // Step D: Very short delay for Persistence of Vision (POV)
    delay(2);
  }
}
