// #include <Arduino.h>

// #define PUL_PIN 2
// #define DIR_PIN 3
// #define ENA_PIN 4

// const int STEPS_PER_MOVE = 50;     // amount moved each time
// const int STEP_DELAY_US = 1000;    // smaller = faster
// const int PAUSE_MS = 10000;         // pause between each small move

// void moveSteps(int steps, bool forward) {
//   digitalWrite(DIR_PIN, forward ? HIGH : LOW);
//   delay(10);  // let direction settle

//   for (int i = 0; i < steps; i++) {
//     digitalWrite(PUL_PIN, HIGH);
//     delayMicroseconds(STEP_DELAY_US);
//     digitalWrite(PUL_PIN, LOW);
//     delayMicroseconds(STEP_DELAY_US);
//   }
// }

// void setup() {
//   pinMode(PUL_PIN, OUTPUT);
//   pinMode(DIR_PIN, OUTPUT);
//   pinMode(ENA_PIN, OUTPUT);

//   digitalWrite(ENA_PIN, HIGH);   // if your driver works opposite, try LOW
// }

// void loop() {
//   // Forward 5 times
//   for (int i = 0; i < 10; i++) {
//     moveSteps(STEPS_PER_MOVE, true);
//     delay(PAUSE_MS);
//   }

//   // Backward 5 times
//   for (int i = 0; i < 10; i++) {
//     moveSteps(STEPS_PER_MOVE, false);
//     delay(PAUSE_MS);
//   }
// }


//NEW

#include <Arduino.h>

#define PUL_PIN 2
#define DIR_PIN 3
#define ENA_PIN 4

const int STEP_DELAY_US = 1000;
const int STEPS_PER_MOVE = 50;

void moveSteps(int steps, bool forward) {
  digitalWrite(DIR_PIN, forward ? HIGH : LOW);
  delay(10); 

  for (int i = 0; i < steps; i++) {
    digitalWrite(PUL_PIN, HIGH);
    delayMicroseconds(STEP_DELAY_US);
    digitalWrite(PUL_PIN, LOW);
    delayMicroseconds(STEP_DELAY_US);
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(PUL_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(ENA_PIN, OUTPUT);

  digitalWrite(ENA_PIN, HIGH);   // if needed, try LOW
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "FWD") {
      moveSteps(STEPS_PER_MOVE, true);
      Serial.println("done");
    } else if (cmd == "BWD") {
      moveSteps(STEPS_PER_MOVE, false);
      Serial.println("done");
    }
  }
}
