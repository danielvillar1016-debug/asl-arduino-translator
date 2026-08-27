#include <LiquidCrystal.h>

LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

void setup() {
  Serial.begin(115200);

  lcd.begin(16, 2);

  lcd.setCursor(0, 0);
  lcd.print("ASL Letter:");
}

void loop() {
  if (Serial.available() > 0) {

    char letter = Serial.read();

    if (letter >= 'A' && letter <= 'Z') {

      lcd.setCursor(0, 1);
      lcd.print("                ");

      lcd.setCursor(0, 1);
      lcd.print(letter);
    }
  }
}