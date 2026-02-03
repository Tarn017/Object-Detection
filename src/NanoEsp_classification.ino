#include <WiFi.h>

const int buttonPin = 2;     // the number of the pushbutton pin

// variables will change:
int buttonState = 0;         // variable for reading the pushbutton status

// WLAN-Zugangsdaten
const char* ssid = "Vodafone123";
const char* password = "123456789";

const uint16_t port = 12345; // Portnummer für die Verbindung

WiFiServer server(port);

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  pinMode(LED_RED, OUTPUT);
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_BLUE, OUTPUT);

  Serial.print("Verbinde mit dem WLAN...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }
  Serial.println("\nVerbunden!");
  Serial.print("IP-Adresse: ");
  Serial.println(WiFi.localIP());

  server.begin();
  Serial.println("Server gestartet, wartet auf Verbindungen...");
  pinMode(buttonPin, INPUT);
}

void loop() {
  WiFiClient client = server.available();
  if (client) {
    Serial.println("Client verbunden.");
    while (client.connected()) {
          buttonState = digitalRead(buttonPin);

    if (buttonState == HIGH) {
      int i = 42;
      Serial.println("pressed");
      client.println(i);
      delay(500);
    }
      if (client.available()) {
        String data = client.readStringUntil('\n');
        Serial.print("Empfangene Daten: ");
        Serial.println(data);

        int commaIndex = data.indexOf(',');

        // Überprüfen, ob ein Komma gefunden wurde
        if (commaIndex != -1) {
          // Klasse extrahieren
          String klasse = data.substring(0, commaIndex);
          // Konfidenz extrahieren und in float konvertieren
          String confStr = data.substring(commaIndex + 1);
          float conf = confStr.toFloat();
          Serial.println(klasse);
          Serial.println(conf);
        }

      }
    }
    client.stop();
    Serial.println("Client getrennt.");
  }
}
