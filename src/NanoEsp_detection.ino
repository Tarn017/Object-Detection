#include <WiFi.h>
#include <ArduinoJson.h>

const int buttonPin = 2;     // the number of the pushbutton pin

// variables will change:
int buttonState = 0;         // variable for reading the pushbutton status

// WLAN-Zugangsdaten
const char* ssid = "TP-Link_92AC";
const char* password = "73125785";

//const char *ssid = "Vodafone-AC24";
//const char *password = "fc6FkCHycEL2qX3n";

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
          String jsonLine = client.readStringUntil('\n');
          jsonLine.trim();
          Serial.println("Empfangen JSON:");
          Serial.println(jsonLine);

          const size_t capacity = 2048; // evtl 1024, 2048, 4096 ... je nach Bedarf
          DynamicJsonDocument doc(capacity);

          DeserializationError err = deserializeJson(doc, jsonLine);
          if (err) {
            Serial.print("deserializeJson failed: ");
            Serial.println(err.c_str());
            return;
          }

          JsonArray arr = doc.as<JsonArray>();
          for (size_t i = 0; i < arr.size(); ++i) {
            JsonObject obj = arr[i];
            const char* label = obj["label"];
            float conf = obj["confidence"];
            JsonArray bb = obj["bbox"];
            int x1 = bb[0], y1 = bb[1], x2 = bb[2], y2 = bb[3];

            Serial.print("Objekt "); Serial.print(i);
            Serial.print(": "); Serial.print(label);
            Serial.print(", conf="); Serial.print(conf);
            Serial.print(", bbox=("); Serial.print(x1); Serial.print(",");
            Serial.print(y1); Serial.print(","); Serial.print(x2); Serial.print(",");
            Serial.print(y2); Serial.println(")");

          }
      }
    }
    client.stop();
    Serial.println("Client getrennt.");
  }
}
