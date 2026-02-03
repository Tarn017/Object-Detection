# Object-Detection

Ziel dieser Anleitung ist es ein neuronales Netz für Object Detection zu trainieren und auf einem Arduino-Microcontroller zu nutzen.

**Steps:**  
1. EspCam einrichten
2. Daten Sammeln
3. Daten annotieren
4. Neuronales Netz trainieren
5. Neuronales Netz testen
6. Arduino Microcontroller einrichten
7. Trainiertes Netz auf Microcontroller deployen

# 1. Einrichten der EspCam

1.	EspCam auf Adapter stecken
2.	In der Arduino IDE: Board-Manager öffnen über Tools -> Board -> Boards Manager
3.	esp32 by Espressif im Boardmanager installieren (Über Suchleiste oben suchen)
4.	In Arduino IDE: File -> Examples -> Esp32 -> Camera -> CameraWebServer
5.	Ergänze Wlan-Daten in CameraWebServer.ino
6.	In der Tableiste oben gegebenenfalls auf board_config.h wechseln (Falls die `#define`-Einstellungen nicht direkt in dieser Datei getroffen werden) 
7.	Vor `#define CAMERA_MODEL_WROVER_KIT` die Striche // hinzufügen
8.	Vor `#define CAMERA_MODEL_AI_THINKER`  // entfernen
9.	Zurück in CameraWebServer.ino passenden Port auswählen und als Board unter esp32 "AI Thinker Esp32-Cam"
10.	Code uploaden
11.	Auf dem Serial Monitor wir nun eine URL ausgegeben:

![cam_url](https://raw.githubusercontent.com/Tarn017/Object-Detection/main/assets/cam_url.png)

12. Kopiere die URL nun in einen Brrowser der mit demselben WLAN, verbundne ist, wie die Kamera, um zu sehen ob alles korrekt funktioniert hat

# 2. Daten Sammeln

**Einrichten:**  
Erstellt in eurer Python IDE ein neues Projekt und legt ein nues Haupt-File an (bspw. main.py).  
Ladet das folgende Skript herunter und kopiert es in euer Python-Projekt sodass es direkt neben eurem Haupt-File zu sehen ist: [project.py](https://github.com/Tarn017/Object-Detection/blob/main/src/project.py)

![project.py Screenshot](https://raw.githubusercontent.com/Tarn017/Object-Detection/main/assets/project_py.png)

**Abhängigkeiten installieren:**
Lade die folgende Datei herunter und kopiere sie in deinen Projektordner: [requirements_det.txt](https://github.com/Tarn017/Object-Detection/blob/main/src/requirements_class.txt)

Gehe anschließend in deiner IDE unten auf das Terminal und installiere alle Pakete mit `pip install -r requirements_class.txt`:

![requirements Screenshot](https://raw.githubusercontent.com/Tarn017/Object-Detection/main/assets/requirements.png)

**Los gehts:**  
Beim einrichten der EspCam wurde eine URL ausgegeben. Diese wird für die nächsten Schritte benötigt.  
Geht in euer Hauptskrip und fügt ganz oben die Zeile `from project import aufnahme` ein. Als nächstes fügt ihr darunter `if __name__ == "__main__":` ein. Alles was hiernah kommt muss nach rechts eingerückt werden. Nun kann `aufnahme` genutzt werden um Bilddaten zu sammeln. Sie nimmt alle paar Sekunden ein Bild auf und speichert dieses automatisch in einer benannten Klasse. Die Funktion ist folgendermaßen aufgebaut:  
`aufnahme(url, interval, klasse, ordner)`: *url* entspricht der URL von der EspCam, diese kann einfach kopiert werden. Wichtig ist nur, dass sie in Anführungszeichen steht. *interval* entspricht der Frequenz, in der Bilder aufgenommen werden (1 bspw. für alle 1 Sekunden). *klasse* sollte dem Klassenname entsprechen, für dessen Klasse gerade Daten gesammelt werden. Für *ordner* kann ebenfalls ein beliebiger Name gewählt werden. Es wird ein Ordner mit selbigem Namen automatisch erstellt in dem die Klassen und Bilder gespeichert werden.

Hier ein Beispiel. Wichtig ist, dass Anführungszeichen übernommen werden, dort wo sie gebraucht werden:  
```python
from project import CNN, aufnahme, testen_classification, neural_network_classification, FFN

if __name__ == "__main__":
    aufnahme(url='http://172.20.10.3', interval=1, klasse='noise', ordner='Objekte')
```

Zum starten kann nun einfach das Skript ausgeführt werden. Für jede Klasse, für die Daten gesammelt werden sollen, muss das Skript separat ausgeführt werden. Ein umbenennen von *klasse* ist während das Skript läuft nicht möglich.

Eine genauere Beschreibung der einzelnen Schritte findet ihr unter [Getting Started With ESP32-CAM](https://lastminuteengineers.com/getting-started-with-esp32-cam/)

