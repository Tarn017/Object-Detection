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
Geht in euer Hauptskrip und fügt ganz oben die Zeile `from project_det import aufnahme` ein. Nun kann `aufnahme` genutzt werden um Bilddaten zu sammeln. Sie nimmt alle paar Sekunden ein Bild auf und speichert dieses automatisch im benannten Ordner. Dieser wird automatisch erstellt und muss daher nicht extra angelegt werden. Die uafgenommenen Bilder dürfen mehrere verschiedene Objekte gleichzeitig oder auch kein Objekt enthalten. Die Funktion ist folgendermaßen aufgebaut:  
`aufnahme(url, interval, ordner)`:  
*url* entspricht der URL von der EspCam, diese kann einfach kopiert werden. Wichtig ist nur, dass sie in Anführungszeichen steht.  
*interval* entspricht der Frequenz, in der Bilder aufgenommen werden (1 bspw. für alle 1 Sekunden).  
*ordner* kann als ein beliebiger Name gewählt werden. Es wird ein Ordner mit selbigem Namen automatisch erstellt in dem die Klassen und Bilder gespeichert werden.

Hier ein Beispiel. Wichtig ist, dass Anführungszeichen übernommen werden, dort wo sie gebraucht werden:  
```python
from project_det import aufnahme

aufnahme(url='http://172.20.10.3', interval=1, ordner='Objekte')
```

# 3. Daten Annotieren
Die gesammelten Daten müssen im nächsten Schritt richtig gelabelt werden. Hierfür wird im Folgenden Roboflow verwendet: [Roboflow](https://roboflow.com/)  
Lege einen Account auf der Seite an und erstelle einen Workspace. Gehe nun auf der linken Seite auf *Projects* und erstelle ein neues Projekt. Wähle für *Project Name* und *Annotation Group* einen beliebigen Namen und als *Project Type* Object Detection. Die Folgende Anleitung wird sich auf den traditionellen Modus beziehen, es kann jedoch auch gerne der *rapid*-Modus ausprobiert werden.  
Nun müssen lediglich die Punkte in der linken Leiste nacheinander abgearbeitet werden:  
1. Ladet den Ordner mit den von euch gesammelten Daten hoch und drückt *Safe & Continue*
2. Annotiert die Daten. Zieht dafür einfach mit der Maus ein Rechteck über das entsprechende Objekt im Bild und gebt diesem das richtige Label. Ist kein Objekt auf dem Bild zu sehen, so muss dies auf der rechten Seite ausgewählt werden. Es kann gerne das Auto-Labeling ausprobiert werden, jedoch sollte danach jedes Bild noch einmal überprüft werden.
3. Fügt die annotierten Bilder dem Datensatz hinzu mit der method *Add All Images To Training Set*
4. Geht nun in der linken Leiste auf *Versions*.
5. Wählt unter *Train/Test Split* das Feld *Rebalance* aus und stellt die von euch gewünschte Verteilung ein. Wichtig ist, dass für train, val und test mindestens ein Bild vorhanden ist. Klickt anschließend auf *continue*.
6. Unter Preprocessing können die Bilder auf ein von euch gewünschtes Format gerisized werden. Ein quadratisches Format ist hierbei empfohlen. Klickt anschließend auf *continue*.
7. Unter Augmentation können nun verschiedene Augmentation-Methoden ausgewählt und hinzugefügt werden. Klickt anschließend auf *continue*.
8. Im letzten Schritt unter Create kann nun ausgewählt werden, wie groß der durch die künstlich veränderten Bilder erweiterte Datensatz werden soll. Klickt anschließend *create*.
9. Kliecke nun auf *Download Dataset* und wähle als Format *YOLOv8* und unten *Show Download Code*.
10. Kopiere den angezeigten Code in dein main-Skript (ohne das pip-install oben).

# 4. Neuronales Netz Trainieren
Nutzt für das Training des Netzes die Funktion `training_detection()`. Kopiert diese dafür unter den Download-Code eures Datensatzes.  
`training_detection(dataset, epochen, img_size)`:  
*dataset* wurde bereits durch den Download-Code definiert-
*epochen* entspricht der Anzahl an Epochen die das Netz trainiert werden soll.
*img_size* entspricht der Bildgröße auf die geresized werden soll.  
**Beispiel:**  
```python
from roboflow import Roboflow
from project_det import training_detection

rf = Roboflow(api_key="Lbp6tBzjKuXWSq7ndLgS")
project = rf.workspace("karlsruher-institut-fr-technologie-7bdnc").project("zml_detect-pemh9")
version = project.version(1)
dataset = version.download("yolov8")

training_detection(dataset, epochen=20, img_size=(640,640))
```

**Training Auswerten:**  
Nach dem Training wird ein Ordner *runs* mit den Ergebnissen angelgt. Werden mehrere Modelle trainiert, so befnden diese sich ebenfalls in dem Ordner (train, train2, ...). Wähle nun den entsprechenden Sub-Ordner des Trainings-Runs, den du dir anschauen möchtest. Unter *weights* sind die Gewichte des Modells gespeichert und daneben noch verschiedene Metriken zu dem Training, die Aufschluss zu der Performanz des Modells geben.

# 5. Neuronales Netz Testen
Mit dieser Methode kann das trainierte Netz live über die EspCam getestet werden. Es wir alle paar Sekunden ein Bild aufgenommen und Object Detection darauf ausgeführt. Neben der reinen Vorhersage wird das durch das Modell annotierte Bild als *latest_capture_annotated.jpg* gespeichert. 
`testen_detection(url, model, conf_thresh, interval, img_size)`:  
*url* ensrpicht der URL der EspCam.
*model* enspricht dem Pfad zu dem Modell welches für die Vorhersage genutzt werden soll. Hiermit ist der zuvor erähnte Pfad zu den Gewichten gemeint. Wichtig ist hierbei das *train* richtig anzupassen (siehe Beispiel).
*conf_thresh* entspricht der Schwelle, ab welcher Sicherheit ein Ojekt erkannt wird (Wert zwischen 0 und 1).
*interval* enspricht dem Abstand in Sekunden bis zur Aufnahme des nächsten Bildes.
*img_size* entspricht der Bildgröße.

**Beispiel:**  
```python
from project_det import testen_detection

testen_detection(url='http://192.168.1.100',
                 model='runs/detect/train16/weights/best.pt',
                 conf_thresh=0.8,
                 interval=10,
                 img_size=(640,640))
```

# 6. Arduino Microcontroller Einrichten

Lade das folgender Arduino-Skript herunter und öffne es in der Arduino-IDE: [NanoEsp_detection.ino](https://github.com/Tarn017/Object-Classification-using-ESP-Cam/blob/main/files/NanoEsp_classification.ino)

Füge die passenden Wlan-Daten ein:

![wlan](https://raw.githubusercontent.com/Tarn017/Object-Classification-using-ESP-Cam/main/assets/wlan.png)

Führe das Skript im nächsten Schritt aus. Öffne anschließend den Serial Monitor. Auf disem sollte nun die IP Adresse des Microcontrollers ausgegeben werden. Ist dort nichts zu sehen, drücke einmal den Reset-Button auf dem Microcontroller.

![arduino_ip](https://raw.githubusercontent.com/Tarn017/Object-Classification-using-ESP-Cam/main/assets/arduino_ip.png)






