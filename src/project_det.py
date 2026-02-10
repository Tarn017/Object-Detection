import requests
import os
import socket
from ultralytics import YOLO
from PIL import Image
import json
import time

def url_capture(url: str) -> str:
    if url.endswith('/capture'):
        return url
    return url.rstrip('/') + '/capture'

def aufnahme(url, interval, ordner):
    # URL des ESP32-CAM /capture Endpunkts
    url = url_capture(url)

    # Intervall zwischen den Aufnahmen in Sekunden
    interval = interval

    # Pfad zum Speicherordner
    save_path = ordner
    os.makedirs(save_path, exist_ok=True)

    while True:
        try:
            # Bild vom /capture Endpunkt abrufen
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                # Aktuellen Zeitstempel für den Dateinamen
                objekte = os.listdir(save_path)

                # Anzahl aller Objekte
                anzahl = len(objekte)
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = os.path.join(save_path, f'capture_{timestamp}.jpg')

                # Bild speichern
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f'Bild gespeichert: {filename} Objekt-Nummer: {anzahl}')
            else:
                print(f'Fehler: Statuscode {response.status_code}')
        except requests.RequestException as e:
            print(f'Anfrage fehlgeschlagen: {e}')

        # Warten bis zum nächsten Abruf
        time.sleep(interval)

def training_detection(dataset, epochen, img_size=(640,640)):

    # YOLO-Modell laden
    model = YOLO('yolov8n.pt')

    # Training starten
    results = model.train(data=os.path.join(dataset.location, 'data.yaml'), epochs=epochen, imgsz=img_size)

    # Modell auf TEST-Daten evaluieren
    metrics = model.val(data=os.path.join(dataset.location, 'data.yaml'), split='test')

    # Vorhersagen auf Testdaten visualisieren
    test_images_path = os.path.join(dataset.location, 'test', 'images')
    model.predict(source=test_images_path, conf=0.2, save=True)

def capture_image(url):
    """Nimmt ein einzelnes Bild von der ESP32-CAM auf und speichert es unter dem gleichen Namen."""
    filename = 'latest_capture.jpg'
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f'Bild gespeichert: {filename}')
            return filename
        else:
            print(f'Fehler: Statuscode {response.status_code}')
    except requests.RequestException as e:
        print(f'Anfrage fehlgeschlagen: {e}')
    return None

def testen_detection(url, model, conf_thresh, interval, img_size=None):
    url = url_capture(url)
    model = YOLO(model)
    while True:
        capture_image(url)
        if img_size is None:
            results = model.predict(source='latest_capture.jpg', conf=conf_thresh)
        else:
            results = model.predict(source='latest_capture.jpg', imgsz=img_size, conf=conf_thresh)

        # Ergebnisse extrahieren (Liste von Result-Objekten, hier nur 1 Bild → results[0])

        result = results[0]
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])  # Klassenindex (z. B. 0 für „ball“)
            label = model.names[cls_id]  # Klassennamen (z. B. "ball")
            conf = float(box.conf[0])  # Konfidenzscore
            xyxy = box.xyxy[0].tolist()  # Bounding Box Koordinaten [x1, y1, x2, y2]
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": xyxy
            })
        print(detections)

        annotated_img = result.plot()
        annotated_pil = Image.fromarray(annotated_img)
        annotated_pil.save('latest_capture_annotated.jpg')

        # 3 Sekunden warten
        if detections:
            # Nur das erste erkannte Objekt verwenden
            det = detections[0]
            label = det['label']
            conf = round(det['confidence'], 2)
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
        else:
            # Kein Objekt erkannt
            label = "none"
            conf = 0
            x1 = y1 = x2 = y2 = 0

        # Arduino-freundlicher, einfacher CSV-String
        result_str = f"{label},{conf},{x1},{y1},{x2},{y2}\n"
        time.sleep(interval)

    #return result_str

def neural_network_detection(url, arduino_ip, model, conf_thresh, port=12345):
    arduino_ip = arduino_ip  # IP-Adresse des Arduino
    port = port  # Muss mit der Portnummer im Arduino-Sketch übereinstimmen
    url = url_capture(url)

    # Socket erstellen
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((arduino_ip, port))

    received_values = []  # Liste zum Speichern der empfangenen Werte

    # Pfad zum Speicherordner
    save_path = 'daten2/obj'

    # Überprüfen, ob der Speicherordner existiert, andernfalls erstellen
    os.makedirs(save_path, exist_ok=True)
    model = YOLO(model)

    def classify_image(filename):
        results = model.predict(source=filename, conf=conf_thresh)
        # Ergebnisse extrahieren (Liste von Result-Objekten, hier nur 1 Bild → results[0])
        result = results[0]
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])  # Klassenindex (z. B. 0 für „ball“)
            label = model.names[cls_id]  # Klassennamen (z. B. "ball")
            conf = float(box.conf[0])  # Konfidenzscore
            xyxy = box.xyxy[0].tolist()  # Bounding Box Koordinaten [x1, y1, x2, y2]

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": xyxy
            })
        print(detections)

        annotated_img = result.plot()
        annotated_pil = Image.fromarray(annotated_img)
        annotated_pil.save('annotated_latest.jpg')

        # 3 Sekunden warten
        if detections:
            compact = []
            for d in detections:
                compact.append({
                    "label": d["label"],
                    "confidence": round(float(d["confidence"]), 2),  # runden um Platz zu sparen
                    "bbox": [int(coord) for coord in d["bbox"]]  # optional: ints statt floats
                })

            json_str = json.dumps(compact, separators=(',', ':'))  # compact JSON
            # Einfach eine Zeilen-terminierte Nachricht senden
            client_socket.sendall((json_str + "\n").encode())
        else:
            # Kein Objekt erkannt
            compact = []
            compact.append({
                "label": "none",
                "confidence": 0,  # runden um Platz zu sparen
                "bbox": [0.0,0.0,0.0,0.0] # optional: ints statt floats
            })
            json_str = json.dumps(compact, separators=(',', ':'))  # compact JSON
            # Einfach eine Zeilen-terminierte Nachricht senden
            client_socket.sendall((json_str + "\n").encode())

    #########################################################################################################

    try:
        while True:
            # Daten vom Arduino empfangen
            response = client_socket.recv(1024).decode().strip()

            if response:  # Falls eine Antwort empfangen wurde
                try:
                    value = int(response)  # Versuche, die Antwort in eine Zahl umzuwandeln
                    received_values.append(value)  # Wert in die Liste speichern
                    print(f"Empfangen und gespeichert: {value}")
                    if (value == 42):
                        image_path = capture_image(url)
                        if image_path:
                            classify_image(image_path)

                except ValueError:
                    print(f"Fehler beim Konvertieren: {response}")  # Falls ungültige Daten empfangen werden

            # Falls du eine Bedingung zum Stoppen willst (z. B. nach 10 Werten)
            if len(received_values) >= 300:
                print("Genug Werte empfangen, beende die Verbindung.")
                break

    finally:
        client_socket.close()


