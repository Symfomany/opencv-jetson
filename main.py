import cv2
import torch
from flask import Flask, Response, redirect, url_for
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO("yolo11n.pt")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)
print("YOLO device:", device)

PERSON_CLASS_ID = 0        # COCO: 0 = person [web:515]
PERSON_CONF_MIN = 0.75

# zoom partagé entre les requêtes
zoom_factor = 1.0   # 1.0 = pas de zoom
ZOOM_MIN = 1.0
ZOOM_MAX = 3.0
ZOOM_STEP = 0.2


def gen_frames():
    global zoom_factor

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        success, frame = cap.read()
        if not success:
            break

        z = max(ZOOM_MIN, min(ZOOM_MAX, zoom_factor))

        if z > 1.0:
            h, w = frame.shape[:2]
            nw, nh = int(w / z), int(h / z)
            x1 = (w - nw) // 2
            y1 = (h - nh) // 2
            crop = frame[y1:y1+nh, x1:x1+nw]
            frame_zoom = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            frame_zoom = frame

        results = model.predict(
            frame_zoom,
            conf=0.5,
            imgsz=640,
            verbose=False,
            device=device,
            classes=[PERSON_CLASS_ID],  # ne garde que person [web:504][web:508]

        )
        r = results[0]

        ### Detected Persons Filtering
        # Si tu veux savoir s'il y a au moins une personne sûre > 0.75
        person_boxes = []
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls_id == PERSON_CLASS_ID and conf >= PERSON_CONF_MIN:
                person_boxes.append((cls_id, conf))

        if person_boxes:
            # ici tu peux logger / déclencher une action
            print(f"{len(person_boxes)} personne(s) détectée(s) ≥ {PERSON_CONF_MIN}")

        annotated_frame = r.plot()

        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()


@app.route("/video")
def video():
    return Response(
        gen_frames(),
    mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/zoom/in", methods=["POST"])
def zoom_in():
    global zoom_factor
    zoom_factor = min(ZOOM_MAX, zoom_factor + ZOOM_STEP)
    return redirect(url_for("index"))


@app.route("/zoom/out", methods=["POST"])
def zoom_out():
    global zoom_factor
    zoom_factor = max(ZOOM_MIN, zoom_factor - ZOOM_STEP)
    return redirect(url_for("index"))


@app.route("/")
def index():
    return """
    <!doctype html>
    <html lang="fr">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Surveillance</title>
        <style>
          * { box-sizing: border-box; margin: 0; padding: 0; }
          body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #050816;
            color: #e5e7eb;
            height: 100vh;
            display: flex;
            flex-direction: column;
          }
          header {
            padding: 0.75rem 1.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(15, 23, 42, 0.9);
            border-bottom: 1px solid rgba(148, 163, 184, 0.3);
            backdrop-filter: blur(10px);
          }
          header h1 {
            font-size: 1rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #a5b4fc;
          }
          header span {
            font-size: 0.8rem;
            color: #9ca3af;
          }
          .container {
            flex: 1;
            padding: 0.75rem;
            display: flex;
            justify-content: center;
            align-items: center;
          }
          .video-card {
            width: 100%;
            max-width: 1200px;
            aspect-ratio: 16 / 9;
            background: radial-gradient(circle at top, #1e293b 0, #020617 60%);
            border-radius: 1rem;
            border: 1px solid rgba(148, 163, 184, 0.3);
            box-shadow: 0 20px 40px rgba(15, 23, 42, 0.8);
            overflow: hidden;
            position: relative;
            display: flex;
            justify-content: center;
            align-items: center;
          }
          .video-card::before {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 0 0, rgba(129, 140, 248, 0.2), transparent 50%),
                        radial-gradient(circle at 100% 100%, rgba(45, 212, 191, 0.18), transparent 50%);
            pointer-events: none;
          }
          .video-wrapper {
            position: relative;
            width: 100%;
            height: 100%;
            padding: 0.35rem;
            display: flex;
            justify-content: center;
            align-items: center;
          }
          .video-wrapper img {
            display: block;
            width: 100%;
            height: 100%;
            object-fit: contain;
            border-radius: 0.75rem;
          }
          .status-pill {
            position: absolute;
            top: 0.75rem;
            left: 0.75rem;
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.3rem 0.65rem;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(52, 211, 153, 0.7);
            font-size: 0.75rem;
            color: #bbf7d0;
          }
          .status-dot {
            width: 0.55rem;
            height: 0.55rem;
            border-radius: 999px;
            background: #22c55e;
            box-shadow: 0 0 10px rgba(34, 197, 94, 0.9);
          }
          .bottom-bar {
            position: absolute;
            bottom: 0.5rem;
            left: 0;
            right: 0;
            display: flex;
            justify-content: center;
            pointer-events: none;
          }
          .bottom-chip {
            pointer-events: auto;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(156, 163, 175, 0.5);
            font-size: 0.75rem;
            color: #e5e7eb;
            display: inline-flex;
            gap: 0.5rem;
            align-items: center;
          }
          .bottom-chip span { opacity: 0.8; }
          .dot {
            width: 0.25rem;
            height: 0.25rem;
            border-radius: 999px;
            background: #6b7280;
          }

          .zoom-controls {
             display: flex;
             gap: 0.5rem;
          }
          .zoom-button {
              border: none;
              border-radius: 999px;
              padding: 0.35rem 0.7rem;
              background: rgba(15, 23, 42, 0.9);
              border: 1px solid rgba(148, 163, 184, 0.8);
              color: #e5e7eb;
              font-size: 0.8rem;
              cursor: pointer;
              display: inline-flex;
              align-items: center;
              gap: 0.25rem;
          }
          .zoom-button:hover {
              background: rgba(30, 64, 175, 0.95);
            order-color: #6366f1;
          }
        </style>
      </head>
      <body>
        <header>
          <div>
            <h1>Jetson Orin · Surveillance</h1>
            <span>Live USB camera</span>
          </div>
          <div class="zoom-controls">
            <form method="post" action="/zoom/out">
              <button class="zoom-button" type="submit">− Zoom out</button>
            </form>
            <form method="post" action="/zoom/in">
              <button class="zoom-button" type="submit">+ Zoom in</button>
            </form>
          </div>
        </header>
        <main class="container">
          <section class="video-card">
            <div class="status-pill">
              <span class="status-dot"></span>
              <span>En direct</span>
            </div>

            <div class="video-wrapper">
              <img src="/video" alt="Flux caméra" />
            </div>

            <div class="bottom-bar">
              <div class="bottom-chip">
                <span>MJPEG stream</span>
                <span class="dot"></span>
                <span>HTTP · port 8000</span>
              </div>
            </div>
          </section>
        </main>
      </body>
    </html>
    """


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
