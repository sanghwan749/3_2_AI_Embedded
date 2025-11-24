from picamera2 import Picamera2
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time

from flask import Flask, Response, render_template_string
from motor import (
    init_servo, setServoPos,
    init_dc_motors, dc_motor_on, dc_motor_off
)
import RPi.GPIO as GPIO


# ======================================
# Face Detection Model
# ======================================
MODEL_PATH = "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"

# ======================================
# Servo Pins & Limits
# ======================================
SERVO_PAN_PIN = 18
SERVO_TILT_PIN = 23

PAN_MIN = 30
PAN_MAX = 150
PAN_INIT = 90

TILT_MIN = 60
TILT_MAX = 120
TILT_INIT = 90

# ======================================
# Camera FOV
# ======================================
CAMERA_FOV_X = 62.2
CAMERA_FOV_Y = 48.8

# ======================================
# Smooth Movement Values
# ======================================
SMOOTH_RATIO_PAN = 0.25
SMOOTH_RATIO_TILT = 0.05

DEADZONE = 0.05


# ======================================
# Load Model (TPU or CPU)
# ======================================
try:
    delegate = tflite.load_delegate("libedgetpu.so.1")
    print("[INFO] TPU loaded.")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH,
                                     experimental_delegates=[delegate])
except:
    print("[WARN] TPU load failed → CPU fallback")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
h_in, w_in = input_details[0]["shape"][1:3]


# ======================================
# Initialize Camera
# ======================================
picam = Picamera2()
config = picam.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"}
)
picam.configure(config)
picam.start()
picam.set_controls({"AeEnable": True, "AwbEnable": True})
time.sleep(1)


# ======================================
# Initialize Servos
# ======================================
servo_pan = init_servo(SERVO_PAN_PIN)
servo_tilt = init_servo(SERVO_TILT_PIN)

current_pan = float(PAN_INIT)
current_tilt = float(TILT_INIT)

setServoPos(servo_pan, current_pan)
setServoPos(servo_tilt, current_tilt)


# ======================================
# Initialize DC Motor (Always ON)
# ======================================
print("[INFO] Initializing DC motors...")
init_dc_motors()

print("[INFO] Turning DC motors ON (Full Power)...")
dc_motor_on()


# ======================================
# Flask Web Interface
# ======================================
app = Flask(__name__)

HTML = """
<!doctype html>
<html>
  <head>
    <title>Face Tracking (Camera Fixed / Smooth Motion)</title>
    <style>
        .big-text {
            font-size: 48px;       /* 기존의 3배 크기 */
            font-weight: bold;
        }
        body {
            background-color: #f5f5f5;
            font-family: Arial, sans-serif;
        }
    </style>
  </head>

  <body>
    <h2>Face Tracking System</h2>

    <p class="big-text">
        PAN: <span id="p"></span>° &nbsp;&nbsp; | &nbsp;&nbsp;
        TILT: <span id="t"></span>°
    </p>

    <img src="{{ url_for('video_feed') }}" width="650"/>

    <script>
      setInterval(async ()=>{
        document.getElementById('p').innerText = await (await fetch('/pan')).text();
        document.getElementById('t').innerText = await (await fetch('/tilt')).text();
      }, 300);
    </script>
  </body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/pan")
def read_pan():
    return f"{current_pan:.1f}"

@app.route("/tilt")
def read_tilt():
    return f"{current_tilt:.1f}"


# ======================================
# Face Selection Logic
# ======================================
def pick_best_face(boxes, scores, threshold=0.4):
    best_idx = -1
    best_score = -1
    for i, s in enumerate(scores):
        if s >= threshold and s > best_score:
            best_score = s
            best_idx = i
    return boxes[best_idx] if best_idx >= 0 else None


# ======================================
# Video Processing Loop
# ======================================
def gen_frames():
    global current_pan, current_tilt

    while True:
        frame_raw = picam.capture_array()
        frame = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(frame, (w_in, h_in))
        input_tensor = np.expand_dims(resized.astype(np.uint8), axis=0)

        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]["index"])[0]
        scores = interpreter.get_tensor(output_details[2]["index"])[0]

        target = pick_best_face(boxes, scores)

        if target is not None:
            ymin, xmin, ymax, xmax = target
            cx_norm = (xmin + xmax) * 0.5
            cy_norm = (ymin + ymax) * 0.5

            offset_x_deg = (cx_norm - 0.5) * CAMERA_FOV_X
            offset_y_deg = (cy_norm - 0.5) * CAMERA_FOV_Y

            target_pan  = PAN_INIT  - offset_x_deg
            target_tilt = TILT_INIT - offset_y_deg

            target_pan  = max(PAN_MIN,  min(PAN_MAX,  target_pan))
            target_tilt = max(TILT_MIN, min(TILT_MAX, target_tilt))

            current_pan  += (target_pan  - current_pan)  * SMOOTH_RATIO_PAN
            current_tilt += (target_tilt - current_tilt) * SMOOTH_RATIO_TILT

        setServoPos(servo_pan, current_pan)
        setServoPos(servo_tilt, current_tilt)

        ret, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )


@app.route("/video")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ======================================
# Main Runtime
# ======================================
if __name__ == "__main__":
    try:
        print("STARTED: http://<pi_ip>:5000")
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

    finally:
        print("[INFO] Turning DC motors OFF...")
        dc_motor_off()

        print("[INFO] Resetting servos...")
        setServoPos(servo_pan, PAN_INIT)
        setServoPos(servo_tilt, TILT_INIT)

        picam.stop()
        GPIO.cleanup()
        print("[INFO] Program ended.")
