import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO
import time

# ===== Constants =====
IMG_SIZE = 416
SCORE_THRESH = 0.2
IOU_THRESH = 0.3

# ===== IoU & NMS =====
def iou(b1, b2):
    y1, x1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    y2, x2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, y2 - y1) * max(0, x2 - x1)
    union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / union if union > 0 else 0

def nms(boxes, scores, classes, iou_t=IOU_THRESH):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        ious = np.array([iou(boxes[i], boxes[j]) for j in idxs[1:]])
        idxs = idxs[1:][~((classes[idxs[1:]] == classes[i]) & (ious > iou_t))]
    return boxes[keep], scores[keep], classes[keep]

# ===== Box Filtering + NMS =====
def filter_boxes(box_xywh, scores):
    boxes, confs, clses = [], [], []
    for i in range(box_xywh.shape[0]):
        cls_scores = scores[i]
        cls_id, score = np.argmax(cls_scores), np.max(cls_scores)
        if score < SCORE_THRESH: continue
        cx, cy, w, h = box_xywh[i]
        xmin, ymin = cx - w/2, cy - h/2
        xmax, ymax = cx + w/2, cy + h/2
        boxes.append([ymin, xmin, ymax, xmax])
        confs.append(score)
        clses.append(cls_id)
    if boxes:
        return nms(np.array(boxes), np.array(confs), np.array(clses))
    return np.array([]), np.array([]), np.array([])

# ===== Visualization =====
def visualize(frame, boxes, scores, classes, labels):
    h, w = frame.shape[:2]
    np.random.seed(42)
    colors = {i: tuple(np.random.randint(0,255,3).tolist()) for i in labels}
    for box, score, cls in zip(boxes, scores, classes):
        ymin, xmin, ymax, xmax = box
        x1, y1 = int(xmin / IMG_SIZE * w), int(ymin / IMG_SIZE * h)
        x2, y2 = int(xmax / IMG_SIZE * w), int(ymax / IMG_SIZE * h)
        color = colors[cls]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{labels.get(cls, str(cls))}:{score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

# ===== Main =====
interpreter = tflite.Interpreter(model_path="yolov4-tiny.tflite")
interpreter.allocate_tensors()
inp, out = interpreter.get_input_details(), interpreter.get_output_details()

labels = {i: n for i, n in enumerate([
    'person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
    'broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed',
    'diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
])}
BUZZER_PIN = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
if not cap.isOpened():
    raise SystemExit("Camera not opened")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    input_data = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)
    interpreter.set_tensor(inp[0]['index'], input_data)
    interpreter.invoke()

    loc = interpreter.get_tensor(out[0]['index'])
    cls = interpreter.get_tensor(out[1]['index'])
    boxes, scores, classes = filter_boxes(loc[0], cls[0])
    boxes, scores, classes = nms(boxes, scores, classes)

    visualize(frame, boxes, scores, classes, labels)

    detected_person = False
    for c in classes:
        if labels.get(c) == "person":
            detected_person = True
            break

    if detected_person:
        cv2.putText(frame, "Person Detected!", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
        time.sleep(0.2)
        GPIO.output(BUZZER_PIN, GPIO.LOW)

    cv2.imshow("YOLOv4-Tiny Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
