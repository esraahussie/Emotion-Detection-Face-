import numpy as np
import cv2
from tensorflow.keras.models import load_model
from model.preprocess import detect_faces, preprocess_face

MODEL_PATH = "model/emotion_model.h5"
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model


def inference(frame_bgr):
    model = get_model()
    faces, gray = detect_faces(frame_bgr)
    results = []
    for box in faces:
        x, y, w, h = box
        face_input = preprocess_face(gray, box)

        preds = model.predict(face_input, verbose=0)[0]
        class_idx = int(np.argmax(preds))
        label = EMOTION_LABELS[class_idx]
        confidence = float(preds[class_idx])

        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{label} ({confidence * 100:.0f}%)"
        cv2.putText(frame_bgr, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        results.append((box, label, confidence))

    return frame_bgr, results