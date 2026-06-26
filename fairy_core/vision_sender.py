from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import requests


DEMO_LABELS = ["水杯", "手机", "键盘", "鼠标", "学生证"]
COCO_LABEL_MAP = {
    "cup": "水杯",
    "cell phone": "手机",
    "keyboard": "键盘",
    "mouse": "鼠标",
}


def send_detection(
    endpoint: str,
    label: str,
    confidence: float,
    *,
    source: str,
    device_id: str,
    api_key: str | None,
) -> dict:
    headers = {"X-Vision-Key": api_key} if api_key else {}
    response = requests.post(
        endpoint,
        headers=headers,
        json={
            "label": label,
            "confidence": round(float(confidence), 4),
            "source": source,
            "device_id": device_id,
        },
        timeout=12,
    )
    response.raise_for_status()
    return response.json()


def run_simulator(args: argparse.Namespace) -> None:
    print(f"Sending simulated detections to {args.endpoint}")
    while True:
        label = random.choice(DEMO_LABELS)
        confidence = random.uniform(0.82, 0.98)
        result = send_detection(
            args.endpoint,
            label,
            confidence,
            source="mac-simulator",
            device_id=args.device_id,
            api_key=args.api_key,
        )
        print(f"{label} {confidence:.1%} -> record #{result['detection']['id']}")
        if args.once:
            return
        time.sleep(args.interval)


def run_camera(args: argparse.Namespace) -> None:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Camera mode needs opencv-python and ultralytics. "
            "Install training/requirements-vision.txt first."
        ) from exc

    model_path = Path(args.model)
    model = YOLO(str(model_path if model_path.exists() else args.model))
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    last_sent: dict[str, float] = {}
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            result = model.predict(frame, imgsz=args.image_size, conf=args.confidence, verbose=False)[0]
            now = time.monotonic()
            for box in result.boxes:
                class_id = int(box.cls[0])
                raw_label = model.names[class_id]
                label = COCO_LABEL_MAP.get(raw_label, raw_label)
                confidence = float(box.conf[0])
                if now - last_sent.get(label, 0) < args.cooldown:
                    continue
                send_detection(
                    args.endpoint,
                    label,
                    confidence,
                    source="mac-yolov8",
                    device_id=args.device_id,
                    api_key=args.api_key,
                )
                last_sent[label] = now
                print(f"{label} {confidence:.1%}")

            annotated = result.plot()
            cv2.imshow("Fairy Vision - press q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send YOLOv8 detections to Fairy")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/api/vision")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--device-id", default="mac-vision-dev")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=4.0)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--cooldown", type=float, default=3.0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.simulate:
        run_simulator(arguments)
    else:
        run_camera(arguments)
