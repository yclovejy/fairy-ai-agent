from libs.PipeLine import PipeLine
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
from libs.Utils import *
from media.media import *
import aidemo
import gc
import image
import network
import nncase_runtime as nn
import os
import socket
import time
import ujson
import ulab.numpy as np


WIFI_SSID = "YOUR_WIFI_NAME"
WIFI_PASSWORD = "YOUR_WIFI_PASSWORD"
FAIRY_HOST = "YOUR_MAC_LAN_IP"
FAIRY_PORT = 8000
DEVICE_ID = "K230-LC-3"

DISPLAY_MODE = "virt"
DISPLAY_SIZE = [640, 480]
RGB888P_SIZE = [224, 224]
MODEL_INPUT_SIZE = [224, 224]
KMODEL_PATH = "/sdcard/examples/kmodel/yolov8n_224.kmodel"
CONFIDENCE_THRESHOLD = 0.32
NMS_THRESHOLD = 0.4
SEND_COOLDOWN_MS = 3000

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

LABEL_MAP = {
    "cup": "水杯",
    "bottle": "水杯",
    "cell phone": "手机",
    "keyboard": "键盘",
    "mouse": "鼠标",
    "laptop": "电脑",
}

ALLOW_LABELS = ("cup", "bottle", "cell phone", "keyboard", "mouse", "laptop")


class ObjectDetectionApp(AIBase):
    def __init__(
        self,
        kmodel_path,
        labels,
        model_input_size,
        max_boxes_num,
        confidence_threshold=0.5,
        nms_threshold=0.2,
        rgb888p_size=[224, 224],
        display_size=[640, 480],
        debug_mode=0,
    ):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        self.labels = labels
        self.model_input_size = model_input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_boxes_num = max_boxes_num
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        self.color_four = get_colors(len(self.labels))
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(
            nn.ai2d_format.NCHW_FMT,
            nn.ai2d_format.NCHW_FMT,
            np.uint8,
            np.uint8,
        )

    def config_preprocess(self, input_image_size=None):
        ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
        top, bottom, left, right, self.scale = letterbox_pad_param(
            self.rgb888p_size,
            self.model_input_size,
        )
        self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [128, 128, 128])
        self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
        self.ai2d.build(
            [1, 3, ai2d_input_size[1], ai2d_input_size[0]],
            [1, 3, self.model_input_size[1], self.model_input_size[0]],
        )

    def preprocess(self, input_np):
        return [nn.from_numpy(input_np)]

    def postprocess(self, results):
        new_result = results[0][0].transpose()
        return aidemo.yolov8_det_postprocess(
            new_result.copy(),
            [self.rgb888p_size[1], self.rgb888p_size[0]],
            [self.model_input_size[1], self.model_input_size[0]],
            [self.display_size[1], self.display_size[0]],
            len(self.labels),
            self.confidence_threshold,
            self.nms_threshold,
            self.max_boxes_num,
        )

    def draw_result(self, pl, dets):
        pl.osd_img.clear()
        if not dets:
            return
        for i in range(len(dets[0])):
            x, y, w, h = map(lambda value: int(round(value, 0)), dets[0][i])
            class_id = dets[1][i]
            score = dets[2][i]
            pl.osd_img.draw_rectangle(
                x,
                y,
                w,
                h,
                color=self.color_four[class_id],
                thickness=4,
            )
            pl.osd_img.draw_string_advanced(
                x,
                y - 42,
                28,
                " " + self.labels[class_id] + " " + str(round(score, 2)),
                color=self.color_four[class_id],
            )


def _scan_wifi_names(wlan):
    names = []
    print("Scanning Wi-Fi...")
    try:
        networks = wlan.scan()
    except Exception as error:
        print("Wi-Fi scan failed:", error)
        return names

    for item in networks:
        ssid = None
        try:
            ssid = item["ssid"]
        except Exception:
            try:
                ssid = item[0]
            except Exception:
                ssid = None
        if isinstance(ssid, bytes):
            try:
                ssid = ssid.decode()
            except Exception:
                ssid = str(ssid)
        if ssid is None:
            ssid = ""
        names.append(ssid)
        print("SSID:", ssid)
    return names


def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    time.sleep(1)

    visible_names = _scan_wifi_names(wlan)
    if WIFI_SSID not in visible_names:
        print("Target Wi-Fi not found:", WIFI_SSID)
        print("Keep the phone hotspot screen open and enable compatibility mode.")

    for attempt in range(1, 4):
        print("Connecting Wi-Fi attempt", attempt, "->", WIFI_SSID)
        try:
            if wlan.isconnected():
                print("Network:", wlan.ifconfig())
                return wlan
            try:
                wlan.disconnect()
                time.sleep(1)
            except Exception:
                pass
            wlan.connect(WIFI_SSID, WIFI_PASSWORD)
        except Exception as error:
            print("connect call failed:", error)

        for _ in range(45):
            if wlan.isconnected():
                print("Network:", wlan.ifconfig())
                return wlan
            time.sleep(1)
            print(".", end="")
        print()

    raise RuntimeError("Wi-Fi connection failed")


def post_detection(label, confidence):
    payload = ujson.dumps(
        {
            "label": label,
            "confidence": round(confidence, 4),
            "source": "k230-yolov8",
            "device_id": DEVICE_ID,
        }
    ).encode()
    request = (
        "POST /api/vision HTTP/1.1\r\n"
        "Host: {}:{}\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: {}\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).format(FAIRY_HOST, FAIRY_PORT, len(payload)).encode() + payload

    address = socket.getaddrinfo(FAIRY_HOST, FAIRY_PORT)[0][-1]
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(3)
    try:
        client.connect(address)
        client.send(request)
        response = client.recv(128)
        print("Fairy:", response.split(b"\r\n", 1)[0])
    except Exception as error:
        print("Fairy post failed:", error)
    finally:
        client.close()


def publish_best_detection(result, last_sent):
    if not result:
        return

    best_label = None
    best_confidence = 0
    for i in range(len(result[0])):
        class_id = int(result[1][i])
        raw_label = COCO_LABELS[class_id]
        confidence = float(result[2][i])
        if raw_label not in ALLOW_LABELS:
            continue
        if confidence > best_confidence:
            best_label = raw_label
            best_confidence = confidence

    if best_label is None:
        return

    now = time.ticks_ms()
    previous = last_sent.get(best_label, None)
    if previous is not None and time.ticks_diff(now, previous) < SEND_COOLDOWN_MS:
        return

    last_sent[best_label] = now
    post_detection(LABEL_MAP.get(best_label, best_label), best_confidence)


def main():
    connect_wifi()
    pl = None
    detector = None

    try:
        pl = PipeLine(
            rgb888p_size=RGB888P_SIZE,
            display_mode=DISPLAY_MODE,
            display_size=DISPLAY_SIZE,
        )
        pl.create(sensor_id=2)
        display_size = pl.get_display_size()
        detector = ObjectDetectionApp(
            KMODEL_PATH,
            labels=COCO_LABELS,
            model_input_size=MODEL_INPUT_SIZE,
            max_boxes_num=30,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            nms_threshold=NMS_THRESHOLD,
            rgb888p_size=RGB888P_SIZE,
            display_size=display_size,
            debug_mode=0,
        )
        detector.config_preprocess()
        last_sent = {}
        print("Fairy YOLOv8 guide ready")

        while True:
            os.exitpoint()
            frame = pl.get_frame()
            result = detector.run(frame)
            detector.draw_result(pl, result)
            pl.show_image()
            publish_best_detection(result, last_sent)
            gc.collect()
    except KeyboardInterrupt:
        print("Stopped")
    finally:
        if detector:
            detector.deinit()
        if pl:
            pl.destroy()


main()
