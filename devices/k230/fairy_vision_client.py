import json
import socket


def post_detection(
    host,
    port,
    path,
    label,
    confidence,
    device_id="lushan-pi-k230",
    api_key=None,
):
    payload = json.dumps(
        {
            "label": label,
            "confidence": confidence,
            "source": "k230-yolov8",
            "device_id": device_id,
        }
    )
    headers = [
        "POST {} HTTP/1.1".format(path),
        "Host: {}:{}".format(host, port),
        "Content-Type: application/json",
        "Content-Length: {}".format(len(payload)),
        "Connection: close",
    ]
    if api_key:
        headers.append("X-Vision-Key: {}".format(api_key))

    request = "\r\n".join(headers) + "\r\n\r\n" + payload
    address = socket.getaddrinfo(host, port)[0][-1]
    client = socket.socket()
    try:
        client.connect(address)
        client.send(request.encode())
        response = client.recv(512)
        return response
    finally:
        client.close()


def publish_yolo_result(detection, fairy_host, fairy_port=8000, api_key=None):
    """Call this from the CanMV YOLOv8 loop after NMS."""
    return post_detection(
        fairy_host,
        fairy_port,
        "/api/vision",
        detection["label"],
        detection["confidence"],
        api_key=api_key,
    )
