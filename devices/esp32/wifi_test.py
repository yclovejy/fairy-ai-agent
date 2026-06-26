import network
import time
import usocket

try:
    import config
except ImportError:
    raise RuntimeError("Upload config.py to the ESP32 first")


def connect_station(timeout_seconds=25):
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print("Connecting to:", config.WIFI_SSID)
        wlan.connect(config.WIFI_SSID, config.WIFI_PASSWORD)
        started = time.ticks_ms()
        while not wlan.isconnected():
            if time.ticks_diff(time.ticks_ms(), started) > timeout_seconds * 1000:
                raise OSError("Wi-Fi connection timed out")
            time.sleep_ms(250)
    print("Wi-Fi connected:", wlan.ifconfig())
    return wlan


def start_access_point():
    station = network.WLAN(network.STA_IF)
    station.active(False)

    access_point = network.WLAN(network.AP_IF)
    access_point.active(True)
    access_point.config(
        essid=config.ACCESS_POINT_SSID,
        authmode=network.AUTH_WPA_WPA2_PSK,
        password=config.ACCESS_POINT_PASSWORD,
    )
    print("Access point ready:", config.ACCESS_POINT_SSID)
    print("Password:", config.ACCESS_POINT_PASSWORD)
    print("ESP32 network:", access_point.ifconfig())
    print("Connect the Mac now. Waiting 20 seconds...")
    time.sleep(20)
    return access_point


def connect_network():
    mode = getattr(config, "WIFI_MODE", "station")
    if mode == "access_point":
        return start_access_point()
    return connect_station()


def ping_fairy():
    address = usocket.getaddrinfo(
        config.FAIRY_HOST,
        config.FAIRY_PORT,
        0,
        usocket.SOCK_STREAM,
    )[0][-1]
    sock = usocket.socket()
    try:
        sock.settimeout(8)
        sock.connect(address)
        request = (
            "GET /ping HTTP/1.1\r\n"
            "Host: %s:%d\r\n"
            "Connection: close\r\n\r\n"
        ) % (config.FAIRY_HOST, config.FAIRY_PORT)
        sock.send(request.encode())
        response = sock.recv(256)
    finally:
        sock.close()

    print("Fairy response:")
    print(response.decode("utf-8", "ignore"))
    if b" 200 " not in response.split(b"\r\n", 1)[0]:
        raise OSError("Fairy server did not return HTTP 200")


connect_network()
ping_fairy()
print("ESP32 -> Fairy connection successful")
