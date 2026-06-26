from machine import ADC, Pin
import time

import ds18x20
import network
import onewire
import ujson
import usocket

try:
    import urandom
except ImportError:
    urandom = None

try:
    import config
except ImportError:
    raise RuntimeError("Save config.example.py to the ESP32 as config.py first")


humidity_value = float(config.HUMIDITY_BASE)


def create_light_sensor():
    sensor = ADC(Pin(config.LIGHT_ADC_PIN))
    try:
        sensor.atten(ADC.ATTN_11DB)
        sensor.width(ADC.WIDTH_12BIT)
    except AttributeError:
        pass
    return sensor


light_sensor = create_light_sensor()
pir_sensor = Pin(config.PIR_PIN, Pin.IN)
temperature_sensor = ds18x20.DS18X20(
    onewire.OneWire(Pin(config.TEMPERATURE_PIN))
)
temperature_roms = temperature_sensor.scan()


def connect_station(timeout_seconds=25):
    wlan = network.WLAN(network.STA_IF)
    if wlan.isconnected():
        return wlan

    # Reset the radio before each attempt. This also clears stale AP-mode state
    # on older MicroPython builds such as the ESP32 v1.11 firmware.
    wlan.active(False)
    time.sleep_ms(300)
    wlan.active(True)
    print("Connecting to Wi-Fi:", config.WIFI_SSID)
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
    print(
        "Access point ready:",
        config.ACCESS_POINT_SSID,
        access_point.ifconfig(),
    )
    print("Connect the Mac, then start Fairy on:", config.FAIRY_HOST)
    return access_point


def connect_network():
    mode = getattr(config, "WIFI_MODE", "station")
    if mode == "access_point":
        return start_access_point(), mode
    return connect_station(), "station"


def read_temperature():
    if not temperature_roms:
        raise OSError("No DS18B20 device found on GPIO%d" % config.TEMPERATURE_PIN)
    temperature_sensor.convert_temp()
    time.sleep_ms(750)
    value = temperature_sensor.read_temp(temperature_roms[0])
    if value is None or value <= -55 or value >= 125:
        raise ValueError("Invalid DS18B20 reading: %s" % value)
    return round(value, 2)


def read_light():
    samples = []
    for _ in range(16):
        samples.append(light_sensor.read())
        time.sleep_ms(8)
    raw_value = int(sum(samples) / len(samples))
    relative_value = int(raw_value * 1000 / 4095)
    if config.LIGHT_INVERT:
        relative_value = 1000 - relative_value
    relative_value = max(0, min(1000, relative_value))
    return raw_value, relative_value


def random_step():
    if urandom:
        return (urandom.getrandbits(8) / 255.0) - 0.5
    return ((time.ticks_ms() % 101) / 100.0) - 0.5


def simulate_humidity(temperature):
    global humidity_value
    target = float(config.HUMIDITY_BASE) - max(0, temperature - 25) * 0.35
    humidity_value += (target - humidity_value) * 0.08 + random_step() * 0.8
    humidity_value = max(38.0, min(68.0, humidity_value))
    return round(humidity_value, 1)


def send_all(sock, data):
    view = memoryview(data)
    sent = 0
    while sent < len(data):
        written = sock.send(view[sent:])
        if not written:
            raise OSError("Socket connection closed during upload")
        sent += written


def post_reading(payload):
    body = ujson.dumps(payload).encode()
    path = "/api/environment/readings"
    headers = [
        "POST %s HTTP/1.1" % path,
        "Host: %s:%d" % (config.FAIRY_HOST, config.FAIRY_PORT),
        "Content-Type: application/json",
        "Content-Length: %d" % len(body),
        "Connection: close",
    ]
    if config.IOT_API_KEY:
        headers.append("X-IoT-Key: %s" % config.IOT_API_KEY)
    request = ("\r\n".join(headers) + "\r\n\r\n").encode() + body

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
        send_all(sock, request)
        response = sock.recv(160)
    finally:
        sock.close()

    first_line = response.split(b"\r\n", 1)[0]
    if b" 200 " not in first_line:
        raise OSError("Fairy upload failed: %s" % first_line)
    return first_line


def collect_reading():
    temperature = read_temperature()
    light_raw, light_level = read_light()
    motion = bool(pir_sensor.value())
    humidity = simulate_humidity(temperature)
    return {
        "device_id": config.DEVICE_ID,
        "temperature": temperature,
        "humidity": humidity,
        "humidity_simulated": True,
        "light": light_level,
        "light_raw": light_raw,
        "motion": motion,
        "firmware_version": config.FIRMWARE_VERSION,
    }


def main():
    print("Fairy Sense", config.FIRMWARE_VERSION)
    print("DS18B20 devices:", temperature_roms)
    print("Fairy server: http://%s:%d" % (config.FAIRY_HOST, config.FAIRY_PORT))
    network_mode = getattr(config, "WIFI_MODE", "station")
    wlan = None
    if network_mode == "access_point":
        wlan = start_access_point()

    while True:
        try:
            if network_mode == "station" and (
                wlan is None or not wlan.isconnected()
            ):
                wlan = connect_station()
            reading = collect_reading()
            response = post_reading(reading)
            print(reading, response)
        except Exception as exc:
            print("Fairy Sense error:", exc)
        time.sleep(config.UPLOAD_INTERVAL_SECONDS)


main()
