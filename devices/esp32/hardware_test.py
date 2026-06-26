from machine import ADC, Pin
import time

import ds18x20
import onewire

try:
    import config
except ImportError:
    raise RuntimeError("Save config.example.py to the ESP32 as config.py first")


def create_light_sensor():
    sensor = ADC(Pin(config.LIGHT_ADC_PIN))
    try:
        sensor.atten(ADC.ATTN_11DB)
        sensor.width(ADC.WIDTH_12BIT)
    except AttributeError:
        pass
    return sensor


def normalize_light(raw_value):
    value = int(raw_value * 1000 / 4095)
    if config.LIGHT_INVERT:
        value = 1000 - value
    return max(0, min(1000, value))


light_sensor = create_light_sensor()
pir_sensor = Pin(config.PIR_PIN, Pin.IN)
temperature_sensor = ds18x20.DS18X20(
    onewire.OneWire(Pin(config.TEMPERATURE_PIN))
)
temperature_roms = temperature_sensor.scan()

print("Fairy Sense hardware test")
print("DS18B20 devices:", temperature_roms)
print("PIR needs about 30-60 seconds to stabilize after power-on.")
print("Press Stop in Thonny to finish.\n")

while True:
    light_samples = [light_sensor.read() for _ in range(12)]
    light_raw = int(sum(light_samples) / len(light_samples))
    light_level = normalize_light(light_raw)
    motion = bool(pir_sensor.value())

    temperature = None
    if temperature_roms:
        temperature_sensor.convert_temp()
        time.sleep_ms(750)
        temperature = temperature_sensor.read_temp(temperature_roms[0])

    print(
        "temperature=",
        temperature,
        "C | light_raw=",
        light_raw,
        "| light_level=",
        light_level,
        "/1000 | motion=",
        motion,
    )
    time.sleep(2)
