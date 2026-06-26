# Copy this file to the ESP32 as config.py.
#
# "access_point": ESP32 creates Fairy-Sense; connect the Mac to that network.
# "station": Mac and ESP32 join the same existing Wi-Fi.
WIFI_MODE = "access_point"
WIFI_SSID = "YOUR_WIFI_NAME"
WIFI_PASSWORD = "YOUR_WIFI_PASSWORD"

ACCESS_POINT_SSID = "Fairy-Sense"
ACCESS_POINT_PASSWORD = "CHANGE_THIS_AP_PASSWORD"

# In access point mode, the first connected computer is normally 192.168.4.2.
# In station mode, use the LAN address printed by run_server.py.
FAIRY_HOST = "192.168.4.2"
FAIRY_PORT = 8000
IOT_API_KEY = ""

DEVICE_ID = "esp32-classroom-01"
FIRMWARE_VERSION = "fairy-sense-1.0"
UPLOAD_INTERVAL_SECONDS = 5

# ESP32 DevKit pin mapping.
TEMPERATURE_PIN = 18
LIGHT_ADC_PIN = 34
PIR_PIN = 27

# Set to True if covering the photoresistor makes the displayed value increase.
LIGHT_INVERT = True

# Humidity is simulated because no humidity sensor is currently available.
HUMIDITY_BASE = 52.0
