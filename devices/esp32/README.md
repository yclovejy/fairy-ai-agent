# Fairy Sense ESP32

## 照片中的硬件

- ESP32 DevKit，ESP-WROOM-32，38 针版本
- 光敏电阻模块，使用 `AO` 模拟输出
- 三针温度模块，照片中引脚为 `VCC / DQ / GND`，按 DS18B20 使用
- HC-SR501 人体红外模块
- 面包板和母对母杜邦线

如果温度模块接好后测试程序显示 `No DS18B20 device found`，先不要更换电压或乱接引脚，
拍摄温度模块正反面的近照确认型号。

## 接线

所有模块必须共地。

| 模块 | 模块引脚 | ESP32 |
| --- | --- | --- |
| 光敏模块 | `VCC` | `3V3` |
| 光敏模块 | `GND` | `GND` |
| 光敏模块 | `AO` | `GPIO34` |
| 光敏模块 | `DO` | 不接 |
| DS18B20 模块 | `VCC` | `3V3` |
| DS18B20 模块 | `GND` | `GND` |
| DS18B20 模块 | `DQ` | `GPIO18` |
| HC-SR501 | `VCC` | `VIN/5V` |
| HC-SR501 | `GND` | `GND` |
| HC-SR501 | `OUT` | `GPIO27` |

注意：ESP32 GPIO 最大输入电压为 3.3V。光敏模块必须使用 `3V3` 供电，避免 `AO` 超压。
HC-SR501 使用 `VIN/5V` 供电，其 `OUT` 信号约为 3.3V，可以连接 ESP32 GPIO。

## Thonny 部署

1. 在 Thonny 打开 `View -> Files`。
2. 将 `config.example.py` 复制一份并命名为 `config.py`。
3. 默认使用 ESP32 自建热点：
   - 热点名：`Fairy-Sense`
   - 密码：`fairy1234`
   - Mac 连接此热点后通常会获得 `192.168.4.2`
   - 因此 `FAIRY_HOST` 默认填写 `192.168.4.2`
4. 若需要使用已有 Wi-Fi，将 `WIFI_MODE` 改为 `station`，再填写 Wi-Fi
   名称、密码和 Mac 的局域网地址。
5. 先运行 `hardware_test.py`，确认：
   - `DS18B20 devices` 不是空列表。
   - 手遮挡光敏电阻时 `light_raw` 明显变化。
   - HC-SR501 预热 30-60 秒后，人在前方移动时 `motion=True`。
6. Mac 端启动 Fairy 后运行 `wifi_test.py`，确认输出
   `ESP32 -> Fairy connection successful`。
7. 将 `config.py` 和 `main.py` 上传并保存到 ESP32 根目录。
8. 按 ESP32 的 `RST/EN` 键。`main.py` 会开机自动运行。

当前配置中，光照被换算为 `0-1000` 的相对值，不是经过仪器标定的 lux。
如果遮挡光敏电阻后 Fairy 页面中的数值反而升高，将 `LIGHT_INVERT` 改为 `True`。

## 启动 Fairy

使用默认热点模式时，先让 Mac 连接 `Fairy-Sense`。Mac 端运行：

```bash
cd "/Users/yongchengwang/Desktop/projects/AI Agent"
conda activate ai_agent
python run_server.py
```

服务器必须监听 `0.0.0.0`，项目的 `run_server.py` 已按此方式启动。
如果 Mac 在 `Fairy-Sense` 下的地址不是 `192.168.4.2`，将实际地址填入
ESP32 的 `config.py`。
