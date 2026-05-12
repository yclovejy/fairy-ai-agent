import os
import socket

import uvicorn


def env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def get_local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def print_access_urls(host: str, port: int) -> None:
    lan_ip = get_local_ip()
    print(f"Local access:   http://127.0.0.1:{port}")

    if host in {"0.0.0.0", "::"}:
        print(f"LAN access:     http://{lan_ip}:{port}")
        print("Tip: Keep this computer and the phone on the same Wi-Fi network.")
    else:
        print(f"Custom access:  http://{host}:{port}")


def main() -> None:
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", os.getenv("PORT", "8000")))
    reload_enabled = env_flag("APP_RELOAD", True)

    print_access_urls(host, port)
    uvicorn.run("app:app", host=host, port=port, reload=reload_enabled)


if __name__ == "__main__":
    main()
