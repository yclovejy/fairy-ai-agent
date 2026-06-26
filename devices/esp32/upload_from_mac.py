"""Upload Fairy Sense files with the Python runtime bundled in Thonny."""

from argparse import ArgumentParser
from pathlib import Path
import time

import serial


FILES = ("config.py", "hardware_test.py", "wifi_test.py", "main.py")
RAW_PROMPT = b"raw REPL; CTRL-B to exit\r\n>"


def read_until(port, marker, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    received = bytearray()
    while time.monotonic() < deadline:
        chunk = port.read(1)
        if chunk:
            received.extend(chunk)
            if received.endswith(marker):
                return bytes(received)
    raise TimeoutError("Timed out waiting for %r; received %r" % (marker, received))


def enter_raw_repl(port):
    time.sleep(0.8)
    port.reset_input_buffer()
    port.write(b"\x03\x03")
    time.sleep(0.5)
    port.reset_input_buffer()
    port.write(b"\x01")
    read_until(port, RAW_PROMPT, 4)
    time.sleep(0.2)


def raw_exec(port, source):
    port.write(source.encode("utf-8"))
    port.write(b"\x04")
    acknowledgement = read_until(port, b"OK", 3)
    if not acknowledgement.endswith(b"OK"):
        raise OSError(
            "ESP32 did not accept raw REPL command: %r" % acknowledgement
        )

    stdout = read_until(port, b"\x04", 8)[:-1]
    stderr = read_until(port, b"\x04", 8)[:-1]
    read_until(port, b">", 3)
    if stderr:
        raise OSError(stderr.decode("utf-8", "replace"))
    return stdout.decode("utf-8", "replace")


def upload_file(port, target, content):
    raw_exec(port, "f=open(%r,'wb');f.close()" % target)
    for offset in range(0, len(content), 192):
        block = content[offset : offset + 192]
        source = (
            "import ubinascii;"
            "f=open(%r,'ab');"
            "f.write(ubinascii.unhexlify(b'%s'));"
            "f.close()"
        ) % (target, block.hex())
        raw_exec(port, source)

    output = raw_exec(
        port,
        "d=open(%r,'rb').read();print(len(d),sum(d)%%65536)" % target,
    ).strip()
    expected = "%d %d" % (len(content), sum(content) % 65536)
    if output != expected:
        raise OSError(
            "Verification failed for %s: expected %s, got %s"
            % (target, expected, output)
        )


def main():
    parser = ArgumentParser()
    parser.add_argument("--port", default="/dev/cu.usbserial-0001")
    args = parser.parse_args()

    source_dir = Path(__file__).resolve().parent
    port = serial.Serial(args.port, 115200, timeout=0.2, write_timeout=3)
    try:
        enter_raw_repl(port)
        for filename in FILES:
            content = (source_dir / filename).read_bytes()
            print("Uploading", filename, "...", flush=True)
            upload_file(port, "/" + filename, content)
            print("Verified", filename, len(content), "bytes", flush=True)

        print("Upload complete. Rebooting ESP32...", flush=True)
        port.write(b"\x02\x04")
        time.sleep(2)
        output = port.read_all().decode("utf-8", "replace").strip()
        if output:
            print(output)
    finally:
        port.close()


if __name__ == "__main__":
    main()
