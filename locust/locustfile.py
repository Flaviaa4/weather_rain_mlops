import csv
import os
import random
import struct
import zlib
from collections import Counter
from pathlib import Path

from locust import HttpUser, between, events, task

SAMPLE_DIR = Path(os.environ.get("LOCUST_SAMPLE_DIR", "data/valid"))
HOSTNAME_CSV = os.environ.get("LOCUST_HOSTNAME_CSV", "")
SAMPLE_LIMIT = int(os.environ.get("LOCUST_SAMPLE_LIMIT", "40"))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
HOSTNAMES = Counter()


def png_chunk(kind, payload):
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def gradient_row(width):
    return b"\x00" + bytes(value for x in range(width) for value in (x * 255 // width, 120, 255 - x * 255 // width))


def synthetic_png(width=300, height=168):
    rows = gradient_row(width) * height
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", header) + png_chunk(b"IDAT", zlib.compress(rows)) + png_chunk(b"IEND", b"")


def load_samples(folder=SAMPLE_DIR, limit=SAMPLE_LIMIT, seed=11):
    paths = sorted(path for path in Path(folder).rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    picked = random.Random(seed).sample(paths, min(limit, len(paths)))
    return [(path.name, path.read_bytes()) for path in picked] or [("synthetic.png", synthetic_png())]


SAMPLES = load_samples()


def record_hostname(response):
    try:
        HOSTNAMES[response.json().get("hostname", "unknown")] += 1
    except ValueError:
        HOSTNAMES["unreadable"] += 1


class PredictUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(9)
    def predict(self):
        name, data = random.choice(SAMPLES)
        with self.client.post("/predict", files={"file": (name, data, "image/jpeg")}, name="/predict",
                              catch_response=True) as response:
            if response.status_code == 200:
                record_hostname(response)
                response.success()
            else:
                response.failure(f"status {response.status_code}")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")


@events.test_stop.add_listener
def save_hostnames(environment, **kwargs):
    if not HOSTNAME_CSV:
        return
    with open(HOSTNAME_CSV, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["hostname", "responses"])
        writer.writerows(sorted(HOSTNAMES.items()))
