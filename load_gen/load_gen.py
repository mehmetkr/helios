"""Configurable traffic generator for Helios simulation.

Reads configuration from environment variables:
    PROFILE: steady | spiky | diurnal (default: steady)
    TARGET_URL: Helios base URL (default: http://localhost:8000)
    MODELS: Number of distinct model IDs (default: 20)
    BURST_MULTIPLIER: Multiplier during spiky bursts (default: 5)
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import time

import httpx

PROFILE = os.getenv("PROFILE", "steady")
TARGET_URL = os.getenv("TARGET_URL", "http://localhost:8000")
MODELS = int(os.getenv("MODELS", "20"))
BURST_MULTIPLIER = int(os.getenv("BURST_MULTIPLIER", "5"))
MODEL_IDS = [f"model-{i:02d}" for i in range(MODELS)]


def requests_per_second(profile: str, elapsed: float) -> float:
    """Return target request rate based on profile and elapsed time."""
    if profile == "steady":
        return 2.0
    if profile == "spiky":
        cycle = elapsed % 60.0
        return float(BURST_MULTIPLIER) if cycle < 10.0 else 1.0
    if profile == "diurnal":
        return 1.75 + 1.25 * math.sin(2 * math.pi * elapsed / 300.0)
    return 1.0


async def main() -> None:
    rng = random.Random(42)
    start = time.monotonic()

    async with httpx.AsyncClient(timeout=60.0) as client:
        while True:
            elapsed = time.monotonic() - start
            rate = requests_per_second(PROFILE, elapsed)
            delay = 1.0 / rate if rate > 0 else 1.0
            await asyncio.sleep(delay)

            model_id = rng.choice(MODEL_IDS)
            try:
                resp = await client.post(
                    f"{TARGET_URL}/v1/infer",
                    json={"model_id": model_id, "payload": "load_gen"},
                )
                status = resp.status_code
            except httpx.HTTPError:
                status = 0

            elapsed_fmt = f"{elapsed:.1f}s"
            print(f"[{elapsed_fmt:>8}] {model_id} -> {status}")


if __name__ == "__main__":
    asyncio.run(main())
