"""FastAPI application entry point for Helios.

Provides POST /v1/infer backed by RequestRouter, a lifespan context manager
for pool lifecycle, Prometheus /metrics endpoint, and typed HTTP error mapping.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from helios.config import InferenceRequest, InferenceResult, PoolConfig, RunnerConfig
from helios.exceptions import (
    AdmissionRejectedError,
    HeliosError,
    MemoryExhaustedError,
    RequestTimeoutError,
    RunnerLoadError,
)
from helios.observability import configure_logging
from helios.observability.metrics import REQUEST_LATENCY, metrics_asgi_app
from helios.pool import HeliosPool
from helios.router import RequestRouter

# ---------------------------------------------------------------------------
# 20-model simulation catalog
# Pre-computed from random.Random(42): load_time_s ~ uniform(2.0, 14.0),
# memory_gb ~ uniform(2.0, 6.0), rounded to 2 decimal places.
# failure_rate=0.0 and infer_time_s=0.05 uniform across all models.
# Sum memory: 72.92 GB. With total_memory_gb=80.0, most models can coexist.
# ---------------------------------------------------------------------------

SIMULATION_MODELS = [
    RunnerConfig(model_id="model-00", memory_gb=2.10, load_time_s=9.67),
    RunnerConfig(model_id="model-01", memory_gb=2.89, load_time_s=5.30),
    RunnerConfig(model_id="model-02", memory_gb=4.71, load_time_s=10.84),
    RunnerConfig(model_id="model-03", memory_gb=2.35, load_time_s=12.71),
    RunnerConfig(model_id="model-04", memory_gb=2.12, load_time_s=7.06),
    RunnerConfig(model_id="model-05", memory_gb=4.02, load_time_s=4.62),
    RunnerConfig(model_id="model-06", memory_gb=2.80, load_time_s=2.32),
    RunnerConfig(model_id="model-07", memory_gb=4.18, load_time_s=9.80),
    RunnerConfig(model_id="model-08", memory_gb=4.36, load_time_s=4.65),
    RunnerConfig(model_id="model-09", memory_gb=2.03, load_time_s=11.71),
    RunnerConfig(model_id="model-10", memory_gb=4.79, load_time_s=11.67),
    RunnerConfig(model_id="model-11", memory_gb=2.62, load_time_s=6.08),
    RunnerConfig(model_id="model-12", memory_gb=3.35, load_time_s=13.49),
    RunnerConfig(model_id="model-13", memory_gb=2.39, load_time_s=3.11),
    RunnerConfig(model_id="model-14", memory_gb=4.41, load_time_s=12.17),
    RunnerConfig(model_id="model-15", memory_gb=4.92, load_time_s=11.69),
    RunnerConfig(model_id="model-16", memory_gb=5.89, load_time_s=8.43),
    RunnerConfig(model_id="model-17", memory_gb=4.21, load_time_s=6.54),
    RunnerConfig(model_id="model-18", memory_gb=4.47, load_time_s=11.95),
    RunnerConfig(model_id="model-19", memory_gb=4.31, load_time_s=12.34),
]


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Build pool, register models, start background tasks, shutdown on exit."""
    configure_logging(json_output=True)

    pool = HeliosPool(config=PoolConfig())
    for config in SIMULATION_MODELS:
        pool.register(config)
    pool.start()

    router = RequestRouter(pool)
    _app.state.pool = pool
    _app.state.router = router

    yield

    await pool.shutdown()


app = FastAPI(title="Helios", lifespan=lifespan)
app.mount("/metrics", metrics_asgi_app)


# ---------------------------------------------------------------------------
# Inference endpoint
# ---------------------------------------------------------------------------


@app.post("/v1/infer")
async def infer(request: InferenceRequest, raw: Request) -> InferenceResult:
    """Route an inference request through the pool."""
    router: RequestRouter = raw.app.state.router

    start = time.monotonic()
    result = await router.route(request)
    duration = time.monotonic() - start

    REQUEST_LATENCY.labels(model_id=request.model_id, cache_status=result.cache_status).observe(
        duration
    )

    return result


# ---------------------------------------------------------------------------
# Exception handlers (most specific first -- FastAPI matches registration order)
# ---------------------------------------------------------------------------


@app.exception_handler(MemoryExhaustedError)
async def memory_exhausted_handler(_request: Request, exc: MemoryExhaustedError) -> JSONResponse:
    status = 400 if exc.permanent else 503
    return JSONResponse(status_code=status, content={"error": str(exc)})


@app.exception_handler(RequestTimeoutError)
async def request_timeout_handler(_request: Request, exc: RequestTimeoutError) -> JSONResponse:
    return JSONResponse(status_code=504, content={"error": str(exc)})


@app.exception_handler(RunnerLoadError)
async def runner_load_handler(_request: Request, exc: RunnerLoadError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"error": str(exc)})


@app.exception_handler(AdmissionRejectedError)
async def admission_rejected_handler(
    _request: Request, exc: AdmissionRejectedError
) -> JSONResponse:
    return JSONResponse(status_code=429, content={"error": str(exc)})


@app.exception_handler(HeliosError)
async def helios_error_handler(_request: Request, exc: HeliosError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": str(exc)})
