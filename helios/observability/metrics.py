"""Prometheus metric objects for Helios observability.

Machine-readable aggregates scraped by Prometheus and displayed in Grafana.
All metric objects are module-level singletons -- import and use directly.
"""

from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

REQUEST_LATENCY = Histogram(
    "helios_request_latency_seconds",
    "Time to first token",
    buckets=[0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0],
    labelnames=["model_id", "cache_status"],  # "warm" | "cold"
)

COLD_STARTS_TOTAL = Counter(
    "helios_cold_starts_total",
    "Total cold starts",
    labelnames=["model_id"],
)

POOL_MEMORY_USED_GB = Gauge(
    "helios_pool_memory_used_gb",
    "Current memory used by warm runners",
)

QUEUE_DEPTH = Gauge(
    "helios_queue_depth",
    "Requests currently buffered awaiting load",
)

EVICTIONS_TOTAL = Counter(
    "helios_evictions_total",
    "Runner evictions",
    labelnames=["model_id", "reason"],  # "cost_based" | "lru" | "health_check"
)

LOAD_FAILURES = Counter(
    "helios_load_failures_total",
    "Runner load failures (incremented on rollback in _initiate_load)",
    labelnames=["model_id"],
)

LOAD_DURATION_SECONDS = Histogram(
    "helios_load_duration_seconds",
    "Model load duration",
    labelnames=["model_id", "outcome"],  # "success" | "failure"
)

metrics_asgi_app = make_asgi_app()
