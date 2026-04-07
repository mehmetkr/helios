# Helios

**An Intelligent Control Plane for Serverless GPU Inference**

Helios is the orchestration layer that decides which AI models to keep warm, when to pre-warm them, and how to route requests during cold starts — solving the hardest operational problem in serverless inference.

---

## Why This Exists

The cold start problem is the central challenge in serverless inference: a warm model responds in under 100ms, a cold start takes 5-20 seconds. Every AI inference company — fal.ai, Modal, Replicate, Baseten, RunPod — is solving it at the hardware layer with faster model loading. But faster loading alone doesn't answer the harder questions: *which* models to load, *when* to load them proactively, and *what* to do with requests that arrive while loading is in progress.

Helios is the control plane that answers those questions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         REQUEST LAYER                           │
│                    FastAPI / asyncio HTTP                       │
│                    GET /metrics (Prometheus scrape)             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       REQUEST ROUTER                            │
│  - Routes to WARM runners immediately (O(1) lookup)             │
│  - Buffers requests when target model is LOADING                │
│  - asyncio.shield prevents load cancellation on caller timeout  │
│  - Idempotent: N concurrent cold requests trigger ONE load      │
│  - Retry up to max_retries on load failure                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        POOL MANAGER                             │
│                                                                 │
│  Two-phase locking (deadlock-free):                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ _memory_lock (pool-wide, brief)                         │    │
│  │  Budget check → iterative eviction → memory reserve     │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ _runner_locks[model_id] (per-model, long)               │    │
│  │  State machine transitions during load lifecycle        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Semaphore(max_concurrent_loads) caps simultaneous loads        │
│  Background tasks: _prewarm_loop, _health_check_loop            │
│  Graceful shutdown: _spawn_task() + set[Task] tracking          │
└──────────┬──────────────────────────────────┬───────────────────┘
           │                                  │
           ▼                                  ▼
┌──────────────────────┐          ┌───────────────────────────────┐
│   EVICTION POLICY    │          │         PREDICTOR             │
│                      │          │                               │
│  BaseEvictionPolicy  │          │  HoltPredictor                │
│  LRUEvictionPolicy   │          │  Holt's linear trend method   │
│  CostBasedPolicy     │          │  Lazy dirty-flag refit        │
│  (5 factors)         │          │  fit() → asyncio.to_thread()  │
└──────────────────────┘          └───────────────────────────────┘
```

**Model Runner Lifecycle FSM:**
```
COLD → DOWNLOADING → LOADING_TO_GPU → WARM → ACTIVE → EVICTING → COLD
```

---

## Key Design Decisions

### Two-Phase Locking (Deadlock-Free)

A pool-wide `_memory_lock` covers budget arithmetic (brief, no I/O). Per-model `_runner_locks` cover load state transitions (long, I/O-bound). The two are never held simultaneously — eliminating the deadlock that naive per-model locking creates when eviction requires cross-model coordination.

### Idempotent Load Orchestration

N concurrent requests for a cold model trigger exactly ONE load. A shared `asyncio.Future` is created under the per-model lock; all waiters `await asyncio.shield(future)`. The shield ensures caller timeouts don't cancel the underlying load — when loading completes, subsequent callers find the runner WARM immediately.

### Three-Path `ensure_loaded`

Fast path 1: model already WARM/ACTIVE — return immediately (no lock). Fast path 2: load in progress — share existing future (no lock contention). Slow path: acquire lock, create future, start load. This eliminates both warm-model reloads and lock contention during I/O-bound loads.

### ACTIVE State with Reference Counting

A hybrid `_active_requests` counter and FSM ACTIVE state tracks concurrent inferences. ACTIVE runners are immune to eviction and health checks — verified by state, not counter, keeping the FSM diagram truthful while handling concurrent inferences on the same model.

### Predictive Pre-Warming

Holt's linear trend method (double exponential smoothing) forecasts per-model demand. EWMA tracks request rates normalized to per-minute. Models are proactively loaded when predicted demand exceeds a configurable threshold — reducing cold starts before they happen.

### Hypothesis Property-Based Testing

A concurrent Hypothesis test drives random request sequences through `asyncio.gather` while a budget monitor checks the memory invariant at every yield point — proving the pool never exceeds its budget under adversarial inputs.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Python | 3.12+ |
| API | FastAPI |
| Async runtime | asyncio |
| API data modeling | Pydantic v2 |
| Internal data models | `@dataclass(frozen=True)` |
| Prediction | statsmodels (Holt's linear trend) |
| CPU-bound offloading | `asyncio.to_thread()` |
| Metrics | prometheus-client |
| Dashboarding | Grafana (auto-provisioned) |
| Testing | pytest, pytest-asyncio, Hypothesis |
| Type checking | mypy (strict) |
| Linting / formatting | ruff |
| Logging | structlog (JSON) |
| Simulation | Docker Compose |
| CI | GitHub Actions |

---

## Development Roadmap

### Phase 1 — Foundation
Exception hierarchy, FSM state definitions, configuration dataclasses with validation, Pydantic API models.

### Phase 2 — Building Blocks
Simulated model runner with failure injection. LRU and cost-based eviction policies (5-factor scoring). Holt predictor with lazy dirty-flag refit via `asyncio.to_thread()`. Unit tests for each component.

### Phase 3 — Core Engine
HeliosPool with two-phase locking, iterative eviction, three-path `ensure_loaded`, ACTIVE-aware dispatch, pre-warm loop with EWMA, health check subsystem, graceful shutdown with timeout budget.

### Phase 4 — Routing and Application
RequestRouter with retry loop, admission control, and typed exception semantics. FastAPI application with lifespan management, 20-model simulation catalog, and HTTP error mapping. Prometheus metrics and structlog observability.

### Phase 5 — Proof of Correctness
Integration tests (router buffering, memory lock safety, eviction, retry paths). Hypothesis property-based invariant tests under concurrent execution. Scenario tests for thundering herd, traffic spikes, budget exhaustion, and timeout resilience.

### Phase 6 — Performance and Production
Benchmark harness comparing LRU vs cost-based eviction under steady, spiky, and diurnal load profiles. Docker Compose simulation harness with Grafana dashboards. GitHub Actions CI pipeline.

---

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run benchmark
python benchmark.py --profile spiky --policy cost_based

# Run with Docker Compose
docker compose up
```

---

## Project Structure

```
helios/
├── helios/
│   ├── app.py                 # FastAPI entry point
│   ├── config.py              # PoolConfig, RunnerConfig, RunnerMetrics, API models
│   ├── exceptions.py          # HeliosError hierarchy
│   ├── fsm.py                 # RunnerLifecycleState enum
│   ├── pool.py                # HeliosPool — core engine
│   ├── router.py              # RequestRouter — retry, admission, timeout
│   ├── policies/
│   │   ├── base.py            # BaseEvictionPolicy ABC
│   │   ├── lru.py             # LRUEvictionPolicy
│   │   └── cost_based.py      # CostBasedEvictionPolicy
│   ├── prediction/
│   │   └── holt.py            # HoltPredictor
│   ├── simulation/
│   │   └── runner.py          # SimulatedModelRunner
│   └── observability/
│       └── metrics.py         # Prometheus metrics
├── tests/
│   ├── unit/                  # Layer 1 — pure, stateless
│   ├── integration/           # Layer 2 — component interactions
│   ├── property/              # Layer 3 — Hypothesis invariants
│   └── scenarios/             # Layer 4 — realistic simulations
├── benchmark.py               # CLI benchmark harness
├── docker-compose.yml
├── grafana/
├── prometheus.yml
└── pyproject.toml
```

---

## License

MIT
