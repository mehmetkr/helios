"""Helios observability -- structlog configuration and Prometheus metrics.

Metrics are defined in helios.observability.metrics (module-level singletons).
Logging is configured by calling configure_logging() from the application
entry point (e.g., FastAPI lifespan).
"""

import structlog


def configure_logging(*, json_output: bool = True) -> None:
    """Configure structlog for Helios.

    Call once at application startup (e.g., in FastAPI lifespan).
    json_output=True: JSON lines for production (machine-parseable).
    json_output=False: colored console output for local development.
    """
    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
