"""TensorRT-LLM Metrics System.

This module provides comprehensive metrics collection and monitoring for TensorRT-LLM,
based on vLLM's metrics implementation.
"""

from .metrics import (
    Metrics,
    Stats,
    SpecDecodeMetrics,
    LoggingStatLogger,
    PrometheusStatLogger,
    build_1_2_5_buckets,
    build_1_2_3_5_8_buckets,
    local_interval_elapsed,
    get_throughput
)

from .collector import (
    MetricsCollector,
    EngineStats
)

from .prometheus_metrics import (
    MetricsMiddleware,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    TOOL_CALL_COUNT,
    TOOL_CALL_LATENCY,
    ACTIVE_REQUESTS,
    ERROR_COUNT,
    TOKEN_THROUGHPUT,
    GPU_MEMORY_USAGE,
    CPU_MEMORY_USAGE,
    GPU_CACHE_USAGE,
    SPEC_DECODE_ACCEPTANCE_RATE,
    SPEC_DECODE_EFFICIENCY
)

__all__ = [
    # Core metrics classes
    "Metrics",
    "Stats", 
    "SpecDecodeMetrics",
    "LoggingStatLogger",
    "PrometheusStatLogger",
    
    # Collector and engine stats
    "MetricsCollector",
    "EngineStats",
    
    # Middleware and utilities
    "MetricsMiddleware",
    
    # Utility functions
    "build_1_2_5_buckets",
    "build_1_2_3_5_8_buckets", 
    "local_interval_elapsed",
    "get_throughput",
    
    # Prometheus metrics
    "REQUEST_COUNT",
    "REQUEST_LATENCY", 
    "TOOL_CALL_COUNT",
    "TOOL_CALL_LATENCY",
    "ACTIVE_REQUESTS",
    "ERROR_COUNT",
    "TOKEN_THROUGHPUT",
    "GPU_MEMORY_USAGE",
    "CPU_MEMORY_USAGE",
    "GPU_CACHE_USAGE",
    "SPEC_DECODE_ACCEPTANCE_RATE",
    "SPEC_DECODE_EFFICIENCY"
] 