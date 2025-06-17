from dataclasses import dataclass
from typing import Dict, List, Optional
import time
from prometheus_client import Counter, Gauge, Histogram, REGISTRY

@dataclass
class Stats:
    """Stats collected from LLM engine."""
    now: float
    
    # System stats
    num_running_requests: int
    num_waiting_requests: int
    gpu_memory_usage: float
    
    # Request stats
    prompt_tokens: int
    generation_tokens: int
    time_to_first_token: List[float]
    time_per_token: List[float]
    e2e_latency: List[float]
    
    # Error stats
    error_count: int
    error_types: Dict[str, int]

class Metrics:
    """Prometheus metrics for TensorRT-LLM."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._init_metrics()
    
    def _init_metrics(self):
        # System metrics
        self.gauge_running_requests = Gauge(
            'tensorrt_llm_running_requests',
            'Number of requests currently running',
            ['model_name']
        )
        
        self.gauge_waiting_requests = Gauge(
            'tensorrt_llm_waiting_requests',
            'Number of requests waiting in queue',
            ['model_name']
        )
        
        self.gauge_gpu_memory_usage = Gauge(
            'tensorrt_llm_gpu_memory_usage',
            'GPU memory usage in percentage',
            ['model_name']
        )
        
        # Token metrics
        self.counter_prompt_tokens = Counter(
            'tensorrt_llm_prompt_tokens_total',
            'Total number of prompt tokens processed',
            ['model_name']
        )
        
        self.counter_generation_tokens = Counter(
            'tensorrt_llm_generation_tokens_total',
            'Total number of generation tokens produced',
            ['model_name']
        )
        
        # Latency metrics
        self.histogram_time_to_first_token = Histogram(
            'tensorrt_llm_time_to_first_token_seconds',
            'Time to first token in seconds',
            ['model_name'],
            buckets=[0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0]
        )
        
        self.histogram_time_per_token = Histogram(
            'tensorrt_llm_time_per_token_seconds',
            'Time per token in seconds',
            ['model_name'],
            buckets=[0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5]
        )
        
        self.histogram_e2e_latency = Histogram(
            'tensorrt_llm_e2e_latency_seconds',
            'End-to-end request latency in seconds',
            ['model_name'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        )
        
        # Error metrics
        self.counter_errors = Counter(
            'tensorrt_llm_errors_total',
            'Total number of errors',
            ['model_name', 'error_type']
        )
    
    def record_stats(self, stats: Stats):
        """Record stats to Prometheus metrics."""
        # System metrics
        self.gauge_running_requests.labels(model_name=self.model_name).set(
            stats.num_running_requests)
        self.gauge_waiting_requests.labels(model_name=self.model_name).set(
            stats.num_waiting_requests)
        self.gauge_gpu_memory_usage.labels(model_name=self.model_name).set(
            stats.gpu_memory_usage)
        
        # Token metrics
        self.counter_prompt_tokens.labels(model_name=self.model_name).inc(
            stats.prompt_tokens)
        self.counter_generation_tokens.labels(model_name=self.model_name).inc(
            stats.generation_tokens)
        
        # Latency metrics
        for latency in stats.time_to_first_token:
            self.histogram_time_to_first_token.labels(
                model_name=self.model_name).observe(latency)
        
        for latency in stats.time_per_token:
            self.histogram_time_per_token.labels(
                model_name=self.model_name).observe(latency)
        
        for latency in stats.e2e_latency:
            self.histogram_e2e_latency.labels(
                model_name=self.model_name).observe(latency)
        
        # Error metrics
        for error_type, count in stats.error_types.items():
            self.counter_errors.labels(
                model_name=self.model_name,
                error_type=error_type
            ).inc(count) 