from prometheus_client import Counter, Histogram, Gauge
import time
import psutil
import torch
from tensorrt_llm.serve.metrics.metrics import Metrics, Stats, SpecDecodeMetrics

# Original metrics (keep these for backward compatibility)
REQUEST_COUNT = Counter(
    'tensorrt_llm_requests_total',
    'Total number of requests',
    ['model', 'endpoint']
)

REQUEST_LATENCY = Histogram(
    'tensorrt_llm_request_latency_seconds',
    'Request latency in seconds',
    ['model', 'endpoint'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 25.0, 50.0, 100.0]
)

TOOL_CALL_COUNT = Counter(
    'tensorrt_llm_tool_calls_total',
    'Total number of tool calls',
    ['model', 'tool_name']
)

TOOL_CALL_LATENCY = Histogram(
    'tensorrt_llm_tool_call_latency_seconds',
    'Tool call latency in seconds',
    ['model', 'tool_name'],
    buckets=[0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0]
)

ACTIVE_REQUESTS = Gauge(
    'tensorrt_llm_active_requests',
    'Number of active requests',
    ['model']
)

ERROR_COUNT = Counter(
    'tensorrt_llm_errors_total',
    'Total number of errors',
    ['model', 'error_type']
)

TOKEN_THROUGHPUT = Histogram(
    'tensorrt_llm_token_throughput_tokens_per_second',
    'Token generation throughput',
    ['model'],
    buckets=[1, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
)

GPU_MEMORY_USAGE = Gauge(
    'tensorrt_llm_gpu_memory_usage_percentage',
    'GPU memory usage percentage',
    ['model']
)

CPU_MEMORY_USAGE = Gauge(
    'tensorrt_llm_cpu_memory_usage_percentage',
    'CPU memory usage percentage',
    ['model']
)

GPU_CACHE_USAGE = Gauge(
    'tensorrt_llm_gpu_cache_usage_percentage',
    'GPU KV cache usage percentage',
    ['model']
)

SPEC_DECODE_ACCEPTANCE_RATE = Gauge(
    'tensorrt_llm_spec_decode_acceptance_rate',
    'Speculative decoding acceptance rate',
    ['model']
)

SPEC_DECODE_EFFICIENCY = Gauge(
    'tensorrt_llm_spec_decode_efficiency',
    'Speculative decoding system efficiency',
    ['model']
)

class MetricsMiddleware:
    def __init__(self, model_name):
        self.model_name = model_name
        self._initialize_metrics()
        self._last_system_update = 0
        self._system_update_interval = 5.0  # Update system metrics every 5 seconds
        
        # Track request statistics for Metrics class
        self._request_stats = {
            'prompt_tokens': [],
            'generation_tokens': [],
            'latencies': [],
            'finish_reasons': [],
            'errors': [],
            'time_to_first_tokens': [],
            'time_per_output_tokens': []
        }

    def _initialize_metrics(self):
        """Initialize both original metrics and Metrics class."""
        # Initialize original metrics with default values
        ACTIVE_REQUESTS.labels(model=self.model_name).set(0)
        GPU_MEMORY_USAGE.labels(model=self.model_name).set(0)
        CPU_MEMORY_USAGE.labels(model=self.model_name).set(0)
        GPU_CACHE_USAGE.labels(model=self.model_name).set(0)
        SPEC_DECODE_ACCEPTANCE_RATE.labels(model=self.model_name).set(0)
        SPEC_DECODE_EFFICIENCY.labels(model=self.model_name).set(0)
        
        # Initialize Metrics class
        self.metrics = Metrics(labelnames=["model"], max_model_len=8192)
        
        # Initialize Metrics class default values
        self.metrics.gauge_scheduler_running.labels(model=self.model_name).set(0)
        self.metrics.gauge_scheduler_waiting.labels(model=self.model_name).set(0)
        self.metrics.gauge_gpu_cache_usage.labels(model=self.model_name).set(0)
        self.metrics.gauge_spec_decode_draft_acceptance_rate.labels(model=self.model_name).set(0)
        self.metrics.gauge_spec_decode_efficiency.labels(model=self.model_name).set(0)

    def _update_system_metrics(self):
        """Update system-level metrics periodically."""
        now = time.time()
        if now - self._last_system_update < self._system_update_interval:
            return
        
        self._last_system_update = now
        
        # Update GPU memory usage
        if torch.cuda.is_available():
            try:
                gpu_memory_allocated = torch.cuda.memory_allocated()
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                gpu_usage = gpu_memory_allocated / gpu_memory_total
                GPU_MEMORY_USAGE.labels(model=self.model_name).set(gpu_usage * 100)
            except Exception:
                pass
        
        # Update CPU memory usage
        try:
            process = psutil.Process()
            cpu_memory_info = process.memory_info()
            cpu_memory_total = psutil.virtual_memory().total
            cpu_usage = cpu_memory_info.rss / cpu_memory_total
            CPU_MEMORY_USAGE.labels(model=self.model_name).set(cpu_usage * 100)
        except Exception:
            pass
        
        # Update CPU utilization
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            # Note: CPU utilization could be added to Metrics class if needed
        except Exception:
            pass

    def _create_stats_object(self) -> Stats:
        """Create a Stats object with current metrics data."""
        now = time.time()
        
        # Get system metrics
        gpu_memory_usage = 0.0
        cpu_memory_usage = 0.0
        gpu_cache_usage = 0.0
        
        if torch.cuda.is_available():
            try:
                gpu_memory_allocated = torch.cuda.memory_allocated()
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_usage = gpu_memory_allocated / gpu_memory_total
            except Exception:
                pass
        
        try:
            process = psutil.Process()
            cpu_memory_info = process.memory_info()
            cpu_memory_total = psutil.virtual_memory().total
            cpu_memory_usage = cpu_memory_info.rss / cpu_memory_total
        except Exception:
            pass
        
        # Create Stats object
        stats = Stats(
            now=now,
            # System stats
            num_running_sys=0,  # Will be updated by active requests
            num_waiting_sys=0,  # Will be updated by queue length
            num_swapped_sys=0,
            gpu_cache_usage_sys=gpu_cache_usage,
            cpu_cache_usage_sys=0.0,
            gpu_memory_usage_sys=gpu_memory_usage,
            cpu_memory_usage_sys=cpu_memory_usage,
            gpu_prefix_cache_hit_rate=0.0,
            cpu_prefix_cache_hit_rate=0.0,
            # LoRA stats
            running_lora_adapters=[],
            waiting_lora_adapters=[],
            max_lora=0,
            # Iteration stats
            num_preemption_iter=0,
            num_prompt_tokens_iter=sum(self._request_stats['prompt_tokens']),
            num_generation_tokens_iter=sum(self._request_stats['generation_tokens']),
            num_tokens_iter=sum(self._request_stats['prompt_tokens']) + sum(self._request_stats['generation_tokens']),
            time_to_first_tokens_iter=self._request_stats['time_to_first_tokens'],
            time_per_output_tokens_iter=self._request_stats['time_per_output_tokens'],
            # Request stats
            time_e2e_requests=self._request_stats['latencies'],
            time_queue_requests=[],  # Not tracked yet
            time_inference_requests=[],  # Not tracked yet
            time_prefill_requests=[],  # Not tracked yet
            time_decode_requests=[],  # Not tracked yet
            num_prompt_tokens_requests=self._request_stats['prompt_tokens'],
            num_generation_tokens_requests=self._request_stats['generation_tokens'],
            n_requests=[1] * len(self._request_stats['latencies']) if self._request_stats['latencies'] else [],
            max_num_generation_tokens_requests=[],  # Not tracked yet
            max_tokens_requests=[],  # Not tracked yet
            finished_reason_requests=self._request_stats['finish_reasons'],
            # Speculative decoding stats
            spec_decode_metrics=None,
            # Tool calling stats
            tool_calls_iter=[],
            tool_call_errors_iter=[]
        )
        
        return stats

    def _log_metrics(self):
        """Log current metrics to Prometheus using the Metrics class."""
        try:
            # Check if we have any data to log (not just latencies)
            has_data = (self._request_stats['prompt_tokens'] or 
                       self._request_stats['generation_tokens'] or 
                       self._request_stats['latencies'])
            
            if not has_data:  # No data to log
                return
                
            stats = self._create_stats_object()
            
            # Create PrometheusStatLogger and log metrics
            from tensorrt_llm.serve.metrics.metrics import PrometheusStatLogger
            logger = PrometheusStatLogger(
                local_interval=1.0,
                labels={"model": self.model_name},
                max_model_len=8192
            )
            logger.log(stats)
            
            # Clear request stats after logging
            self._request_stats = {
                'prompt_tokens': [],
                'generation_tokens': [],
                'latencies': [],
                'finish_reasons': [],
                'errors': [],
                'time_to_first_tokens': [],
                'time_per_output_tokens': []
            }
        except Exception as e:
            print(f"Error logging metrics: {e}")

    def track_request(self, endpoint):
        """Track a new request."""
        self._update_system_metrics()
        
        # Update original metrics
        REQUEST_COUNT.labels(model=self.model_name, endpoint=endpoint).inc()
        ACTIVE_REQUESTS.labels(model=self.model_name).inc()
        
        # Update Metrics class
        current_running = self.metrics.gauge_scheduler_running.labels(model=self.model_name)._value.get()
        self.metrics.gauge_scheduler_running.labels(model=self.model_name).set(current_running + 1)

    def track_latency(self, endpoint, start_time):
        """Track request latency."""
        latency = time.time() - start_time
        
        # Update original metrics
        REQUEST_LATENCY.labels(model=self.model_name, endpoint=endpoint).observe(latency)
        ACTIVE_REQUESTS.labels(model=self.model_name).dec()
        
        # Update Metrics class
        self._request_stats['latencies'].append(latency)
        current_running = self.metrics.gauge_scheduler_running.labels(model=self.model_name)._value.get()
        self.metrics.gauge_scheduler_running.labels(model=self.model_name).set(max(0, current_running - 1))
        
        # Log metrics periodically
        if len(self._request_stats['latencies']) >= 10:  # Log every 10 requests
            self._log_metrics()

    def track_tokens(self, prompt_tokens, generation_tokens):
        """Track token counts."""
        # Update original metrics immediately
        if prompt_tokens > 0:
            # Update the original prompt_tokens_total counter using existing metrics instance
            self.metrics.counter_prompt_tokens.labels(model=self.model_name).inc(prompt_tokens)
            
            # Update the request_prompt_tokens histogram
            self.metrics.histogram_num_prompt_tokens_request.labels(model=self.model_name).observe(prompt_tokens)
            
            # Also store for Metrics class batch logging
            self._request_stats['prompt_tokens'].append(prompt_tokens)
        
        if generation_tokens > 0:
            # Update the original generation_tokens_total counter using existing metrics instance
            self.metrics.counter_generation_tokens.labels(model=self.model_name).inc(generation_tokens)
            
            # Update the request_generation_tokens histogram
            self.metrics.histogram_num_generation_tokens_request.labels(model=self.model_name).observe(generation_tokens)
            
            # Also store for Metrics class batch logging
            self._request_stats['generation_tokens'].append(generation_tokens)
        
        # Force log metrics if we have accumulated enough data
        if len(self._request_stats['prompt_tokens']) + len(self._request_stats['generation_tokens']) >= 5:
            self._log_metrics()

    def track_time_to_first_token(self, ttft):
        """Track time to first token."""
        self._request_stats['time_to_first_tokens'].append(ttft)

    def track_time_per_token(self, tpt):
        """Track time per token."""
        self._request_stats['time_per_output_tokens'].append(tpt)

    def track_tool_call(self, tool_name):
        """Track a tool call."""
        # Update original metrics
        TOOL_CALL_COUNT.labels(model=self.model_name, tool_name=tool_name).inc()
        
        # Add to tool calls list for Metrics class
        self._request_stats.setdefault('tool_calls', []).append({'tool_name': tool_name})

    def track_tool_latency(self, tool_name, start_time):
        """Track tool call latency."""
        latency = time.time() - start_time
        TOOL_CALL_LATENCY.labels(model=self.model_name, tool_name=tool_name).observe(latency)

    def track_error(self, error_type):
        """Track an error."""
        # Update original metrics
        ERROR_COUNT.labels(model=self.model_name, error_type=error_type).inc()
        
        # Add to errors list for Metrics class
        self._request_stats['errors'].append(error_type)

    def track_success(self, finish_reason="stop"):
        """Track a successful request."""
        # Add to finish reasons list for Metrics class
        self._request_stats['finish_reasons'].append(finish_reason)

    def track_throughput(self, tokens_per_second):
        """Track token throughput."""
        TOKEN_THROUGHPUT.labels(model=self.model_name).observe(tokens_per_second)

    def track_gpu_memory_usage(self, usage_percentage):
        """Track GPU memory usage."""
        GPU_MEMORY_USAGE.labels(model=self.model_name).set(usage_percentage)

    def track_cpu_memory_usage(self, usage_percentage):
        """Track CPU memory usage."""
        CPU_MEMORY_USAGE.labels(model=self.model_name).set(usage_percentage)

    def track_gpu_cache_usage(self, usage_percentage):
        """Track GPU cache usage."""
        GPU_CACHE_USAGE.labels(model=self.model_name).set(usage_percentage)
        self.metrics.gauge_gpu_cache_usage.labels(model=self.model_name).set(usage_percentage)

    def track_gpu_utilization(self, utilization_percentage):
        """Track GPU utilization."""
        # This could be added to the Metrics class if needed
        pass

    def track_queue_length(self, length):
        """Track queue length."""
        self.metrics.gauge_scheduler_waiting.labels(model=self.model_name).set(length)

    def track_queue_time(self, queue_time):
        """Track time spent in queue."""
        # This will be handled through the Stats object
        pass

    def track_spec_decode_metrics(self, acceptance_rate, efficiency):
        """Track speculative decoding metrics."""
        # Update original metrics
        SPEC_DECODE_ACCEPTANCE_RATE.labels(model=self.model_name).set(acceptance_rate)
        SPEC_DECODE_EFFICIENCY.labels(model=self.model_name).set(efficiency)
        
        # Update Metrics class
        self.metrics.gauge_spec_decode_draft_acceptance_rate.labels(model=self.model_name).set(acceptance_rate)
        self.metrics.gauge_spec_decode_efficiency.labels(model=self.model_name).set(efficiency)

    def force_log_metrics(self):
        """Force log current metrics to Prometheus."""
        self._log_metrics() 