from prometheus_client import Counter, Histogram, Gauge
import time

REQUEST_COUNT = Counter(
    'tensorrt_llm_requests_total',
    'Total number of requests',
    ['model', 'endpoint']
)

REQUEST_LATENCY = Histogram(
    'tensorrt_llm_request_latency_seconds',
    'Request latency in seconds',
    ['model', 'endpoint']
)

TOOL_CALL_COUNT = Counter(
    'tensorrt_llm_tool_calls_total',
    'Total number of tool calls',
    ['model', 'tool_name']
)

TOOL_CALL_LATENCY = Histogram(
    'tensorrt_llm_tool_call_latency_seconds',
    'Tool call latency in seconds',
    ['model', 'tool_name']
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

class MetricsMiddleware:
    def __init__(self, model_name):
        self.model_name = model_name

    def track_request(self, endpoint):
        REQUEST_COUNT.labels(model=self.model_name, endpoint=endpoint).inc()
        ACTIVE_REQUESTS.labels(model=self.model_name).inc()

    def track_latency(self, endpoint, start_time):
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(model=self.model_name, endpoint=endpoint).observe(latency)
        ACTIVE_REQUESTS.labels(model=self.model_name).dec()

    def track_tool_call(self, tool_name):
        TOOL_CALL_COUNT.labels(model=self.model_name, tool_name=tool_name).inc()

    def track_tool_latency(self, tool_name, start_time):
        latency = time.time() - start_time
        TOOL_CALL_LATENCY.labels(model=self.model_name, tool_name=tool_name).observe(latency)

    def track_error(self, error_type):
        ERROR_COUNT.labels(model=self.model_name, error_type=error_type).inc() 