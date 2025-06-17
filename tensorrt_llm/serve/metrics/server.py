from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import logging
from typing import Optional

from .collector import MetricsCollector

logger = logging.getLogger(__name__)

class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for metrics endpoint."""
    
    collector: Optional[MetricsCollector] = None
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            
            if self.collector:
                stats = self.collector.get_stats()
                response = self._format_metrics(stats)
                self.wfile.write(response.encode())
            else:
                self.wfile.write(b'# No metrics collector available\n')
        else:
            self.send_response(404)
            self.end_headers()
    
    def _format_metrics(self, stats):
        """Format metrics in Prometheus format."""
        if not stats:
            return '# No metrics available\n'
        
        # Get the latest stats
        latest = stats[-1]
        
        # Format metrics
        metrics = []
        
        # System metrics
        metrics.append(f'# HELP tensorrt_llm_num_running_requests Number of requests currently running')
        metrics.append(f'# TYPE tensorrt_llm_num_running_requests gauge')
        metrics.append(f'tensorrt_llm_num_running_requests {latest.num_running_requests}')
        
        metrics.append(f'# HELP tensorrt_llm_num_waiting_requests Number of requests waiting in queue')
        metrics.append(f'# TYPE tensorrt_llm_num_waiting_requests gauge')
        metrics.append(f'tensorrt_llm_num_waiting_requests {latest.num_waiting_requests}')
        
        metrics.append(f'# HELP tensorrt_llm_gpu_memory_usage GPU memory usage in bytes')
        metrics.append(f'# TYPE tensorrt_llm_gpu_memory_usage gauge')
        metrics.append(f'tensorrt_llm_gpu_memory_usage {latest.gpu_memory_usage}')
        
        # Token metrics
        metrics.append(f'# HELP tensorrt_llm_prompt_tokens_total Total number of prompt tokens processed')
        metrics.append(f'# TYPE tensorrt_llm_prompt_tokens_total counter')
        metrics.append(f'tensorrt_llm_prompt_tokens_total {latest.prompt_tokens}')
        
        metrics.append(f'# HELP tensorrt_llm_generation_tokens_total Total number of generated tokens')
        metrics.append(f'# TYPE tensorrt_llm_generation_tokens_total counter')
        metrics.append(f'tensorrt_llm_generation_tokens_total {latest.generation_tokens}')
        
        # Latency metrics
        if latest.time_to_first_token:
            metrics.append(f'# HELP tensorrt_llm_time_to_first_token_seconds Time to first token in seconds')
            metrics.append(f'# TYPE tensorrt_llm_time_to_first_token_seconds histogram')
            for t in latest.time_to_first_token:
                metrics.append(f'tensorrt_llm_time_to_first_token_seconds_bucket{{le="{t}"}} 1')
        
        if latest.time_per_token:
            metrics.append(f'# HELP tensorrt_llm_time_per_token_seconds Time per token in seconds')
            metrics.append(f'# TYPE tensorrt_llm_time_per_token_seconds histogram')
            for t in latest.time_per_token:
                metrics.append(f'tensorrt_llm_time_per_token_seconds_bucket{{le="{t}"}} 1')
        
        if latest.e2e_latency:
            metrics.append(f'# HELP tensorrt_llm_e2e_latency_seconds End-to-end latency in seconds')
            metrics.append(f'# TYPE tensorrt_llm_e2e_latency_seconds histogram')
            for t in latest.e2e_latency:
                metrics.append(f'tensorrt_llm_e2e_latency_seconds_bucket{{le="{t}"}} 1')
        
        # Error metrics
        if latest.error_count > 0:
            metrics.append(f'# HELP tensorrt_llm_error_count_total Total number of errors')
            metrics.append(f'# TYPE tensorrt_llm_error_count_total counter')
            metrics.append(f'tensorrt_llm_error_count_total {latest.error_count}')
            
            for error_type, count in latest.error_types.items():
                metrics.append(f'# HELP tensorrt_llm_error_types_total Number of errors by type')
                metrics.append(f'# TYPE tensorrt_llm_error_types_total counter')
                metrics.append(f'tensorrt_llm_error_types_total{{type="{error_type}"}} {count}')
        
        return '\n'.join(metrics) + '\n'

def start_metrics_server(collector: MetricsCollector, host: str = 'localhost', port: int = 8000):
    """Start the metrics server."""
    MetricsHandler.collector = collector
    
    server = HTTPServer((host, port), MetricsHandler)
    logger.info(f"Starting metrics server on {host}:{port}")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping metrics server")
        server.server_close() 