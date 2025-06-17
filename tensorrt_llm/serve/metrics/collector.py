import time
from typing import Dict, List, Optional
from prometheus_client import start_http_server
import threading
import logging

from .metrics import Metrics, Stats

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and exposes metrics for TensorRT-LLM."""
    
    def __init__(self, model_name: str, port: int = 18002):
        self.metrics = Metrics(model_name)
        self.port = port
        self.server_thread = None
        self._start_time = time.time()
        self._stats_buffer = []
    
    def start(self):
        """Start the metrics server in a separate thread."""
        def run_server():
            try:
                start_http_server(self.port)
                logger.info(f"Metrics server started on port {self.port}")
            except Exception as e:
                logger.error(f"Failed to start metrics server: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
    
    def collect_stats(self,
                     num_running_requests: int,
                     num_waiting_requests: int,
                     gpu_memory_usage: float,
                     prompt_tokens: int = 0,
                     generation_tokens: int = 0,
                     time_to_first_token: Optional[List[float]] = None,
                     time_per_token: Optional[List[float]] = None,
                     e2e_latency: Optional[List[float]] = None,
                     error_type: Optional[str] = None):
        """Collect stats from the LLM engine."""
        stats = Stats(
            now=time.time(),
            num_running_requests=num_running_requests,
            num_waiting_requests=num_waiting_requests,
            gpu_memory_usage=gpu_memory_usage,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            time_to_first_token=time_to_first_token or [],
            time_per_token=time_per_token or [],
            e2e_latency=e2e_latency or [],
            error_count=1 if error_type else 0,
            error_types={error_type: 1} if error_type else {}
        )
        
        self._stats_buffer.append(stats)
        self.metrics.record_stats(stats)
    
    def get_stats(self) -> List[Stats]:
        """Get collected stats."""
        return self._stats_buffer
    
    def clear_stats(self):
        """Clear collected stats."""
        self._stats_buffer = []
    
    def stop(self):
        """Stop the metrics server."""
        if self.server_thread and self.server_thread.is_alive():
            logger.info("Metrics server will be stopped with the main process") 