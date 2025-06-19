import time
import psutil
import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from prometheus_client import start_http_server
import threading
import logging

from .metrics import Stats, SpecDecodeMetrics, LoggingStatLogger, PrometheusStatLogger

logger = logging.getLogger(__name__)

@dataclass
class EngineStats:
    """Engine statistics for metrics collection."""
    num_running_requests: int = 0
    num_waiting_requests: int = 0
    num_swapped_requests: int = 0
    gpu_memory_usage: float = 0.0
    cpu_memory_usage: float = 0.0
    gpu_cache_usage: float = 0.0
    cpu_cache_usage: float = 0.0
    gpu_prefix_cache_hit_rate: float = -1.0
    cpu_prefix_cache_hit_rate: float = -1.0
    running_lora_adapters: List[str] = None
    waiting_lora_adapters: List[str] = None
    max_lora: int = 0

class MetricsCollector:
    """Collects metrics from TensorRT-LLM engine and system."""
    
    def __init__(self, model_name: str, max_model_len: int = 8192):
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.start_time = time.time()
        
        # Initialize stat loggers
        self.logging_logger = LoggingStatLogger(local_interval=10.0)
        self.prometheus_logger = PrometheusStatLogger(
            local_interval=10.0,
            labels={"model": model_name},
            max_model_len=max_model_len
        )
        
        # Track cumulative stats
        self.total_prompt_tokens = 0
        self.total_generation_tokens = 0
        self.total_requests = 0
        self.total_errors = 0
        
        # Track request stats
        self.request_start_times: Dict[str, float] = {}
        self.request_stats: List[Dict] = []
        
        # Track tool calling stats
        self.tool_calls: List[Dict] = []
        self.tool_call_errors: List[str] = []
        
        # Track speculative decoding stats
        self.spec_decode_stats: Optional[SpecDecodeMetrics] = None

    def get_system_stats(self) -> EngineStats:
        """Collect system-level statistics."""
        stats = EngineStats()
        
        # Get GPU memory usage
        if torch.cuda.is_available():
            try:
                gpu_memory_allocated = torch.cuda.memory_allocated()
                gpu_memory_reserved = torch.cuda.memory_reserved()
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                stats.gpu_memory_usage = gpu_memory_allocated / gpu_memory_total
            except Exception:
                stats.gpu_memory_usage = 0.0
        
        # Get CPU memory usage
        try:
            process = psutil.Process()
            cpu_memory_info = process.memory_info()
            cpu_memory_total = psutil.virtual_memory().total
            stats.cpu_memory_usage = cpu_memory_info.rss / cpu_memory_total
        except Exception:
            stats.cpu_memory_usage = 0.0
        
        return stats

    def record_request_start(self, request_id: str):
        """Record the start of a request."""
        self.request_start_times[request_id] = time.time()
        self.total_requests += 1

    def record_request_end(self, request_id: str, 
                          prompt_tokens: int,
                          generation_tokens: int,
                          finish_reason: str = "stop"):
        """Record the end of a request."""
        if request_id in self.request_start_times:
            start_time = self.request_start_times[request_id]
            end_time = time.time()
            e2e_latency = end_time - start_time
            
            # Record request stats
            request_stat = {
                "e2e_latency": e2e_latency,
                "prompt_tokens": prompt_tokens,
                "generation_tokens": generation_tokens,
                "finish_reason": finish_reason
            }
            self.request_stats.append(request_stat)
            
            # Update cumulative stats
            self.total_prompt_tokens += prompt_tokens
            self.total_generation_tokens += generation_tokens
            
            # Clean up
            del self.request_start_times[request_id]

    def record_tool_call(self, tool_name: str, start_time: float, end_time: float):
        """Record a tool call."""
        latency = end_time - start_time
        tool_call = {
            "name": tool_name,
            "latency": latency,
            "timestamp": start_time
        }
        self.tool_calls.append(tool_call)

    def record_tool_call_error(self, error_type: str):
        """Record a tool call error."""
        self.tool_call_errors.append(error_type)
        self.total_errors += 1

    def record_spec_decode_metrics(self, 
                                 draft_acceptance_rate: float,
                                 system_efficiency: float,
                                 num_spec_tokens: int,
                                 accepted_tokens: int,
                                 draft_tokens: int,
                                 emitted_tokens: int):
        """Record speculative decoding metrics."""
        self.spec_decode_stats = SpecDecodeMetrics(
            draft_acceptance_rate=draft_acceptance_rate,
            system_efficiency=system_efficiency,
            num_spec_tokens=num_spec_tokens,
            accepted_tokens=accepted_tokens,
            draft_tokens=draft_tokens,
            emitted_tokens=emitted_tokens
        )

    def collect_stats(self, engine_stats: EngineStats) -> Stats:
        """Collect all statistics into a Stats object."""
        now = time.time()
        
        # Calculate throughput
        elapsed_time = now - self.start_time
        prompt_throughput = self.total_prompt_tokens / elapsed_time if elapsed_time > 0 else 0
        generation_throughput = self.total_generation_tokens / elapsed_time if elapsed_time > 0 else 0
        
        # Extract request stats
        e2e_latencies = [stat["e2e_latency"] for stat in self.request_stats[-100:]]  # Last 100 requests
        prompt_tokens_list = [stat["prompt_tokens"] for stat in self.request_stats[-100:]]
        generation_tokens_list = [stat["generation_tokens"] for stat in self.request_stats[-100:]]
        finish_reasons = [stat["finish_reason"] for stat in self.request_stats[-100:]]
        
        # Create Stats object
        stats = Stats(
            now=now,
            
            # System stats
            num_running_sys=engine_stats.num_running_requests,
            num_waiting_sys=engine_stats.num_waiting_requests,
            num_swapped_sys=engine_stats.num_swapped_requests,
            gpu_cache_usage_sys=engine_stats.gpu_cache_usage,
            cpu_cache_usage_sys=engine_stats.cpu_cache_usage,
            gpu_memory_usage_sys=engine_stats.gpu_memory_usage,
            cpu_memory_usage_sys=engine_stats.cpu_memory_usage,
            gpu_prefix_cache_hit_rate=engine_stats.gpu_prefix_cache_hit_rate,
            cpu_prefix_cache_hit_rate=engine_stats.cpu_prefix_cache_hit_rate,
            
            # LoRA stats
            running_lora_adapters=engine_stats.running_lora_adapters or [],
            waiting_lora_adapters=engine_stats.waiting_lora_adapters or [],
            max_lora=engine_stats.max_lora,
            
            # Iteration stats (simplified for now)
            num_preemption_iter=0,
            num_prompt_tokens_iter=self.total_prompt_tokens,
            num_generation_tokens_iter=self.total_generation_tokens,
            num_tokens_iter=self.total_prompt_tokens + self.total_generation_tokens,
            time_to_first_tokens_iter=[0.1],  # Placeholder
            time_per_output_tokens_iter=[0.05],  # Placeholder
            
            # Request stats
            time_e2e_requests=e2e_latencies,
            time_queue_requests=[0.01] * len(e2e_latencies),  # Placeholder
            time_inference_requests=[lat - 0.01 for lat in e2e_latencies],  # Placeholder
            time_prefill_requests=[lat * 0.3 for lat in e2e_latencies],  # Placeholder
            time_decode_requests=[lat * 0.7 for lat in e2e_latencies],  # Placeholder
            num_prompt_tokens_requests=prompt_tokens_list,
            num_generation_tokens_requests=generation_tokens_list,
            n_requests=[1] * len(e2e_latencies),  # Placeholder
            max_num_generation_tokens_requests=[100] * len(e2e_latencies),  # Placeholder
            max_tokens_requests=[100] * len(e2e_latencies),  # Placeholder
            finished_reason_requests=finish_reasons,
            
            # Speculative decoding stats
            spec_decode_metrics=self.spec_decode_stats,
            
            # Tool calling stats
            tool_calls_iter=self.tool_calls[-10:],  # Last 10 tool calls
            tool_call_errors_iter=self.tool_call_errors[-10:]  # Last 10 errors
        )
        
        return stats

    def log_stats(self, engine_stats: EngineStats):
        """Log statistics using both loggers."""
        stats = self.collect_stats(engine_stats)
        
        # Log to both loggers
        self.logging_logger.log(stats)
        self.prometheus_logger.log(stats)
        
        # Clear temporary stats
        self.tool_calls = self.tool_calls[-100:]  # Keep last 100
        self.tool_call_errors = self.tool_call_errors[-100:]  # Keep last 100

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get a summary of current statistics."""
        elapsed_time = time.time() - self.start_time
        
        return {
            "model_name": self.model_name,
            "uptime_seconds": elapsed_time,
            "total_requests": self.total_requests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_generation_tokens": self.total_generation_tokens,
            "total_errors": self.total_errors,
            "avg_prompt_throughput": self.total_prompt_tokens / elapsed_time if elapsed_time > 0 else 0,
            "avg_generation_throughput": self.total_generation_tokens / elapsed_time if elapsed_time > 0 else 0,
            "total_tool_calls": len(self.tool_calls),
            "total_tool_call_errors": len(self.tool_call_errors)
        }

    def start(self):
        """Start the metrics server in a separate thread."""
        def run_server():
            try:
                start_http_server(8000)
                logger.info("Metrics server started on port 8000")
            except Exception as e:
                logger.error(f"Failed to start metrics server: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
    
    def stop(self):
        """Stop the metrics server."""
        if self.server_thread and self.server_thread.is_alive():
            logger.info("Metrics server will be stopped with the main process") 