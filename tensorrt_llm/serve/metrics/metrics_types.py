# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the TensorRT-LLM project

import time
import psutil
import torch
from typing import Optional

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.metrics.metrics import Metrics, Stats, PrometheusStatLogger
from tensorrt_llm.serve.metrics.engine_metrics_collector import EngineMetricsCollector


class TensorRTMetrics:
    """
    TensorRT-LLM metrics system that directly uses EngineMetricsCollector to gather data.
    Removes redundancy from MetricsMiddleware and provides a cleaner interface.
    """
    
    def __init__(self, model_name: str, llm=None):
        self.model_name = model_name
        self.llm = llm
        
        # Initialize engine metrics collector
        if llm:
            self.engine_collector = EngineMetricsCollector(llm)
        else:
            self.engine_collector = None
            logger.warning("No LLM instance provided, engine metrics collection disabled")
        
        # Initialize Prometheus metrics logger
        self.prometheus_logger = PrometheusStatLogger(
            local_interval=1.0,
            labels={"model": self.model_name},
            max_model_len=8192
        )
        
        # Initialize Metrics class (for backward compatibility)
        self.metrics = Metrics(labelnames=["model"], max_model_len=8192)
        
        # System metrics update interval
        self._last_system_update = 0
        self._system_update_interval = 5.0
        
        # Request tracking
        self._active_requests = 0
        self._request_start_times = {}
    
    def track_request_start(self, request_id: str, prompt_tokens: int = 0, max_tokens: int = 100):
        """Track request start"""
        self._active_requests += 1
        self._request_start_times[request_id] = time.time()
        
        # Update Metrics class indicators
        self.metrics.gauge_scheduler_running.labels(model=self.model_name).set(self._active_requests)
        
        # Track request in engine collector if available
        if self.engine_collector:
            self.engine_collector.track_request_start(request_id, prompt_tokens, max_tokens)
        
        logger.debug(f"Request {request_id} started, active requests: {self._active_requests}")
    
    def track_request_completion(self, request_id: str, prompt_tokens: int = 0, generation_tokens: int = 0, finish_reason: str = "stop"):
        """Track request completion"""
        if request_id in self._request_start_times:
            latency = time.time() - self._request_start_times[request_id]
            del self._request_start_times[request_id]
        else:
            latency = 0.0
        
        self._active_requests = max(0, self._active_requests - 1)
        
        # Update Metrics class indicators
        self.metrics.gauge_scheduler_running.labels(model=self.model_name).set(self._active_requests)
        
        # Update token counts
        if prompt_tokens > 0:
            self.metrics.counter_prompt_tokens.labels(model=self.model_name).inc(prompt_tokens)
            self.metrics.histogram_num_prompt_tokens_request.labels(model=self.model_name).observe(prompt_tokens)
        
        if generation_tokens > 0:
            self.metrics.counter_generation_tokens.labels(model=self.model_name).inc(generation_tokens)
            self.metrics.histogram_num_generation_tokens_request.labels(model=self.model_name).observe(generation_tokens)
        
        # Update latency metrics
        if latency > 0:
            self.metrics.histogram_e2e_time_request.labels(model=self.model_name).observe(latency)
        
        # Track completion in engine collector if available
        if self.engine_collector:
            self.engine_collector.track_request_completion(request_id, finish_reason)
        
        logger.debug(f"Request {request_id} completed, latency: {latency:.3f}s, active requests: {self._active_requests}")
    
    def track_token_generation(self, request_id: str, num_tokens: int):
        """Track token generation"""
        if num_tokens > 0:
            self.metrics.counter_generation_tokens.labels(model=self.model_name).inc(num_tokens)
            self.metrics.histogram_num_generation_tokens_request.labels(model=self.model_name).observe(num_tokens)
        
        # Track token generation in engine collector if available
        if self.engine_collector:
            self.engine_collector.track_token_generation(request_id, num_tokens)
    
    def track_first_token(self, request_id: str):
        """Track first token generation"""
        if self.engine_collector:
            self.engine_collector.track_first_token(request_id)
    
    def track_error(self, request_id: str, error_type: str):
        """Track error"""
        self.metrics.counter_request_errors.labels(model=self.model_name, error_type=error_type).inc()
        
        # Decrease active request count
        if request_id in self._request_start_times:
            del self._request_start_times[request_id]
            self._active_requests = max(0, self._active_requests - 1)
            self.metrics.gauge_scheduler_running.labels(model=self.model_name).set(self._active_requests)
    
    def update_system_metrics(self):
        """Update system metrics"""
        now = time.time()
        if now - self._last_system_update < self._system_update_interval:
            return
        
        self._last_system_update = now
        
        # GPU memory usage
        if torch.cuda.is_available():
            try:
                gpu_memory_allocated = torch.cuda.memory_allocated()
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                gpu_usage = gpu_memory_allocated / gpu_memory_total
                self.metrics.gauge_gpu_memory_usage.labels(model=self.model_name).set(gpu_usage * 100)
            except Exception:
                pass
        
        # CPU memory usage
        try:
            process = psutil.Process()
            cpu_memory_info = process.memory_info()
            cpu_memory_total = psutil.virtual_memory().total
            cpu_usage = cpu_memory_info.rss / cpu_memory_total
            self.metrics.gauge_cpu_memory_usage.labels(model=self.model_name).set(cpu_usage * 100)
        except Exception:
            pass
    
    def log_metrics(self):
        """Log metrics to Prometheus"""
        try:
            # Update system metrics
            self.update_system_metrics()
            
            # Use engine collector statistics if available
            if self.engine_collector and self.engine_collector.should_collect_stats():
                stats = self.engine_collector.get_stats()
                self.prometheus_logger.log(stats)
                logger.debug("Logged metrics using engine collector data")
            else:
                # Create basic statistics object
                stats = self._create_basic_stats()
                self.prometheus_logger.log(stats)
                logger.debug("Logged metrics using basic stats")
                
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    def _create_basic_stats(self) -> Stats:
        """Create basic statistics object"""
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
            # System statistics
            num_running_sys=self._active_requests,
            num_waiting_sys=0,
            num_swapped_sys=0,
            gpu_cache_usage_sys=gpu_cache_usage,
            cpu_cache_usage_sys=0.0,
            gpu_memory_usage_sys=gpu_memory_usage,
            cpu_memory_usage_sys=cpu_memory_usage,
            gpu_prefix_cache_hit_rate=0.0,
            cpu_prefix_cache_hit_rate=0.0,
            # LoRA statistics
            running_lora_adapters=[],
            waiting_lora_adapters=[],
            max_lora=0,
            # Iteration statistics
            num_preemption_iter=0,
            num_prompt_tokens_iter=0,
            num_generation_tokens_iter=0,
            num_tokens_iter=0,
            time_to_first_tokens_iter=[],
            time_per_output_tokens_iter=[],
            # Request statistics
            time_e2e_requests=[],
            time_queue_requests=[],
            time_inference_requests=[],
            time_prefill_requests=[],
            time_decode_requests=[],
            num_prompt_tokens_requests=[],
            num_generation_tokens_requests=[],
            n_requests=[],
            max_num_generation_tokens_requests=[],
            max_tokens_requests=[],
            finished_reason_requests=[],
            # Speculative decoding statistics
            spec_decode_metrics=None,
            # Tool calling statistics
            tool_calls_iter=[],
            tool_call_errors_iter=[]
        )
        
        return stats
    
    def get_active_requests(self) -> int:
        """Get number of active requests"""
        return self._active_requests
    
    def force_log_metrics(self):
        """Force log metrics"""
        self.log_metrics() 