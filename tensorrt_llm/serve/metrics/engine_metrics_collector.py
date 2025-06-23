# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the TensorRT-LLM project

import time
import psutil
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from collections import defaultdict

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.metrics.metrics import Stats, SpecDecodeMetrics


@dataclass
class EngineStats:
    """Engine statistics collected from TensorRT-LLM executor."""
    now: float
    
    # System stats
    num_running_requests: int
    num_waiting_requests: int
    num_swapped_requests: int
    gpu_cache_usage: float
    cpu_cache_usage: float
    gpu_memory_usage: float
    cpu_memory_usage: float
    
    # Iteration stats
    num_prompt_tokens: int
    num_generation_tokens: int
    num_tokens_total: int
    time_to_first_tokens: List[float]
    time_per_output_tokens: List[float]
    num_preemptions: int
    
    # Request stats
    request_latencies: List[float]
    queue_times: List[float]
    inference_times: List[float]
    prefill_times: List[float]
    decode_times: List[float]
    prompt_tokens_per_request: List[int]
    generation_tokens_per_request: List[int]
    max_tokens_per_request: List[int]
    finish_reasons: List[str]
    
    # Speculative decoding stats
    spec_decode_metrics: Optional[SpecDecodeMetrics] = None


class EngineMetricsCollector:
    """Collects metrics directly from TensorRT-LLM engine components."""
    
    def __init__(self, llm):
        self.llm = llm
        self._request_timestamps = {}  # request_id -> arrival_time
        self._request_start_times = {}  # request_id -> first_token_time
        self._request_metrics = {}  # request_id -> metrics
        self._last_stats_time = time.time()
        self._stats_interval = 1.0  # Collect stats every second
        
    def _get_executor_stats(self) -> Dict[str, Any]:
        """Get statistics from the executor."""
        try:
            # Try to access executor stats if available
            if hasattr(self.llm, '_executor') and self.llm._executor:
                executor = self.llm._executor
                
                # Get basic stats from executor
                stats = {
                    'num_running': 0,
                    'num_waiting': 0,
                    'num_swapped': 0,
                    'gpu_cache_usage': 0.0,
                    'cpu_cache_usage': 0.0,
                }
                
                # Try to get more detailed stats if available
                if hasattr(executor, 'get_stats'):
                    executor_stats = executor.get_stats()
                    stats.update(executor_stats)
                
                return stats
        except Exception as e:
            logger.debug(f"Could not get executor stats: {e}")
        
        return {
            'num_running': 0,
            'num_waiting': 0,
            'num_swapped': 0,
            'gpu_cache_usage': 0.0,
            'cpu_cache_usage': 0.0,
        }
    
    def _get_system_stats(self) -> Dict[str, float]:
        """Get system-level statistics."""
        stats = {
            'gpu_memory_usage': 0.0,
            'cpu_memory_usage': 0.0,
        }
        
        # GPU memory usage
        if torch.cuda.is_available():
            try:
                gpu_memory_allocated = torch.cuda.memory_allocated()
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                stats['gpu_memory_usage'] = gpu_memory_allocated / gpu_memory_total
            except Exception as e:
                logger.debug(f"Could not get GPU memory stats: {e}")
        
        # CPU memory usage
        try:
            process = psutil.Process()
            cpu_memory_info = process.memory_info()
            cpu_memory_total = psutil.virtual_memory().total
            stats['cpu_memory_usage'] = cpu_memory_info.rss / cpu_memory_total
        except Exception as e:
            logger.debug(f"Could not get CPU memory stats: {e}")
        
        return stats
    
    def _get_request_stats(self) -> Dict[str, Any]:
        """Get request-level statistics."""
        now = time.time()
        request_stats = {
            'latencies': [],
            'queue_times': [],
            'inference_times': [],
            'prefill_times': [],
            'decode_times': [],
            'prompt_tokens': [],
            'generation_tokens': [],
            'max_tokens': [],
            'finish_reasons': [],
            'time_to_first_tokens': [],
            'time_per_output_tokens': [],
        }
        
        # Process completed requests
        completed_requests = []
        for request_id, arrival_time in self._request_timestamps.items():
            if request_id in self._request_metrics:
                metrics = self._request_metrics[request_id]
                if metrics.get('finished', False):
                    # Calculate latencies
                    latency = now - arrival_time
                    request_stats['latencies'].append(latency)
                    
                    # Get other metrics
                    if 'prompt_tokens' in metrics:
                        request_stats['prompt_tokens'].append(metrics['prompt_tokens'])
                    if 'generation_tokens' in metrics:
                        request_stats['generation_tokens'].append(metrics['generation_tokens'])
                    if 'max_tokens' in metrics:
                        request_stats['max_tokens'].append(metrics['max_tokens'])
                    if 'finish_reason' in metrics:
                        request_stats['finish_reasons'].append(metrics['finish_reason'])
                    
                    # Calculate timing metrics
                    if 'first_token_time' in metrics:
                        ttft = metrics['first_token_time'] - arrival_time
                        request_stats['time_to_first_tokens'].append(ttft)
                    
                    if 'last_token_time' in metrics and 'first_token_time' in metrics:
                        tpot = (metrics['last_token_time'] - metrics['first_token_time']) / max(1, metrics.get('generation_tokens', 1))
                        request_stats['time_per_output_tokens'].append(tpot)
                    
                    completed_requests.append(request_id)
        
        # Clean up completed requests
        for request_id in completed_requests:
            del self._request_timestamps[request_id]
            del self._request_metrics[request_id]
            if request_id in self._request_start_times:
                del self._request_start_times[request_id]
        
        return request_stats
    
    def track_request_start(self, request_id: str, prompt_tokens: int, max_tokens: int):
        """Track when a request starts processing."""
        now = time.time()
        self._request_timestamps[request_id] = now
        self._request_metrics[request_id] = {
            'prompt_tokens': prompt_tokens,
            'max_tokens': max_tokens,
            'finished': False,
        }
    
    def track_first_token(self, request_id: str):
        """Track when the first token is generated."""
        if request_id in self._request_timestamps:
            now = time.time()
            self._request_start_times[request_id] = now
            if request_id in self._request_metrics:
                self._request_metrics[request_id]['first_token_time'] = now
    
    def track_token_generation(self, request_id: str, num_tokens: int):
        """Track token generation."""
        if request_id in self._request_metrics:
            self._request_metrics[request_id]['generation_tokens'] = num_tokens
            self._request_metrics[request_id]['last_token_time'] = time.time()
    
    def track_request_completion(self, request_id: str, finish_reason: str):
        """Track when a request completes."""
        if request_id in self._request_metrics:
            self._request_metrics[request_id]['finished'] = True
            self._request_metrics[request_id]['finish_reason'] = finish_reason
    
    def get_stats(self) -> Stats:
        """Get comprehensive statistics similar to vLLM's _get_stats."""
        now = time.time()
        
        # Get executor stats
        executor_stats = self._get_executor_stats()
        
        # Get system stats
        system_stats = self._get_system_stats()
        
        # Get request stats
        request_stats = self._get_request_stats()
        
        # Calculate iteration stats
        num_prompt_tokens = sum(request_stats['prompt_tokens'])
        num_generation_tokens = sum(request_stats['generation_tokens'])
        num_tokens_total = num_prompt_tokens + num_generation_tokens
        
        # Create Stats object
        stats = Stats(
            now=now,
            # System stats
            num_running_sys=executor_stats.get('num_running', 0),
            num_waiting_sys=executor_stats.get('num_waiting', 0),
            num_swapped_sys=executor_stats.get('num_swapped', 0),
            gpu_cache_usage_sys=executor_stats.get('gpu_cache_usage', 0.0),
            cpu_cache_usage_sys=executor_stats.get('cpu_cache_usage', 0.0),
            gpu_memory_usage_sys=system_stats['gpu_memory_usage'],
            cpu_memory_usage_sys=system_stats['cpu_memory_usage'],
            gpu_prefix_cache_hit_rate=0.0,  # Not available in TensorRT-LLM
            cpu_prefix_cache_hit_rate=0.0,  # Not available in TensorRT-LLM
            # LoRA stats
            running_lora_adapters=[],
            waiting_lora_adapters=[],
            max_lora=0,
            # Iteration stats
            num_preemption_iter=0,  # Not tracked yet
            num_prompt_tokens_iter=num_prompt_tokens,
            num_generation_tokens_iter=num_generation_tokens,
            num_tokens_iter=num_tokens_total,
            time_to_first_tokens_iter=request_stats['time_to_first_tokens'],
            time_per_output_tokens_iter=request_stats['time_per_output_tokens'],
            # Request stats
            time_e2e_requests=request_stats['latencies'],
            time_queue_requests=request_stats['queue_times'],
            time_inference_requests=request_stats['inference_times'],
            time_prefill_requests=request_stats['prefill_times'],
            time_decode_requests=request_stats['decode_times'],
            num_prompt_tokens_requests=request_stats['prompt_tokens'],
            num_generation_tokens_requests=request_stats['generation_tokens'],
            n_requests=[1] * len(request_stats['latencies']) if request_stats['latencies'] else [],
            max_num_generation_tokens_requests=request_stats['max_tokens'],
            max_tokens_requests=request_stats['max_tokens'],
            finished_reason_requests=request_stats['finish_reasons'],
            # Speculative decoding stats
            spec_decode_metrics=None,
            # Tool calling stats
            tool_calls_iter=[],
            tool_call_errors_iter=[]
        )
        
        return stats
    
    def should_collect_stats(self) -> bool:
        """Check if it's time to collect stats."""
        now = time.time()
        if now - self._last_stats_time >= self._stats_interval:
            self._last_stats_time = now
            return True
        return False
    
    def reset_stats(self):
        """Reset collected statistics."""
        self._request_timestamps.clear()
        self._request_metrics.clear()
        self._request_start_times.clear()
        self._last_stats_time = time.time() 