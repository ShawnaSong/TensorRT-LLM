# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Callable, Optional
from typing import Dict, List, Optional, Type, Union, cast
from typing import Counter as CollectionsCounter


import numpy as np
import prometheus_client

from tensorrt_llm.logger import logger
from .collector import Stats
from .metrics import Metrics
from .collector import SpecDecodeMetrics

class LoggingStatLogger:
    """LoggingStatLogger is used in LLMEngine to log to Stdout."""

    def __init__(self, local_interval: float) -> None:
        self.local_interval = local_interval
        self.last_local_log = 0.0
        self.num_prompt_tokens: List[int] = []
        self.num_generation_tokens: List[int] = []
        self.last_prompt_throughput: Optional[float] = None
        self.last_generation_throughput: Optional[float] = None
        self.spec_decode_metrics: Optional[SpecDecodeMetrics] = None

    def log(self, stats: Stats) -> None:
        """Called by LLMEngine.
           Logs to Stdout every self.local_interval seconds."""

        # Save tracked stats for token counters.
        self.num_prompt_tokens.append(stats.num_prompt_tokens_iter)
        self.num_generation_tokens.append(stats.num_generation_tokens_iter)

        # Update spec decode metrics
        self.spec_decode_metrics = stats.spec_decode_metrics

        # Log locally every local_interval seconds.
        if local_interval_elapsed(stats.now, self.last_local_log,
                                  self.local_interval):
            # Compute summary metrics for tracked stats (and log them
            # to promethus if applicable).
            prompt_throughput = get_throughput(self.num_prompt_tokens,
                                               now=stats.now,
                                               last_log=self.last_local_log)
            generation_throughput = get_throughput(
                self.num_generation_tokens,
                now=stats.now,
                last_log=self.last_local_log)

            log_fn = logger.info
            if not any((prompt_throughput, generation_throughput,
                        self.last_prompt_throughput,
                        self.last_generation_throughput)):
                # Avoid log noise on an idle production system
                log_fn = logger.debug

            log_fn(
                "Avg prompt throughput: %.1f tokens/s, "
                "Avg generation throughput: %.1f tokens/s, "
                "Running: %d reqs, Swapped: %d reqs, "
                "Pending: %d reqs, GPU KV cache usage: %.1f%%, "
                "CPU KV cache usage: %.1f%%.",
                prompt_throughput,
                generation_throughput,
                stats.num_running_sys,
                stats.num_swapped_sys,
                stats.num_waiting_sys,
                stats.gpu_cache_usage_sys * 100,
                stats.cpu_cache_usage_sys * 100,
            )
            if (stats.cpu_prefix_cache_hit_rate >= 0
                    or stats.gpu_prefix_cache_hit_rate >= 0):
                log_fn(
                    "Prefix cache hit rate: GPU: %.2f%%, CPU: %.2f%%",
                    stats.gpu_prefix_cache_hit_rate * 100,
                    stats.cpu_prefix_cache_hit_rate * 100,
                )
            if self.spec_decode_metrics is not None:
                log_fn(
                    self._format_spec_decode_metrics_str(
                        self.spec_decode_metrics))

            self._reset(stats, prompt_throughput, generation_throughput)

    def _reset(self, stats, prompt_throughput, generation_throughput) -> None:
        # Reset tracked stats for next interval.
        self.num_prompt_tokens = []
        self.num_generation_tokens = []
        self.last_local_log = stats.now
        self.spec_decode_metrics = None
        self.last_prompt_throughput = prompt_throughput
        self.last_generation_throughput = generation_throughput

    def _format_spec_decode_metrics_str(
            self, metrics: SpecDecodeMetrics) -> str:

        return ("Speculative metrics: "
                f"Draft acceptance rate: {metrics.draft_acceptance_rate:.3f}, "
                f"System efficiency: {metrics.system_efficiency:.3f}, "
                f"Number of speculative tokens: {metrics.num_spec_tokens}, "
                f"Number of accepted tokens: {metrics.accepted_tokens}, "
                f"Number of draft tokens: {metrics.draft_tokens}, "
                f"Number of emitted tokens: {metrics.emitted_tokens}.")


class PrometheusStatLogger:
    """PrometheusStatLogger is used LLMEngine to log to Promethus."""
    _metrics_cls = Metrics
    _gauge_cls = prometheus_client.Gauge

    def __init__(self, local_interval: float, labels: Dict[str, str],
                 max_model_len: int = 8192) -> None:
        self.local_interval = local_interval
        self.last_local_log = 0.0
        self.num_prompt_tokens: List[int] = []
        self.num_generation_tokens: List[int] = []
        self.spec_decode_metrics: Optional[SpecDecodeMetrics] = None
        
        # Prometheus metrics
        self.labels = labels
        self.metrics = self._metrics_cls(labelnames=list(labels.keys()),
                                         max_model_len=max_model_len)

    def _log_gauge(self, gauge, data: Union[int, float]) -> None:
        # Convenience function for logging to gauge.
        gauge.labels(**self.labels).set(data)

    def _log_counter(self, counter, data: Union[int, float]) -> None:
        # Convenience function for logging to counter.
        # Prevent ValueError from negative increment
        if data < 0:
            logger.warning("Skipping negative increment of %g to %s", data,
                           counter)
            return
        counter.labels(**self.labels).inc(data)

    def _log_counter_labels(self, counter, data: CollectionsCounter,
                            label_key: str) -> None:
        # Convenience function for collection counter of labels.
        for label, count in data.items():
            counter.labels(**{**self.labels, label_key: label}).inc(count)

    def _log_histogram(self, histogram, data: Union[List[int],
                                                    List[float]]) -> None:
        # Convenience function for logging list to histogram.
        for datum in data:
            histogram.labels(**self.labels).observe(datum)

    def _log_gauge_string(self, gauge, data: Dict[str, str]) -> None:
        gauge.labels(**data).set_to_current_time()

    def _log_prometheus(self, stats: Stats) -> None:
        # System state data
        self._log_gauge(self.metrics.gauge_scheduler_running,
                        stats.num_running_sys)
        self._log_gauge(self.metrics.gauge_scheduler_waiting,
                        stats.num_waiting_sys)
        self._log_gauge(self.metrics.gauge_gpu_cache_usage,
                        stats.gpu_cache_usage_sys)
        self._log_gauge(self.metrics.gauge_gpu_memory_usage,
                        stats.gpu_memory_usage_sys)
        self._log_gauge(self.metrics.gauge_cpu_memory_usage,
                        stats.cpu_memory_usage_sys)
        
        # Including max-lora in metric, in future this property of lora
        # config maybe extended to be dynamic.
        lora_info = {
            self.metrics.labelname_running_lora_adapters:
            ",".join(stats.running_lora_adapters),
            self.metrics.labelname_waiting_lora_adapters:
            ",".join(stats.waiting_lora_adapters),
            self.metrics.labelname_max_lora:
            stats.max_lora,
        }
        self._log_gauge_string(self.metrics.gauge_lora_info, lora_info)
        
        # Iteration level data
        self._log_counter(self.metrics.counter_num_preemption,
                          stats.num_preemption_iter)
        self._log_counter(self.metrics.counter_prompt_tokens,
                          stats.num_prompt_tokens_iter)
        self._log_counter(self.metrics.counter_generation_tokens,
                          stats.num_generation_tokens_iter)
        self._log_histogram(self.metrics.histogram_iteration_tokens,
                            [stats.num_tokens_iter])
        self._log_histogram(self.metrics.histogram_time_to_first_token,
                            stats.time_to_first_tokens_iter)
        self._log_histogram(self.metrics.histogram_time_per_output_token,
                            stats.time_per_output_tokens_iter)

        # Request level data
        # Latency
        self._log_histogram(self.metrics.histogram_e2e_time_request,
                            stats.time_e2e_requests)
        self._log_histogram(self.metrics.histogram_queue_time_request,
                            stats.time_queue_requests)
        self._log_histogram(self.metrics.histogram_inference_time_request,
                            stats.time_inference_requests)
        self._log_histogram(self.metrics.histogram_prefill_time_request,
                            stats.time_prefill_requests)
        self._log_histogram(self.metrics.histogram_decode_time_request,
                            stats.time_decode_requests)
        # Metadata
        finished_reason_counter = CollectionsCounter(
            stats.finished_reason_requests)
        self._log_counter_labels(self.metrics.counter_request_success,
                                 finished_reason_counter,
                                 Metrics.labelname_finish_reason)
        self._log_histogram(self.metrics.histogram_num_prompt_tokens_request,
                            stats.num_prompt_tokens_requests)
        self._log_histogram(
            self.metrics.histogram_num_generation_tokens_request,
            stats.num_generation_tokens_requests)
        self._log_histogram(self.metrics.histogram_n_request, stats.n_requests)
        self._log_histogram(
            self.metrics.histogram_max_num_generation_tokens_request,
            stats.max_num_generation_tokens_requests)
        self._log_histogram(self.metrics.histogram_max_tokens_request,
                            stats.max_tokens_requests)

        # Tool calling stats
        if stats.tool_calls_iter:
            for tool_call in stats.tool_calls_iter:
                tool_name = tool_call.get('name', 'unknown')
                self.metrics.counter_tool_calls_total.labels(
                    **{**self.labels, 'tool_name': tool_name}).inc(1)
        
        if stats.tool_call_errors_iter:
            for error_type in stats.tool_call_errors_iter:
                self.metrics.counter_tool_call_errors.labels(
                    **{**self.labels, 'error_type': error_type}).inc(1)

    def log(self, stats: Stats):
        """Logs to prometheus and tracked stats every iteration."""
        # Log to prometheus.
        self._log_prometheus(stats)

        # Save tracked stats for token counters.
        self.num_prompt_tokens.append(stats.num_prompt_tokens_iter)
        self.num_generation_tokens.append(stats.num_generation_tokens_iter)

        # Update spec decode metrics
        self.spec_decode_metrics = stats.spec_decode_metrics

        # Log locally every local_interval seconds.
        if local_interval_elapsed(stats.now, self.last_local_log,
                                  self.local_interval):
            if self.spec_decode_metrics is not None:
                self._log_gauge(
                    self.metrics.gauge_spec_decode_draft_acceptance_rate,
                    self.spec_decode_metrics.draft_acceptance_rate)
                self._log_gauge(self.metrics.gauge_spec_decode_efficiency,
                                self.spec_decode_metrics.system_efficiency)
                self._log_counter(
                    self.metrics.counter_spec_decode_num_accepted_tokens,
                    self.spec_decode_metrics.accepted_tokens)
                self._log_counter(
                    self.metrics.counter_spec_decode_num_draft_tokens,
                    self.spec_decode_metrics.draft_tokens)
                self._log_counter(
                    self.metrics.counter_spec_decode_num_emitted_tokens,
                    self.spec_decode_metrics.emitted_tokens)

            # Reset tracked stats for next interval.
            self.num_prompt_tokens = []
            self.num_generation_tokens = []
            self.last_local_log = stats.now
            self.spec_decode_metrics = None

    def info(self, type: str, obj) -> None:
        # Info type metrics are syntactic sugar for a gauge permanently set to 1
        # Since prometheus multiprocessing mode does not support Info, emulate
        # info here with a gauge.
        if type == "cache_config":
            metrics_info = obj.metrics_info()
            info_gauge = self._gauge_cls(
                name="tensorrt_llm:cache_config_info",
                documentation="Information of the LLMEngine CacheConfig",
                labelnames=metrics_info.keys(),
                multiprocess_mode="mostrecent")
            info_gauge.labels(**metrics_info).set(1) 


def local_interval_elapsed(now: float, last_log: float,
                           local_interval: float) -> bool:
    elapsed_time = now - last_log
    return elapsed_time > local_interval


def get_throughput(tracked_stats: List[int], now: float,
                   last_log: float) -> float:
    return float(np.sum(tracked_stats) / (now - last_log))
