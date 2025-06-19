# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the TensorRT-LLM project

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Counter as CollectionsCounter
from typing import Dict, List, Optional, Type, Union, cast

import numpy as np
import prometheus_client

from tensorrt_llm.logger import logger

prometheus_client.disable_created_metrics()

# The begin-* and end* here are used by the documentation generator
# to extract the metrics definitions.


# --8<-- [start:metrics-definitions]
class Metrics:
    """
    TensorRT-LLM uses a multiprocessing-based frontend for the OpenAI server.
    This means that we need to run prometheus_client in multiprocessing mode
    See https://prometheus.github.io/client_python/multiprocess/ for more
    details on limitations.
    """

    labelname_finish_reason = "finished_reason"
    labelname_waiting_lora_adapters = "waiting_lora_adapters"
    labelname_running_lora_adapters = "running_lora_adapters"
    labelname_max_lora = "max_lora"
    _gauge_cls = prometheus_client.Gauge
    _counter_cls = prometheus_client.Counter
    _histogram_cls = prometheus_client.Histogram

    def __init__(self, labelnames: List[str], max_model_len: int = 8192):
        # Unregister any existing TensorRT-LLM collectors (for CI/CD)
        self._unregister_tensorrt_llm_metrics()

        # Use this flag to hide metrics that were deprecated in
        # a previous release and which will be removed future
        self.show_hidden_metrics = True

        # System stats
        #   Scheduler State
        self.gauge_scheduler_running = self._gauge_cls(
            name="tensorrt_llm:num_requests_running",
            documentation="Number of requests currently running on GPU.",
            labelnames=labelnames,
            multiprocess_mode="sum")
        self.gauge_scheduler_waiting = self._gauge_cls(
            name="tensorrt_llm:num_requests_waiting",
            documentation="Number of requests waiting to be processed.",
            labelnames=labelnames,
            multiprocess_mode="sum")
        self.gauge_lora_info = self._gauge_cls(
            name="tensorrt_llm:lora_requests_info",
            documentation="Running stats on lora requests.",
            labelnames=[
                self.labelname_running_lora_adapters,
                self.labelname_max_lora,
                self.labelname_waiting_lora_adapters,
            ],
            multiprocess_mode="livemostrecent",
        )

        #   KV Cache Usage in %
        self.gauge_gpu_cache_usage = self._gauge_cls(
            name="tensorrt_llm:gpu_cache_usage_perc",
            documentation="GPU KV-cache usage. 1 means 100 percent usage.",
            labelnames=labelnames,
            multiprocess_mode="sum")

        # Iteration stats
        self.counter_num_preemption = self._counter_cls(
            name="tensorrt_llm:num_preemptions_total",
            documentation="Cumulative number of preemption from the engine.",
            labelnames=labelnames)
        self.counter_prompt_tokens = self._counter_cls(
            name="tensorrt_llm:prompt_tokens_total",
            documentation="Number of prefill tokens processed.",
            labelnames=labelnames)
        self.counter_generation_tokens = self._counter_cls(
            name="tensorrt_llm:generation_tokens_total",
            documentation="Number of generation tokens processed.",
            labelnames=labelnames)
        self.histogram_iteration_tokens = self._histogram_cls(
            name="tensorrt_llm:iteration_tokens_total",
            documentation="Histogram of number of tokens per engine_step.",
            labelnames=labelnames,
            buckets=[
                1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
            ])
        self.histogram_time_to_first_token = self._histogram_cls(
            name="tensorrt_llm:time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            labelnames=labelnames,
            buckets=[
                0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0,
                2560.0
            ])
        self.histogram_time_per_output_token = self._histogram_cls(
            name="tensorrt_llm:time_per_output_token_seconds",
            documentation="Histogram of time per output token in seconds.",
            labelnames=labelnames,
            buckets=[
                0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75,
                1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0
            ])

        # Request stats
        #   Latency
        request_latency_buckets = [
            0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
            40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0
        ]
        self.histogram_e2e_time_request = self._histogram_cls(
            name="tensorrt_llm:e2e_request_latency_seconds",
            documentation="Histogram of end to end request latency in seconds.",
            labelnames=labelnames,
            buckets=request_latency_buckets)
        self.histogram_queue_time_request = self._histogram_cls(
            name="tensorrt_llm:request_queue_time_seconds",
            documentation=
            "Histogram of time spent in WAITING phase for request.",
            labelnames=labelnames,
            buckets=request_latency_buckets)
        self.histogram_inference_time_request = self._histogram_cls(
            name="tensorrt_llm:request_inference_time_seconds",
            documentation=
            "Histogram of time spent in RUNNING phase for request.",
            labelnames=labelnames,
            buckets=request_latency_buckets)
        self.histogram_prefill_time_request = self._histogram_cls(
            name="tensorrt_llm:request_prefill_time_seconds",
            documentation=
            "Histogram of time spent in PREFILL phase for request.",
            labelnames=labelnames,
            buckets=request_latency_buckets)
        self.histogram_decode_time_request = self._histogram_cls(
            name="tensorrt_llm:request_decode_time_seconds",
            documentation=
            "Histogram of time spent in DECODE phase for request.",
            labelnames=labelnames,
            buckets=request_latency_buckets)

        #   Metadata
        self.histogram_num_prompt_tokens_request = self._histogram_cls(
            name="tensorrt_llm:request_prompt_tokens",
            documentation="Number of prefill tokens processed.",
            labelnames=labelnames,
            buckets=build_1_2_5_buckets(max_model_len),
        )
        self.histogram_num_generation_tokens_request = \
            self._histogram_cls(
                name="tensorrt_llm:request_generation_tokens",
                documentation="Number of generation tokens processed.",
                labelnames=labelnames,
                buckets=build_1_2_5_buckets(max_model_len),
            )
        self.histogram_max_num_generation_tokens_request = self._histogram_cls(
            name="tensorrt_llm:request_max_num_generation_tokens",
            documentation=
            "Histogram of maximum number of requested generation tokens.",
            labelnames=labelnames,
            buckets=build_1_2_5_buckets(max_model_len))
        self.histogram_n_request = self._histogram_cls(
            name="tensorrt_llm:request_params_n",
            documentation="Histogram of the n request parameter.",
            labelnames=labelnames,
            buckets=[1, 2, 5, 10, 20],
        )
        self.histogram_max_tokens_request = self._histogram_cls(
            name="tensorrt_llm:request_params_max_tokens",
            documentation="Histogram of the max_tokens request parameter.",
            labelnames=labelnames,
            buckets=build_1_2_5_buckets(max_model_len),
        )
        self.counter_request_success = self._counter_cls(
            name="tensorrt_llm:request_success_total",
            documentation="Count of successfully processed requests.",
            labelnames=labelnames + [Metrics.labelname_finish_reason])

        # Speculative decoding stats
        self.gauge_spec_decode_draft_acceptance_rate = self._gauge_cls(
            name="tensorrt_llm:spec_decode_draft_acceptance_rate",
            documentation="Speulative token acceptance rate.",
            labelnames=labelnames,
            multiprocess_mode="sum")
        self.gauge_spec_decode_efficiency = self._gauge_cls(
            name="tensorrt_llm:spec_decode_efficiency",
            documentation="Speculative decoding system efficiency.",
            labelnames=labelnames,
            multiprocess_mode="sum")
        self.counter_spec_decode_num_accepted_tokens = (self._counter_cls(
            name="tensorrt_llm:spec_decode_num_accepted_tokens_total",
            documentation="Number of accepted tokens.",
            labelnames=labelnames))
        self.counter_spec_decode_num_draft_tokens = self._counter_cls(
            name="tensorrt_llm:spec_decode_num_draft_tokens_total",
            documentation="Number of draft tokens.",
            labelnames=labelnames)
        self.counter_spec_decode_num_emitted_tokens = (self._counter_cls(
            name="tensorrt_llm:spec_decode_num_emitted_tokens_total",
            documentation="Number of emitted tokens.",
            labelnames=labelnames))

        # Tool calling stats
        self.counter_tool_calls_total = self._counter_cls(
            name="tensorrt_llm:tool_calls_total",
            documentation="Total number of tool calls.",
            labelnames=labelnames + ["tool_name"])
        self.histogram_tool_call_latency = self._histogram_cls(
            name="tensorrt_llm:tool_call_latency_seconds",
            documentation="Tool call latency in seconds.",
            labelnames=labelnames + ["tool_name"],
            buckets=[0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0])
        self.counter_tool_call_errors = self._counter_cls(
            name="tensorrt_llm:tool_call_errors_total",
            documentation="Total number of tool call errors.",
            labelnames=labelnames + ["error_type"])

        # Memory and performance stats
        self.gauge_gpu_memory_usage = self._gauge_cls(
            name="tensorrt_llm:gpu_memory_usage_perc",
            documentation="GPU memory usage percentage.",
            labelnames=labelnames,
            multiprocess_mode="sum")
        self.gauge_cpu_memory_usage = self._gauge_cls(
            name="tensorrt_llm:cpu_memory_usage_perc",
            documentation="CPU memory usage percentage.",
            labelnames=labelnames,
            multiprocess_mode="sum")
        self.histogram_throughput_tokens_per_second = self._histogram_cls(
            name="tensorrt_llm:throughput_tokens_per_second",
            documentation="Tokens generated per second.",
            labelnames=labelnames,
            buckets=[1, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000])


# --8<-- [end:metrics-definitions]

    def _unregister_tensorrt_llm_metrics(self) -> None:
        for collector in list(prometheus_client.REGISTRY._collector_to_names):
            if hasattr(collector, "_name") and "tensorrt_llm" in collector._name:
                prometheus_client.REGISTRY.unregister(collector)


def build_buckets(mantissa_lst: List[int], max_value: int) -> List[int]:
    """
    Builds a list of buckets with increasing powers of 10 multiplied by
    mantissa values until the value exceeds the specified maximum.

    """
    exponent = 0
    buckets: List[int] = []
    while True:
        for m in mantissa_lst:
            value = m * 10**exponent
            if value <= max_value:
                buckets.append(value)
            else:
                return buckets
        exponent += 1


def build_1_2_5_buckets(max_value: int) -> List[int]:
    """
    Example:
    >>> build_1_2_5_buckets(100)
    [1, 2, 5, 10, 20, 50, 100]
    """
    return build_buckets([1, 2, 5], max_value)


def build_1_2_3_5_8_buckets(max_value: int) -> List[int]:
    """
    Example:
    >>> build_1_2_3_5_8_buckets(100)
    [1, 2, 3, 5, 8, 10, 20, 30, 50, 80, 100]
    """
    return build_buckets([1, 2, 3, 5, 8], max_value)


def local_interval_elapsed(now: float, last_log: float,
                           local_interval: float) -> bool:
    elapsed_time = now - last_log
    return elapsed_time > local_interval


def get_throughput(tracked_stats: List[int], now: float,
                   last_log: float) -> float:
    return float(np.sum(tracked_stats) / (now - last_log))


@dataclass
class SpecDecodeMetrics:
    """Metrics for speculative decoding."""
    draft_acceptance_rate: float
    system_efficiency: float
    num_spec_tokens: int
    accepted_tokens: int
    draft_tokens: int
    emitted_tokens: int


@dataclass
class Stats:
    """Stats collected from LLM engine."""
    now: float
    
    # System stats
    num_running_sys: int
    num_waiting_sys: int
    num_swapped_sys: int
    gpu_cache_usage_sys: float
    cpu_cache_usage_sys: float
    gpu_memory_usage_sys: float
    cpu_memory_usage_sys: float
    gpu_prefix_cache_hit_rate: float
    cpu_prefix_cache_hit_rate: float
    
    # LoRA stats
    running_lora_adapters: List[str]
    waiting_lora_adapters: List[str]
    max_lora: int
    
    # Iteration stats
    num_preemption_iter: int
    num_prompt_tokens_iter: int
    num_generation_tokens_iter: int
    num_tokens_iter: int
    time_to_first_tokens_iter: List[float]
    time_per_output_tokens_iter: List[float]
    
    # Request stats
    time_e2e_requests: List[float]
    time_queue_requests: List[float]
    time_inference_requests: List[float]
    time_prefill_requests: List[float]
    time_decode_requests: List[float]
    num_prompt_tokens_requests: List[int]
    num_generation_tokens_requests: List[int]
    n_requests: List[int]
    max_num_generation_tokens_requests: List[int]
    max_tokens_requests: List[int]
    finished_reason_requests: List[str]
    
    # Speculative decoding stats
    spec_decode_metrics: Optional[SpecDecodeMetrics] = None
    
    # Tool calling stats
    tool_calls_iter: List[Dict] = None
    tool_call_errors_iter: List[str] = None


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