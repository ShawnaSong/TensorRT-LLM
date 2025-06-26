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
