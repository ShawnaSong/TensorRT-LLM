# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the TensorRT-LLM project
from __future__ import annotations

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.structured_output.backend_guidance import GuidanceBackend
from tensorrt_llm.serve.structured_output.backend_types import (StructuredOutputBackend,
                                                     StructuredOutputGrammar)
from tensorrt_llm.serve.structured_output.backend_xgrammar import XgrammarBackend
from tensorrt_llm.serve.structured_output.request import StructuredOutputRequest

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch
    from tensorrt_llm.serve.structured_output.backend_types import StructuredOutputKey
else:
    torch = None

logger = logger


class StructuredOutputManager:
    """Engine-level manager for structured output requests."""

    def __init__(self, llm_config=None):
        self.backend: Optional[StructuredOutputBackend] = None
        self.llm_config = llm_config
        self.tokenizer = None

        self._grammar_bitmask: Optional[torch.Tensor] = None
        if torch is not None:
            self._full_mask = torch.tensor(-1, dtype=torch.int32)
        else:
            self._full_mask = None

        # The default max_workers if not specified is the number of CPUs * 5,
        # which is way too high since these tasks are CPU-bound, not I/O bound.
        # We also know we would never dominate CPU usage with just grammar
        # compilation, so we set it to half the number of CPUs.
        max_workers = max(1, (multiprocessing.cpu_count() + 1) // 2)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def set_tokenizer(self, tokenizer):
        """Set the tokenizer for the manager."""
        self.tokenizer = tokenizer

    def _get_vocab_size(self) -> int:
        """Get vocabulary size from tokenizer or config."""
        if self.tokenizer is not None:
            # Try to get vocab size from the underlying tokenizer
            if hasattr(self.tokenizer, 'tokenizer'):
                underlying_tokenizer = self.tokenizer.tokenizer
                if hasattr(underlying_tokenizer, 'get_vocab'):
                    return len(underlying_tokenizer.get_vocab())
        
        # Fallback to config if available
        if self.llm_config and hasattr(self.llm_config, 'vocab_size'):
            return self.llm_config.vocab_size
        
        # Default fallback
        return 32000

    def grammar_init(self, request, server=None) -> None:
        """Initialize grammar for a request."""
        # Get structured output info from server if available
        if server is not None and hasattr(server, '_get_structured_output_info'):
            info = server._get_structured_output_info(request)
            structured_output_request = info.get('structured_output_request')
            use_structured_output = info.get('use_structured_output', False)
        else:
            # Fallback to old method
            structured_output_request = getattr(request, 'structured_output_request', None)
            use_structured_output = getattr(request, 'use_structured_output', False)
        
        if not use_structured_output or structured_output_request is None:
            return

        # Initialize the backend the first time it is needed.
        if self.backend is None:
            if not hasattr(structured_output_request, 'sampling_params') or structured_output_request.sampling_params is None:
                return
                
            guided_decoding = structured_output_request.sampling_params.guided_decoding
            if guided_decoding is None:
                return
                
            backend_type = getattr(guided_decoding, 'backend', 'xgrammar')
            vocab_size = self._get_vocab_size()
            
            if backend_type == "xgrammar":
                self.backend = XgrammarBackend(
                    self.llm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend_type == "guidance":
                self.backend = GuidanceBackend(
                    self.llm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            else:
                raise ValueError(
                    f"Unsupported structured output backend: {backend_type}")

        grammar = self.executor.submit(self._async_create_grammar, structured_output_request)
        structured_output_request.grammar = grammar

    def _async_create_grammar(
        self,
        structured_output_request: StructuredOutputRequest,
    ) -> StructuredOutputGrammar:
        """Asynchronously create grammar for a request."""
        if structured_output_request is None:
            raise ValueError("Structured output request is None")
            
        key = structured_output_request.structured_output_key

        # Note that the request was validated in the engine core client,
        # so at this point we know it is a supported type of request.
        request_type, grammar_spec = key

        assert self.backend is not None
        return self.backend.compile_grammar(request_type, grammar_spec)

    def grammar_bitmask(
        self,
        requests: dict[str, any],
        structured_output_request_ids: dict[str, int],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ) -> Optional[npt.NDArray[np.int32]]:
        """Generate grammar bitmask for batch processing."""
        # Prepare the structured output bitmask for this batch.
        if not structured_output_request_ids:
            return None

        max_num_spec_tokens = 0
        if self.llm_config and hasattr(self.llm_config, 'speculative_config'):
            max_num_spec_tokens = getattr(self.llm_config.speculative_config, 'num_speculative_tokens', 0)

        if self._grammar_bitmask is None:
            assert self.backend is not None
            max_batch_size = getattr(self.llm_config, 'max_num_seqs', 256) if self.llm_config else 256

            # Allocate a bitmask for each token needing to be checked:
            # one for each speculative position, and one more for the
            # bonus token / non-speculative token.
            self._grammar_bitmask = \
                self.backend.allocate_token_bitmask(
                    max_batch_size * (1 + max_num_spec_tokens))

        bitmask_tensor = self._grammar_bitmask
        # Generate a batched bitmask for all structured output requests.
        # When speculative decoding is enabled, we need to include multiple
        # masks for each request, one for each possible bonus token position.
        # These are stored inline in the tensor and unpacked by the gpu runner.
        cumulative_index = 0
        ordered_seq = sorted(structured_output_request_ids.items(),
                             key=lambda x: x[1])

        # Note that for thinking support, we will need to
        # reset the relevant part of the bitmask for consequent
        # request here.
        bitmask_tensor[:(len(ordered_seq) * (1 + max_num_spec_tokens))].fill_(
            self._full_mask)

        # NOTE: This outer loop can likely be parallelized to improve
        # performance of bitmask generation for large batches.
        for req_id, _ in ordered_seq:
            request = requests[req_id]
            structured_output_request = request.structured_output_request

            if structured_output_request is None or structured_output_request.grammar is None:
                continue
                
            apply_bitmask: bool = True
            # Note: Reasoning support is not implemented in TensorRT-LLM yet
            # if self.reasoner is not None:
            #     if structured_output_request.reasoning_ended is None:
            #         structured_output_request.reasoning_ended = \
            #             self.reasoner.is_reasoning_end(request.prompt_token_ids)
            #     apply_bitmask = structured_output_request.reasoning_ended

            state_advancements = 0
            req_tokens = scheduled_spec_decode_tokens.get(req_id, []) + [None]
            for i, token in enumerate(req_tokens):
                if apply_bitmask and not \
                    structured_output_request.grammar.is_terminated():
                    structured_output_request.grammar.fill_bitmask(
                        bitmask_tensor, cumulative_index)
                    if token is not None:
                        # In order to generate the correct bitmask for each
                        # position in the speculative sequence, we advance
                        # the FSM state for each speculative token and rollback
                        # to restore the previous state when we are finished.
                        assert structured_output_request.grammar.accept_tokens(
                            req_id, [token])
                        state_advancements += 1
                cumulative_index += 1
            if state_advancements > 0:
                structured_output_request.grammar.rollback(state_advancements)

        if cumulative_index < bitmask_tensor.shape[0]:
            bitmask_tensor = bitmask_tensor[:cumulative_index]

        # After finishing with the xgrammar operations, we convert to
        # np.ndarray, because that is much more efficient for serialization
        # and deserialization when sending this to the GPU workers.
        return bitmask_tensor.numpy()

    def should_advance(self, request) -> bool:
        """Determine if the FSM should advance."""
        if not hasattr(request, 'use_structured_output') or not request.use_structured_output:
            return False

        # To determine whether we can advance the FSM.
        # Supports thinking usage where we skip the reasoning components.
        if not hasattr(request, 'structured_output_request') or request.structured_output_request is None:
            return True
            
        # by default, we should always advance
        # for cases that doesn't uses thinking mode.
        # Note: Reasoning support is not implemented in TensorRT-LLM yet
        # if self.reasoner is not None:
        #     structured_req = request.structured_output_request
        #     if structured_req.reasoning_ended:
        #         return True
        #     # Check if reasoning ends in *this* step
        #     if self.reasoner.is_reasoning_end(request.all_token_ids):
        #         # Reasoning just ended, so we shouldn't advanced til
        #         # next pass
        #         structured_req.reasoning_ended = True
        #     return False
        # else:
        return True

    def clear_backend(self) -> None:
        """Clean up backend resources."""
        if self.backend is not None:
            self.backend.destroy()


__all__ = [
    'StructuredOutputManager',
    'StructuredOutputRequest',
    'StructuredOutputBackend',
    'StructuredOutputGrammar',
    'XgrammarBackend',
    'GuidanceBackend'
]
