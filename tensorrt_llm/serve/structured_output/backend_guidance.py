# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the TensorRT-LLM project

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.structured_output.backend_types import (StructuredOutputBackend,
                                                     StructuredOutputGrammar,
                                                     StructuredOutputOptions)
from tensorrt_llm.serve.structured_output.request import get_structured_output_key

if TYPE_CHECKING:
    import llguidance
    import llguidance.hf as llguidance_hf
    import llguidance.torch as llguidance_torch
else:
    try:
        import llguidance
        import llguidance.hf as llguidance_hf
        import llguidance.torch as llguidance_torch
    except ImportError:
        llguidance = None
        llguidance_hf = None
        llguidance_torch = None
        logger.warning("llguidance not available. Structured output with guidance backend will not work.")

logger = logger


def _walk_json_for_additional_properties(data: object):
    if isinstance(data, dict):
        for value in data.values():
            _walk_json_for_additional_properties(value)
        if 'additionalProperties' not in data and \
            ('properties' in data or 'patternProperties' in data):
            data['additionalProperties'] = False
    elif isinstance(data, list):
        for item in data:
            _walk_json_for_additional_properties(item)


def process_for_additional_properties(
        guide_json: Union[str, dict[str, Any]]) -> dict[str, Any]:
    if isinstance(guide_json, str):
        guide_json_obj = json.loads(guide_json)
    else:
        # copy for modifications
        guide_json_obj = copy.deepcopy(guide_json)
    _walk_json_for_additional_properties(guide_json_obj)
    return guide_json_obj


@dataclass
class GuidanceBackend(StructuredOutputBackend):

    def __post_init__(self):
        if llguidance is None:
            raise ImportError("llguidance is required for GuidanceBackend")
            
        self.disable_any_whitespace = getattr(self.llm_config, 'disable_any_whitespace', False)
        self.disable_additional_properties = getattr(self.llm_config, 'disable_additional_properties', False)

        self.ll_tokenizer = llguidance_hf.from_tokenizer(
            self.tokenizer, self.vocab_size)

    def compile_grammar(self, request_type: StructuredOutputOptions,
                        grammar_spec: str) -> StructuredOutputGrammar:
        self.serialized_grammar = serialize_guidance_grammar(
            request_type, grammar_spec, self.disable_any_whitespace,
            self.disable_additional_properties)

        ll_matcher = llguidance.LLMatcher(
            self.ll_tokenizer,
            self.serialized_grammar,
            log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")),
        )

        r = GuidanceGrammar(
            ll_matcher=ll_matcher,
            ll_tokenizer=self.ll_tokenizer,
            vocab_size=self.vocab_size,
        )

        r.check_error()
        return r

    def allocate_token_bitmask(self, max_num_seqs: int):
        return llguidance_torch.allocate_token_bitmask(
            max_num_seqs, self.ll_tokenizer.vocab_size)

    def destroy(self):
        pass


@dataclass
class GuidanceGrammar(StructuredOutputGrammar):
    ll_matcher: llguidance.LLMatcher
    ll_tokenizer: llguidance.LLTokenizer
    vocab_size: int
    printed_error: bool = False
    terminated: bool = False

    def check_error(self):
        if not self.printed_error:
            err = self.ll_matcher.get_error()
            if err:
                self.printed_error = True
                logger.warning("LLMatcher error: %s", err)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the parser.

        Returns True if the parser was advanced successfully.
        Returns False if the parser failed to advance.
        """

        if self.ll_tokenizer.eos_token in tokens:
            self.terminated = True

        if self.ll_matcher.is_stopped():
            return True

        # TODO - Add jump decoding support in the future:
        # self.ll_matcher.compute_ff_bytes() - this should always work
        # self.ll_matcher.compute_ff_tokens() - this only works for
        #   "canonical" tokenizers
        # For conversion between the two, see
        # https://github.com/guidance-ai/llguidance/blob/main/docs/fast_forward.md

        r = self.ll_matcher.consume_tokens(tokens)

        self.check_error()

        return r

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the parser in sequence.
        Will not advance the parser.

        Returns the prefix list of tokens that are accepted by the parser.
        """
        if len(tokens) == 0:
            return []
        if self.ll_matcher.is_stopped():
            return []

        num_tokens = self.ll_matcher.validate_tokens(tokens)

        self.check_error()

        return tokens[:num_tokens]

    def rollback(self, num_tokens: int) -> None:
        self.ll_matcher.rollback(num_tokens)
        self.check_error()

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        # this will automatically return [EOS] mask if the matcher is stopped
        # or otherwise in an error state
        llguidance_torch.fill_next_token_bitmask(self.ll_matcher, bitmask, idx)
        self.check_error()

    def is_terminated(self) -> bool:
        return self.terminated

    def reset(self):
        # This method may be not needed anymore? TODO
        self.ll_matcher.reset()


def serialize_guidance_grammar(
    request_type: StructuredOutputOptions,
    grammar_spec: Union[str, dict[str, Any]],
    disable_any_whitespace: bool = False,
    disable_additional_properties: bool = False,
) -> str:

    def _process_schema(grammar_spec: Union[str, dict[str, Any]], ) -> str:
        if disable_additional_properties:
            grammar_spec = process_for_additional_properties(grammar_spec)
        return llguidance.LLMatcher.grammar_from_json_schema(
            grammar_spec,
            defaults={
                "whitespace_flexible": not disable_any_whitespace,
            })

    if request_type == StructuredOutputOptions.JSON:
        return _process_schema(grammar_spec)
    elif request_type == StructuredOutputOptions.JSON_OBJECT:
        return llguidance.LLMatcher.grammar_from_json_schema(
            '{"type": "object"}',
            defaults={
                "whitespace_flexible": not disable_any_whitespace,
            })
    else:
        if request_type == StructuredOutputOptions.REGEX:
            return llguidance.LLMatcher.grammar_from_regex(grammar_spec)
        elif request_type == StructuredOutputOptions.GRAMMAR:
            return llguidance.LLMatcher.grammar_from_ebnf(grammar_spec)
        elif request_type == StructuredOutputOptions.CHOICE:
            if isinstance(grammar_spec, str):
                choice_list = json.loads(grammar_spec)
            else:
                choice_list = grammar_spec
            return llguidance.LLMatcher.grammar_from_choice(choice_list)
        elif request_type == StructuredOutputOptions.STRUCTURAL_TAG:
            if isinstance(grammar_spec, str):
                s_tag = json.loads(grammar_spec)
            else:
                s_tag = grammar_spec
            tags = [
                llguidance.StructuralTagItem(
                    begin=s["begin"],
                    schema=json.dumps(s["schema"]),
                    end=s["end"],
                ) for s in s_tag["structures"]
            ]
            return llguidance.LLMatcher.grammar_from_structural_tag(tags, s_tag["triggers"])
        else:
            raise ValueError(f"Unsupported request type: {request_type}")


def validate_guidance_grammar(
        sampling_params: SamplingParams,
        tokenizer: Optional[llguidance.LLTokenizer] = None) -> None:
    """Validate guidance grammar before compilation."""
    if sampling_params.guided_decoding is None:
        return

    guided_decoding = sampling_params.guided_decoding

    if guided_decoding.json is not None:
        if isinstance(guided_decoding.json, str):
            try:
                schema = json.loads(guided_decoding.json)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON schema: {e}")
        else:
            schema = guided_decoding.json

        # Validate JSON schema with llguidance
        try:
            llguidance.LLMatcher.grammar_from_json_schema(schema)
        except Exception as e:
            raise ValueError(f"Invalid JSON schema for guidance: {e}")

    elif guided_decoding.regex is not None:
        try:
            llguidance.LLMatcher.grammar_from_regex(guided_decoding.regex)
        except Exception as e:
            raise ValueError(f"Invalid regex for guidance: {e}")

    elif guided_decoding.grammar is not None:
        try:
            llguidance.LLMatcher.grammar_from_ebnf(guided_decoding.grammar)
        except Exception as e:
            raise ValueError(f"Invalid grammar for guidance: {e}")

    elif guided_decoding.choice is not None:
        if not isinstance(guided_decoding.choice, (list, str)):
            raise ValueError("Choice must be a list of strings or a JSON string")
        try:
            if isinstance(guided_decoding.choice, str):
                choice_list = json.loads(guided_decoding.choice)
            else:
                choice_list = guided_decoding.choice
            llguidance.LLMatcher.grammar_from_choice(choice_list)
        except Exception as e:
            raise ValueError(f"Invalid choice for guidance: {e}")

    elif guided_decoding.structural_tag is not None:
        try:
            if isinstance(guided_decoding.structural_tag, str):
                s_tag = json.loads(guided_decoding.structural_tag)
            else:
                s_tag = guided_decoding.structural_tag
            tags = [
                llguidance.StructuralTagItem(
                    begin=s["begin"],
                    schema=json.dumps(s["schema"]),
                    end=s["end"],
                ) for s in s_tag["structures"]
            ]
            llguidance.LLMatcher.grammar_from_structural_tag(tags, s_tag["triggers"])
        except Exception as e:
            raise ValueError(f"Invalid structural tag for guidance: {e}")
