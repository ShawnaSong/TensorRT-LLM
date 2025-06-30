# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the TensorRT-LLM project

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.structured_output.backend_types import (StructuredOutputBackend,
                                                     StructuredOutputGrammar,
                                                     StructuredOutputOptions)
from tensorrt_llm.serve.structured_output.utils import (choice_as_grammar,
                                             convert_lark_to_ebnf,
                                             grammar_is_likely_lark)

if TYPE_CHECKING:
    import xgrammar as xgr
else:
    try:
        import xgrammar as xgr
    except ImportError:
        xgr = None
        logger.warning("xgrammar not available. Structured output with xgrammar backend will not work.")

logger = logger


@dataclass
class XgrammarBackend(StructuredOutputBackend):

    def __post_init__(self):
        if xgr is None:
            raise ImportError("xgrammar is required for XgrammarBackend")
            
        self.disable_any_whitespace = getattr(self.llm_config, 'disable_any_whitespace', False)

        # For TensorRT-LLM, we'll use a simpler tokenizer approach
        try:
            # Try to get vocabulary from tokenizer
            if hasattr(self.tokenizer, 'tokenizer') and hasattr(self.tokenizer.tokenizer, 'get_vocab'):
                # Use the underlying tokenizer
                encoded_vocab = [
                    token for token, _ in sorted(
                        self.tokenizer.tokenizer.get_vocab().items(),
                        key=lambda x: x[1],
                    )
                ]
            elif hasattr(self.tokenizer, 'get_vocab'):
                # Direct access if available
                encoded_vocab = [
                    token for token, _ in sorted(
                        self.tokenizer.get_vocab().items(),
                        key=lambda x: x[1],
                    )
                ]
            else:
                # Fallback for tokenizers without get_vocab method
                encoded_vocab = [f"token_{i}" for i in range(self.vocab_size)]
                
            stop_token_ids = None
            if (hasattr(self.tokenizer, "eos_token_id") and 
                self.tokenizer.eos_token_id is not None):
                stop_token_ids = [self.tokenizer.eos_token_id]
                
        except AttributeError as e:
            raise ValueError(
                f"Cannot get the vocabulary of the tokenizer "
                f"{type(self.tokenizer)}. The tokenizer should have a "
                "get_vocab method.") from e
                
        tokenizer_info = xgr.TokenizerInfo(
            encoded_vocab=encoded_vocab,
            vocab_type=xgr.VocabType.BYTE_FALLBACK,
            vocab_size=self.vocab_size,
            stop_token_ids=stop_token_ids,
            add_prefix_space=True,
        )
        
        # Get cache size from environment or use default
        cache_mb = int(os.environ.get('TLLM_XGRAMMAR_CACHE_MB', '1024'))
        
        self.compiler = xgr.GrammarCompiler(
            tokenizer_info,
            max_threads=8,
            cache_enabled=True,
        )

        self.num_speculative_tokens = 0
        if self.llm_config and hasattr(self.llm_config, 'speculative_config'):
            self.num_speculative_tokens = getattr(
                self.llm_config.speculative_config, 'num_speculative_tokens', 0)

    def compile_grammar(self, request_type: StructuredOutputOptions,
                        grammar_spec: str) -> StructuredOutputGrammar:
        if request_type == StructuredOutputOptions.JSON:
            ctx = self.compiler.compile_json_schema(
                grammar_spec, any_whitespace=not self.disable_any_whitespace)
        elif request_type == StructuredOutputOptions.JSON_OBJECT:
            ctx = self.compiler.compile_json_schema(
                '{"type": "object"}',
                any_whitespace=not self.disable_any_whitespace)
        elif request_type == StructuredOutputOptions.GRAMMAR:
            ctx = self.compiler.compile_grammar(grammar_spec)
        elif request_type == StructuredOutputOptions.REGEX:
            ctx = self.compiler.compile_regex(grammar_spec)
        elif request_type == StructuredOutputOptions.STRUCTURAL_TAG:
            s_tag = json.loads(grammar_spec)
            tags = [
                xgr.StructuralTagItem(
                    begin=s["begin"],
                    schema=json.dumps(s["schema"]),
                    end=s["end"],
                ) for s in s_tag["structures"]
            ]
            ctx = self.compiler.compile_structural_tag(tags, s_tag["triggers"])
        else:
            logger.error(
                "Validation should have already occurred. Please file an issue."
            )
            raise ValueError(
                f"grammar is not of valid supported types. ({request_type!s})")

        return XgrammarGrammar(
            matcher=xgr.GrammarMatcher(
                ctx,
                max_rollback_tokens=self.num_speculative_tokens,
            ),
            vocab_size=self.vocab_size,
            ctx=ctx,
        )

    def allocate_token_bitmask(self, max_num_seqs: int):
        return xgr.allocate_token_bitmask(max_num_seqs, self.vocab_size)

    def destroy(self):
        del self.compiler


@dataclass
class XgrammarGrammar(StructuredOutputGrammar):
    # NOTE: This would be a generic-enough class for
    # supporting different backends, in the future.
    # For now, just xgrammar.
    #
    # https://xgrammar.mlc.ai/docs/api/python/index.html#xgrammar.GrammarMatcher.find_jump_forward_string
    # for jump-forward decoding

    vocab_size: int
    matcher: xgr.GrammarMatcher = field(hash=False)
    ctx: xgr.CompiledGrammar = field(hash=False)
    num_processed_tokens: int = field(default_factory=lambda: 0,
                                      repr=False,
                                      hash=False,
                                      init=False)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.", request_id, token)
                return False
            self.num_processed_tokens += 1
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the FSM in sequence.
        Will not advance the FSM.

        Returns the prefix list of tokens that are accepted by the FSM.
        """
        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
            else:
                break
        if len(accepted_tokens) > 0:
            # Rollback the FSM to the initial state
            self.matcher.rollback(len(accepted_tokens))
        return accepted_tokens

    def rollback(self, num_tokens: int) -> None:
        self.matcher.rollback(num_tokens)
        self.num_processed_tokens -= num_tokens

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        self.matcher.fill_next_token_bitmask(bitmask, idx)

    def is_terminated(self) -> bool:
        return self.matcher.is_terminated()

    def reset(self):
        self.num_processed_tokens = 0
        self.matcher.reset()


def has_xgrammar_unsupported_json_features(schema: dict[str, Any]) -> bool:
    """Check if JSON schema contains features unsupported by xgrammar."""
    def check_object(obj: dict[str, Any]) -> bool:
        for key, value in obj.items():
            if key in ["$ref", "$schema", "$id", "$defs", "definitions"]:
                return True
            if key == "type" and value == "array":
                if "items" in obj:
                    if isinstance(obj["items"], dict):
                        if check_object(obj["items"]):
                            return True
                    elif isinstance(obj["items"], list):
                        for item in obj["items"]:
                            if isinstance(item, dict) and check_object(item):
                                return True
            elif key == "properties" and isinstance(value, dict):
                for prop_value in value.values():
                    if isinstance(prop_value, dict) and check_object(prop_value):
                        return True
            elif key == "allOf" and isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and check_object(item):
                        return True
            elif key == "anyOf" and isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and check_object(item):
                        return True
            elif key == "oneOf" and isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and check_object(item):
                        return True
            elif key == "not" and isinstance(value, dict):
                if check_object(value):
                    return True
        return False

    return check_object(schema)


def validate_xgrammar_grammar(sampling_params: SamplingParams) -> None:
    """Validate xgrammar grammar before compilation."""
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

        if has_xgrammar_unsupported_json_features(schema):
            raise ValueError(
                "JSON schema contains features unsupported by xgrammar. "
                "Please check the xgrammar documentation for supported features."
            )

    elif guided_decoding.grammar is not None:
        grammar_str = guided_decoding.grammar
        if grammar_is_likely_lark(grammar_str):
            try:
                convert_lark_to_ebnf(grammar_str)
            except Exception as e:
                raise ValueError(f"Invalid Lark grammar: {e}")

    elif guided_decoding.choice is not None:
        if not isinstance(guided_decoding.choice, (list, str)):
            raise ValueError("Choice must be a list of strings or a JSON string")
