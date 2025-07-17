# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: TensorRT-LLM Tool Parsers

import json
import re
import logging
from typing import Any, Dict, List, Optional

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager
)
from ..openai_protocol import ChatCompletionRequest, FunctionCall, ToolCall

logger = logging.getLogger(__name__)


@ToolParserManager.register_module(["kimi", "kimik2", "kimi-k2"])
class KimiK2ToolParser(ToolParser):
    """
    Tool parser for Kimi K2 format.
    
    Kimi K2 uses a specific format with special tokens:
    <|tool_calls_section_begin|><|tool_call_begin|>functions.function_name:index<|tool_call_argument_begin|>{"param": "value"}<|tool_call_end|><|tool_calls_section_end|>
    """
    
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        
        # KimiK2 special tokens from tokenizer config
        self.tool_calls_start_token = "<|tool_calls_section_begin|>"
        self.tool_calls_end_token = "<|tool_calls_section_end|>"
        self.tool_call_start_token = "<|tool_call_begin|>"
        self.tool_call_end_token = "<|tool_call_end|>"
        self.tool_argument_start_token = "<|tool_call_argument_begin|>"
        
        # Get token IDs from vocabulary using the correct method
        vocab = self.tokenizer.get_vocab() if hasattr(self.tokenizer, 'get_vocab') else {}
        self.tool_calls_start_token_id = vocab.get(self.tool_calls_start_token)
        self.tool_calls_end_token_id = vocab.get(self.tool_calls_end_token)
        self.tool_call_start_token_id = vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = vocab.get(self.tool_call_end_token)
        
        if (self.tool_calls_start_token_id is None or self.tool_calls_end_token_id is None):
            logger.warning(
                "Kimi-K2 Tool parser could not locate tool call start/end "
                "tokens in the tokenizer! This may be normal if the model "
                "doesn't use these specific tokens.")
        
        # Updated regex patterns to match vLLM implementation
        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*?)\s*<\|tool_call_end\|>",
            re.DOTALL
        )
        
        # Pattern for incomplete tool calls (streaming)
        self.partial_tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)(?:\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*?))?",
            re.DOTALL
        )
        
        # Streaming state tracking (matching vLLM approach)
        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.streamed_args_for_tool = []
        self.streaming_buffer = ""
    
    def can_parse(self, text: str) -> bool:
        """Check if text contains KimiK2 tool call format."""
        return self.tool_calls_start_token in text or self.tool_call_start_token in text
    
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from KimiK2 format output."""
        
        # Sanity check; avoid unnecessary processing
        if self.tool_calls_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )
        
        try:
            # Find all complete tool calls
            function_call_tuples = self.tool_call_regex.findall(model_output)
            
            logger.debug("function_call_tuples: %s", function_call_tuples)
            
            tool_calls = []
            for match in function_call_tuples:
                function_id, function_args = match
                # function_id: functions.get_weather:0
                function_name = function_id.split('.')[1].split(':')[0]
                
                try:
                    # Parse JSON arguments
                    if function_args.strip():
                        arguments = json.loads(function_args.strip())
                    else:
                        arguments = {}
                    
                    # Create ToolCall with proper ID
                    tool_call = ToolCall(
                        id=function_id,
                        type="function",
                        function=FunctionCall(
                            name=function_name,
                            arguments=json.dumps(arguments, ensure_ascii=False)
                        )
                    )
                    tool_calls.append(tool_call)
                    
                except json.JSONDecodeError as json_err:
                    logger.warning(f"Failed to parse JSON arguments for function {function_name}: {json_err}")
                    # If JSON parsing fails, treat as string
                    tool_call = ToolCall(
                        id=function_id,
                        type="function",
                        function=FunctionCall(
                            name=function_name,
                            arguments=json.dumps({"raw_args": function_args.strip()}, ensure_ascii=False)
                        )
                    )
                    tool_calls.append(tool_call)
            
            # Extract content before tool calls
            content = model_output[:model_output.find(self.tool_calls_start_token)]
            
            return ExtractedToolCallInformation(
                tools_called=len(tool_calls) > 0,
                tool_calls=tool_calls,
                content=content if content else None
            )
            
        except Exception as e:
            logger.error(f"Error parsing KimiK2 tool calls: {e}", exc_info=True)
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )
    
    def _clean_content(self, content: str) -> str:
        """Clean KimiK2 special tokens from content."""
        # Remove any KimiK2 special tokens that might appear in content
        tokens_to_remove = [
            self.tool_calls_start_token,
            self.tool_calls_end_token,
            self.tool_call_start_token,
            self.tool_call_end_token,
            self.tool_argument_start_token
        ]
        
        for token in tokens_to_remove:
            content = content.replace(token, "")
        
        # Clean up extra whitespace
        content = re.sub(r'\n\s*\n', '\n', content).strip()
        
        return content
    
    def format_tools_for_prompt(
        self,
        tools: List[Any]
    ) -> str:
        """Format tools for KimiK2 prompt template."""
        if not tools:
            return ""
        
        formatted_tools = super().format_tools_for_prompt(tools)
        
        tools_text = "You have access to the following functions. "
        tools_text += "Use them if required:\n\n"
        
        for tool in formatted_tools:
            tools_text += f"Function: {tool['name']}\n"
            tools_text += f"Description: {tool['description']}\n"
            if tool.get('parameters'):
                tools_text += f"Parameters: {json.dumps(tool['parameters'], indent=2)}\n"
            tools_text += "\n"
        
        tools_text += ("To call a function, use this format:\n"
                      f"{self.tool_calls_start_token}\n"
                      f"{self.tool_call_start_token}functions.function_name:0{self.tool_argument_start_token}\n"
                      '{"param1": "value1", "param2": "value2"}\n'
                      f"{self.tool_call_end_token}\n"
                      f"{self.tool_calls_end_token}\n\n")
        
        return tools_text
    
    def extract_tool_calls_streaming(
        self,
        delta_text: str,
        full_text: str,
        request: ChatCompletionRequest
    ) -> Optional[Dict[str, Any]]:
        """Extract tool calls from streaming KimiK2 output."""
        
        logger.debug("delta_text: %s", delta_text)
        
        # Update streaming buffer
        self.streaming_buffer += delta_text
        
        # Check if we have any complete tool calls
        if self.tool_calls_end_token in self.streaming_buffer:
            # We have complete tool calls, parse them
            result = self.extract_tool_calls(self.streaming_buffer, request)
            if result.tools_called:
                return {
                    "type": "tool_calls",
                    "tool_calls": result.tool_calls,
                    "content": result.content
                }
        
        # Check for partial tool calls
        partial_matches = self.partial_tool_call_regex.findall(self.streaming_buffer)
        if partial_matches:
            # We have a partial tool call in progress
            return {"type": "partial_tool_call", "content": delta_text}
        
        # Regular content
        return {"type": "content", "content": delta_text}
    
    def reset_state(self):
        """Reset parser state for new conversations."""
        self.streaming_buffer = ""
        self.current_tool_calls = []
        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.streamed_args_for_tool = []