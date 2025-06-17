# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: TensorRT-LLM Tool Call Manager

import json
from typing import Any, Dict, List, Optional, Union

from .tool_parsers import ToolParserManager
from .openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionCall,
    ToolCall
)


class ToolCallManager:
    """
    Manager for handling tool calling functionality in TensorRT-LLM.
    
    This class integrates with different tool parsers to extract and process
    tool calls from model outputs.
    """
    
    def __init__(self, tokenizer, tool_parser_name: str = None):
        """
        Initialize the tool call manager.
        
        Args:
            tokenizer: The tokenizer used by the model
            tool_parser_name: Name of the tool parser to use (auto, hermes, llama, qwen, mistral)
                             If None, tool calling functionality is disabled
        """
        self.tokenizer = tokenizer
        self.tool_parser_name = tool_parser_name
        self.tool_parser = None
        
        # If tool_parser_name is None, disable tool calling
        if tool_parser_name is None:
            self.enabled = False
            return
        
        self.enabled = True
        
        # Auto-detect parser if not specified
        if tool_parser_name != "auto":
            self.tool_parser = ToolParserManager.get_parser(tool_parser_name, tokenizer)
    
    def _auto_detect_parser(self, text: str) -> str:
        """Auto-detect the best parser for the given text format."""
        
        # Try parsers in order of specificity
        parser_names = ["deepseek", "hermes", "mistral", "qwen", "llama"]
        
        for parser_name in parser_names:
            try:
                parser = ToolParserManager.get_parser(parser_name, self.tokenizer)
                if parser.can_parse(text):
                    return parser_name
            except Exception:
                continue
        
        # Default to hermes if nothing else works
        return "hermes"
    
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest
    ) -> Dict[str, Any]:
        """
        Extract tool calls from model output.
        
        Args:
            model_output: The complete text output from the model
            request: The original chat completion request
            
        Returns:
            Dictionary containing:
            - tools_called: bool
            - tool_calls: List[ToolCall]
            - content: Optional[str]
        """
        
        # If tool calling is disabled, return no tool calls
        if not self.enabled:
            return {
                "tools_called": False,
                "tool_calls": [],
                "content": model_output,
                "parser_used": None
            }
        
        # Auto-detect parser if needed
        if self.tool_parser is None:
            parser_name = self._auto_detect_parser(model_output)
            self.tool_parser = ToolParserManager.get_parser(parser_name, self.tokenizer)
            self.tool_parser_name = parser_name
        
        # Extract tool calls using the selected parser
        result = self.tool_parser.extract_tool_calls(model_output, request)
        
        return {
            "tools_called": result.tools_called,
            "tool_calls": result.tool_calls,
            "content": result.content,
            "parser_used": self.tool_parser_name
        }
    
    def extract_tool_calls_streaming(
        self,
        delta_text: str,
        full_text: str,
        request: ChatCompletionRequest
    ) -> Optional[Dict[str, Any]]:
        """
        Extract tool calls from streaming model output.
        
        Args:
            delta_text: New text chunk
            full_text: Complete text so far
            request: The original chat completion request
            
        Returns:
            Dictionary with streaming information or None
        """
        
        # Auto-detect parser if needed
        if self.tool_parser is None:
            if len(full_text.strip()) > 10:  # Only auto-detect with enough content
                parser_name = self._auto_detect_parser(full_text)
                self.tool_parser = ToolParserManager.get_parser(parser_name, self.tokenizer)
                self.tool_parser_name = parser_name
            else:
                return {"type": "content", "content": delta_text}
        
        # Use streaming extraction
        return self.tool_parser.extract_tool_calls_streaming(delta_text, full_text, request)
    
    def format_tools_for_chat_template(
        self,
        tools: List[ChatCompletionToolsParam],
        parser_name: Optional[str] = None
    ) -> str:
        """
        Format tools for inclusion in chat templates.
        
        Args:
            tools: List of available tools
            parser_name: Specific parser to use for formatting
            
        Returns:
            Formatted tools string for the chat template
        """
        
        if not tools:
            return ""
        
        # Use specified parser or current one
        if parser_name:
            parser = ToolParserManager.get_parser(parser_name, self.tokenizer)
        elif self.tool_parser:
            parser = self.tool_parser
        else:
            # Default to hermes format
            parser = ToolParserManager.get_parser("hermes", self.tokenizer)
        
        return parser.format_tools_for_prompt(tools)
    
    def validate_tool_calls(
        self,
        tool_calls: List[ToolCall],
        available_tools: List[ChatCompletionToolsParam]
    ) -> List[Dict[str, Any]]:
        """
        Validate extracted tool calls against available tools.
        
        Args:
            tool_calls: Extracted tool calls
            available_tools: Available tools from the request
            
        Returns:
            List of validation results
        """
        
        # Create a lookup of available tools
        available_tool_names = {
            tool.function.name for tool in available_tools if tool.function.name
        }
        
        validation_results = []
        
        for tool_call in tool_calls:
            result = {
                "tool_call": tool_call,
                "valid": True,
                "errors": []
            }
            
            # Check if tool exists
            if tool_call.function.name not in available_tool_names:
                result["valid"] = False
                result["errors"].append(f"Unknown function: {tool_call.function.name}")
            
            # Validate arguments are valid JSON
            try:
                json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                result["valid"] = False
                result["errors"].append(f"Invalid JSON arguments: {e}")
            
            validation_results.append(result)
        
        return validation_results
    
    def reset_state(self):
        """Reset parser state (useful for new conversations)."""
        if self.tool_parser:
            self.tool_parser.reset_state()
    
    def get_supported_parsers(self) -> List[str]:
        """Get list of supported tool parsers."""
        return list(ToolParserManager._parsers.keys())
    
    def set_parser(self, parser_name: str):
        """
        Manually set the tool parser to use.
        
        Args:
            parser_name: Name of the parser to use
        """
        self.tool_parser = ToolParserManager.get_parser(parser_name, self.tokenizer)
        self.tool_parser_name = parser_name 