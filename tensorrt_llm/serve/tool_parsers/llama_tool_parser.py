# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: TensorRT-LLM Tool Parsers

import ast
import json
import re
from typing import Any, Dict, List, Optional

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager
)
from ..openai_protocol import ChatCompletionRequest


@ToolParserManager.register_module(["llama", "llama3", "llama32", "llama4"])
class LlamaToolParser(ToolParser):
    """
    Tool parser for Llama format.
    
    Supports both JSON format and pythonic format:
    
    JSON format:
    {"name": "function_name", "parameters": {"param1": "value1"}}
    
    Pythonic format:
    [function_name(param1="value1", param2="value2")]
    """
    
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        
        # Tool call patterns 
        self.bot_token = "<|python_tag|>"
        self.eot_token = "<|eot_id|>"
        
        # Regex for pythonic format
        self.pythonic_regex = re.compile(r'\[(.*?)\]', re.DOTALL)
        
        # Support both formats
        self.format_type = "auto"  # auto, json, pythonic
    
    def can_parse(self, text: str) -> bool:
        """Check if text contains Llama-style tool calls."""
        # Check for pythonic format
        if re.search(r'\[[\w_]+\(', text):
            return True
        
        # Check for OpenAI JSON format
        if re.search(r'\{"type":\s*"function"', text):
            return True
        
        # Check for simple JSON format
        if text.strip().startswith('{') and '"name"' in text:
            return True
            
        # Check for bot token
        if self.bot_token in text:
            return True
            
        return False
    
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete Llama format output."""
        
        if not self.can_parse(model_output):
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )
        
        try:
            # Try pythonic format first
            if '[' in model_output and '(' in model_output:
                result = self._parse_pythonic_format(model_output)
                if result.tools_called:
                    return result
            
            # Try JSON format
            result = self._parse_json_format(model_output)
            if result.tools_called:
                return result
            
            # If neither format worked, return as content
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )
            
        except Exception as e:
            print(f"Error parsing Llama tool calls: {e}")
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )
    
    def _parse_pythonic_format(self, text: str) -> ExtractedToolCallInformation:
        """Parse pythonic format: [function_name(param1="value1")]"""
        
        # Find content that looks like a function call list
        matches = self.pythonic_regex.findall(text)
        
        for match in matches:
            try:
                # Try to parse as Python AST
                module = ast.parse(f"[{match}]")
                parsed = getattr(module.body[0], "value", None)
                
                if (isinstance(parsed, ast.List) and 
                    all(isinstance(e, ast.Call) for e in parsed.elts)):
                    
                    tool_calls = []
                    for call in parsed.elts:
                        if isinstance(call.func, ast.Name):
                            function_name = call.func.id
                            arguments = {}
                            
                            # Extract keyword arguments
                            for keyword in call.keywords:
                                arguments[keyword.arg] = self._get_ast_value(keyword.value)
                            
                            tool_call = self._create_tool_call(
                                name=function_name,
                                arguments=arguments
                            )
                            tool_calls.append(tool_call)
                    
                    if tool_calls:
                        # Extract content before tool calls
                        tool_call_start = text.find('[')
                        content = text[:tool_call_start].strip() if tool_call_start > 0 else None
                        
                        return ExtractedToolCallInformation(
                            tools_called=True,
                            tool_calls=tool_calls,
                            content=content
                        )
                        
            except (SyntaxError, ValueError):
                continue
        
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=text
        )
    
    def _parse_json_format(self, text: str) -> ExtractedToolCallInformation:
        """Parse JSON format: {"name": "function_name", "parameters": {...}} or OpenAI format"""
        
        # Remove bot token if present
        if self.bot_token in text:
            text = text.replace(self.bot_token, "").strip()
        
        # Clean up template tokens that might appear
        template_tokens = ["<|eom_id|>", "<|start_header_id|>", "<|end_header_id|>", "<|python_tag|>"]
        for token in template_tokens:
            text = text.replace(token, "")
        text = text.strip()
        
        # Extract JSON from mixed content - handle nested braces
        json_matches = []
        
        # Look for complete JSON objects with proper nesting
        brace_count = 0
        start_pos = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_pos = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_pos != -1:
                    potential_json = text[start_pos:i+1]
                    # Check if it looks like a tool call
                    if '"type"' in potential_json or '"name"' in potential_json:
                        json_matches.append(potential_json)
                    start_pos = -1
        
        # Use the first valid JSON match, or the whole text
        json_text = json_matches[0] if json_matches else text
        
        # Try to parse as single JSON object or array
        try:
            # First try parsing as single object
            if json_text.strip().startswith('{'):
                parsed = json.loads(json_text.strip())
                if isinstance(parsed, dict):
                    # Handle OpenAI format: {"type": "function", "name": "...", "parameters": {...}}
                    if parsed.get("type") == "function" and "name" in parsed:
                        tool_call = self._create_tool_call(
                            name=parsed["name"],
                            arguments=parsed.get("parameters", parsed.get("arguments", {}))
                        )
                        
                        # Clean content by removing the JSON tool call
                        clean_content = text
                        if json_matches:
                            clean_content = text.replace(json_matches[0], "").strip()
                        
                        return ExtractedToolCallInformation(
                            tools_called=True,
                            tool_calls=[tool_call],
                            content=clean_content if clean_content else None
                        )
                    
                    # Handle simple format: {"name": "function_name", "parameters": {...}}
                    elif "name" in parsed:
                        tool_call = self._create_tool_call(
                            name=parsed["name"],
                            arguments=parsed.get("parameters", parsed.get("arguments", {}))
                        )
                        
                        # Clean content by removing the JSON tool call
                        clean_content = text
                        if json_matches:
                            clean_content = text.replace(json_matches[0], "").strip()
                        
                        return ExtractedToolCallInformation(
                            tools_called=True,
                            tool_calls=[tool_call],
                            content=clean_content if clean_content else None
                        )
            
            # Try parsing as array
            elif json_text.strip().startswith('['):
                parsed = json.loads(json_text.strip())
                if isinstance(parsed, list):
                    tool_calls = []
                    for item in parsed:
                        if isinstance(item, dict):
                            # Handle both OpenAI format and simple format
                            if item.get("type") == "function" and "name" in item:
                                tool_call = self._create_tool_call(
                                    name=item["name"],
                                    arguments=item.get("parameters", item.get("arguments", {}))
                                )
                                tool_calls.append(tool_call)
                            elif "name" in item:
                                tool_call = self._create_tool_call(
                                    name=item["name"],
                                    arguments=item.get("parameters", item.get("arguments", {}))
                                )
                                tool_calls.append(tool_call)
                    
                    if tool_calls:
                        # Clean content by removing the JSON tool call
                        clean_content = text
                        if json_matches:
                            clean_content = text.replace(json_matches[0], "").strip()
                        
                        return ExtractedToolCallInformation(
                            tools_called=True,
                            tool_calls=tool_calls,
                            content=clean_content if clean_content else None
                        )
                        
        except (json.JSONDecodeError, ValueError):
            pass
        
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=text
        )
    
    def _get_ast_value(self, node):
        """Extract value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Str):  # For older Python versions
            return node.s
        elif isinstance(node, ast.Num):  # For older Python versions
            return node.n
        elif isinstance(node, ast.Dict):
            return {
                self._get_ast_value(k): self._get_ast_value(v)
                for k, v in zip(node.keys, node.values)
            }
        elif isinstance(node, ast.List):
            return [self._get_ast_value(item) for item in node.elts]
        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")
    
    def format_tools_for_prompt(
        self,
        tools: List[Any]
    ) -> str:
        """Format tools for Llama prompt template."""
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
                      "[function_name(param1=\"value1\", param2=\"value2\")]\n\n"
                      "Or JSON format:\n"
                      '{"name": "function_name", "parameters": {"param1": "value1"}}\n\n')
        
        return tools_text 