#!/usr/bin/env python
import asyncio
import signal
import traceback
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
import time
from typing import AsyncGenerator, AsyncIterator, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from transformers import AutoConfig, AutoProcessor

# yapf: disable
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.executor.postproc_worker import PostprocParams
from tensorrt_llm.inputs import prompt_inputs
from tensorrt_llm.inputs.utils import ConversationMessage, apply_chat_template
from tensorrt_llm.llmapi import LLM
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.chat_utils import (check_multiple_response,
                                           parse_chat_messages_coroutines)
from tensorrt_llm.serve.openai_protocol import (ChatCompletionRequest,
                                                ChatCompletionResponse,
                                                CompletionRequest,
                                                CompletionResponse,
                                                CompletionResponseChoice,
                                                ErrorResponse, ModelCard,
                                                ModelList, UsageInfo,
                                                to_llm_disaggregated_params)
from tensorrt_llm.serve.postprocess_handlers import (
    ChatPostprocArgs, CompletionPostprocArgs, chat_response_post_processor,
    chat_stream_post_processor, completion_response_post_processor,
    completion_stream_post_processor)
from tensorrt_llm.serve.tool_call_manager import ToolCallManager
from tensorrt_llm.version import __version__ as VERSION
from tensorrt_llm.serve.metrics import TensorRTMetrics
from tensorrt_llm.serve.metrics.prometheus_server import PrometheusServer

from .._utils import nvtx_mark

# yapf: enale
TIMEOUT_KEEP_ALIVE = 5  # seconds.

import os
ERROR_LOG_PATH = os.path.expanduser('~/.cache/trtllm-error.log')


class OpenAIServer:

    def __init__(self,
                 llm: LLM,
                 model: str,
                 tool_parser: str,
                 server_role: Optional["ServerRole"],
                 metadata_server_cfg: "MetadataServerConfig",
                 prometheus_port: int = None):
        self.llm = llm
        self.tokenizer = llm.tokenizer
        #self.metadata_server = create_metadata_server(metadata_server_cfg)
        self.server_role = server_role
        self.binding_addr = None  # Will be set in __call__
        self.tool_parser = tool_parser
        self.metrics = TensorRTMetrics(model_name=model, llm=llm)
        self.prometheus_server = PrometheusServer(port=prometheus_port) if prometheus_port else None

        # Initialize tool call manager
        self.tool_call_manager = ToolCallManager(self.tokenizer, self.tool_parser)
        hf_tokenizer_path = llm._hf_model_dir or self.tokenizer.tokenizer.name_or_path
        trust_remote_code = llm.args.trust_remote_code
        try:
            self.processor = AutoProcessor.from_pretrained(hf_tokenizer_path, trust_remote_code=trust_remote_code)
            self.model_config = AutoConfig.from_pretrained(hf_tokenizer_path)
        except Exception:
            logger.debug("Failed to load AutoProcessor or AutoConfig for %s", hf_tokenizer_path)
            self.processor = None
            self.model_config = None

        model_dir = Path(model)
        if model_dir.exists() and model_dir.is_dir():
            self.model = model_dir.name
        else:
            self.model = model

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # terminate rank0 worker
            yield
            self.llm.shutdown()

        self.app = FastAPI(lifespan=lifespan)

        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(_, exc):
            return self.create_error_response(message=str(exc))

        self.register_routes()

        self.last_yield_time = time.time()

        # Delete error log file if it exists
        if os.path.exists(ERROR_LOG_PATH):
            try:
                os.remove(ERROR_LOG_PATH)
            except Exception as e:
                pass

    async def await_disconnected(self, raw_request: Request, promise):
        while not await raw_request.is_disconnected():
            await asyncio.sleep(1)
        if not promise.finished:
            promise.abort()
            logger.info(
                f"{raw_request.client} is disconnected, abort {promise.request_id}")
            self.metrics.track_request_completion(promise.request_id, finish_reason="disconnected")

    @property
    def postproc_worker_enabled(self) -> bool:
        return True if self.llm.args.num_postprocess_workers > 0 else False

    @staticmethod
    def create_error_response(
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> ErrorResponse:
        error_response = ErrorResponse(message=message,
                                       type=err_type,
                                       code=status_code.value)
        return JSONResponse(content=error_response.model_dump(),
                            status_code=error_response.code)


    def register_routes(self):
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/version", self.version, methods=["GET"])
        self.app.add_api_route("/v1/models", self.get_model, methods=["GET"])
        # TODO: the metrics endpoint only reports iteration stats, not the runtime stats for now
        self.app.add_api_route("/metrics", self.get_iteration_stats, methods=["GET"])
        # TODO: workaround before ETCD support
        self.app.add_api_route("/kv_cache_events", self.get_kv_cache_events, methods=["POST"])
        self.app.add_api_route("/v1/completions",
                               self.openai_completion,
                               methods=["POST"])
        self.app.add_api_route("/v1/chat/completions",
                               self.openai_chat,
                               methods=["POST"])

    async def health(self) -> Response:
        # check error log
        if os.path.exists(ERROR_LOG_PATH):
            mtime = os.path.getmtime(ERROR_LOG_PATH)
            if time.time() - mtime < 300:
                with open(ERROR_LOG_PATH, "r") as f:
                    lines = f.readlines()
                    last_line = lines[-1].strip() if lines else ""
                    logger.warning(f'Fatal error from worker detected: {last_line}', )
                return Response(content=last_line, status_code=500)

        # Check for pending requests and yield timeout
        waiting_time = 0
        active_requests = self.metrics._active_requests
        if active_requests > 0:
            waiting_time = time.time() - self.last_yield_time
            if waiting_time > 60:
                logger.error(f"Critical timeout, pending requests: {active_requests}, waiting timeout: {waiting_time}.")

                time.sleep(1)
                os._exit(-1)
            elif active_requests > 1 and waiting_time > 1:
                logger.warning(f"Pending requests: {active_requests}, waiting timeout: {waiting_time}.")
                return Response(
                    content=f"Pending request timeout, {waiting_time} seconds.",
                    status_code=500,
                )
            else:
                logger.info(f"Pending requests: {active_requests}, waiting time: {waiting_time} seconds.")

        return Response(status_code=200)

    async def version(self) -> JSONResponse:
        ver = {"version": VERSION}
        return JSONResponse(content=ver)

    async def get_model(self) -> JSONResponse:
        model_list = ModelList(data=[ModelCard(id=self.model)])
        return JSONResponse(content=model_list.model_dump())

    async def get_iteration_stats(self) -> JSONResponse:
        stats = []
        async for stat in self.llm.get_stats_async(2):
            stats.append(stat)
        return JSONResponse(content=stats)

    async def get_kv_cache_events(self) -> JSONResponse:
        events = []
        try:
            async for event in self.llm.get_kv_cache_events_async(2):
                events.append(event)
        except IndexError:
            # queue is empty, no more events
            pass
        return JSONResponse(content=events)

    async def openai_chat(self, request: ChatCompletionRequest, raw_request: Request) -> Response:

        def get_role() -> str:
            if request.add_generation_prompt:
                role = "assistant"
            else:
                role = request.messages[-1]["role"]
            return role

        async def chat_stream_generator(
                promise: RequestOutput, postproc_params: PostprocParams) -> AsyncGenerator[str, None]:
            if not self.postproc_worker_enabled:
                post_processor, args = postproc_params.post_processor, postproc_params.postproc_args

            prompt_tokens_len = len(promise.prompt_token_ids)
            logger.info(f">> {prompt_tokens_len=}")

            self.metrics.track_first_token(request_id)
            token_count = 0

            async for res in promise:
                pp_results = res.outputs[0]._postprocess_result if self.postproc_worker_enabled else post_processor(res, args)
                for pp_res in pp_results:
                    self.metrics.track_token_generation(request_id, len(promise.outputs[0].token_ids))
                    yield pp_res
                    self.last_yield_time = time.time()

            finish_reason = "stop"
            if promise.outputs:
                finish_reason = promise.outputs[0].finish_reason or "stop"
            self.metrics.track_request_completion(request_id,
                                                  prompt_tokens=len(promise.prompt_token_ids),
                                                  generation_tokens=token_count,
                                                  finish_reason=finish_reason)

            yield f"data: [DONE]\n\n"
            nvtx_mark("generation ends")

            logger.info(f"<< {prompt_tokens_len}:{token_count}")

        async def create_chat_response(
                promise: RequestOutput, postproc_params: PostprocParams) -> ChatCompletionResponse:
            await promise.aresult()
            if self.postproc_worker_enabled:
                return promise.outputs[0]._postprocess_result
            else:
                post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
                chat_response = post_processor(promise, args)

            chat_response.prompt_token_ids = promise.prompt_token_ids
            finish_reason = "stop"
            if chat_response.choices:
                finish_reason = chat_response.choices[0].finish_reason or "stop"
            self.metrics.track_request_completion(request_id,
                                                  prompt_tokens=chat_response.usage.prompt_tokens,
                                                  generation_tokens=chat_response.usage.completion_tokens,
                                                  finish_reason=finish_reason)

            # Process tool calls if tools are available and no tool calls were found
            if (request.tools and len(request.tools) > 0 and 
                chat_response.choices and len(chat_response.choices) > 0):
                
                for choice in chat_response.choices:
                    if not choice.message.tool_calls or len(choice.message.tool_calls) == 0:
                        # Use tool manager to extract tool calls from content
                        try:
                            tool_result = self.tool_call_manager.extract_tool_calls(
                                choice.message.content or "", request
                            )
                            if tool_result["tools_called"]:
                                choice.message.tool_calls = tool_result["tool_calls"]
                                choice.message.content = tool_result["content"] or ""
                                logger.info(f"Tool manager extracted {len(tool_result['tool_calls'])} tool calls using {tool_result['parser_used']} parser")
                        except Exception as e:
                            logger.warning(f"Tool call extraction failed: {e}")

            # Add prompt_tokens_ids to the response
            #chat_response.prompt_token_ids = promise.prompt_token_ids
            return chat_response

        try:
            conversation: List[ConversationMessage] = []
            tool_dicts = None if request.tools is None else [
                tool.model_dump() for tool in request.tools
            ]
            sampling_params = request.to_sampling_params()
            postproc_args = ChatPostprocArgs.from_request(request)
            disaggregated_params = to_llm_disaggregated_params(request.disaggregated_params)

            conversation, mm_coroutines, mm_placeholder_counts = parse_chat_messages_coroutines(request.messages, self.model_config)

            if request.prompt_token_ids is not None:
                prompt = request.prompt_token_ids
            else:
                prompt: str = apply_chat_template(
                    model_type=self.model_config.model_type,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    conversation=conversation,
                    add_generation_prompt=request.add_generation_prompt,
                    mm_placeholder_counts=mm_placeholder_counts,
                    tools=tool_dicts,
                    documents=request.documents,
                    chat_template=request.chat_template,
                    chat_template_kwargs=request.chat_template_kwargs or {},
                )
            prompt = prompt_inputs(prompt)

            mm_data = await mm_coroutines
            if mm_data is not None:
                prompt["multi_modal_data"] = mm_data

            postproc_args.reasoning_parser = self.llm.args.reasoning_parser
            if conversation and conversation[-1].get(
                    "content") and conversation[-1].get("role") == get_role():
                postproc_args.last_message_content = conversation[-1]["content"]
            postproc_params = PostprocParams(
                post_processor=chat_stream_post_processor
                if request.stream else chat_response_post_processor,
                postproc_args=postproc_args,
            )

            promise = self.llm.generate_async(
                inputs=prompt,
                sampling_params=sampling_params,
                _postproc_params=postproc_params if self.postproc_worker_enabled else None,
                streaming=request.stream,
                disaggregated_params=disaggregated_params
            )
            request_id = promise.request_id
            self.metrics.track_request_start(promise.request_id, request.max_completion_tokens)
            asyncio.create_task(self.await_disconnected(raw_request, promise))
            if not self.postproc_worker_enabled:
                postproc_args.tokenizer = self.tokenizer
                postproc_args.num_prompt_tokens = len(promise.prompt_token_ids)

            if request.stream:
                response_generator = chat_stream_generator(promise, postproc_params)
                return StreamingResponse(content=response_generator,
                                         media_type="text/event-stream")
            else:
                response = await create_chat_response(promise, postproc_params)
                return JSONResponse(content=response.model_dump())
        except CppExecutorError as e:
            # If internal executor error is raised, shutdown the server
            self.metrics.track_error(request_id, "CppExecutorError")
            signal.raise_signal(signal.SIGINT)
            raise e
        except Exception as e:
            self.metrics.track_error(request_id, type(e).__name__)
            traceback.print_exc()
            return self.create_error_response(str(e))

    async def openai_completion(self, request: CompletionRequest, raw_request: Request) -> Response:

        def merge_promises(
            promises: List[RequestOutput],
            postproc_params_collections: List[Optional[PostprocParams]]
        ) -> AsyncIterator[Tuple[RequestOutput, Optional[PostprocParams]]]:
            outputs = asyncio.Queue()
            finished = [False] * len(promises)

            async def producer(i: int, promise: RequestOutput, postproc_params: Optional[PostprocParams]):
                async for output in promise:
                    await outputs.put((output, postproc_params))
                finished[i] = True

            _tasks = [
                asyncio.create_task(producer(i, promise, postproc_params))
                for i, (promise, postproc_params) in enumerate(zip(promises, postproc_params_collections))
            ]

            async def consumer():
                while not all(finished) or not outputs.empty():
                    item = await outputs.get()
                    yield item
                await asyncio.gather(*_tasks)

            return consumer()

        generate_length_recorder = {}
        prompt_length_recorder = {}
        async def create_completion_generator(
                generator: AsyncIterator[Tuple[RequestOutput, Optional[PostprocParams]]]):
            total_tokens = 0
            prompt_length_recorder = {}
            async for request_output, postproc_params in generator:
                rid = request_output.request_id
                prompt_length_recorder[rid] = len(request_output.prompt_token_ids)
                generate_length_recorder[rid] = len(request_output.outputs[0].token_ids)
                if not self.postproc_worker_enabled:
                    post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
                    pp_result = post_processor(request_output, args)
                else:
                    pp_result = request_output.outputs[0]._postprocess_result
                for pp_res in pp_result:
                    # total_tokens += 1	# TODO: plus 1 is wrong for speculative decoding
                    yield pp_res
                    self.last_yield_time = time.time()
			# TODO: merge num_gen_tokens?
            for rid in generate_length_recorder.keys():
               self.metrics.track_request_completion(rid, prompt_tokens=prompt_length_recorder[rid], generation_tokens=generate_length_recorder[rid], finish_reason="stop")
            yield f"data: [DONE]\n\n"
        async def create_completion_response(
                generator: AsyncIterator[Tuple[RequestOutput, Optional[PostprocParams]]]) -> CompletionResponse:
            all_choices: List[CompletionResponseChoice] = []
            num_prompt_tokens = num_gen_tokens = 0
            async for request_output, postproc_params in generator:
                pp_result: CompletionResponse
                if not self.postproc_worker_enabled:
                    post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
                    pp_result = post_processor(request_output, args)
                else:
                    pp_result = request_output.outputs[0]._postprocess_result

                rid = request_output.request_id
                generate_length_recorder[rid] = len(request_output.outputs[0].token_ids)

                choices, usage = pp_result.choices, pp_result.usage
                all_choices.extend(choices)
                num_prompt_tokens += usage.prompt_tokens
                num_gen_tokens += usage.completion_tokens

            usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_gen_tokens,
                total_tokens=num_gen_tokens + num_prompt_tokens,
            )
            response = CompletionResponse(
                model=self.model,
                choices=all_choices,
                usage=usage_info,
            )
			# TODO: merge num_gen_tokens?
            for rid in generate_length_recorder.keys():
               self.metrics.track_request_completion(
                   rid,
                   prompt_tokens=num_prompt_tokens,
                   generation_tokens=num_gen_tokens,
                   finish_reason="stop"
               )
            return response

        try:
            if isinstance(request.prompt, str) or \
                (isinstance(request.prompt, list) and isinstance(request.prompt[0], int)):
                prompts = [request.prompt]
            else:
                prompts = request.prompt

            promises: List[RequestOutput] = []
            postproc_params_collection: List[Optional[PostprocParams]] = []
            sampling_params = request.to_sampling_params()
            disaggregated_params = to_llm_disaggregated_params(request.disaggregated_params)
            for idx, prompt in enumerate(prompts):
                postproc_args = CompletionPostprocArgs.from_request(request)
                postproc_args.prompt_idx = idx
                if request.echo:
                    postproc_args.prompt = prompt
                postproc_params = PostprocParams(
                    post_processor=completion_stream_post_processor
                    if request.stream else completion_response_post_processor,
                    postproc_args=postproc_args,
                )
                promise = self.llm.generate_async(
                    inputs=prompt,
                    sampling_params=sampling_params,
                    _postproc_params=postproc_params,
                    streaming=request.stream,
                    disaggregated_params=disaggregated_params
                )
                # print(f"Request ID: {promise.request_id}, Prompt: {prompt}")
                # # request_ids.append(promise.request_id)
                self.metrics.track_request_start(promise.request_id, request.max_tokens)
                asyncio.create_task(self.await_disconnected(raw_request, promise))
                if not self.postproc_worker_enabled:
                    postproc_args.tokenizer = self.tokenizer
                    postproc_args.num_prompt_tokens = len(promise.prompt_token_ids)
                promises.append(promise)
                postproc_params_collection.append(None if self.postproc_worker_enabled else postproc_params)

            generator = merge_promises(promises, postproc_params_collection)
            if request.stream:
                response_generator = create_completion_generator(
                    generator)
                return StreamingResponse(content=response_generator,
                                            media_type="text/event-stream")
            else:
                response = await create_completion_response(
                    generator)
                return JSONResponse(content=response.model_dump())
        except CppExecutorError as e:
            # If internal executor error is raised, shutdown the server
            for request_id in generate_length_recorder.keys():
                self.metrics.track_error(request_id, "CppExecutorError")
            signal.raise_signal(signal.SIGINT)
            raise e
        except Exception as e:
            for request_id in generate_length_recorder.keys():
                self.metrics.track_error(request_id, type(e).__name__)
            traceback.print_exc()
            return self.create_error_response(str(e))

    async def __call__(self, host, port):
        if self.prometheus_server is not None:
            self.prometheus_server.start()

        config = uvicorn.Config(self.app,
                                host=host,
                                port=port,
                                log_level="info",
                                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
        await uvicorn.Server(config).serve()
