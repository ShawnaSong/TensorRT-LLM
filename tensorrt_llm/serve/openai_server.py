#!/usr/bin/env python
import asyncio
import signal
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from typing import AsyncGenerator, AsyncIterator, List, Optional, Tuple
import time

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
from tensorrt_llm.llmapi.disagg_utils import MetadataServerConfig, ServerRole
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.chat_utils import (check_multiple_response,
                                           parse_chat_messages_coroutines)
from tensorrt_llm.serve.metadata_server import create_metadata_server
from tensorrt_llm.serve.metrics import TensorRTMetrics
from tensorrt_llm.serve.metrics.prometheus_server import PrometheusServer
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

from .._utils import nvtx_mark

# yapf: enale
TIMEOUT_KEEP_ALIVE = 5  # seconds.


class OpenAIServer:

    def __init__(self,
                 llm: LLM,
                 model: str,
                 tool_parser: str,
                 server_role: Optional[ServerRole],
                 metadata_server_cfg: MetadataServerConfig,
                 prometheus_port: int = 18002):
        self.llm = llm
        self.tokenizer = llm.tokenizer
        self.metadata_server = create_metadata_server(metadata_server_cfg)
        self.server_role = server_role
        self.binding_addr = None  # Will be set in __call__
        self.metrics = TensorRTMetrics(model_name=model, llm=llm)
        self.prometheus_server = PrometheusServer(port=prometheus_port)
        hf_tokenizer_path = llm._hf_model_dir or self.tokenizer.tokenizer.name_or_path
        trust_remote_code = llm.args.trust_remote_code
        try:
            self.processor = AutoProcessor.from_pretrained(hf_tokenizer_path, trust_remote_code=trust_remote_code)
        except Exception:
            logger.debug("Failed to load AutoProcessor or AutoConfig for %s", hf_tokenizer_path)
            self.processor = None
        try:
            self.model_config = AutoConfig.from_pretrained(hf_tokenizer_path)
        except Exception:
            logger.debug("Failed to load AutoConfig for %s", hf_tokenizer_path)
            self.model_config = None

        model_dir = Path(model)
        if model_dir.exists() and model_dir.is_dir():
            self.model = model_dir.name
        else:
            self.model = model

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            if self.metadata_server is not None:
                metadata = {
                    "model": self.model,
                    "version": VERSION,
                    "timestamp": datetime.now().isoformat(),
                    "server_role": server_role.name,
                    "url": self.binding_addr
                }
                self.metadata_server.put(f"trtllm/{self.llm.llm_id}", metadata)
                logger.info(f"trtllm/{self.llm.llm_id} is registered")

            yield

            if self.metadata_server is not None:
                self.metadata_server.remove(f"trtllm/{self.llm.llm_id}")
                logger.info(f"trtllm/{self.llm.llm_id} is unregistered")
            self.llm.shutdown()

        self.app = FastAPI(lifespan=lifespan)

        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(_, exc):
            return self.create_error_response(message=str(exc))

        self.register_routes()

    async def await_disconnected(self, raw_request: Request, promise):
        if raw_request is None:
            return
        while not await raw_request.is_disconnected():
            await asyncio.sleep(1)
        if not promise.finished:
            promise.abort()
            logger.info(f"{raw_request.client} is disconnected, abort {promise.request_id}")

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
        self.app.add_api_route("/health_generate", self.health_generate, methods=["GET"])
        self.app.add_api_route("/version", self.version, methods=["GET"])
        self.app.add_api_route("/v1/models", self.get_model, methods=["GET"])
        self.app.add_api_route("/metrics", self.get_iteration_stats, methods=["GET"])
        self.app.add_api_route("/kv_cache_events", self.get_kv_cache_events, methods=["POST"])
        self.app.add_api_route("/v1/completions",
                               self.openai_completion,
                               methods=["POST"])
        self.app.add_api_route("/v1/chat/completions",
                               self.openai_chat,
                               methods=["POST"])

    async def health(self) -> Response:
        return Response(status_code=200)

    async def health_generate(self) -> Response:
        try:
            health_request = ChatCompletionRequest(
                messages=[{"role": "user", "content": "hi"}],
                model=self.model,
                max_completion_tokens=1,
                stream=False,
                temperature=0.0
            )
            mock_request = None
            response = await self.openai_chat(health_request, mock_request)
            if response.status_code == 200:
                return Response(status_code=200, content="Generation health check OK")
            else:
                logger.error(f"Health generate check failed with status code: {response.status_code}")
                return Response(status_code=500, content="Generation health check failed")
        except Exception as e:
            logger.error(f"Health generate check encountered exception: {e}", exc_info=True)
            return Response(status_code=500, content=f"Generation health check failed: {str(e)}")

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
            pass
        return JSONResponse(content=events)

    async def openai_chat(self, request: ChatCompletionRequest, raw_request: Request) -> Response:
        request_id = f"chatcmpl-{int(time.time() * 1000)}"
        try:
            prompt_tokens = 0
            if request.messages and isinstance(request.messages, list):
                for message in request.messages:
                    if isinstance(message, dict) and 'content' in message and isinstance(message['content'], str):
                        prompt_tokens += len(message['content']) // 3
            self.metrics.track_request_start(request_id, prompt_tokens, request.max_completion_tokens or 1024)

            def get_role() -> str:
                if request.add_generation_prompt:
                    return "assistant"
                else:
                    return request.messages[-1]["role"]

            async def chat_stream_generator(promise: RequestOutput, postproc_params: PostprocParams) -> AsyncGenerator[str, None]:
                if not self.postproc_worker_enabled:
                    post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
                
                self.metrics.track_first_token(request_id)
                token_count = 0
                async for res in promise:
                    pp_results = res.outputs[0]._postprocess_result if self.postproc_worker_enabled else post_processor(res, args)
                    for pp_res in pp_results:
                        self.metrics.track_token_generation(request_id, 1)
                        token_count += 1
                        yield pp_res

                finish_reason = "stop"
                if promise.outputs:
                    finish_reason = promise.outputs[0].finish_reason or "stop"
                self.metrics.track_request_completion(request_id,
                                                      prompt_tokens=len(promise.prompt_token_ids),
                                                      generation_tokens=token_count,
                                                      finish_reason=finish_reason)
                yield "data: [DONE]\n\n"
                nvtx_mark("generation ends")

            async def create_chat_response(promise: RequestOutput, postproc_params: PostprocParams) -> ChatCompletionResponse:
                await promise.aresult()
                if self.postproc_worker_enabled:
                    chat_response = promise.outputs[0]._postprocess_result
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
                return chat_response

            check_multiple_response(request.n, self.llm.args.backend)
            conversation: List[ConversationMessage] = []
            tool_dicts = None if request.tools is None else [t.model_dump() for t in request.tools]
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
            if conversation and conversation[-1].get("content") and conversation[-1].get("role") == get_role():
                postproc_args.last_message_content = conversation[-1]["content"]
            postproc_params = PostprocParams(
                post_processor=chat_stream_post_processor if request.stream else chat_response_post_processor,
                postproc_args=postproc_args,
            )

            promise = self.llm.generate_async(
                inputs=prompt,
                sampling_params=sampling_params,
                _postproc_params=postproc_params if self.postproc_worker_enabled else None,
                streaming=request.stream,
                disaggregated_params=disaggregated_params
            )
            asyncio.create_task(self.await_disconnected(raw_request, promise))
            if not self.postproc_worker_enabled:
                postproc_args.tokenizer = self.tokenizer
                postproc_args.num_prompt_tokens = len(promise.prompt_token_ids)

            if request.stream:
                response_generator = chat_stream_generator(promise, postproc_params)
                return StreamingResponse(content=response_generator, media_type="text/event-stream")
            else:
                return await create_chat_response(promise, postproc_params)

        except CppExecutorError as e:
            self.metrics.track_error(request_id, "CppExecutorError")
            signal.raise_signal(signal.SIGINT)
            raise e
        except Exception as e:
            self.metrics.track_error(request_id, type(e).__name__)
            traceback.print_exc()
            return self.create_error_response(str(e))

    async def openai_completion(self, request: CompletionRequest, raw_request: Request) -> Response:
        request_id = f"cmpl-{int(time.time() * 1000)}"
        try:
            prompt_tokens = 0
            if isinstance(request.prompt, str):
                prompt_tokens = len(request.prompt) // 3
            elif isinstance(request.prompt, list):
                prompt_tokens = len(request.prompt)
            self.metrics.track_request_start(request_id, prompt_tokens, request.max_completion_tokens or 1024)

            def merge_promises(promises: List[RequestOutput], postproc_params_collections: List[Optional[PostprocParams]]) -> AsyncIterator[Tuple[RequestOutput, Optional[PostprocParams]]]:
                outputs = asyncio.Queue()
                finished = [False] * len(promises)
                async def producer(i: int, promise: RequestOutput, postproc_params: Optional[PostprocParams]):
                    async for output in promise:
                        await outputs.put((output, postproc_params))
                    finished[i] = True
                _tasks = [asyncio.create_task(producer(i, promise, postproc_params)) for i, (promise, postproc_params) in enumerate(zip(promises, postproc_params_collections))]
                async def consumer():
                    while not all(finished) or not outputs.empty():
                        item = await outputs.get()
                        yield item
                    await asyncio.gather(*_tasks)
                return consumer()

            async def create_completion_generator(generator: AsyncIterator[Tuple[RequestOutput, Optional[PostprocParams]]]):
                self.metrics.track_first_token(request_id)
                total_tokens = 0
                async for request_output, postproc_params in generator:
                    if not self.postproc_worker_enabled:
                        post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
                        pp_result = post_processor(request_output, args)
                    else:
                        pp_result = request_output.outputs[0]._postprocess_result
                    for pp_res in pp_result:
                        self.metrics.track_token_generation(request_id, 1)
                        total_tokens += 1
                        yield pp_res
                self.metrics.track_request_completion(request_id, generation_tokens=total_tokens, finish_reason="stop")
                yield "data: [DONE]\n\n"

            async def create_completion_response(generator: AsyncIterator[Tuple[RequestOutput, Optional[PostprocParams]]]) -> CompletionResponse:
                all_choices: List[CompletionResponseChoice] = []
                all_prompt_token_ids: List[List[int]] = []
                num_prompt_tokens = num_gen_tokens = 0
                async for request_output, postproc_params in generator:
                    pp_result: CompletionResponse
                    if not self.postproc_worker_enabled:
                        post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
                        pp_result = post_processor(request_output, args)
                    else:
                        pp_result = request_output.outputs[0]._postprocess_result
                    choices, usage = pp_result.choices, pp_result.usage
                    all_choices.extend(choices)
                    num_prompt_tokens += usage.prompt_tokens
                    num_gen_tokens += usage.completion_tokens
                    all_prompt_token_ids.append(request_output.prompt_token_ids)

                usage_info = UsageInfo(prompt_tokens=num_prompt_tokens, completion_tokens=num_gen_tokens, total_tokens=num_gen_tokens + num_prompt_tokens)
                response = CompletionResponse(model=self.model, choices=all_choices, usage=usage_info, prompt_token_ids=all_prompt_token_ids)
                self.metrics.track_request_completion(request_id, prompt_tokens=num_prompt_tokens, generation_tokens=num_gen_tokens, finish_reason="stop")
                return response

            check_multiple_response(request.n, self.llm.args.backend)
            if isinstance(request.prompt, str) or (isinstance(request.prompt, list) and isinstance(request.prompt[0], int)):
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
                    post_processor=completion_stream_post_processor if request.stream else completion_response_post_processor,
                    postproc_args=postproc_args,
                )
                promise = self.llm.generate_async(
                    inputs=prompt,
                    sampling_params=sampling_params,
                    _postproc_params=postproc_params,
                    streaming=request.stream,
                    disaggregated_params=disaggregated_params
                )
                asyncio.create_task(self.await_disconnected(raw_request, promise))
                if not self.postproc_worker_enabled:
                    postproc_args.tokenizer = self.tokenizer
                    postproc_args.num_prompt_tokens = len(promise.prompt_token_ids)
                promises.append(promise)
                postproc_params_collection.append(None if self.postproc_worker_enabled else postproc_params)

            generator = merge_promises(promises, postproc_params_collection)
            if request.stream:
                response_generator = create_completion_generator(generator)
                return StreamingResponse(content=response_generator, media_type="text/event-stream")
            else:
                return await create_completion_response(generator)
        except CppExecutorError as e:
            self.metrics.track_error(request_id, "CppExecutorError")
            signal.raise_signal(signal.SIGINT)
            raise e
        except Exception as e:
            self.metrics.track_error(request_id, type(e).__name__)
            traceback.print_exc()
            return self.create_error_response(str(e))

    async def __call__(self, host, port):
        self.prometheus_server.start()
        self.binding_addr = f"http://{host}:{port}"
        config = uvicorn.Config(self.app, host=host, port=port, log_level="info", timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
        await uvicorn.Server(config).serve()