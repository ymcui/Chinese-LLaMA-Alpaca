from typing import Optional, List, Dict, Any, Union

import time

import shortuuid
from pydantic import BaseModel, Field

class ChatCompletionRequest(BaseModel):
    model: str = "chinese-llama-alpaca"
    messages: Union[str, List[Dict[str, str]]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 40
    n: Optional[int] = 1
    max_tokens: Optional[int] = 128
    num_beams: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    repetition_penalty: Optional[float] = 1.0
    user: Optional[str] = None
    do_sample: Optional[bool] = True


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "chinese-llama-alpaca"
    choices: List[ChatCompletionResponseChoice]


class EmbeddingsRequest(BaseModel):
    input: Union[str, List[Any]]
    user: Optional[str] = None


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str = "chinese-llama-alpaca"


class CompletionRequest(BaseModel):
    prompt: Union[str, List[Any]]
    temperature: Optional[float] = 0.1
    n: Optional[int] = 1
    max_tokens: Optional[int] = 128
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = 0.75
    top_k: Optional[int] = 40
    num_beams: Optional[int] = 1
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    repetition_penalty: Optional[float] = 1.0
    user: Optional[str] = None
    do_sample: Optional[bool] = True


class CompletionResponseChoice(BaseModel):
    index: int
    text: str


class CompletionResponse(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: Optional[str] = "text_completion"
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    model: Optional[str] = 'chinese-llama-alpaca'
    choices: List[CompletionResponseChoice]

