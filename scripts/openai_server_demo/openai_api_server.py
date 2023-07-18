import argparse
import os
from fastapi import FastAPI
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--lora_model', default=None, type=str,help="If None, perform inference on the base model")
parser.add_argument('--tokenizer_path',default=None,type=str)
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--load_in_8bit',action='store_true', help='use 8 bit model')
parser.add_argument('--only_cpu',action='store_true',help='only use CPU for inference')
parser.add_argument('--alpha',type=str,default="1.0", help="The scaling factor of NTK method, can be a float or 'auto'. ")
args = parser.parse_args()
load_in_8bit = args.load_in_8bit
if args.only_cpu is True:
    args.gpus = ""
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from peft import PeftModel

from patches import apply_attention_patch, apply_ntk_scaling_patch
apply_attention_patch(use_memory_efficient_attention=True)
apply_ntk_scaling_patch(args.alpha)

from openai_api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    EmbeddingsRequest,
    EmbeddingsResponse,
)

load_type = torch.float16
if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device('cpu')
if args.tokenizer_path is None:
    args.tokenizer_path = args.lora_model
    if args.lora_model is None:
        args.tokenizer_path = args.base_model
tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)

base_model = LlamaForCausalLM.from_pretrained(
    args.base_model,
    load_in_8bit=load_in_8bit,
    torch_dtype=load_type,
    low_cpu_mem_usage=True,
    device_map='auto' if not args.only_cpu else None,
    )

model_vocab_size = base_model.get_input_embeddings().weight.size(0)
tokenzier_vocab_size = len(tokenizer)
print(f"Vocab of the base model: {model_vocab_size}")
print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
if model_vocab_size!=tokenzier_vocab_size:
    assert tokenzier_vocab_size > model_vocab_size
    print("Resize model embeddings to fit tokenizer")
    base_model.resize_token_embeddings(tokenzier_vocab_size)
if args.lora_model is not None:
    print("loading peft model")
    model = PeftModel.from_pretrained(base_model, args.lora_model,torch_dtype=load_type,device_map='auto',)
else:
    model = base_model

if device==torch.device('cpu'):
    model.float()

model.eval()

def generate_completion_prompt(instruction: str):
    """Generate prompt for completion"""
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
    
### Instruction:
{instruction}

### Response: """

def generate_chat_prompt(messages: list):
    """Generate prompt for chat completion"""
    system_msg = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.'''
    for msg in messages:
        if msg.role == 'system':
            system_msg = msg.content
    prompt = f"{system_msg}\n\n"
    for msg in messages:
        if msg.role == 'system':
            continue
        if msg.role == 'assistant':
            prompt += f"### Response: {msg.content}\n\n"
        if msg.role == 'user':
            prompt += f"### Instruction:\n{msg.content}\n\n"
    prompt += "### Response: "
    return prompt

def predict(
    input,
    max_new_tokens=128,
    top_p=0.75,
    temperature=0.1,
    top_k=40,
    num_beams=4,
    repetition_penalty=1.0,
    do_sample=True,
    **kwargs,
):
    """
    Main inference method
    type(input) == str -> /v1/completions
    type(input) == list -> /v1/chat/completions
    """
    if isinstance(input, str):
        prompt = generate_completion_prompt(input)
    else:
        prompt = generate_chat_prompt(input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        do_sample=do_sample,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
            max_new_tokens=max_new_tokens,
            repetition_penalty=float(repetition_penalty),
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    output = output.split("### Response:")[-1].strip()
    return output

def get_embedding(input):
    """Get embedding main function"""
    with torch.no_grad():
        if tokenizer.pad_token == None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        encoding = tokenizer(
            input, padding=True, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        model_output = model(
            input_ids, attention_mask, output_hidden_states=True
        )
        data = model_output.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).expand(data.size()).float()
        masked_embeddings = data * mask
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        seq_length = torch.sum(mask, dim=1)
        embedding = sum_embeddings / seq_length
        normalized_embeddings = F.normalize(embedding, p=2, dim=1)
        ret = normalized_embeddings.squeeze(0).tolist()
    return ret

app = FastAPI()

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Creates a completion for the chat message"""
    msgs = request.messages
    if isinstance(msgs, str):
        msgs = [ChatMessage(role='user',content=msgs)]
    else:
        msgs = [ChatMessage(role=x['role'],content=x['message']) for x in msgs]
    output = predict(
        input=msgs,
        max_new_tokens=request.max_tokens,
        top_p=request.top_p,
        top_k=request.top_k,
        temperature=request.temperature,
        num_beams=request.num_beams,
        repetition_penalty=request.repetition_penalty,
        do_sample=request.do_sample,
    )
    choices = [ChatCompletionResponseChoice(index = i, message = msg) for i, msg in enumerate(msgs)]
    choices += [ChatCompletionResponseChoice(index = len(choices), message = ChatMessage(role='assistant',content=output))]
    return ChatCompletionResponse(choices = choices)

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Creates a completion"""
    output = predict(
        input=request.prompt,
        max_new_tokens=request.max_tokens,
        top_p=request.top_p,
        top_k=request.top_k,
        temperature=request.temperature,
        num_beams=request.num_beams,
        repetition_penalty=request.repetition_penalty,
        do_sample=request.do_sample,
    )
    choices = [CompletionResponseChoice(index = 0, text = output)]
    return CompletionResponse(choices = choices)

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingsRequest):
    """Creates text embedding"""
    embedding = get_embedding(request.input)
    data = [{
        "object": "embedding",
        "embedding": embedding,
        "index": 0
    }]
    return EmbeddingsResponse(data=data)


if __name__ == "__main__":
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    uvicorn.run(app, host='0.0.0.0', port=19327, workers=1, log_config=log_config)
