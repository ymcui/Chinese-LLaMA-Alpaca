"""
Borrowed and modified from https://github.com/tloen/alpaca-lora
"""

import os
import json
import gc

import torch
import peft
from peft import PeftModel

import transformers

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--base_model',default=None,required=True,type=str,help="Please specify a base_model")
parser.add_argument('--lora_model',default=None,required=True,type=str,help="Please specify a lora_model")

# deprecated; the script infers the model size from the checkpoint
parser.add_argument('--model_size',default='7B',type=str,help="Size of the LLaMA model",choices=['7B','13B'])

parser.add_argument('--offload_dir',default=None,type=str,help="(Optional) Please specify a temp folder for offloading (useful for low-RAM machines). Default None (disable offload).")
parser.add_argument('--output_dir',default='./',type=str)
args = parser.parse_args()


assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM

BASE_MODEL = args.base_model
LORA_MODEL = args.lora_model
output_dir = args.output_dir

assert (
    BASE_MODEL
), "Please specify a BASE_MODEL in the script, e.g. 'decapoda-research/llama-7b-hf'"

tokenizer = LlamaTokenizer.from_pretrained(LORA_MODEL)
if args.offload_dir is not None:
    # Load with offloading, which is useful for low-RAM machines.
    # Note that if you have enough RAM, please use original method instead, as it is faster.
    base_model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        offload_folder=args.offload_dir,
        offload_state_dict=True,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
    )
else:
    # Original method without offloading
    base_model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )

base_model.resize_token_embeddings(len(tokenizer))
assert base_model.get_input_embeddings().weight.size(0) == len(tokenizer)
tokenizer.save_pretrained(output_dir)
print(f"Extended vocabulary size: {len(tokenizer)}")

## infer the model size from the checkpoint
emb_to_model_size = {
    4096 : '7B',
    5120 : '13B',
    6656 : '30B',
    8192 : '65B',
}
embedding_size = base_model.get_input_embeddings().weight.size(1)
model_size = emb_to_model_size[embedding_size]
print(f"Loading LoRA for {model_size} model")

lora_model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL,
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)

# merge weights
print(f"Peft version: {peft.__version__}")
print(f"Merging model")
if peft.__version__ > '0.2.0':
    # merge weights - new merging method from peft
    lora_model = lora_model.merge_and_unload()
else:
    # merge weights
    for layer in lora_model.base_model.model.model.layers:
        if hasattr(layer.self_attn.q_proj,'merge_weights'):
            layer.self_attn.q_proj.merge_weights = True
        if hasattr(layer.self_attn.v_proj,'merge_weights'):
            layer.self_attn.v_proj.merge_weights = True
        if hasattr(layer.self_attn.k_proj,'merge_weights'):
            layer.self_attn.k_proj.merge_weights = True
        if hasattr(layer.self_attn.o_proj,'merge_weights'):
            layer.self_attn.o_proj.merge_weights = True
        if hasattr(layer.mlp.gate_proj,'merge_weights'):
            layer.mlp.gate_proj.merge_weights = True
        if hasattr(layer.mlp.down_proj,'merge_weights'):
            layer.mlp.down_proj.merge_weights = True
        if hasattr(layer.mlp.up_proj,'merge_weights'):
            layer.mlp.up_proj.merge_weights = True

lora_model.train(False)

lora_model_sd = lora_model.state_dict()
del lora_model, base_model

num_shards_of_models = {'7B': 1, '13B': 2}
params_of_models = {
    '7B':
        {
        "dim": 4096,
        "multiple_of": 256,
        "n_heads": 32,
        "n_layers": 32,
        "norm_eps": 1e-06,
        "vocab_size": -1,
        },
    '13B':
        {
        "dim": 5120,
        "multiple_of": 256,
        "n_heads": 40,
        "n_layers": 40,
        "norm_eps": 1e-06,
        "vocab_size": -1,
        },
}

params = params_of_models[model_size]
num_shards = num_shards_of_models[model_size]


n_layers = params["n_layers"]
n_heads = params["n_heads"]
dim = params["dim"]
dims_per_head = dim // n_heads
base = 10000.0
inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))


def permute(w):
    return (
        w.view(n_heads, dim // n_heads // 2, 2, dim).transpose(1, 2).reshape(dim, dim)
    )


def unpermute(w):
    return (
        w.view(n_heads, 2, dim // n_heads // 2, dim).transpose(1, 2).reshape(dim, dim)
    )


def translate_state_dict_key(k):
    k = k.replace("base_model.model.", "")
    if k == "model.embed_tokens.weight":
        return "tok_embeddings.weight"
    elif k == "model.norm.weight":
        return "norm.weight"
    elif k == "lm_head.weight":
        return "output.weight"
    elif k.startswith("model.layers."):
        layer = k.split(".")[2]
        if k.endswith(".self_attn.q_proj.weight"):
            return f"layers.{layer}.attention.wq.weight"
        elif k.endswith(".self_attn.k_proj.weight"):
            return f"layers.{layer}.attention.wk.weight"
        elif k.endswith(".self_attn.v_proj.weight"):
            return f"layers.{layer}.attention.wv.weight"
        elif k.endswith(".self_attn.o_proj.weight"):
            return f"layers.{layer}.attention.wo.weight"
        elif k.endswith(".mlp.gate_proj.weight"):
            return f"layers.{layer}.feed_forward.w1.weight"
        elif k.endswith(".mlp.down_proj.weight"):
            return f"layers.{layer}.feed_forward.w2.weight"
        elif k.endswith(".mlp.up_proj.weight"):
            return f"layers.{layer}.feed_forward.w3.weight"
        elif k.endswith(".input_layernorm.weight"):
            return f"layers.{layer}.attention_norm.weight"
        elif k.endswith(".post_attention_layernorm.weight"):
            return f"layers.{layer}.ffn_norm.weight"
        elif k.endswith("rotary_emb.inv_freq") or "lora" in k:
            return None
        else:
            print(layer, k)
            raise NotImplementedError
    else:
        print(k)
        raise NotImplementedError


def save_shards(lora_model_sd, num_shards: int):
    # Add the no_grad context manager
    with torch.no_grad():
        if num_shards == 1:
            new_state_dict = {}
            for k, v in lora_model_sd.items():
                new_k = translate_state_dict_key(k)
                if new_k is not None:
                    if "wq" in new_k or "wk" in new_k:
                        new_state_dict[new_k] = unpermute(v)
                    else:
                        new_state_dict[new_k] = v

            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving shard 1 of {num_shards} into {output_dir}/consolidated.00.pth")
            torch.save(new_state_dict, output_dir + "/consolidated.00.pth")
            with open(output_dir + "/params.json", "w") as f:
                json.dump(params, f)
        else:
            new_state_dicts = [dict() for _ in range(num_shards)]
            for k in list(lora_model_sd.keys()):
                v = lora_model_sd[k]
                new_k = translate_state_dict_key(k)
                if new_k is not None:
                    if new_k=='tok_embeddings.weight':
                        print(f"Processing {new_k}")
                        assert v.size(1)%num_shards==0
                        splits = v.split(v.size(1)//num_shards,dim=1)
                    elif new_k=='output.weight':
                        print(f"Processing {new_k}")
                        splits = v.split(v.size(0)//num_shards,dim=0)

                    elif new_k=='norm.weight':
                        print(f"Processing {new_k}")
                        splits = [v] * num_shards
                    elif 'ffn_norm.weight' in new_k:
                        print(f"Processing {new_k}")
                        splits = [v] * num_shards
                    elif 'attention_norm.weight' in new_k:
                        print(f"Processing {new_k}")
                        splits = [v] * num_shards


                    elif 'w1.weight' in new_k:
                        print(f"Processing {new_k}")
                        splits = v.split(v.size(0)//num_shards,dim=0)
                    elif 'w2.weight' in new_k:
                        print(f"Processing {new_k}")
                        splits = v.split(v.size(1)//num_shards,dim=1)
                    elif 'w3.weight' in new_k:
                        print(f"Processing {new_k}")
                        splits = v.split(v.size(0)//num_shards,dim=0)


                    elif 'wo.weight' in new_k:
                        print(f"Processing {new_k}")
                        splits = v.split(v.size(1)//num_shards,dim=1)

                    elif 'wv.weight' in new_k:
                        print(f"Processing {new_k}")
                        splits = v.split(v.size(0)//num_shards,dim=0)

                    elif "wq.weight" in new_k or "wk.weight" in new_k:
                        print(f"Processing {new_k}")
                        v = unpermute(v)
                        splits = v.split(v.size(0)//num_shards,dim=0)
                    else:
                        print(f"Unexpected key {new_k}")
                        raise ValueError
                    for sd,split in zip(new_state_dicts,splits):
                        sd[new_k] = split.clone()
                        del split
                    del splits
                del lora_model_sd[k],v
                gc.collect()    # Effectively enforce garbage collection

            os.makedirs(output_dir, exist_ok=True)
            for i,new_state_dict in enumerate(new_state_dicts):
                print(f"Saving shard {i+1} of {num_shards} into {output_dir}/consolidated.0{i}.pth")
                torch.save(new_state_dict, output_dir + f"/consolidated.0{i}.pth")
            with open(output_dir + "/params.json", "w") as f:
                print(f"Saving params.json into {output_dir}/params.json")
                json.dump(params, f)


save_shards(lora_model_sd=lora_model_sd, num_shards=num_shards)
