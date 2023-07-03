"""
Usage: 
python merge_llama_with_chinese_lora_low_mem.py \
    --base_model path/to/llama/model \
    --lora_model path/to/first/lora[,path/to/second/lora] \
    --output_type [pth|huggingface] \
    --output_dir path/to/output/dir
"""
import argparse
import json
import os
import gc
import torch
import peft
from transformers import LlamaTokenizer
from transformers.modeling_utils import dtype_byte_size
from huggingface_hub import snapshot_download
import re

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, required=True,
                    type=str, help="Please specify a base model")
parser.add_argument('--lora_model', default=None, required=True,
                    type=str, help="Please specify LoRA models to be merged (ordered); use commas to separate multiple LoRA models")
parser.add_argument('--output_type', default='pth',choices=['pth','huggingface'],
                    type=str, help="Save the merged model in pth or huggingface format")
parser.add_argument('--output_dir', default='./merged_model',
                    type=str, help="The output folder to save the merged model")
parser.add_argument('--verbose', default=False, action='store_true',
                    help="Show detailed messages")


emb_to_model_size = {
    4096 : '7B',
    5120 : '13B',
    6656 : '33B',
    8192 : '65B',
}
num_shards_of_models = {'7B': 1, '13B': 2, '33B': 4, '65B': 8}
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
    '33B':
        {
        "dim": 6656,
        "multiple_of": 256,
        "n_heads": 52,
        "n_layers": 60,
        "norm_eps": 1e-06,
        "vocab_size": -1,
        },
    '65B':
        {
        "dim": 8192,
        "multiple_of": 256,
        "n_heads": 64,
        "n_layers": 80,
        "norm_eps": 1e-05,
        "vocab_size": -1,
        },
}

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

# Borrowed and modified from https://github.com/tloen/alpaca-lora
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


def unpermute(w):
    return (
        w.view(n_heads, 2, dim // n_heads // 2, dim).transpose(1, 2).reshape(dim, dim)
    )


def save_shards(model_sd, num_shards: int, prefix="", verbose=False):
    """
    Convert and save the HF format weights to PTH format weights
    """
    with torch.no_grad():
        if num_shards == 1:
            new_state_dict = {}
            for k, v in model_sd.items():
                new_k = translate_state_dict_key(k)
                if new_k is not None:
                    if "wq" in new_k or "wk" in new_k:
                        new_state_dict[new_k] = unpermute(v)
                    else:
                        new_state_dict[new_k] = v

            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving shard 1 of {num_shards} into {output_dir}/{prefix}consolidated.00.pth")
            torch.save(new_state_dict, output_dir + f"/{prefix}consolidated.00.pth")
        else:
            new_state_dicts = [dict() for _ in range(num_shards)]
            for k in list(model_sd.keys()):
                v = model_sd[k]
                new_k = translate_state_dict_key(k)
                if new_k is not None:
                    if new_k=='tok_embeddings.weight':
                        assert v.size(1)%num_shards==0
                        splits = v.split(v.size(1)//num_shards,dim=1)
                    elif new_k=='output.weight':
                        if v.size(0)%num_shards==0:
                            splits = v.split(v.size(0)//num_shards,dim=0)
                        else:
                            size_list = [v.size(0)//num_shards] * num_shards
                            size_list[-1] += v.size(0)%num_shards
                            splits = v.split(size_list, dim=0) # 13B: size_list == [24976,24977]
                    elif new_k=='norm.weight':
                        splits = [v] * num_shards
                    elif 'ffn_norm.weight' in new_k:
                        splits = [v] * num_shards
                    elif 'attention_norm.weight' in new_k:
                        splits = [v] * num_shards


                    elif 'w1.weight' in new_k:
                        splits = v.split(v.size(0)//num_shards,dim=0)
                    elif 'w2.weight' in new_k:
                        splits = v.split(v.size(1)//num_shards,dim=1)
                    elif 'w3.weight' in new_k:
                        splits = v.split(v.size(0)//num_shards,dim=0)


                    elif 'wo.weight' in new_k:
                        splits = v.split(v.size(1)//num_shards,dim=1)

                    elif 'wv.weight' in new_k:
                        splits = v.split(v.size(0)//num_shards,dim=0)

                    elif "wq.weight" in new_k or "wk.weight" in new_k:
                        v = unpermute(v)
                        splits = v.split(v.size(0)//num_shards,dim=0)
                    else:
                        print(f"Unexpected key {new_k}")
                        raise ValueError
                    if verbose:
                        print(f"Processing {new_k}")
                    for sd,split in zip(new_state_dicts,splits):
                        sd[new_k] = split.clone()
                        del split
                    del splits
                del model_sd[k],v
                gc.collect()    # Effectively enforce garbage collection

            os.makedirs(output_dir, exist_ok=True)
            for i,new_state_dict in enumerate(new_state_dicts):
                print(f"Saving shard {i+1} of {num_shards} into {output_dir}/{prefix}consolidated.0{i}.pth")
                torch.save(new_state_dict, output_dir + f"/{prefix}consolidated.0{i}.pth")

def merge_shards(output_dir, num_shards: int):
    ckpt_filenames = sorted([f for f in os.listdir(output_dir) if re.match('L(\d+)-consolidated.(\d+).pth',f)])

    for i in range(num_shards):
        shards_filenames = sorted([f for f in ckpt_filenames if re.match(f'L(\d+)-consolidated.0{i}.pth',f)])
        print(f"Loading {shards_filenames} ...")
        shards_dicts = [torch.load(os.path.join(output_dir,fn)) for fn in shards_filenames]
        shards_merged = {}
        for d in shards_dicts:
            shards_merged |= d

        print(f"Saving the merged shard to " + os.path.join(output_dir, f"consolidated.0{i}.pth"))
        torch.save(shards_merged, os.path.join(output_dir, f"consolidated.0{i}.pth"))

        print("Cleaning up...")
        del shards_merged
        for d in shards_dicts:
            del d
        del shards_dicts
        gc.collect()    # Effectively enforce garbage collection
        for fn in shards_filenames:
            os.remove(os.path.join(output_dir,fn))

if __name__=='__main__':

    args = parser.parse_args()
    base_model_path = args.base_model
    lora_model_paths = [s.strip() for s in args.lora_model.split(',') if len(s.strip())!=0]
    output_dir = args.output_dir
    output_type = args.output_type
    os.makedirs(output_dir, exist_ok=True)

    print(f"Base model: {base_model_path}")
    print(f"LoRA model(s) {lora_model_paths}:")

    tokenizers_and_loras = []
    for lora_model_path in lora_model_paths:
        print(f"Loading {lora_model_path}")
        if not os.path.exists(lora_model_path):
            print("Cannot find lora model on the disk. Downloading lora model from hub...")
            lora_model_path = snapshot_download(repo_id=lora_model_path)
        tokenizer = LlamaTokenizer.from_pretrained(lora_model_path)
        lora_config = peft.LoraConfig.from_pretrained(lora_model_path)
        lora_state_dict = torch.load(os.path.join(lora_model_path,'adapter_model.bin'),map_location='cpu')
        if 'base_model.model.model.embed_tokens.weight' in lora_state_dict:
            lora_vocab_size = lora_state_dict['base_model.model.model.embed_tokens.weight'].shape[0]
            assert lora_vocab_size==len(tokenizer), \
            (f"The vocab size of the tokenizer {len(tokenizer)} does not match the vocab size of the LoRA weight {lora_vocab_size}.\n"
            "Make sure that you use LLaMA tokenizer with the LLaMA-LoRA weight and Alpaca tokenizer with the Alpaca-LoRA weight!")
        tokenizers_and_loras.append(
            {
                "tokenizer"  :tokenizer,
                "state_dict" :lora_state_dict,
                "config": lora_config,
                "scaling": lora_config.lora_alpha / lora_config.r,
                "fan_in_fan_out" : lora_config.fan_in_fan_out,
            })
    if len(tokenizers_and_loras)==2:
        t1_vocab_size = len(tokenizers_and_loras[0]["tokenizer"])
        t2_vocab_size = len(tokenizers_and_loras[1]["tokenizer"])
        assert t1_vocab_size<=t2_vocab_size, \
        (f"The vocab size of the first tokenizer is {t1_vocab_size}\n"
         f"The vocab size of the second tokenizer is {t2_vocab_size}, found to be smaller than {t1_vocab_size}\n"
        "This is not the intended use. Please check your model and tokenizer.")

    if not os.path.exists(base_model_path):
        print("Cannot find lora model on the disk. Downloading lora model from hub...")
        base_model_path = snapshot_download(repo_id=base_model_path)
    ckpt_filenames = sorted([f for f in os.listdir(base_model_path) if re.match('pytorch_model-(\d+)-of-(\d+).bin',f)])

    embedding_size = None
    model_size = None


    total_size = 0
    for index, filename in enumerate(ckpt_filenames):
        print(f"Loading ckpt {filename}")
        state_dict = torch.load(os.path.join(base_model_path,filename), map_location='cpu')
        if index == 0:
            embedding_size = state_dict['model.embed_tokens.weight'].shape[1]
            model_size = emb_to_model_size[embedding_size]
            if output_type=='pth':
                params = params_of_models[model_size]
                num_shards = num_shards_of_models[model_size]
                n_layers = params["n_layers"]
                n_heads = params["n_heads"]
                dim = params["dim"]
                dims_per_head = dim // n_heads
                base = 10000.0
                inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
        print("Merging...")
        for k in state_dict:
            for tl_idx, t_and_l in enumerate(tokenizers_and_loras):
                saved_key = 'base_model.model.'+k
                lora_key_A = saved_key.replace('.weight','.lora_A.weight')
                if saved_key in t_and_l['state_dict']:
                    if args.verbose:
                        print(f"copying {saved_key} from {tl_idx}-th LoRA weight to {k}")
                    state_dict[k] = t_and_l['state_dict'][saved_key].half().clone() # do we need half()?
                if lora_key_A in t_and_l['state_dict']:
                    lora_key_B = lora_key_A.replace('lora_A.weight','lora_B.weight')
                    if args.verbose:
                        print(f"merging {lora_key_A} and lora_B.weight form {tl_idx}-th LoRA weight to {k}")
                    state_dict[k] += (
                        transpose(
                            t_and_l['state_dict'][lora_key_B].float()
                          @ t_and_l['state_dict'][lora_key_A].float(), t_and_l['fan_in_fan_out']) * t_and_l['scaling']
                    )
            weight_size = state_dict[k].numel() * dtype_byte_size(state_dict[k].dtype)
            total_size += weight_size

        if output_type=='huggingface':
            print(f"Saving ckpt {filename} to {output_dir} in HF format...")
            torch.save(state_dict,os.path.join(output_dir, filename))
        elif output_type=='pth':
            print(f"Converting to pth format...")
            save_shards(model_sd=state_dict, num_shards=num_shards,prefix=f"L{index+1}-", verbose=args.verbose)
        del state_dict
        gc.collect()    # Effectively enforce garbage collection


    print(f"Saving tokenizer")
    tokenizers_and_loras[-1]['tokenizer'].save_pretrained(output_dir)
    if output_type == 'pth':
        with open(output_dir + "/params.json", "w") as f:
            print(f"Saving params.json into {output_dir}/params.json")
            json.dump(params, f)
        merge_shards(output_dir, num_shards=num_shards)

    if output_type=='huggingface':
        configs = ('config.json', 'generation_config.json', 'pytorch_model.bin.index.json')
        for config in configs:
            if os.path.exists(os.path.join(base_model_path, config)):
                print(f"Saving {config}")
                with open(os.path.join(base_model_path, config),'r') as f:
                    obj = json.load(f)
                if config=='config.json':
                    obj['vocab_size'] = len(tokenizers_and_loras[-1]['tokenizer'])
                if config=='pytorch_model.bin.index.json':
                    obj['metadata']['total_size'] = total_size
                with open(os.path.join(output_dir, config), 'w') as f:
                    json.dump(obj, f, indent=2)
    print("Done.")
