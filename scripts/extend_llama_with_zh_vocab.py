'''
NOTE: As we have simplified the mergeing steps,
this script is no longer needed and will be removed in the next release.
'''
from transformers import LlamaForCausalLM, LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--llama_model',default=None,required=True,type=str)
parser.add_argument('--tokenizer',default=None,required=True,type=str)
parser.add_argument('--output_dir',default=None,required=True,type=str)

DEFAULT_PAD_TOKEN = '[PAD]'

if __name__ == '__main__':
    args = parser.parse_args()

    model = LlamaForCausalLM.from_pretrained(args.llama_model,torch_dtype="auto")
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer)
    model.resize_token_embeddings(len(tokenizer))

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    assert model.get_input_embeddings().weight.size(0) == len(tokenizer)
    print(f"Extended vocabulary size: {len(tokenizer)}")
