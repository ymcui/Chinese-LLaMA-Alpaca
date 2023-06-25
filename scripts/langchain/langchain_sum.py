import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--file_path',required=True,type=str)
parser.add_argument('--model_path',required=True,type=str)
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--chain_type', default="refine", type=str)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
file_path = args.file_path
model_path = args.model_path

import torch
from langchain import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

prompt_template = ("Below is an instruction that describes a task. "
                   "Write a response that appropriately completes the request.\n\n"
                   "### Instruction:\n请为以下文字写一段摘要:\n{text}\n\n### Response: ")
refine_template = (
    "Below is an instruction that describes a task."
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "已有一段摘要：{existing_answer}\n"
    "现在还有一些文字，（如果有需要）你可以根据它们完善现有的摘要。"
    "\n"
    "{text}\n"
    "\n"
    "如果这段文字没有用，返回原来的摘要即可。请你生成一个最终的摘要。"
    "\n\n### Response: "
)


if __name__ == '__main__':
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100, length_function=len)
    with open(file_path) as f:
        text = f.read()
    docs = text_splitter.create_documents([text])

    print("loading LLM...")
    model = HuggingFacePipeline.from_model_id(model_id=model_path,
            task="text-generation",
            model_kwargs={
                          "torch_dtype" : load_type,
                          "low_cpu_mem_usage" : True,
                          "temperature": 0.2,
                          "max_length": 1000,
                          "device_map": "auto",
                          "repetition_penalty":1.1}
            )

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    REFINE_PROMPT = PromptTemplate(
        template=refine_template,input_variables=["existing_answer", "text"],
    )

    if args.chain_type == "stuff":
        chain = load_summarize_chain(model, chain_type="stuff", prompt=PROMPT)
    elif args.chain_type == "refine":
        chain = load_summarize_chain(model, chain_type="refine", question_prompt=PROMPT, refine_prompt=REFINE_PROMPT)
    print(chain.run(docs))
