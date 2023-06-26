import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--file_path',required=True,type=str)
parser.add_argument('--embedding_path',required=True,type=str)
parser.add_argument('--model_path',required=True,type=str)
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--chain_type', default="refine", type=str)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
file_path = args.file_path
embedding_path = args.embedding_path
model_path = args.model_path

import torch
from langchain import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

prompt_template = ("Below is an instruction that describes a task. "
                   "Write a response that appropriately completes the request.\n\n"
                   "### Instruction:\n{context}\n{question}\n\n### Response: ")


refine_prompt_template = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "这是原始问题: {question}\n"
    "已有的回答: {existing_answer}\n"
    "现在还有一些文字，（如果有需要）你可以根据它们完善现有的回答。"
    "\n\n"
    "{context_str}\n"
    "\\nn"
    "请根据新的文段，进一步完善你的回答。\n\n"
    "### Response: "
)

initial_qa_template = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "以下为背景知识：\n"
    "{context_str}"
    "\n"
    "请根据以上背景知识, 回答这个问题：{question}。\n\n"
    "### Response: "
)


if __name__ == '__main__':
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    print("Loading the embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_path)
    docsearch = FAISS.from_documents(texts, embeddings)

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

    if args.chain_type == "stuff":
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": PROMPT}
        qa = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=docsearch.as_retriever(search_kwargs={"k": 1}),
            chain_type_kwargs=chain_type_kwargs)

    elif args.chain_type == "refine":
        refine_prompt = PromptTemplate(
            input_variables=["question", "existing_answer", "context_str"],
            template=refine_prompt_template,
        )
        initial_qa_prompt = PromptTemplate(
            input_variables=["context_str", "question"],
            template=initial_qa_template,
        )
        chain_type_kwargs = {"question_prompt": initial_qa_prompt, "refine_prompt": refine_prompt}
        qa = RetrievalQA.from_chain_type(
            llm=model, chain_type="refine",
            retriever=docsearch.as_retriever(search_kwargs={"k": 1}),
            chain_type_kwargs=chain_type_kwargs)

    while True:
        query = input("请输入问题：")
        if len(query.strip())==0:
            break
        print(qa.run(query))
