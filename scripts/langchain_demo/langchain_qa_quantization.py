import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--file_path',required=True,type=str)
parser.add_argument('--model_path',required=True,type=str)

args = parser.parse_args()
file_path = args.file_path
model_path = args.model_path

import torch
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from fuzzywuzzy import fuzz
def get_prompt(reranked_content, question):
    chat_prompt = """
You will be provided some context from a document.
Based on this context, answer the user question.
Only answer based on the given context.
the answer which with profound viewpoints and sufficient facts.
If you cannot answer, say 'I don't know' and recommend a different question.
If the question is in Chinese, please also answer it in Chinese.

"""
    message = f'{chat_prompt} ---------- Context: {reranked_content} -------- User Question: {question} ---------- Response:'
    return message

def edit_distance(query, topkdocs):
    scores_and_docs = []
    for doc in topkdocs:
        score = fuzz.partial_ratio(query, doc.page_content)
        scores_and_docs.append((score, doc))
    scores_and_docs.sort(key=lambda x: x[0], reverse=True)
    reranked = [doc for score, doc in scores_and_docs]
    return reranked
def get_doc_source(context):
    # return the doc source of the context
    doc_source_strs = []
    for doc in context:
        source_name = doc.metadata['source'].split('\\')[-1]
        source_content = doc.page_content.replace('\n', ' ')
        formatted_string = f"|||出自: {source_name} \n\n||| 原文: '{source_content}'"
        doc_source_strs.append(formatted_string)
    retrieval_string = ('--'*50 + "\n\n").join(doc_source_strs)
    return retrieval_string

if __name__ == '__main__':
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    print("Loading the embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name='GanymedeNil/text2vec-large-chinese')
    docsearch = FAISS.from_documents(texts, embeddings)

    print("loading LLM...")
    llm = LlamaCpp(model_path=model_path, n_ctx=2048)
    n = 0
    while True:
        query = input("请输入问题：\n\n>")
        if len(query.strip())==0:
            n += 1
            print("连续三次【无】问题输入自动退出\n\n")
            if n == 3:
                print("No query, exit, now")
                break
        else:
            print("Query with similarity search...")
            topkdocs = docsearch.similarity_search(query, k=10)
            # then find "edit distance", so that look like origin raw
            reranked_docs = edit_distance(query, topkdocs)
            reranked_content = [result.page_content for result in reranked_docs]
            # context, up & below, called context, so num is 2.
            context = reranked_content[:2]
            prompt = get_prompt(context, query)
            retrieval_strings = get_doc_source(reranked_docs[:2])
            print("分析中...")
            output = llm(prompt)
            print(output)
            print("*" * 100)
            print("以上结论出自\n\n")
            print(retrieval_strings)
            print("*" * 100)
            print('\n* 本次检索结束 *\n')