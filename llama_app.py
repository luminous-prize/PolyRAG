from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain.chains.question_answering import load_qa_chain
import os
import sys
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import faiss
from sentence_transformers import SentenceTransformer, util
import time
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain

import warnings
warnings.filterwarnings('ignore')

def get_pdf_text(pdf_path):
    loader = PyPDFDirectoryLoader(pdf_path)
    data = loader.load()
    return data


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    chunk_embeddings = []
    
    for text_chunk in chunks:
        # Generate embeddings for each text chunk using the embeddings model
        chunk_embedding = embeddings.embed_query(text_chunk.page_content)
        chunk_embeddings.append(chunk_embedding)

    # Convert embeddings to numpy array
    chunk_embeddings_np = np.array(chunk_embeddings)

    index = faiss.IndexFlatL2(chunk_embeddings_np.shape[1]) 

    index.add(chunk_embeddings_np.astype(np.float32))

    return index




def get_conversational_chain():

    prompt_template = """<s>[INST] <<SYS>>
    {{Use the following pieces of context to answer the question at the end.If you don't know the answer, just say that you don't know, don't try to make up an answer.}}<</SYS>>
    ###

    Previous Conversation:
    '''
    {history}
    '''

    {{{input}}}[/INST]

    """

    model = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=1000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0.3})

    prompt = PromptTemplate(template=prompt_template, input_variables=['input', 'history'])

    chain = ConversationChain(llm=llm, prompt=prompt)

    return chain


def user_input(question, index, chain):

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

    query_embedding = embeddings.embed_query(question)

    query_embedding_array = np.array([query_embedding])

    query_embedding_array = query_embedding_array.astype(np.float32)

    D, I = index.search(query_embedding_array, k=1)

    similar_documents = [docs[i] for i in I[0]]
    docs = similar_documents[0].page_content
    
    ip = f"Context : {docs} . Question: {question} "

    return chain.run(ip)


def main():

    pdf_path = "/documentation/"

    # pdf -> extract text -> divided into chunks -> taking those chunks & converting tnto vectors, similarity searhc using faiss only
    raw_text = get_pdf_text(pdf_path)
    text_chunks = get_text_chunks(raw_text)
    print('Chunk 1 is :',text_chunks[0].page_content)
    index = get_vector_store(text_chunks)
    chain = get_conversational_chain()


    while True:
        input_question=input(f"prompt:")
        if input_question=='exit':
            print('Exiting')
            sys.exit()
        if input_question=='':
            continue
        result=user_input(input_question,index,chain)
        print(f"Answer:{result}")


if __name__ == "__main__":
    main()
