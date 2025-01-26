import streamlit as st
import sys
import numpy as np
import faiss
import torch
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
import sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os
import uuid
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_pdf_text(pdf_path):
    loader = PyPDFDirectoryLoader(pdf_path)
    data = loader.load()
    return data


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    chunk_embeddings = []

    for text_chunk in chunks[:300]:
        # Generate embeddings for each text chunk using the embeddings model
        chunk_embedding = embeddings.embed_query(text_chunk.page_content)
        chunk_embeddings.append(chunk_embedding)

    # Convert embeddings to numpy array
    chunk_embeddings_np = np.array(chunk_embeddings)

    index = faiss.IndexFlatL2(chunk_embeddings_np.shape[1])

    index.add(chunk_embeddings_np.astype(np.float32))

    return index


def get_small_model():

    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    return tokenizer, model

def conversational_chain(rel_docs, tokenizer, model, input_question):
    instruction = f"Instruction: Only give the AI answer in response. Use the information in Context to answer the Question: {input_question}"

    user_query = instruction + " " + "Context: " + rel_docs

    device = torch.device('cpu')
    tokenized_text = tokenizer.encode(user_query, return_tensors="pt").to(device)

    summary_ids = model.generate(tokenized_text,
                                 num_beams=14,
                                 no_repeat_ngram_size=5,
                                 min_length=30,
                                 max_length=100,
                                 early_stopping=False,do_sample=True)
    
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return output


def user_input(question, index,text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

    query_embedding = embeddings.embed_query(question)

    query_embedding_array = np.array([query_embedding])

    query_embedding_array = query_embedding_array.astype(np.float32)

    D, I = index.search(query_embedding_array, k=2)

    similar_documents = [text_chunks[i] for i in I[0]]
    docs1 = similar_documents[0].page_content
    docs2 = similar_documents[1].page_content
    docs = docs1 + " " + docs2

    return docs


def main():

    pdf_path = "/Documents/"

    #pdf -> extract text -> divided into chunks -> taking those chunks & converting tnto vectors, similarity searhc using faiss only
    raw_text = get_pdf_text(pdf_path)
    text_chunks = get_text_chunks(raw_text)
    index = get_vector_store(text_chunks)
    tokenizer, model = get_small_model()

    st.set_page_config(page_title="IBM's Ceph ChatBot",page_icon=":gem:")
    st.header("IBM Ceph Chatbot")

    input_question = st.text_input("What would you like to know about Ceph ?")
    if input_question:
        rel_docs = user_input(input_question, index, text_chunks)
        result = conversational_chain(rel_docs, tokenizer, model, input_question)
        st.write("", result)

if __name__ == "__main__":
    main()
