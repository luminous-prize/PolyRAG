
import streamlit as st 
from langchain.schema import(SystemMessage, HumanMessage, AIMessage)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain_ibm import WatsonxLLM
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": os.getenv("WML_PI_KEY")
}

project_id = os.getenv("PROJECT_ID")

def get_pdf_text(pdf_path):
    loader = PyPDFDirectoryLoader(pdf_path)
    documents = loader.load()
    return documents


def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    return texts

def get_vector_store(texts):

    embeddings = HuggingFaceEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)

    return embeddings,docsearch


def get_model():

    model_id = ModelTypes.GRANITE_13B_CHAT_V2

    parameters = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MIN_NEW_TOKENS: 30,
        GenParams.MAX_NEW_TOKENS: 250,
        GenParams.STOP_SEQUENCES: ["<|endoftext|>"]
    }

    watsonx_granite = WatsonxLLM(
        model_id=model_id.value,
        url=credentials.get("url"),
        apikey=credentials.get("apikey"),
        project_id=project_id,
        params=parameters
    )

    return watsonx_granite

def conversation_chain(docsearch,watsonx_granite,input_question):

    qa = RetrievalQA.from_chain_type(llm=watsonx_granite, chain_type="stuff", retriever=docsearch.as_retriever())

    answer = qa.invoke(input_question)

    return answer['result']


def init_page():
  
  st.set_page_config(
    page_title="ChatBot",
    page_icon=":gem:",
    initial_sidebar_state="collapsed"
    )
  st.header("ChatBot")
  st.sidebar.title("History")


def init_messages():
  clear_button = st.sidebar.button("Clear Conversation", key="clear")
  if clear_button or "messages" not in st.session_state:
    st.session_state.messages = [
      SystemMessage(
        content="You are a helpful AI assistant. Reply your answer in markdown format."
      )
    ]


def main():

    init_page()
    init_messages()

    pdf_path = "/Documents/"

    #pdf -> extract text -> divided into chunks -> taking those chunks & converting tnto vectors, similarity searhc using faiss only
    documents = get_pdf_text(pdf_path)
    texts = get_text_chunks(documents)
    embeddings,docsearch = get_vector_store(texts)
    watsonx_granite = get_model()

    if user_input := st.chat_input("What would you like to know about ?"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Bot is typing ..."):

            answer = conversation_chain(docsearch,watsonx_granite,user_input)
            print(answer)

        st.session_state.messages.append(AIMessage(content=answer))
        

    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)


if __name__ == "__main__":
  main()
 
