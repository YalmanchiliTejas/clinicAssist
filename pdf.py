import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
import pickle

# Load environment variables from a .env file
load_dotenv()

@st.cache_data
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

@st.cache_data
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

@st.cache_resource
def get_conversation_chain(vector_store):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"max_length": 512, "temperature": 0.5})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation

def handleInput(question):
    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response['chat_history']
    for message in st.session_state.chat_history:
        st.write(message.content)

def main():
    st.set_page_config(page_title="PDF Information", page_icon=":books:")
    st.header("Chat with Multiple PDFs :books:")

    # Initialize session state variables if not already set
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Main area for asking questions once processing is done
    question = st.text_input("Question?")
    if question and st.session_state.conversation:
        handleInput(question)
    
    # Sidebar for document upload and processing
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your documents here", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_texts = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_texts)
                # Optionally, persist the chunks for debugging
                pickle.dump(text_chunks, open("text_chunks.pkl", "wb"))
                vector_store = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vector_store)
                st.success("Document processed successfully!")

if __name__ == "__main__":
    main()
