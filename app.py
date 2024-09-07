import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import os

# Define paths to save and load the vectorstore
VECTORSTORE_PATH = "vectorstore.faiss"
PDF_DIRECTORY = "PDFs"  # Change this to your PDF directory

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks=None):
    if os.path.exists(VECTORSTORE_PATH):
        # Load existing vectorstore from disk
        vectorstore = FAISS.load_local(VECTORSTORE_PATH)
    else:
        # Create vectorstore if it doesn't exist
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        # Save the vectorstore to disk
        vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatGroq(temperature=0, max_tokens=500)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Buzz", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    # Load or create vectorstore and conversation chain only once
    if st.session_state.conversation is None:
        with st.spinner("Setting up conversation chain..."):
            # Load PDFs from the specified directory
            pdf_files = [os.path.join(PDF_DIRECTORY, f) for f in os.listdir(PDF_DIRECTORY) if f.endswith('.pdf')]
            raw_text = get_pdf_text(pdf_files)  # Get text from PDFs
            text_chunks = get_text_chunks(raw_text)  # Get text chunks
            vectorstore = get_vectorstore(text_chunks)  # Create or load vectorstore
            st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
