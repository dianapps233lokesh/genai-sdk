## this project is related to chat with multipe pdf docs with langchain and google gemini pro


import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import asyncio
from dotenv import load_dotenv

load_dotenv()

client=genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text_metadata(pdf_docs):
    documents = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            text = page.extract_text()      #extracts the text from page
            if text:  # Ensure there is text on the page
                documents.append(Document(
                    page_content=text,
                    metadata={'page': page_num, 'source': pdf.name}
                ))
    return documents


def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
    chunks=text_splitter.split_documents(text)
    return chunks


def  get_vector_store(text_chunks):
    async def run():
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    asyncio.run(run())




def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    For each piece of information you provide, you MUST cite the source PDF and the page number(s) from which it was sourced.
    Format your citations clearly at the end of each relevant sentence, for example: (Source: [PDF Name], Page: 5).
    If the context includes diagrams or tables, mention them in your answer if they are relevant.
    If the answer is not available in the provided context, you must say, "The answer is not available in the provided documents." Do not provide an incorrect answer.

    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    model=ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type='stuff',prompt=prompt)
    return chain

def user_input(user_question):
    async def run():
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        new_db=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
        docs=new_db.similarity_search(user_question)
        chain=get_conversational_chain()

        response=chain.invoke(
            {"input_documents":docs,"question":user_question}, return_only_outputs=True
        )

        st.write("Reply:",response['output_text'])
    asyncio.run(run())

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with multiple PDF using Gemini...")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_docs = get_pdf_text_metadata(pdf_docs)
                    text_chunks = get_text_chunks(raw_docs)
                    get_vector_store(text_chunks)
                    st.success("Done")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__=="__main__":
    main()