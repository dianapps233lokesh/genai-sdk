## this project is related to chat with multipe pdf docs with langchain and google gemini pro

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import asyncio
from dotenv import load_dotenv
## Tools import
from langchain.agents import create_react_agent,AgentExecutor
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub
from langchain import hub
from src.applications.constant import LLM_PROMPT
from langchain.callbacks.streamlit import StreamlitCallbackHandler


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
                    metadata={'page': page_num, 'source': pdf.name}))
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




def get_conversational_chain(st_callback):
    prompt_template = LLM_PROMPT
    model=ChatGoogleGenerativeAI(model="gemini-2.5-pro",streaming=True,callbacks=[st_callback])
    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type='stuff',prompt=prompt)
    return chain


@tool("pdf_rag_tool")
def rag_tool(user_question:str):
    """Always use this tool FIRST when answering questions.
    It retrieves answers from the uploaded PDF documents with page citations.
    Only use web search if the answer is NOT found in the PDFs."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_db=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_question)

    st_callback = StreamlitCallbackHandler(st.container())        #live display
    chain=get_conversational_chain(st_callback)
    
    response=chain.invoke(
            {"input_documents":docs,"question":user_question}, return_only_outputs=True
        )
    return response['output_text']

search_tool=DuckDuckGoSearchRun()

def get_agent_executor(tools,prompt,st_callback):
    llm=ChatGoogleGenerativeAI(model="gemini-2.5-pro",streaming=True,callbacks=[st_callback])

    #create react agent
    agent=create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    #wrap into executor(the thing that we actually call)
    agent_executor=AgentExecutor(agent=agent,tools=tools,handle_parsing_errors=True)
    return agent_executor


def user_input(user_question):
    st_callback = StreamlitCallbackHandler(st.container())

    prompt = hub.pull("hwchase17/react")
    tools=[rag_tool,search_tool]

    agent_executor=get_agent_executor(tools=tools,prompt=prompt,st_callback=st_callback)

    response=agent_executor.invoke({"input":user_question})
    st.write("Reply:",response['output'])
    

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