# PyMuPDF + Tesseract + pdfplumber

import os
from typing import TypedDict, Optional, Any
from typing import Annotated
from langgraph.graph.message import add_messages

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.parsers import TesseractBlobParser
from langchain_community.tools import DuckDuckGoSearchRun


from langgraph.graph import StateGraph, END, add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", streaming=False)
search_tool = DuckDuckGoSearchRun()

# client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


class AppState(TypedDict):
    query: str
    domain: Optional[str]
    messages: Annotated[list[Any], add_messages]


def get_pdf_vector_store(pdf_dir="pdfs", index_path="faiss_index_"):
    if os.path.exists(index_path):
        return FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )

    docs = []
    for fname in os.listdir(pdf_dir):
        if fname.endswith(".pdf"):
            loader = PyMuPDFLoader(
                os.path.join(pdf_dir, fname),
                mode="page",
                images_inner_format="html-img",
                images_parser=TesseractBlobParser(),
                extract_tables="markdown",
            )
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore


vectorstore = get_pdf_vector_store()


def OrchestratorParser(state: AppState) -> AppState:
    """LLM decides if query is about coal_mining, sports, or other."""
    query = state["query"]

    prompt = ChatPromptTemplate.from_template(
        """
    You are a strict domain classifier.
    Decide if the query is about:
    - coal_mining
    - sports
    - other
    
    Query: "{query}"
    
    Respond with ONLY one of these values (no explanations):
    coal_mining, sports, other
    """
    )

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"query": query}).strip().lower()

    if result not in ("coal_mining", "sports", "other"):
        result = "other"

    state["domain"] = result

    state["messages"] = state["messages"] + [
        SystemMessage(content=f"Domain classified as: {result}")
    ]

    print("state in orchestrator:::", state)
    return state


# Nodes


def coal_mining_node(state: AppState) -> AppState:
    """RAG over PDF vectorstore"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(state["query"])

    context = "\n\n".join(d.page_content for d in docs)
    prompt = f"Answer the query using this coal-mining context:\n\n{context}\n\nQuestion: {state['query']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    ai_content = response.content if hasattr(response, "content") else str(response)
    state["messages"] = state["messages"] + [AIMessage(content=ai_content)]
    print("state in coal:::", state)

    return state


def sports_node(state: AppState) -> AppState:
    """DuckDuckGo search for sports for the sports"""
    search_result = search_tool.run(state["query"])
    prompt = f"Answer the sports query using this info:\n{search_result}\n\nQuestion: {state['query']}"
    response = llm.invoke([HumanMessage(content=prompt)])

    ai_content = response.content if hasattr(response, "content") else str(response)
    add_messages(state["messages"], [AIMessage(content=ai_content)])
    print("state in sport:::", state)

    return state


def restricted_node(state: AppState) -> AppState:
    """Reject unrelated queries."""
    msg = "This system only supports questions about sports or coal-mining."
    add_messages(state["messages"], [AIMessage(content=msg)])
    print("state in restricted:::", state)

    return state


graph = StateGraph(AppState)

# Add nodes
graph.add_node("orchestrator", OrchestratorParser)
graph.add_node("coal_mining", coal_mining_node)
graph.add_node("sports", sports_node)
graph.add_node("restricted", restricted_node)

graph.set_entry_point("orchestrator")

# Conditional routing
graph.add_conditional_edges(
    "orchestrator",
    lambda s: s["domain"],
    {
        "coal_mining": "coal_mining",
        "sports": "sports",
        "other": "restricted",
    },
)

# End edges
graph.add_edge("coal_mining", END)
graph.add_edge("sports", END)
graph.add_edge("restricted", END)

graph_executor = graph.compile()


if __name__ == "__main__":
    state = {"query": "Coal Despatch to Different Sectors in july 2025", "messages": []}
    final_state = graph_executor.invoke(state)

    for m in final_state["messages"]:
        print(m.content)
