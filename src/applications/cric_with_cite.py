import os
from typing import TypedDict, Optional, Any, List
from typing import Annotated

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.parsers import TesseractBlobParser

from langgraph.graph import StateGraph, END, add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

# ------------------------------
# Embeddings, LLM, Search
# ------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", model_kwargs={"streaming": False}
)
search_tool = TavilySearchResults()


# ------------------------------
# App State
# ------------------------------
class AppState(TypedDict):
    query: str
    domain: Optional[str]  # sports / other
    domain_figure: Optional[str]  # virat / rohit / other / invalid
    messages: Annotated[list[Any], add_messages]
    user_preference: Optional[str]


# ------------------------------
# PDF Vector Store
# ------------------------------
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


# ------------------------------
# Orchestrator: Sports / Other
# ------------------------------
def OrchestratorParser(state: AppState) -> AppState:
    """Classifies query as sports/other, considering previous context for follow-ups."""
    query = state["query"]
    previous_domain = state.get("domain") or "other"

    prompt = ChatPromptTemplate.from_template(
    """
    You are a strict domain classifier.

    Rules:
    1. Classify as "sports" if the query is about:
       - Any sport, sports event, player, team, tournament, score, or match
       - OR about a sports figure’s life, background, travel, family, or personal details
    2. Otherwise, classify as "other".
    3. If the query is vague, a follow-up, or depends on previous conversation context,
       you should consider the previous domain to decide.

    Previous domain: "{previous_domain}"
    Current query: "{query}"

    Respond with ONLY one of these values:
    sports, other
    """
)

    chain = prompt | llm | StrOutputParser()
    result = (
        chain.invoke({"query": query, "previous_domain": previous_domain})
        .strip()
        .lower()
    )

    if result not in ("sports", "other"):
        # fallback to previous domain
        result = previous_domain

    state["domain"] = result
    state["messages"] += [SystemMessage(content=f"Domain classified as: {result}")]
    return state


# ------------------------------
# Sports Figure Detector
# ------------------------------
def SportsFigureParser(state: AppState) -> AppState:
    """Detect if query is about Virat, Rohit, another player, or invalid.
    Uses previous figure and conversation context for follow-ups.
    """
    query = state["query"]
    previous_figure = state.get("domain_figure")  # previous sports figure if any

    prompt = ChatPromptTemplate.from_template(
    """
    You are a strict sports figure detector.

    Rules:
    1. If the query explicitly mentions Virat Kohli → respond exactly "virat".
    2. If the query explicitly mentions Rohit Sharma → respond exactly "rohit".
    3. If the query explicitly mentions another named sports player 
       (e.g., Messi, Dhoni, Federer) → respond exactly "other".
    4. If the query refers to a sports player generically 
       (e.g., "a football player", "the captain", "his wife"):
       - Use the previous sports figure if available.
       - Otherwise, respond "other".
    5. If the query is not about sports figures at all → respond "invalid".
    6. Always prioritize continuity with the previous figure for follow-up queries.

    Previous sports figure: "{previous_figure}"
    Previous query/messages: "{messages}"
    Current query: "{query}"

    Respond with ONLY one of these values:
    virat, rohit, other, invalid
    """
)


    chain = prompt | llm | StrOutputParser()
    result = (
        chain.invoke(
            {
                "query": query,
                "previous_figure": previous_figure,
                "messages": "\n".join([m.content for m in state["messages"]]),
            }
        )
        .strip()
        .lower()
    )

    if result not in ("virat", "rohit", "other", "invalid"):
        # fallback to previous figure if possible
        result = previous_figure if previous_figure else "invalid"

    state["domain_figure"] = result
    state["messages"] += [
        SystemMessage(content=f"Sports figure classified as: {result}")
    ]

    return state


# ------------------------------
# Nodes
# ------------------------------
def invalid_sports_node(state: AppState) -> AppState:
    msg = "⚠️ Your query is not about a valid sports figure. Please ask about a real player."
    state["messages"] += [AIMessage(content=msg)]
    # print(f"State in the invalid node is {state}\n\n*******************\n\n")
    return state


def sports_rag_node(state: AppState) -> AppState:
    query = state["query"]
    figure = state["domain_figure"]

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)

    context_with_citations = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata.get("source", f"doc{i}")
        context_with_citations.append(f"{d.page_content}\n[Source: {meta}]")

    context = "\n\n".join(context_with_citations)

    # Incorporate previous conversation for context
    conversation_history = "\n".join([m.content for m in state["messages"]])

    prompt = f"""
        You are a precise and factual assistant answering questions about {figure.title()}.

        Instructions:
        1. Consider the entire conversation so far to understand the context and follow-up queries:
        {conversation_history}

        2. Check the provided knowledge base context for relevant information:
        {context}

        3. If the context contains the answer, use it to respond.
        - Include [Source] tags exactly as they appear in the context.
        - Do not add information not present in the context.

        4. If the context does not contain the answer:
        - Provide a concise, factual answer based on your general knowledge about {figure.title()}.
        - Do not invent sources or fabricate information.

        5. Never hallucinate, speculate, or provide irrelevant information.
        6. Always respond clearly and concisely, in complete sentences.

        Question:
        {query}

        Answer:
        """

    response = llm.invoke([HumanMessage(content=prompt)])
    ai_content = response.content if hasattr(response, "content") else str(response)

    state["messages"] += [AIMessage(content=ai_content)]
    return state


def sports_web_node(state: AppState) -> AppState:
    query = state["query"]
    search_result = search_tool.invoke(query)
    print("Response from the duckduckgo is -----------",search_result)

    # Include previous conversation for context
    conversation_history = "\n".join([m.content for m in state["messages"]])

    prompt = f"""
        You are a factual assistant answering questions about a sports figure.

        Instructions:
        1. Consider the entire conversation so far to understand the context and follow-up queries:
        {conversation_history}

        2. Use the following DuckDuckGo search results to answer the current question:
        {search_result}

        3. At the end of your response, include [Source: DuckDuckGo].

        4. Provide a clear, concise answer.
        5. If the search results do not clarify, politely say so.
        6. Do NOT hallucinate or invent information.
        7. Always respond clearly and concisely, in complete sentences.

        Current Question:
        {query}

        Answer:
        """

    response = llm.invoke([HumanMessage(content=prompt)])
    # print("Response from the duckduckgo is -----------",response)
    ai_content = response.content if hasattr(response, "content") else str(response)

    state["messages"] += [AIMessage(content=ai_content)]
    return state


# ------------------------------
# Graph Wiring
# ------------------------------
graph = StateGraph(AppState)

graph.add_node("orchestrator", OrchestratorParser)
graph.add_node("figure_detector", SportsFigureParser)
graph.add_node("invalid_sports", invalid_sports_node)
graph.add_node("sports_rag", sports_rag_node)
graph.add_node("sports_web", sports_web_node)


graph.set_entry_point("orchestrator")

graph.add_conditional_edges(
    "orchestrator",
    lambda s: s["domain"],
    {
        "sports": "figure_detector",
        "other": "invalid_sports",
    },
)
graph.add_conditional_edges(
    "figure_detector",
    lambda s: s["domain_figure"],
    {
        "virat": "sports_rag",
        "rohit": "sports_rag",
        "other": "sports_web",
        "invalid": "invalid_sports",
    },
)

graph.add_edge("invalid_sports", END)
graph.add_edge("sports_rag", END)
graph.add_edge("sports_web", END)

graph_executor = graph.compile()


# if __name__ == "__main__":
#     print("Welcome to the Sports/Other Query Assistant!")
#     print("Type 'exit' to quit.\n")

#     # Initialize conversation state once
#     state = {"queries": [], "messages": []}

#     while True:
#         query = input("Enter your query: ").strip()
#         if query.lower() == "exit":
#             print("Goodbye!")
#             break

#         # Append the new query to the list
#         state["queries"].append(query)

#         # You can also keep the latest query separate for processing if needed
#         current_state = {"query": query, "messages": state["messages"]}

#         # Invoke the graph
#         final_state = graph_executor.invoke(current_state)

#         # Update the global messages list
#         state["messages"] = final_state["messages"]

#         # Print AI/system messages
#         print("\n--- Response ---")
#         for m in final_state["messages"]:
#             print(m.content)

#         print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    print("Welcome to the Sports/Other Query Assistant!")
    print("Type 'exit' to quit.\n")

    # Initialize conversation state once
    state = {
        "messages": [],
        "domain": None,
        "domain_figure": None,
    }

    while True:
        query = input("Enter your query: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break

        # Add the current query as a HumanMessage to the conversation
        state["messages"].append(HumanMessage(content=query))

        # Prepare query-specific state with updated info
        query_state = {
            "query": query,
            "domain": state["domain"],  # inherit previous domain
            "domain_figure": state["domain_figure"],  # inherit previous figure
            "messages": state["messages"],  # include full conversation so far
        }

        # Invoke the graph
        final_state = graph_executor.invoke(query_state)

        # Update accumulated messages and domain info
        state["messages"] = final_state["messages"]
        state["domain"] = final_state.get("domain", state["domain"])
        state["domain_figure"] = final_state.get(
            "domain_figure", state["domain_figure"]
        )

        # Print AI/system messages
        print("\n--- Response ---")
        for m in final_state["messages"]:
            print(m.content)
        print("\n" + "-" * 50 + "\n")
