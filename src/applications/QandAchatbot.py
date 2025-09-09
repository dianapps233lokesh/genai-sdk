from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import os
from google import genai


client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

chat = client.chats.create(model="gemini-2.0-flash")


def get_gemini_response(question):
    response = chat.send_message_stream(question)
    return response


st.set_page_config(page_title="Q&A ChatBot")
st.header("Gemini LLM application")

# initialize session state for chat history if it doesn't exist

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

input = st.text_input("Input:", key="input")
submit = st.button("Ask")

if submit and input:
    response = get_gemini_response(input)
    ##adding user query and response to session chat history
    st.session_state["chat_history"].append(("You", input))
    st.subheader("The response is")

    for chunk in response:
        st.write(chunk.text)
        st.session_state["chat_history"].append(("LLM", chunk.text))

    st.subheader("Chat history is")

for role, text in st.session_state["chat_history"]:
    st.write(f"{role}:{text}")
