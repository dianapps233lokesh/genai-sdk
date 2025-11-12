from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import os
from google import genai


client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def get_gemini_response(question):
    response = client.models.generate_content(model="gemini-2.5-pro", contents=question)
    return response.text


st.set_page_config(page_title="Q&A demo")

st.header("Gemini Application")

input = st.text_input("Input: ", key="input")

submit = st.button("Ask the question")

##when submit is clicked
if submit:
    response = get_gemini_response(input)
    st.write(response)
