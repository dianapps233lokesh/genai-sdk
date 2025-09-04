from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import os
from google import genai
from PIL import Image


client=genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def get_gemini_response(input,image,prompt):
    response=client.models.generate_content(model="gemini-2.5-flash-lite",contents=[input,image,prompt])
    return response.text

st.set_page_config(page_title="Invoice extractor")

st.header("Gemini Application for invoice extractor")

input=st.text_input("Input: ", key="input")

uploaded_file=st.file_uploader("Choose an image",type=["jpg","jpeg","png"])
image=""
if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image,caption="uploaded image",width="stretch")

submit=st.button("Tell me about invoice")

input_prompt='''
    You are an expert in understanding invoices. 
    We'll upload a image as invoice and you will have to answer any question on the uploaded invoices image
'''

#if submit clicked
if submit:
    response=get_gemini_response(input_prompt,image,input)
    st.subheader("The response is")
    st.write(response)