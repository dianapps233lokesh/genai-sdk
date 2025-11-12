from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import os
from google import genai
from PIL import Image


client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def get_gemini_response(input, image):
    if input != "":
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", contents=[input, image]
        )
    else:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", contents=image
        )

    return response.text


st.set_page_config(page_title="Gemini image demo")

st.header("Gemini Application")

input = st.text_input("Input: ", key="input")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", width="stretch")


submit = st.button("Info about image")

##when submit is clicked
if submit:
    response = get_gemini_response(input, image)
    st.write(response)
