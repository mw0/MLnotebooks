import streamlit as st
from transformers import pipeline

summarizer = pipeline("summarization")

userText = """As Data Science and Web developers try to collaborate, API’s become an essential piece of the puzzle to make codes as well as skills more modular. In fact, in the same way, that a data scientist can’t be expected to know much about Javascript or nodeJS, a frontend developer should be able to get by without knowing any Data Science Language. And APIs do play a considerable role in this abstraction.
But, APIs are confusing. I myself have been confused a lot while creating and sharing them with my development teams who talk in their API terminology like GET request, PUT request, endpoint, Payloads, etc."""

summarizer(userText, min_length=25, max_length=65)

# Now for the Streamlit interface:

st.sidebar.title("About")

st.sidebar.info("This little app uses the default HuggingFace summarization "
                "pipeline to summarize text that you post.\n\nFor additional"
                " information, see "
                "https://github.com/mw0/MLnotebooks/HuggingFace/README.md.")

st.sidebar.header('Set summarization output range (words).')
minLength = st.sidebar.slider('min. word count', 25, 175, 70)
maxLength = st.sidebar.slider('max. word count', 50, 310, 90)

userText = st.text_input('Input text you want summarized:')

st.write('summary: ', summarizer(userText, min_length=minLength,
                                 max_length=maxLength))
