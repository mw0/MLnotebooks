#!/usr/bin/python3
# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import bs4

import requests
import streamlit as st
import datetime
from os import environ
from time import perf_counter

from pynytimes import NYTAPI

from transformers import pipeline
import torch

print(f"torch sees cuda: {torch.cuda.is_available()}")
print(f"torch device named: {torch.cuda.get_device_name(0)}")

@st.cache(allow_output_mutation=True)
def initializeSummarizer():
    return pipeline("summarization")


# @st.cache(suppress_st_warning=True)
def fetchTop5TitlesURLs():
    top5WorldStories = nyt.top_stories(section="world")[:5]

    titles = []
    URLs = dict()
    for i, top in enumerate(top5WorldStories):
        if i == 0:
            latest = top["updated_date"]
            date = latest[:10]
            date = date.split("-")
        title = top["title"]
        titles.append(title)
        URLs[title] = top["url"]

    return titles, URLs, latest


@st.cache(suppress_st_warning=True)
def getArticle(URLs, title):
    return requests.get(URLs[title])


@st.cache(suppress_st_warning=True)
def soupifyArticle(all):
    doc = BeautifulSoup(all.text, "html.parser")
    soup = doc.findAll("p", {"class", "css-158dogj evys1bk0"})

    story = []
    for paraSoup in soup:
        paragraph = " ".join(paraSoup.text.split()) + "\n"
        print(paragraph)
        story.append(paragraph)

    return story


@st.cache(suppress_st_warning=True)
def soupifyArticle(all):
    doc = BeautifulSoup(all.text, "html.parser")
    soup = doc.findAll("p", {"class", "css-158dogj evys1bk0"})

    story = []
    for paraSoup in soup:
        paragraph = " ".join(paraSoup.text.split()) + "\n"
        print(paragraph)
        story.append(paragraph)

    return story

@st.cache(suppress_st_warning=True)
def summarizeArticle(toSummarize, minLength, maxLength):
    return summarizer(toSummarize, min_length=minLength,
                      max_length=maxLength)[0]["summary_text"]


# NY Times API

NYTimesAPIkey = environ.get("NYTimesAPIkey")
if NYTimesAPIkey is None:
    raise KeyError("'NYTimesAPIkey' not an environment variable name.")

nyt = NYTAPI(NYTimesAPIkey)

t0 = perf_counter()
summarizer = initializeSummarizer()
t1 = perf_counter()
Δt01 = t1 - t0

# Now for the Streamlit interface:

st.sidebar.title("About")

st.sidebar.info(
    "This streamlit app uses the default HuggingFace summarization "
    "pipeline (Facebook's BART model) to summarize text from selected "
    "NY Times articles.\n\n"
    "The actual summarization time takes on the order of 20 seconds, although"
    " increasing the summary length will extend this significantly.\n"
    "\nFor additional information, see the "
    "[README.md](https://github.com/mw0/MLnotebooks/tree/master/HuggingFace)."
)

st.sidebar.header("Set summarization output range (words)")
minLength = st.sidebar.slider("min. word count", 25, 250, 125)
maxLength = st.sidebar.slider("max. word count", 45, 310, 175)
st.sidebar.header("Article truncation size (words)")
truncateWords = st.sidebar.slider("truncate size", 300, 720, 500)

st.sidebar.title("Top 5 New York Times world news articles")

t2 = perf_counter()
titles, URLs, latest = fetchTop5TitlesURLs()
t3 = perf_counter()
Δt23 = t3 - t2

title = st.sidebar.selectbox(f"at {latest}", titles)
st.write(f"You selected: {title}, {URLs[title]}")

t4 = perf_counter()
all = getArticle(URLs, title)
t5 = perf_counter()
Δt45 = t5 - t4

t6 = perf_counter()
story = soupifyArticle(all)
t7 = perf_counter()
Δt67 = t7 - t6

userText = "\n\n".join(story)
print(f"len(userText): {len(userText)}")

# Ensure that there are not too many tokens for BART model. The following
# kludge, which truncates the story, seems to work:
words = userText.split()
print(f"len(words): {len(words)}")
if len(words) > truncateWords:
    words = words[:truncateWords]
toSummarize = " ".join(words)
print(len(toSummarize))

st.title("Summary")
t8 = perf_counter()
summary = summarizeArticle(toSummarize, minLength, maxLength)
st.write(summary)
t9 = perf_counter()
Δt89 = t9 - t8

t10 = perf_counter()
st.title("Full article")
st.write(userText)
t11 = perf_counter()
Δt10 = t11 - t10

print(f"Δt to fetch top 5 article metadatums: {Δt01:4.1f}s")
print(f"Δt to generate sidebar dropdown: {Δt23:4.1f}s")
print(f"Δt to fetch article: {Δt45:4.1f}s")
print(f"Δt to soupify article: {Δt67:4.1f}s")
print(f"Δt to summarize article: {Δt89:4.1f}s")
print(f"Δt to write article: {Δt10:4.1f}s")

if not st.sidebar.button('Hide profiling information'):
    st.sidebar.info(f"* initialize summarizer Δt: {Δt01:4.1f}s\n"
                    f"* fetch top 5 article metada Δt: {Δt23:4.1f}s\n"
                    f"* fetch selected article Δt: {Δt45:4.1f}s\n"
                    f"* soupify article Δt: {Δt67:4.1f}s\n"
                    f"* summarize article Δt: {Δt89:4.1f}s")


# if __name__ == "__main__":

#     t0 = perf_counter()
#     summarizer = initializeSummarizer()
#     t1 = perf_counter()
#     Δt01 = t1 - t0
