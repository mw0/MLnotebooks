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

cudaDetected = torch.cuda.is_available()
print(f"torch sees cuda: {cudaDetected}")
if cudaDetected:
    cudaDeviceCt = torch.cuda.device_count()
for i in range(cudaDeviceCt):
    print(f"cuda device[{i}]: {torch.cuda.get_device_name(i)}")


@st.cache(allow_output_mutation=True)
def initializeSummarizer():
    return pipeline("summarization", device=0)


@st.cache(ttl=60.0*3.0, max_entries=20)		# clear cache every 3 minutes
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
    "The actual summarization time takes a few seconds, although"
    " increasing the summary length will extend this.\n"
    "\nFor additional information, see the "
    "[README.md](https://github.com/mw0/MLnotebooks/tree/master/HuggingFace)."
)

st.sidebar.header("Set summarization output range (words)")
minLength = st.sidebar.slider("min. word count", 25, 350, 200)
maxLength = st.sidebar.slider("max. word count", 45, 400, 250)
st.sidebar.header("Article truncation size (words)")
truncateWords = st.sidebar.slider("truncate size", 300, 720, 720)

st.sidebar.title("Top 5 New York Times world news articles")

t2 = perf_counter()
titles, URLs, latest = fetchTop5TitlesURLs()
t3 = perf_counter()
Δt23 = t3 - t2

title = st.sidebar.selectbox(f"at {latest}", titles)
st.write(f"You selected: *{title}*, {URLs[title]}")

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

print(f"Δt to fetch top 5 article meta: {Δt01:5.2f}s")
print(f"Δt to generate sidebar dropdown: {Δt23:5.2f}s")
print(f"Δt to fetch article: {Δt45:5.2f}s")
print(f"Δt to soupify article: {Δt67:5.2f}s")
print(f"Δt to summarize article: {Δt89:5.2f}s")
print(f"Δt to write article: {Δt10:5.2f}s")

if not st.sidebar.button("Hide profiling information"):
    st.sidebar.header("Profiling information")
    sbInfoStr = (
        f"* initialize summarizer: {Δt01:5.2f}s\n"
        f"* fetch top 5 article metadata: {Δt23:5.2f}s\n"
        f"* fetch selected article: {Δt45:5.2f}s\n"
        f"* soupify article: {Δt67:5.2f}s\n"
        f"* summarize article: {Δt89:5.2f}s"
    )
    if cudaDetected:
        sbInfoStr += "\n"
        for i in range(cudaDeviceCt):
            allocated = round(torch.cuda.memory_allocated(i) / 1024 ** 3, 1)
            cached = round(torch.cuda.memory_cached(i) / 1024 ** 3, 1)
            sbInfoStr += (
                f"\n\ncuda device[{i}]:"
                # f" {torch.cuda.get_device_name(i)}"
                f"\n* Allocated memory: {allocated:5.3f} GB\n"
                f"* Cached memory: {cached:5.3f} GB"
            )
    print(sbInfoStr)
    st.sidebar.info(sbInfoStr)
