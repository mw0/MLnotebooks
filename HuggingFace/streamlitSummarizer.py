#!/usr/bin/python3
# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import bs4

import requests
import streamlit as st
import datetime
from os import environ
from time import perf_counter()
import re

from pynytimes import NYTAPI

from transformers import pipeline

NYTimesAPIkey = environ.get("NYTimesAPIkey")
if NYTimesAPIkey is None:
    raise KeyError("'NYTimesAPIkey' not an environment variable name.")

# nyt = articleAPI(NYTimesAPIkey)
nyt = NYTAPI(NYTimesAPIkey)

summarizer = pipeline("summarization")

# Now for the Streamlit interface:

st.sidebar.title("About")

st.sidebar.info(
    "This little app uses the default HuggingFace summarization "
    "pipeline to summarize text that you post.\n\nFor additional"
    " information, see "
    "https://github.com/mw0/MLnotebooks/HuggingFace/README.md."
)

st.sidebar.header("Set summarization output range (words).")
minLength = st.sidebar.slider("min. word count", 25, 175, 120)
maxLength = st.sidebar.slider("max. word count", 50, 310, 250)

st.sidebar.title("Top 5 New York Times world news articles")

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

title = st.sidebar.selectbox(f"at {latest}", titles)
st.write(f"You selected: {title}, {URLs[title]}")

t0 = perf_counter()
all = requests.get(URLs[title])
t1 = perf_counter()
Δt01 = t1 - t0
print(f"Δt to fetch article: {Δt01:.1f}s")

t2 = perf_counter()
doc = BeautifulSoup(all.text, "html.parser")
soup = doc.findAll("p", {"class", "css-158dogj evys1bk0"})
t3 = perf_counter()
Δt23 = t3 - t2
print(f"Δt to soupify article: {Δt23:.1f}s")

story = []
for paraSoup in soup:
    thing = paraSoup.contents[0]
    if isinstance(thing, bs4.element.NavigableString):
        story.append(thing)
        print(thing, "\n")

userText = "\n\n".join(story)
print(len(userText))

toSummarize = userText[:2000]
print(len(toSummarize))

st.title("Summary")
t4 = perf_counter()
st.write(
    summarizer(toSummarize, min_length=minLength, max_length=maxLength)[0][
        "summary_text"
    ]
)
t5 = perf_counter()
Δt45 = t5 - t4
print(f"Δt to summarize article: {Δt23:.1f}s")

st.title('Full article')
st.write(userText)
