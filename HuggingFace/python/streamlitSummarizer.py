#!/usr/bin/python3
# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import bs4

import requests
import streamlit as st
import datetime
from os import environ
from time import perf_counter
import re

from pynytimes import NYTAPI

from transformers import pipeline

summarizer = pipeline("summarization")


def getSummary(story):
    return summarizer(story, min_length=minLength, max_length=maxLength)[0][
        "summary_text"
    ]


# NY Times API

NYTimesAPIkey = environ.get("NYTimesAPIkey")
if NYTimesAPIkey is None:
    raise KeyError("'NYTimesAPIkey' not an environment variable name.")

nyt = NYTAPI(NYTimesAPIkey)

# Now for the Streamlit interface:

st.sidebar.title("About")

st.sidebar.info(
    "This streamlit app uses the default HuggingFace summarization "
    "pipeline (Facebook's Bart model) to summarize text from selected "
    "NY Times articles.\n\n"
    "The actual summarization time takes on the order of a half minute.\n"
    "\nFor additional information, see the "
    "[README.md](https://github.com/mw0/MLnotebooks/tree/master/HuggingFace)."
)

st.sidebar.header("Set summarization output range (words)")
minLength = st.sidebar.slider("min. word count", 25, 250, 120)
maxLength = st.sidebar.slider("max. word count", 45, 310, 250)

st.sidebar.title("Top 5 New York Times world news articles")

t0 = perf_counter()
top5WorldStories = nyt.top_stories(section="world")[:5]
t1 = perf_counter()
Δt01 = t1 - t0

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

t2 = perf_counter()
title = st.sidebar.selectbox(f"at {latest}", titles)
st.write(f"You selected: {title}, {URLs[title]}")
t3 = perf_counter()
Δt23 = t3 - t2

t4 = perf_counter()
all = requests.get(URLs[title])
t5 = perf_counter()
Δt45 = t5 - t4

t6 = perf_counter()
doc = BeautifulSoup(all.text, "html.parser")
soup = doc.findAll("p", {"class", "css-158dogj evys1bk0"})
t7 = perf_counter()
Δt67 = t7 - t6

story = []
for paraSoup in soup:
    print(len(paraSoup), paraSoup)
    if len(paraSoup) > 0:
        thing = paraSoup.contents[0]
        if isinstance(thing, bs4.element.NavigableString):
            story.append(thing)
            print(thing, "\n")

userText = "\n\n".join(story)
print(len(userText))

toSummarize = userText[:2500]
print(len(toSummarize))

st.title("Summary")
t8 = perf_counter()

summary = getSummary(toSummarize)
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
