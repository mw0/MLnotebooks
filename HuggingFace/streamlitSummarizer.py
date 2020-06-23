#!/usr/bin/python3
# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import bs4

import requests
import streamlit as st
import datetime
from os import environ

import re

paraExtract = re.compile(r'<p class="[^"]*">([^<]*)</p>')
htmlDrop = re.compile(
    r'([^<]*)<a class=\"[^"]*"\W+href="[^"]*"\W+title="[^"]*">([^<]*)</a>'
)

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
minLength = st.sidebar.slider("min. word count", 25, 175, 70)
maxLength = st.sidebar.slider("max. word count", 50, 310, 110)

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

all = requests.get(URLs[title])
doc = BeautifulSoup(all.text, "html.parser")
soup = doc.findAll("p", {"class", "css-158dogj evys1bk0"})

story = []
for paraSoup in soup:
    thing = paraSoup.contents[0]
    if isinstance(thing, bs4.element.NavigableString):
        story.append(thing)
        print(thing, "\n")

st.write("\n\n".join(story))

summarizer("\n\n".join(story), min_length=25, max_length=65)

st.write("summary: ", summarizer(userText, min_length=minLength, max_length=maxLength))
