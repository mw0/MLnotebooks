#!/usr/bin/python3
# -*- coding: utf-8 -*-

# import requests
# import re

import streamlit as st
import datetime
# from os import environ
from time import perf_counter

import pandas as pd
import numpy as np

from pathlib import Path

from PIL import Image, ImageDraw
import pytesseract

import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# widthHeight = re.compile(r"^[^(].\((\d*)\, (\d*)\).*$")

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# @st.cache(allow_output_mutation=True)
# def initializeSummarizer():
#     return pipeline("summarization", device=0)


# @st.cache(ttl=60.0*3.0, max_entries=20)		# clear cache every 3 minutes
# def fetchTop5TitlesURLs():
#     top5WorldStories = nyt.top_stories(section="world")[:5]

#     titles = []
#     URLs = dict()
#     for i, top in enumerate(top5WorldStories):
#         if i == 0:
#             latest = top["updated_date"]
#             date = latest[:10]
#             date = date.split("-")
#         title = top["title"]
#         titles.append(title)
#         URLs[title] = top["url"]

#     return titles, URLs, latest

# @st.cache(suppress_st_warning=True)
# def getArticle(URLs, title):
#     return requests.get(URLs[title])


# @st.cache(suppress_st_warning=True)
# def soupifyArticle(all):
#     doc = BeautifulSoup(all.text, "html.parser")
#     soup = doc.findAll("p", {"class", "css-158dogj evys1bk0"})

#     story = []
#     for paraSoup in soup:
#         paragraph = " ".join(paraSoup.text.split()) + "\n"
#         print(paragraph)
#         story.append(paragraph)

#     return story


# @st.cache(suppress_st_warning=True)
# def soupifyArticle(all):
#     doc = BeautifulSoup(all.text, "html.parser")
#     soup = doc.findAll("p", {"class", "css-158dogj evys1bk0"})

#     story = []
#     for paraSoup in soup:
#         paragraph = " ".join(paraSoup.text.split()) + "\n"
#         print(paragraph)
#         story.append(paragraph)

#     return story


# @st.cache(suppress_st_warning=True)
# def summarizeArticle(toSummarize, minLength, maxLength):
#     return summarizer(toSummarize, min_length=minLength,
#                       max_length=maxLength)[0]["summary_text"]


# # NY Times API

# NYTimesAPIkey = environ.get("NYTimesAPIkey")
# if NYTimesAPIkey is None:
#     raise KeyError("'NYTimesAPIkey' not an environment variable name.")

# nyt = NYTAPI(NYTimesAPIkey)

# t0 = perf_counter()
# summarizer = initializeSummarizer()
# t1 = perf_counter()
# Δt01 = t1 - t0

# # Now for the Streamlit interface:

st.title('scan2text')

st.title("About")

st.info(
    "This streamlit app uses tesseract to extract text from scanned "
    "documents that you upload. Your image is displayed with overlays "
    "of boxes showing where tesseract infers text.\n\n"
    "If the 'Autocorrect' checkbox is selected, Symspell is applied "
    "to the text in an attempt to remove character and spacing misreads."
    "\nFor additional information, see the "
    "[README.md](https://github.com/mw0/MLnotebooks/tree/master/OCRapp)."
)

st.checkbox("Autocorrect", ['yes', 'no'])

# print(help(st.sidebar.file_uploader))
myBytesIO = st.file_uploader('Upload a local scan file for text extraction',
                             encoding='auto',
                             key='userFile')
print(type(myBytesIO))
print(myBytesIO)
image = Image.open(myBytesIO)		# .convert('RBGA')
print(image)
print(image.format, image.mode, image.size, image.palette)
print(type(image.size), image.size)

width, height = image.size
print(f"width: {width}, height: {height}")
# # plt.figure(figsize=(width/72, height/72))		# , dpi=72)
# if image.mode == 'L':
#     plt.imshow(image, cmap='gray', vmin=0, vmax=255)
# else:
#     plt.imshow(image)
# st.pyplot()
st.image(image, caption='Scanned image (raw)')  #,
         # use_column_width=True)


# t2 = perf_counter()
# titles, URLs, latest = fetchTop5TitlesURLs()
# t3 = perf_counter()
# Δt23 = t3 - t2

# title = st.sidebar.selectbox(f"at {latest}", titles)
# st.write(f"You selected: {title}, {URLs[title]}")

# t4 = perf_counter()
# all = getArticle(URLs, title)
# t5 = perf_counter()
# Δt45 = t5 - t4

# t6 = perf_counter()
# story = soupifyArticle(all)
# t7 = perf_counter()
# Δt67 = t7 - t6

# userText = "\n\n".join(story)
# print(f"len(userText): {len(userText)}")

# # Ensure that there are not too many tokens for BART model. The following
# # kludge, which truncates the story, seems to work:
# words = userText.split()
# print(f"len(words): {len(words)}")
# if len(words) > truncateWords:
#     words = words[:truncateWords]
# toSummarize = " ".join(words)
# print(len(toSummarize))

# st.title("Summary")
# t8 = perf_counter()
# summary = summarizeArticle(toSummarize, minLength, maxLength)
# st.write(summary)
# t9 = perf_counter()
# Δt89 = t9 - t8

# t10 = perf_counter()
# st.title("Full article")
# st.write(userText)
# t11 = perf_counter()
# Δt10 = t11 - t10

# print(f"Δt to fetch top 5 article meta: {Δt01:5.2f}s")
# print(f"Δt to generate sidebar dropdown: {Δt23:5.2f}s")
# print(f"Δt to fetch article: {Δt45:5.2f}s")
# print(f"Δt to soupify article: {Δt67:5.2f}s")
# print(f"Δt to summarize article: {Δt89:5.2f}s")
# print(f"Δt to write article: {Δt10:5.2f}s")

# if not st.sidebar.button("Hide profiling information"):
#     st.sidebar.header("Profiling information")
#     sbInfoStr = (
#         f"* initialize summarizer: {Δt01:5.2f}s\n"
#         f"* fetch top 5 article metadata: {Δt23:5.2f}s\n"
#         f"* fetch selected article: {Δt45:5.2f}s\n"
#         f"* soupify article: {Δt67:5.2f}s\n"
#         f"* summarize article: {Δt89:5.2f}s"
#     )
#     if cudaDetected:
#         sbInfoStr += "\n"
#         for i in range(cudaDeviceCt):
#             allocated = round(torch.cuda.memory_allocated(i) / 1024 ** 3, 1)
#             cached = round(torch.cuda.memory_cached(i) / 1024 ** 3, 1)
#             sbInfoStr += (
#                 f"\n\ncuda device[{i}]:"
#                 # f" {torch.cuda.get_device_name(i)}"
#                 f"\n* Allocated memory: {allocated:5.3f} GB\n"
#                 f"* Cached memory: {cached:5.3f} GB"
#             )
#     print(sbInfoStr)
#     st.sidebar.info(sbInfoStr)

