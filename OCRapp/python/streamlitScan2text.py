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
import io

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
#	.
#	.
#	.
#     return titles, URLs, latest

@st.cache(suppress_st_warning=True)
def extractBoundingBoxDatums(image):
    data = pytesseract.image_to_data(image).split('\n')
    df = pd.DataFrame([x.split('\t') for x in data[1:]],
                  columns=data[0].split('\t'))

    print(df.head(20))

    # convert to int, replace -1 conf values with pd.NA, remove boxes that
    # contain no text:

    cols = df.columns
    print(cols)
    for col in cols[:-1]:
        df[col] = df[col].astype(int)
    print(df.info())

    df.loc[df.conf == -1, 'conf'] = pd.NA
    df = df[~df.conf.isna()].reset_index()
    print(df.head(10))

    return df

@st.cache(suppress_st_warning=True)
def drawBoxesOnCopy(df, copy):
    # Draw boxes surrounding text on copy
    for ind, row in df.iterrows():
        xy = [(row['left'], row['top']),
              (row['left'] + row['width'], row['top'] + row['height'])]
        # print(xy)
        draw.rectangle(xy, fill=None, width=3, outline='#FF0000')

# def getArticle(URLs, title):
#     return requests.get(URLs[title])

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

st.sidebar.title("About")

st.sidebar.info(
    "This streamlit app uses tesseract to extract text from scanned "
    "documents that you upload. Your image is displayed with overlays "
    "of boxes showing where tesseract infers text.\n\n"
    "If the 'Autocorrect' checkbox is selected, Symspell is applied "
    "to the text in an attempt to remove character and spacing misreads."
    "\nFor additional information, see the "
    "[README.md](https://github.com/mw0/MLnotebooks/tree/master/OCRapp)."
)

autocorrect = st.sidebar.checkbox("Autocorrect", ['yes', 'no'])
showBoundingBoxes = st.sidebar.checkbox("Show bounding boxes", ['yes', 'no'])

# print(help(st.sidebar.file_uploader))
myBytesIO = st.sidebar.file_uploader('Upload a local scan file for'
                                     ' text extraction',
                                     encoding='auto',
                                     key='userFile')

t0 = perf_counter()
print(type(myBytesIO))
print(myBytesIO)
image = Image.open(myBytesIO)		# .convert('RBGA')
print(image)
print(image.format, image.mode, image.size, image.palette)
print(type(image.size), image.size)

width, height = image.size
print(f"width: {width}, height: {height}")

st.image(image, caption='Scanned image (raw)',
         use_column_width=True)
t1 = perf_counter()
Δt01 = t1 - t0

# Create a copies to prevent overwriting of original image
copy = image.copy()
draw = ImageDraw.Draw(copy)

if showBoundingBoxes:
    t2 = perf_counter()
    df = extractBoundingBoxDatums(image)
    t3 = perf_counter()
    Δt23 = t3 - t2
    print(f"extractBoundingBoxDatums() Δt23: {Δt23: 4.1f}s")

    t4 = perf_counter()
    drawBoxesOnCopy(df, copy)
    t5 = perf_counter()
    Δt45 = t5 - t4
    print(f"drawBoxesOnCopy() Δt45: {Δt45: 4.1f}s")

    t6 = perf_counter()
    st.image(copy, caption='Scanned image (bounding boxes)',
             use_column_width=True)
    t7 = perf_counter()
    Δt67 = t7 - t6
    print(f"st.image(copy) Δt67: {Δt67: 4.1f}s")

t8 = perf_counter()
text = pytesseract.image_to_string(image)
t9 = perf_counter()
Δt89 = t9 - t8
print(f"pytesseract.image_to_string() Δt89: {Δt89: 4.1f}")
st.write(text)
print(text)

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

