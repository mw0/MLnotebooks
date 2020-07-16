#!/usr/bin/python3
# -*- coding: utf-8 -*-

# import requests
# import re

import streamlit as st
import datetime
from os import environ
from time import perf_counter

import pandas as pd
import numpy as np

from pathlib import Path

from PIL import Image, ImageDraw
import io

import pytesseract
from nltk.tokenize import word_tokenize, sent_tokenize

import spacy
import contextualSpellCheck

from symspellpy.symspellpy import SymSpell
import pkg_resources
from itertools import islice

import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# widthHeight = re.compile(r"^[^(].\((\d*)\, (\d*)\).*$")

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

@st.cache(allow_output_mutation=True)
def initializeContextualSpellCheck():
    spacy.require_gpu()
    nlp = spacy.load('en_core_web_sm')
    contextualSpellCheck.add_to_pipe(nlp)
    return nlp


@st.cache(allow_output_mutation=True)
def initializeSymspell():
    print("inside initializeSymspell()")
    symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    print("symspell created")
    resourceNames = ["symspellpy", "frequency_dictionary_en_82_765.txt",
                     "frequency_bigramdictionary_en_243_342.txt"]
    dictionaryPath = pkg_resources.resource_filename(resourceNames[0],
                                                     resourceNames[1])
    bigramPath = pkg_resources.resource_filename(resourceNames[0],
                                                 resourceNames[2])
    print("dictionaryPath created")
    symspell.load_dictionary(dictionaryPath, 0, 1)
    symspell.create_dictionary_entry(key='ap', count=500000000)
    print(list(islice(symspell.words.items(), 5)))
    print("symspell.load_ditionary() done")
    symspell.load_bigram_dictionary(bigramPath, 0, 1)
    print(list(islice(symspell.bigrams.items(), 5)))
    print("symspell.load_bigram_ditionary() done")

    # Create vocab
    vocab = set([w for w, f in symspell.words.items()])

    return symspell, vocab


# @st.cache(suppress_st_warning=True)
def doSpacySpellCheck(nlp, text):
    doc = nlp(text)
    test = doc._.performed_spellCheck
    return doc._outcome_spellCheck


# @st.cache(ttl=60.0*3.0, max_entries=20)  # clear cache every 3 minutes
@st.cache(suppress_st_warning=True)
def correctSpellingUsingSymspell(symSpell, vocab, text):
    sentences = sent_tokenize(text)
    lines = []
    for sent in sentences:
        OK = True
        words = word_tokenize(sent)
        for word in words:
            if word not in vocab:
                OK = False
                break
        if OK:
            lines.append(sent)
        else:
            suggestions = symSpell.lookup_compound(sent, max_edit_distance=2,
                                                   transfer_casing=True)
            for i, suggestion in enumerate(suggestions):
                lines.append(suggestion._term)
                print(f"{i:02d}: {type(suggestion)}\t{suggestion}")

    return " ".join(lines)


@st.cache(suppress_st_warning=True)
def correctSpellingUsingSymspellOri(symSpell, vocab, text):
    suggestions = symSpell.lookup_compound(text, max_edit_distance=2,
                                           transfer_casing=True)
    lines = []
    for i, suggestion in enumerate(suggestions):
        lines.append(suggestion._term)
        print(f"{i:02d}: {type(suggestion)}\t{suggestion}")
    return " ".join(lines)


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

# @st.cache(suppress_st_warning=True)
def drawBoxesOnCopy(df, draw):
    # Draw boxes surrounding text on copy
    for ind, row in df.iterrows():
        xy = [(row['left'], row['top']),
              (row['left'] + row['width'], row['top'] + row['height'])]
        # print(xy)
        draw.rectangle(xy, fill=None, width=3, outline='#FF0000')

Δt01 = 0.0
Δt23 = 0.0
Δt45 = 0.0
Δt67 = 0.0
Δt89 = 0.0
Δt10 = 0.0
Δt12 = 0.0
Δt14 = 0.0

# # Now for the Streamlit interface:

st.markdown('## scan2text')

st.sidebar.title("About")

st.sidebar.info(
    "This streamlit app uses tesseract to extract text from scanned "
    "documents that you upload. Your image is displayed with overlays "
    "of boxes showing where tesseract infers text.\n\n"
    "Start by uploading local scan file (below).\n\n"
    "If the 'Show bounding boxes' checkbox is selected, will shows a "
    "copy of the image with all of the text bounding boxes found by "
    "tesseract. Note: displaying bounding boxes takes a lot of time!\n\n"
    # "If the 'Autocorrect' checkbox is selected, Symspell is applied "
    # "to the text in an attempt to remove character and spacing misreads.\n\n"
    "For additional information, see the "
    "[README.md](https://github.com/mw0/MLnotebooks/tree/master/OCRapp)."
)

showBoundingBoxes = False
showBoundingBoxes = st.sidebar.checkbox("Show bounding boxes", ['yes', 'no'])

autocorrect = False
autocorrect = st.sidebar.checkbox("Autocorrect (slow!)", ['yes', 'no'])

# print(help(st.sidebar.file_uploader))
st.sidebar.markdown('## Upload a local scan file')
myBytesIO = st.sidebar.file_uploader(' ',
                                     encoding='auto',
                                     key='userFile')
st.markdown('### Original image')

if myBytesIO is None:
    localImageLocation = environ.get("LocalImageLocation")
    print(f"localImageLocation: {localImageLocation}")
    t0 = perf_counter()
    image = Image.open(localImageLocation)
else:
    t0 = perf_counter()
    print(type(myBytesIO))
    print(myBytesIO)
    image = Image.open(myBytesIO)		# .convert('RBGA')

print(image)
print(image.format, image.mode, image.size, image.palette)
print(type(image.size), image.size)

width, height = image.size
print(f"width: {width}, height: {height}")

st.image(image, use_column_width=True)
t1 = perf_counter()
Δt01 = t1 - t0

# Create a copies to prevent overwriting of original image
copy = image.copy()

# Extract text from image:
t2 = perf_counter()
text = pytesseract.image_to_string(image)
t3 = perf_counter()
Δt23 = t3 - t2

if showBoundingBoxes:
    draw = ImageDraw.Draw(copy)

    t4 = perf_counter()
    df = extractBoundingBoxDatums(image)
    t5 = perf_counter()
    Δt45 = t5 - t4

    t6 = perf_counter()
    drawBoxesOnCopy(df, draw)
    t7 = perf_counter()
    Δt67 = t7 - t6

    t8 = perf_counter()
    st.markdown('### Image with bounding boxes')
    st.image(copy, caption='Scanned image (bounding boxes)',
             use_column_width=True)
    t9 = perf_counter()
    Δt89 = t9 - t8

    st.markdown('### Extracted text')
    st.write(text)
    print(text)

    if autocorrect:
        t10 = perf_counter()
        # symSpell, vocab = initializeSymspell()
        nlp = initializeContextualSpellCheck()
        t11 = perf_counter()
        Δt10 = t11 - t10

        t12 = perf_counter()
        # corrected = correctSpellingUsingSymspell(symSpell, vocab, text)
        corrected = doSpacySpellCheck(nlp, text)
        t12 = perf_counter()
        Δt12 = t13 - t12

        st.markdown('### Text corrected with Symspell')
        st.write(corrected)
        print(corrected)

print(f"\n\nload, format, display user image: {Δt23: 4.1f}s")
print(f"extract text using tesseract: {Δt45: 4.1f}s")
if showBoundingBoxes:
    print(f"extract bounding boxes: {Δt45: 4.1f}s")
    print(f"draw boxes on image: {Δt67: 4.1f}s")
    print(f"display image with boxes: {Δt89: 4.1f}s")
    print(f"summarize article: {Δt89:5.2f}s")
if autocorrect:
    print(f"initialize symspell: {Δt10: 4.1f}s")
    print(f"spell correct with symspell: {Δt12: 4.1f}s")

if not st.sidebar.button("Hide profiling information"):
    st.sidebar.header("Profiling information")
    sbInfoStr = (
        f"* load, format, display user image: {Δt23: 4.1f}s\n"
        f"* extract text using tesseract: {Δt45: 4.1f}s")
    if showBoundingBoxes:
        sbInfoStr += (f"\n* extract bounding boxes: {Δt45: 4.1f}s\n"
                      f"* draw boxes on image: {Δt67: 4.1f}s\n"
                      f"* display image with boxes: {Δt89: 4.1f}s\n"
                      f"* summarize article: {Δt89:5.2f}s")
    if autocorrect:
        sbInfoStr += (f"\n* initialize symspell: {Δt10: 4.1f}s\n"
                      f"* spell correct with symspell: {Δt12: 4.1f}s")
    st.sidebar.info(sbInfoStr)
