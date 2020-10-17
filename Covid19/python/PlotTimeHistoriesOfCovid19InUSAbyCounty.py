#!/usr/bin/python3
# coding: utf-8

# ## Setup

# ### Library import
# We import all the required Python libraries

from time import time, asctime, gmtime, localtime

# from platform import node
import requests
import os
from os.path import exists
from copy import deepcopy
from random import random
import gc		# garbage collection module
from pathlib import Path
import timeit
import sys
from urllib.request import urlopen

# chart_studio is part of plotly, but does not have a
# separate __version__ variable
# import chart_studio
# import chart_studio.plotly as py

from dateutil import __version__ as duVersion
from dateutil.parser import parse
import urllib3

import numpy as np

import pandas as pd

import pyreadr
import pydot_ng
import graphviz

# Visualizations
from matplotlib import __version__ as mpVersion
import matplotlib.pyplot as plt

import json
from plotly import __version__ as plVersion
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
# cf.go_offline(connected=True)
cf.set_config_file(theme='white')
cf.set_config_file(offline=True)

print("Python version: ", sys.version_info[:])
print("Un-versioned imports:\n")
prefixStr = ''
if 'chart_studio' in sys.modules:
    print(prefixStr + 'chart_studio', end="")
    prefixStr = ', '
if 'collections' in sys.modules:
    print(prefixStr + 'collections', end="")
    prefixStr = ', '
if 'copy' in sys.modules:
    print(prefixStr + 'copy', end="")
    prefixStr = ', '
if 'descartes' in sys.modules:
    print(prefixStr + 'descartes', end="")
    prefixStr = ', '
if 'gc' in sys.modules:
    print(prefixStr + 'gc', end="")
    prefixStr = ', '
if 'glob' in sys.modules:
    print(prefixStr + 'glob', end="")
    prefixStr = ', '
if 'io' in sys.modules:
    print(prefixStr + 'io', end="")
    prefixStr = ', '
if 'pathlib' in sys.modules:
    print(prefixStr + 'pathlib', end="")
    prefixStr = ', '
if 'pickle' in sys.modules:
    print(prefixStr + 'pickle', end="")
    prefixStr = ', '
if 'platform' in sys.modules:
    print(prefixStr + 'platform', end="")
    prefixStr = ', '
if 'plotHelpers' in sys.modules:
    print(prefixStr + 'plotHelpers', end="")
    prefixStr = ', '
if 'pprint' in sys.modules:
    print(prefixStr + 'pprint', end="")
    prefixStr = ', '
if 'os' in sys.modules:
    print(prefixStr + 'os', end="")
    prefixStr = ', '
if 'os.path' in sys.modules:
    print(prefixStr + 'os.path', end="")
    prefixStr = ', '
if 'random' in sys.modules:
    print(prefixStr + 'random', end="")
    prefixStr = ', '
if 'shutil' in sys.modules:
    print(prefixStr + 'shutil', end="")
    prefixStr = ', '
if 'sys' in sys.modules:
    print(prefixStr + 'sys', end="")
    prefixStr = ', '
if 'timeit' in sys.modules:
    print(prefixStr + 'timeit', end="")
    prefixStr = ', '
if 'utility' in sys.modules:
    print(prefixStr + 'utility', end="")
    # prefixStr = ', '

print("\n")
if 'colorcet' in sys.modules:
    print(f"colorcet: {cc.__version__}", end="\t")
if 'cufflinks' in sys.modules:
    print(f"cufflinks: {cf.__version__}", end="\t")
if 'dateutil' in sys.modules:
    print(f"dateutil: {duVersion}", end="\t")
if 'graphviz' in sys.modules:
    print(f"graphviz: {graphviz.__version__}", end="\t")
if 'joblib' in sys.modules:
    print(f"joblib: {jlVersion}", end="\t")
if 'json' in sys.modules:
    print(f"json: {json.__version__}", end="\t")
if 'matplotlib' in sys.modules:
    print(f"matplotlib: {mpVersion}", end="\t")
if 'numpy' in sys.modules:
    print(f"numpy: {np.__version__}", end="\t")
if 'pandas' in sys.modules:
    print(f"pandas: {pd.__version__}", end="\t")
if 'plotly' in sys.modules:
    print(f"plotly: {plVersion}", end="\t")
if 'pydot' in sys.modules:
    print(f"pydot: {pd.__version__}", end="\t")
if 'pyreader' in sys.modules:
    print(f"pyreader: {pyreader.__version__}", end="\t")
if 'requests' in sys.modules:
    print(f"requests: {requests.__version__}", end="\t")
if 'urllib3' in sys.modules:
    print(f"urllib3: {urllib3.__version__}", end="\t")

# get_ipython().run_line_magic('matplotlib', 'inline')

# Options for pandas
pd.options.display.max_columns = 30
pd.options.display.max_rows = 50

# Autoreload extension
# if 'autoreload' not in get_ipython().extension_manager.loaded:
#     get_ipython().run_line_magic('load_ext', 'autoreload')

# get_ipython().run_line_magic('autoreload', '2')
print(asctime(localtime()))

# ### Helper function

# #### `choroplethCovidUSA()`

# Creates choropleths from data in df, using plotly geojson formatted county
# shapes.

# Modeled after Plotly's [Choropleth map using GeoJSON]
# (https://plotly.com/python/choropleth-maps/)


def choroplethCovidUSA(df, myGeoJSON, myDecimalRange, myTopOfRange,
                       myAnimationVar=None,
                       myLocs='fips',
                       myColscaleVar='log10cases',
                       myColscale='Portland',
                       myScope='usa',
                       myZlabel='Total cases',
                       myHoverDescription='log10(cases)',
                       myHoverVar='cases',
                       myTitle='Covid-19 Total Cases to Date'):
    """
    INPUT:
        df			pd.DataFrame, containing data to be plotted
        myGeoJSON		plotly shape file dict containing county shapes
        myDecimalRange		list type(int), containing decimal orders of
                                magnitude for color bar range. (E.g., [-3, 4]
                                for 0.001–10000.)
        myTopOfRange		float, containing 'extra' range on top of
                                decimal scale. (E.g. 30000.0 to extend
                                myDecimalRange by less than an order of
                                magnitude.) If None, entire range will
                                be specified by myDecimalRange.
        myAnimationVar		str, feature to use to select individual
                                animation frames (typically a time-based
                                variable, such as 'weekStr'). If None, no
                                animation, which is the default.
        myLocs			str, feature used to select shapes from
                                myGeoJSON dict, default: 'fips'.
        myColscaleVar		str, column name in df for z-scale,
                                default: 'log10cases'.
        myColscale		str, valid Plotly color scale name,
                                default: 'Portland'
        myScope			str, identifier for geographic range,
                                default: 'usa'
        myZlabel		str, label on top of color bar
        myHoverDescription	str, label, in addition to myLocs, to describe
                                values in DataFrame column used for colorbar
                                intensity, default: 'Total cases'
        myHoverVar		str, feature values to be shown when hovering
                                over a region.
        myTitle			str, figure title,
                                default: "Covid-19 Total Cases to Date"
    """

    minval = df[myColscaleVar].replace(0, np.nan).min()
    maxval = np.max(df[myColscaleVar])
    print(f"minval: {minval}, maxval: {maxval}")

    myBottomOfRange = 10**myDecimalRange[0]
    if maxval > myTopOfRange:
        print("WARNING: maximum value exceeds top of your range."
              f" {maxval} > {myTopOfRange}.")
    if minval < 10**myDecimalRange[0]:
        print("WARNING: minimum value is below bottom of your range."
              f" {minval} < {myBottomOfRange}.")

    linVals = np.logspace(myDecimalRange[0], myDecimalRange[1],
                          int(myDecimalRange[1] - myDecimalRange[0]) + 1)
    if myTopOfRange is not None:
        linVals = np.append(linVals, myTopOfRange)
    linVals = [round(lv, 4) for lv in linVals]
    logVals = [round(lv, 4) for lv in np.log10(linVals)]

    # coloraxis_colorbar dict needs ticktext to an array of strings:
    linVals = [str(lv) for lv in linVals]
    print(linVals)
    print(logVals)

    fig = px.choropleth(df,
                        animation_frame=myAnimationVar,
                        geojson=myGeoJSON,
                        locations=myLocs,
                        color=myColscaleVar,
                        color_continuous_scale=myColscale,
                        range_color=(logVals[0], logVals[-1]),
                        scope=myScope,
                        hover_name=myHoverVar,
                        labels={myColscaleVar: myHoverDescription})
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      title={'text': myTitle,
                             'x': 0.5, 'xanchor': 'center',
                             'y': 0.97, 'yanchor': 'top'},
                      coloraxis_colorbar=dict(title=myZlabel,
                                              tickvals=logVals,
                                              ticktext=linVals))

    fig.show("notebook")

    return fig


# # Generate Plots from MungedCovid19 data

# **<font color="darkred">This uses data created by
# [NewYorkTimesCovid19DataMunging.ipynb]
# (NewYorkTimesCovid19DataMunging.ipynb)</font>**

# ## Data import

# ### Load munged Covid-19 data

# * data are stored in feather format

basePath = Path.cwd()
dataPath = basePath.parent / 'data'
imgPath = basePath.parent / 'png'

df = pd.read_feather(dataPath / 'MungedCovid19v1.1.ftr', use_threads=True)
df.head().T

# #### Remove columns not currently needed to reduce the size of data file for
# web page

df = df[['fips', 'weekStr', 'cases', 'deaths', 'casesk', 'deathsk',
        'log10cases', 'log10deaths', 'log10casesk', 'log10deathsk']]
df.info()

# ### Get JSON file from plotly github repo

# * if first time, fetch and save to disk
# * else, read file from disk

commonDataPath = basePath.parent.parent / 'commonData'
countyShapesFile = commonDataPath / 'UScountyShapes.json'
if countyShapesFile.is_file():
    print(f"Found shape file {countyShapesFile}")
    with open(countyShapesFile, 'r') as shapesHandle:
        counties = json.load(shapesHandle)
else:
    print(f"Fetching shape file {countyShapesFile}")
    URL = ('https://raw.githubusercontent.com/plotly/datasets/master'
           '/geojson-counties-fips.json')
    with urlopen(URL) as response:
        counties = json.load(response)
    with open(countyShapesFile, 'w') as outFile:
        json.dump(counties, outFile)

print(df.columns)

# ## Plots

# ### Credentials for uploading to plotly.com

# * two values are stored as environment variables that are passed to available
# to jupyter instance

# username = os.environ['PlotlyUsername']
# api_key = os.environ['PlotlyAPIkey']
# chart_studio.tools.set_credentials_file(username=username, api_key=api_key)

# ### Animated all-data plots

# * Modeled after Plotly's [Choropleth map using GeoJSON]
# (https://plotly.com/python/choropleth-maps/).
# * Animations are based upon `weekStr` column values, which are strings
# containing the first Monday of the week plotted. (Typically, the maximum
# values taken will arrive later in the week.)


pio.kaleido.scope.default_format = "png"
# pio.orca.config.executable = '/usr/local/bin/orca'

# ## Generate single frames for each week in data file:

t0 = time()
weekStrs = df.weekStr
for weekStr in weekStrs:

    # #### Case counts

    weekTitle = ", ".join(["Covid-19 Total Cases", weekStr])
    fileOut = '../img/' +  ".".join(['Covid19TotalCases', weekStr, 'png'])
    figCaseCts = choroplethCovidUSA(df[df.weekStr == weekStr], counties,
                                    [-1, 5], 300000.0,
                                    myTitle=weekTitle)
    figCaseCts.write_image(fileOut, engine="kaleido")
    # pio.write_image(figCaseCts, fileOut)

    # #### Death counts

    weekTitle = ", ".join(["Covid-19 Total Deaths", weekStr])
    # fileOut = imgPath / ".".join(['Covid19TotalDeaths', weekStr, 'png'])
    fileOut = '../img/' +  ".".join(['Covid19TotalDeaths', weekStr, 'png'])
    figDeathCts = choroplethCovidUSA(df[df.weekStr == weekStr], counties,
                                     [-1, 5], 300000.0,
                                     myColscaleVar='log10deaths',
                                     myZlabel='Total deaths',
                                     myHoverDescription='Total deaths',
                                     myHoverVar='deaths',
                                     myTitle=weekTitle)
    figDeathCts.write_image(fileOut, engine="kaleido")
    # plotly.io.write_image(figDeathCts, fileOut)

    # # #### Cases per thousand

    weekTitle = ", ".join(["Covid-19 Cases per Thousand", weekStr])
    # fileOut = imgPath / ".".join(['Covid19CasesPer1000', weekStr, 'png'])
    fileOut = '../img/' +  ".".join(['Covid19CasesPer1000', weekStr, 'png'])
    myHoverDescr = 'Total cases per 1000'
    figCasesPerK = choroplethCovidUSA(df[df.weekStr == weekStr], counties,
                                      [-1, 4], 25000.0,
                                      myColscaleVar='log10casesk',
                                      myZlabel='Total cases per 1000',
                                      myHoverDescription=myHoverDescr,
                                      myHoverVar='casesk',
                                      myTitle=weekTitle)
    figCasesPerK.write_image(fileOut, engine="kaleido")
    # plotly.io.write_image(figCasesPerK, fileOut)

    # # #### Deaths per thousand

    weekTitle = ", ".join(["Covid-19 Deaths per Thousand", weekStr])
    # fileOut = imgPath / ".".join(['Covid19DeathsPer1000', weekStr, 'png'])
    fileOut = '../img/' + ".".join(['Covid19DeathsPer1000', weekStr, 'png'])
    myHoverDescr = 'Total deaths per 1000'
    myTitle = "Covid-19 Total Deaths per 1000"
    figDeathsPerK = choroplethCovidUSA(df[df.weekStr == weekStr], counties,
                                       [-1, 5], 300000.0,
                                       myColscaleVar='log10deathsk',
                                       myZlabel='Total deaths per 1000',
                                       myHoverDescription=myHoverDescr,
                                       myHoverVar='deathsk',
                                       myTitle=myTitle)
    figDeathsPerK.write_image(fileOut, engine="kaleido")
    # plotly.io.write_image(figDeathsPerK, fileOut)

Δt = time() - t0
print(f"\n\nTime to generate single frame images Δt: {Δt: 4.1f}s.")

# ## Generate monolithic JavaScript-fueled animations in HTML documents.

# # #### Case counts

# figCaseCts = choroplethCovidUSA(df, counties, [-1, 5], 300000.0,
#                                 myAnimationVar='weekStr',
#                                 myTitle="Covid-19 Total Cases")

# # ##### Export HTML w/ JavaScript

# # py.plot(figCaseCts, filename='Covid19CaseCtsUScounties', auto_open=True)
# pio.write_html(figCaseCts, file=imgPath / 'Covid19CaseCtsUScounties.html',
#                auto_open=True)

# # #### Death counts

# figDeathCts = choroplethCovidUSA(df, counties, [-1, 4], 25000.0,
#                                  myAnimationVar='weekStr',
#                                  myColscaleVar='log10deaths',
#                                  myZlabel='Total deaths',
#                                  myHoverDescription='Total deaths',
#                                  myHoverVar='deaths',
#                                  myTitle="Covid-19 Total Deaths")

# # py.plot(figCaseCts, filename='Covid19DeathCtsUScounties', auto_open=True)
# pio.write_html(figDeathCts, file=imgPath / 'Covid19DeathCtsUScounties.html',
#                auto_open=True)

# # #### Cases per thousand

# figCasesPerK = choroplethCovidUSA(df, counties, [0, 2], 170.0,
#                                   myAnimationVar='weekStr',
#                                   myColscaleVar='log10casesk',
#                                   myZlabel='Total cases per 1000',
#                                   myHoverDescription='Total cases per 1000',
#                                   myHoverVar='casesk',
#                                   myTitle="Covid-19 Total Cases per 1000")

# # py.plot(figCaseCts, filename='Covid19DeathCtsUScounties', auto_open=True)
# pio.write_html(figCasesPerK,
#                file=imgPath / 'Covid19CasesPerThousandUScounties.html',
#                auto_open=True)

# # #### Deaths per thousand

# figDeathsPerK = choroplethCovidUSA(df, counties, [-2, 0], 5.2,
#                                    myAnimationVar='weekStr',
#                                    myColscaleVar='log10deathsk',
#                                    myZlabel='Total deaths per 1000',
#                                    myHoverDescription='Total deaths per 1000',
#                                    myHoverVar='deathsk',
#                                    myTitle="Covid-19 Total Deaths per 1000")

# # py.plot(figCaseCts, filename='Covid19DeathCtsUScounties', auto_open=True)
# pio.write_html(figDeathsPerK,
#                file=imgPath / 'Covid19DeathsPerThousandUScounties.html',
#                auto_open=True)
