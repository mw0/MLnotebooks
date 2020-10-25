#!/usr/bin/python3
# coding: utf-8

# ## Setup

# ### Library import
# Import required Python libraries

import os
from os.path import exists
from copy import deepcopy
from pathlib import Path
import sys
from urllib.request import urlopen

# Versioned modules:

import requests
import numpy as np
import pandas as pd

# Visualizations

from matplotlib import __version__ as mpVersion
import matplotlib.pyplot as plt

from plotly import __version__ as plVersion
import chart_studio.plotly as py
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from IPython.core.interactiveshell import InteractiveShell

# cf.go_offline(connected=True)
cf.set_config_file(theme='white')
cf.set_config_file(offline=True)

print("Python version: ", sys.version_info[:])
print("Un-versioned imports:\n")
prefixStr = ''
if 'copy' in sys.modules:
    print(prefixStr + 'copy', end="")
    prefixStr = ', '
if 'pathlib' in sys.modules:
    print(prefixStr + 'pathlib', end="")
    prefixStr = ', '
if 'os' in sys.modules:
    print(prefixStr + 'os', end="")
    prefixStr = ', '
if 'os.path' in sys.modules:
    print(prefixStr + 'os.path', end="")
    prefixStr = ', '
if 'sys' in sys.modules:
    print(prefixStr + 'sys', end="")
    prefixStr = ', '
if 'urllib' in sys.modules:
    print(prefixStr + 'urllib', end="")

print("\n")
if 'matplotlib' in sys.modules:
    print(f"matplotlib: {mpVersion}", end="\t")
if 'numpy' in sys.modules:
    print(f"numpy: {np.__version__}", end="\t")
if 'pandas' in sys.modules:
    print(f"pandas: {pd.__version__}", end="\t")
if 'plotly' in sys.modules:
    print(f"plotly: {plVersion}", end="\t")
if 'requests' in sys.modules:
    print(f"requests: {requests.__version__}", end="\t")

# get_ipython().run_line_magic('matplotlib', 'inline')

# Options for pandas
pd.options.display.max_columns = 30
pd.options.display.max_rows = 50

# Autoreload extension
if 'autoreload' not in get_ipython().extension_manager.loaded:
    get_ipython().run_line_magic('load_ext', 'autoreload')

get_ipython().run_line_magic('autoreload', '2')


# **Next two lines are for pretty output for all prints in a Pandas cell, not
# just the last**

InteractiveShell.ast_node_interactivity = "all"

# ### Get county shape files to be used by plotly.express

# #### Get JSON file from plotly github repo

# * if first time, fetch and save to disk
# * else, read file from disk

basePath = Path.cwd()
dataPath = basePath.parent / 'data'
commonDataPath = basePath.parent.parent / 'commonData'
countyShapesFile = commonDataPath / 'UScountyShapes.json'
if countyShapesFile.is_file():
    with open(countyShapesFile, 'r') as shapesHandle:
        counties = json.load(shapesHandle)
else:
    URL = ('https://raw.githubusercontent.com/plotly/datasets/master/'
           'geojson-counties-fips.json')
    with urlopen(URL) as response:
        counties = json.load(response)
    with open(countyShapesFile, 'w') as outFile:
        json.dump(counties, outFile)

for feature in counties["features"]:
    if (feature['properties']['STATE'] == '29'):
        print(feature['properties']['COUNTY'], feature['properties']['NAME'])

# ## Prepare Data

# ### Data import

# * Data are from the *NY Times* Covid-19 data repository
# * Want to use freshest data, so each time this notebook is run the data are
#   downloaded directly from the repo
# * pandas v1.x does permit direct reads of remote data using `.read_csv(URL)`,
#   but this fails if any of the many lines in the file have incorrect
#    formatting. In such cases:
#   * download the data first
#   * manually correct dodgy lines
#   * read in the data directly from disk

URL = ('https://raw.githubusercontent.com/nytimes/covid-19-data/master/'
       'us-counties.csv')
covidData = requests.get(URL).content

with open(dataPath / 'CovidRaw.csv', 'wb') as source:
    source.write(covidData)

df0 = pd.read_csv(dataPath / 'CovidRaw.csv')
df0.info()

# #### Get the most recent day's data:

URL = ('https://raw.githubusercontent.com/nytimes/covid-19-data/master/live/'
       'us-counties.csv')
covidData = requests.get(URL).content
with open(dataPath / 'CovidRawToday.csv', 'wb') as source:
    source.write(covidData)

df1 = pd.read_csv(dataPath / 'CovidRawToday.csv')
print(df1.info())

# ### Union data and get aggregate statistics

# * After the union, set the date column to datetimes.

df2 = pd.concat([df0, df1]).reset_index(drop=True)

df2['date'] = pd.to_datetime(df2.date)
print("\n", df2.info())
print("\n", df2.describe(percentiles=[0.01, 0.05, 0.20, 0.50,
                                      0.80, 0.95, 0.99]))

print("df2.head(10)")
print("\n", df2.tail(10))

# ### Get fips values

# #### By county

dffc = pd.read_csv(commonDataPath / 'nrcsFIPS.tsv', delimiter='\t')
print("\n", dffc.info())
print("\n", dffc.head(9))
print("\n", dffc.tail(9))

# #### By state

dffs = pd.read_csv(commonDataPath / 'nrcsStateFIPS.tsv', delimiter='\t')
print("\n", dffs.info())
print("\n", dffs.head())
print("\n", dffs.tail())

# ##### Create a State ⟶ StateCode dict

# This is used in [Replace 'Unknown' counties with StateCode-Unknown values]
# (#Replace-'Unknown'-counties-with-StateCode-Unknown-values)

State2StateCode = {s: sc for s, sc in zip(dffs.State, dffs.StateCode)}
print(State2StateCode)

# **There is no state with FIPS code 11, corresponding to Warshington, D.C.**

# dffs[dffs.StateFIPS == 11]

# #### Create dict for state "Unknown"  FIPS codes

# * Take the two-digit state code and add 999
# * The FIPS code for the 'county' District of Columbia, 11, is missing and
#   needs to be added after creating the dict

stateFIPS = {s: str(f) + '999' for s, f in zip(dffs.State, dffs.StateFIPS)}
stateFIPS['District of Columbia'] = '11999'
sorted(stateFIPS.items(), key=lambda kv: kv[1])
print("\n", stateFIPS)

# ### Get county Census data (by FIPS code)

# The file [PopulationEstimates.xls]
# (https://www.ers.usda.gov/webdocs/DataFiles/48747
#  /PopulationEstimates.xls?v=5425) is from the
# [USDA Economic Research Service]
# (https://www.ers.usda.gov/data-products/county-level-data-sets
#  /download-data/). This has more data than the file [co-est2019-annres.xlsx]
# (https://www2.census.gov/programs-surveys/popest/tables/2010-2019/counties
#  /totals/co-est2019-annres.xlsx) directly from the [US Census Bureau]
# (https://www.census.gov/data/datasets/time-series/demo/popest
#  /2010s-counties-total.html).

# Differences:

# * ERS data includes data for Puerto Rico (80 more lines)
# * ERSS data includes total counts for each state and PR (51 more lines)
# * ERS data includes births, deaths, and counts for natural increases,
#   international migration, domestic migration, net migration, etc. (many
#   multiples of the number of columns)

# Rural-urban continuum codes are found in ruralurbancodes2013.xls and
# pr2003.xls (for Puerto Rico), available from [USDA's ERS]
# (https://www.ers.usda.gov/data-products/rural-urban-continuum-codes/). The
# short summary:

# > The 2013 Rural-Urban Continuum Codes form a classification scheme that
# > distinguishes metropolitan counties by the population size of their metro
# > area, and nonmetropolitan counties by degree of urbanization and adjacency
# > to a metro area. The official Office of Management and Budget (OMB) metro
# > and nonmetro categories have been subdivided into three metro and six
# > nonmetro categories. Each county in the U.S. is assigned one of the 9
# > codes. This scheme allows researchers to break county data into finer
# > residential groups, beyond metro and nonmetro, particularly for the
# > analysis of trends in nonmetro areas that are related to population density
# > and metro influence. The Rural-Urban Continuum Codes were originally
# > developed in 1974. They have been updated each decennial since (1983, 1993,
# > 2003, 2013), and slightly revised in 1988. Note that the 2013 Rural-Urban
# > Continuum Codes are not directly comparable with the codes prior to 2000
# > because of the new methodology used in developing the 2000 metropolitan
# > areas. See the Documentation for details and a map of the codes.

# Information on urban influence codes are found in UrbanInfluenceCodes2013.xls
# and pr2003UrbInf.xls (for Puerto Rico), available from [USDA's ERS]
# (https://www.ers.usda.gov/data-products/urban-influence-codes/). The short
# summary:

# > The 2013 Urban Influence Codes form a classification scheme that
# > distinguishes metropolitan counties by population size of their metro
# > area, and nonmetropolitan counties by size of the largest city or town and
# > proximity to metro and micropolitan areas. The standard Office of
# > Management and Budget (OMB) metro and nonmetro categories have been
# > subdivided into two metro and 10 nonmetro categories, resulting in a
# > 12-part county classification. This scheme was originally developed in
# > 1993. This scheme allows researchers to break county data into finer
# > residential groups, beyond metro and nonmetro, particularly for the
# > analysis of trends in nonmetro areas that are related to population
# > density and metro influence.

# Information on county economic type codes are found in
# ERSCountyTypology2015Edition.xls, available from [USDA's ERS]
# (https://www.ers.usda.gov/data-products/county-typology-codes/). The short
# summary:

# > An area's economic and social characteristics have significant effects on
# > its development and need for various types of public programs. To provide
# > policy-relevant information about diverse county conditions to
# > policymakers, public officials, and researchers, ERS has developed a set of
# > county-level typology codes that captures a range of economic and social
# > characteristics.

# > The 2015 County Typology Codes classify all U.S. counties according to six
# > mutually exclusive categories of economic dependence and six overlapping
# > categories of policy-relevant themes. The economic dependence types include
# > farming, mining, manufacturing, Federal/State government, recreation, and
# > nonspecialized counties. The policy-relevant types include low education,
# > low employment, persistent poverty, persistent child poverty, population
# > loss, and retirement destination.

# (Used in [Create per thousand case values]
#  (#Create-per-thousand-case-values).)

dfCensus = pd.read_excel(commonDataPath / 'PopulationEstimates.xls',
                         sheet_name='Population Estimates 2010-19',
                         header=2)
keepColumns = ['FIPStxt', 'State', 'Area_Name',
               'Rural-urban_Continuum Code_2013', 'Urban_Influence_Code_2013',
               'Economic_typology_2015', 'POP_ESTIMATE_2019',
               'N_POP_CHG_2019', 'Births_2019', 'Deaths_2019']
dfCensus = dfCensus[keepColumns]
print(dfCensus.head(8))

# ### Data Munging

# **Beat data into shape**

# <font color="darkgreen">**For details with data handling, refer to the
# [Readme.md](https://github.com/nytimes/covid-19-data) for the /NY Times/
# data repository. In particular, note:**

# * [Methodology and Definitions]
#   (https://github.com/nytimes/covid-19-data#methodology-and-definitions)
# * [Geographic Exceptions]
#   (https://github.com/nytimes/covid-19-data#geographic-exceptions)
# </font>

# #### Missing fips codes:

# An incomplete discussion of this is outlined in
# [FIPS code for New York City and Rhode Island Missing? #105]
# (https://github.com/nytimes/covid-19-data/issues/105), submitted to issues
# for the *NY Times* github repo. These are addressed indirectly in the
# Geographic Exceptions linked above.

# ##### New York City

# * New York City has 5 boroughs, and for each a fips code. However, the
#   /NY Times/ has aggregated all of these into 'New York City' county. Set the
#   fips code to 36061, which is correct for Manhattan.
# * Later will duplicate values for the other boroughs, represented by FIPS
#   codes 36005, 36047, 36081, and 36085.

print("\n", df2[(df2.county == 'New York City')]['fips'].head())
df2['fips'] = [36061 if (c == 'New York City') else f
               for c, f in zip(df2.county, df2.fips)]
print("\n", df2[(df2.county == 'New York City')]['fips'].head())

# ##### Kansas City

# From the Geographic Exceptions section of the NY Times repo Readme.md:

# > Four counties (Cass, Clay, Jackson and Platte) overlap the municipality of
# > Kansas City, Mo. The cases and deaths that we show for these four counties
# > are only for the portions exclusive of Kansas City. Cases and deaths for
# > Kansas City are reported as their own line.

# * For purposes of assigning to a FIPS code, we use 29096, which has not been
#   assigned to any county. (It follows the FIPS code for Jackson County, which
#   contains the largest share of Kansas City.)

print("\n", df2[(df2.county == 'Kansas City')]['fips'].head())
df2['fips'] = [29096 if (c == 'Kansas City') else f
               for c, f in zip(df2.county, df2.fips)]
print("\n", df2[(df2.county == 'Kansas City')]['fips'].head())

# ##### Joplin, MO

# > Starting June 25, cases and deaths for Joplin are reported separately from
# > Jasper and Newton counties. The cases and deaths reported for those
# > counties are only for the portions exclusive of Joplin. Joplin cases and
# > deaths previously appeared in the counts for those counties or as Unknown.

# * As Joplin is not a county, there is no county FIPS code. Use 29375,
#   otherwise un-used, instead of its actual code: 29-37592

print("\n", df2[df2.fips == 29375])

print("\n", df2[(df2.county == 'Joplin')]['fips'].head())
df2['fips'] = [29375 if (c == 'Joplin') else f
               for c, f in zip(df2.county, df2.fips)]
print("\n", df2[(df2.county == 'Joplin')]['fips'].head())

# ##### Unknowns

# Most states in the database have cases not assigned to any county.

# * Assign to these 'Unknown' counties a FIPS `df2['fips'] = StateFIPS[state]`,
#   where that contains those 999 county codes.
# * Oregon, North Carolina and Alabama keep good track of cases, and as of
#   11 Oct 2020, there were no cases assigned to 'Unknown' county.

states = df2.state.unique()
for state in states:
    print(state)
    fips = stateFIPS[state]
    print(df2[(df2.county == 'Unknown')
              & (df2.state == state)]['fips'].head(4))
    df2['fips'] = [fips if ((c == 'Unknown') and (s == state)) else f
                   for c, s, f in zip(df2.county, df2.state, df2.fips)]
    print(df2[(df2.county == 'Unknown')
              & (df2.state == state)]['fips'].head(4))
    print("")

# **Verify that all null fips values are gone**

dfnulls = df2[df2.fips.isnull()]
print("\n", dfnulls.head())
del dfnulls

# #### Convert FIPS codes from float type to 5-character strings

print("\n", df2[df2.state == 'Alabama'].head(3))
df2['fips'] = [str(int(x)).zfill(5) for x in df2['fips']]
print("\n", df2.loc[1619:1621, :])

# df2bak = deepcopy(df2)
# df2 = deepcopy(df2bak)

# #### Replace 'Unknown' counties with StateCode-Unknown values

# (`StateCode` is created above in
# [Create a State ⟶ StateCode dict](#Create-a-State-%E2%9F%B6-StateCode-dict))

print("\n", df2[df2.county == 'Unknown'].tail(3))
df2['county'] = [State2StateCode[s] + '-Unknown' if c == 'Unknown' else c
                 for s, c in zip(df2.state, df2.county)]
print("\n", df2[df2.county.str.contains('-Unknown')].tail(3))

# #### Create duplicate values for Manhattan for each of 4 other boroughs

# * Duplicate values for New York City, FIPS 36061, for the other four
#   boroughs having FIPS values of 36005, 36047, 36081, and 36085
# * Want all 5 boroughs to match Manhattan, so that the whole city will show
#   up in choropleths with the same values.

dfNYC = df2[df2.fips == '36061'].copy()
dfNYC['fips'] = '36005'
df2 = pd.concat([df2, dfNYC])
dfNYC['fips'] = '36047'
df2 = pd.concat([df2, dfNYC])
dfNYC['fips'] = '36081'
df2 = pd.concat([df2, dfNYC])
dfNYC['fips'] = '36085'
df2 = pd.concat([df2, dfNYC]).reset_index(drop=True)
print("\n", df2[df2.fips == '36005'].head(3))
print("\n", df2[df2.fips == '36047'].head(3))
print("\n", df2[df2.fips == '36081'].head(3))
print("\n", df2[df2.fips == '36085'].head(3))

# #### Create reverse dict, providing county from fips

# Rather than using dffc, need to work from fips values in df2, since that
# DataFrame includes those weird cases, above, for which FIPS codes were
# manually added.

# * Extract a unique set of `['county', 'fips']` labels using
#   `.drop_duplicates()`
# * *Note that there is a degeneracy of 5 for NY City FIPS values*

NYCityFIPS = ['36061', '36005', '36047', '36081', '36085']

uniqCountyFIPS = df2[['county', 'fips']].drop_duplicates()
FIPScounty = dict()
for c, f in zip(uniqCountyFIPS.county, uniqCountyFIPS.fips):
    FIPScounty[f] = c

# Print 10 examples
i = 0
for k, v in FIPScounty.items():
    if ('-Unknown' in v) and (i < 10):
        print(k, v)
        i += 1

for k, v in FIPScounty.items():
    if k in NYCityFIPS:
        print(k, v)


# #### Group by county and aggregate values by week, starting on Mondays

# Want to group by fips (county), and then aggregate values by week starting
# on Mondays, such that for a given Monday the sums are for the week to follow.

# * Create a new datetime variable `'weekStart'` to indicate the Monday of the
#   week for each `date`. Described in
#   [Get Week Start Date (Monday) from a date column in Python (pandas)?]
#   (https://stackoverflow.com/questions/27989120
#    /get-week-start-date-monday-from-a-date-column-in-python-pandas)

quantities = ['cases', 'confirmed_cases', 'confirmed_deaths',
              'deaths', 'probable_cases', 'probable_deaths']
df2['weekStart'] = df2['date'].dt.to_period('W').apply(lambda r: r.start_time)
print("\n", df2.head())
print("\n", df2[df2.county.str.contains('Unknown')].head())

# * Group by `'fips'` and by new `'weekStart'` variable, having a frequency of
#   one week
# * Aggregate using `.max()` to get maximum for the week starting on
#   `weekStart`.
# * `.reset_index()` to clean up.
# * Create a 'county' variable that contains county name or `<state>-Unknown`.

dfg = df2.groupby(['fips', 'weekStart']).max() \
      .sort_values(by=['state', 'county', 'weekStart']).reset_index()
dfg['county'] = dfg['fips'].apply(lambda f: FIPScounty[f])
print("\n", dfg.tail())

dftmp = df2[df2.fips == '56045'].sort_values(by=['date']).tail(21)
print("\n", dftmp[(dftmp.date >= pd.to_datetime('2020-10-05'))
                  & (dftmp.date <= pd.to_datetime('2020-10-11'))])
print("\n", dfg[dfg.weekStart == pd.to_datetime('2020-10-05')].head())

dftmp = df2[df2.fips == '01001'].sort_values(by=['date']).tail(9)
print("\n", dftmp[(dftmp.date >= pd.to_datetime('2020-10-05'))
                  & (dftmp.date <= pd.to_datetime('2020-10-11'))])

# ### Difference totals to get new cases/deaths

dfg['new_cases'] = dfg.groupby(['fips'])['cases'].diff().fillna(0)
dfg['new_deaths'] = dfg.groupby(['fips'])['deaths'].diff().fillna(0)
print("\n", dfg.tail())

# ### Create per thousand case values

# `dfCensus` was imported in [Get county Census data (by FIPS code)]
# (#Get-county-Census-data-(by-FIPS-code))

# * start by creating a modified version, `dfc`, of the census DataFrame
#   `dfCensus`.
#   * cast FIPtxt astype(str)
#   * aggregate the population values for the 5 NY City boroughs
#   * duplicate the aggregate values for each of the 5 boroughs, so all
#     boroughs will have those totals
# * LEFT JOIN dfg with dfc to get population values for each FIPS value
# * create new columns in dfg to get per capita cases, confirmed cases, deaths,
#   confirmed deaths, probable cases, probable deaths

# #### Set all NY City populations to total

dfc = dfCensus[['FIPStxt', 'State', 'Area_Name', 'POP_ESTIMATE_2019']].copy()
dfc['FIPStxt'] = [str(int(x)).zfill(5) for x in dfc['FIPStxt']]
NYCpop = dfc[dfc.FIPStxt.isin(NYCityFIPS)]['POP_ESTIMATE_2019'].sum()
print("\n", dfc[dfc.FIPStxt.isin(NYCityFIPS)])
dfc['POP_ESTIMATE_2019'] = [NYCpop if f in NYCityFIPS else p
                            for f, p in
                            zip(dfc.FIPStxt, dfc.POP_ESTIMATE_2019)]
print("\n", dfc[dfc.FIPStxt.isin(NYCityFIPS)])

# #### Join dfC with dfg to add populations

df = pd.merge(dfg, dfc[['FIPStxt', 'POP_ESTIMATE_2019']], how='left',
              left_on='fips', right_on='FIPStxt')
del df['FIPStxt']
df = df.sort_values(by=['weekStart', 'fips']).reset_index(drop=True)
df['weekStr'] = df.weekStart.apply(lambda x: x.strftime('%Y-%m-%d'))

print("\n", df.head(4))
print("\n", df.tail(4))

# #### Create the variables

df['casesk'] = 1000.0*df.cases / df.POP_ESTIMATE_2019
df['deathsk'] = 1000.0*df.deaths / df.POP_ESTIMATE_2019
df['casesConfk'] = 1000.0*df.confirmed_cases / df.POP_ESTIMATE_2019
df['deathsConfk'] = 1000.0*df.confirmed_deaths / df.POP_ESTIMATE_2019
df['casesProbk'] = 1000.0*df.probable_cases / df.POP_ESTIMATE_2019
df['deathsProbk'] = 1000.0*df.probable_deaths / df.POP_ESTIMATE_2019
df['log10cases'] = np.log10(df.cases)
df['log10deaths'] = np.log10(df.deaths)
df['log10casesk'] = np.log10(df.casesk)
df['log10deathsk'] = np.log10(df.deathsk)
df['newcasesK'] = 1000.0*df.new_cases / df.POP_ESTIMATE_2019
df['newdeathsK'] = 1000.0*df.new_deaths / df.POP_ESTIMATE_2019
print("\n", df.head().T)

# ## Save `df`

# * Currently using feather format

df.to_feather(dataPath / 'MungedCovid19v1.2.ftr')
