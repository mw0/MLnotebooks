## Apps based upon HuggingFace API

### Contents

* [Summarization](summarization)
* []()

### Summarization

This is a streamlit app that does several things:

* uses the NY Times Top Stories API to get metadata for the current top stories
* extracts URLs and titles for the top 5, creates dropdown in sidebar
* when user selects a title:
  * fetches the full article directly from the Times website
  * extracts body of article using BeautifulSoup
  * submits first 2500 characters of body to HuggingFace's summarizer
  * prints the resulting summary
  * prints content of full article

See [streamlitSummarizer.py](https://github.com/mw0/MLnotebooks/blob/master/HuggingFace/python/streamlitSummarizer.py)
