## Apps based upon HuggingFace API

### Contents

* [Summarization](#summarization)
* []()

### Summarization

Applies Facebook's BART model, as implemented by HuggingFace, to summarize articles from the NY Times.

<table>
<tr>
<td>This is a streamlit app that does several things:

* uses the NY Times Top Stories API to get metadata for the current top stories
* extracts URLs and titles for the top 5, creates dropdown in sidebar
* when user selects a title:
  * fetches the full article directly from the Times website
  * extracts body of article using BeautifulSoup
  * NY Times articles are typically too long for the summarizer, so the body is truncated to a maximum of 720 words
  * truncated text is sent to HuggingFace's summarizer
  * prints the resulting summary
  * prints content of full article
* Using streamlit's caching capabilities, most steps (fetching and extracting text from an article, for example) are repeated only as needed.

See [streamlitSummarizer.py](https://github.com/mw0/MLnotebooks/blob/master/HuggingFace/python/streamlitSummarizer.py) for source code.
</td><td>![image](https://github.com/mw0/MLnotebooks/blob/master/HuggingFace/SummarizerAppScreenshot.png)</td>
</tr>
</table>
