## Apps based upon HuggingFace API

### Contents

* [Summarization](#summarization)
* []()

### Summarization

Applies Facebook's BART model, as implemented by HuggingFace, to summarize articles from the NY Times.

<table>
<tr valign="top">
<td>This is a streamlit app that does several things:

* uses the <em>NY Times</em> Top Stories API to get metadata for the current top stories
* extracts URLs and titles for the top 5, creates sidebar dropdown
* when user selects a title:
  * fetches the article directly from <em>Times</em> website
  * extracts body of article using BeautifulSoup
  * <em>NY Times</em> articles are typically too long for the summarizer, so body is truncated to a maximum of 720 words
  * applies summarizer to truncated text
  * prints the resulting summary
  * prints content of full article
* Using streamlit's caching capabilities, most steps (fetching and extracting text from an article, for example) are repeated only as needed.

See [streamlitSummarizer.py](https://github.com/mw0/MLnotebooks/blob/master/HuggingFace/python/streamlitSummarizer.py) for source code.
</td><td width="743"><img src="SummarizerAppScreenshot.png" width="743" height="982"</td>
</tr>
</table>
