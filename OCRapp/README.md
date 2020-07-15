## scan2text

### Extract text content from scanned file

Uses tesseract optical character recognition software to extract text from images. Optionally displays bounding box for each word identified, and optionally attempts to correct spelling errors.

<table>
<tr valign="top">
<td>This is a streamlit app that does several things:

* user uploads file containing scanned text
* uses tesseract to extract text content
* optionally redraws figure displaying bounding boxes for each word found
* optionally uses symspell to correct spelling errors
* Streamlit's caching capabilities obviate repeating steps &mdash; e.g., extracting text from an article already parsed.
  * currently has caching problems with the routine that draws bounding boxes, and this is pretty slow

See [streamlitScan2text.py](https://github.com/mw0/MLnotebooks/blob/master/OCRapp/python/streamlitScan2text.py) for source code.
</td><td width="700"><img src="OCRappScreenshot0.png" width="899" height="1206"</td>
</tr>
</table>
