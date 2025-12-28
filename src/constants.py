#!/usr/bin/env python3

# Standard library modules

# Third-party modules


ISO639_LANGUAGE_NAMES = {
    "ara": "Arabic",
    "dan": "Danish",
    "deu": "German",
    "dut": "Dutch",
    "eng": "English",
    "fas": "Persian",
    "fra": "French",
    "gre": "Greek",
    "hin": "Hindi",
    "ita": "Italian",
    "kan": "Kannada",
    "mal": "Malayalam",
    "por": "Portuguese",
    "rus": "Russian",
    "spa": "Spanish",
    "swe": "Swedish",
    "tam": "Tamil",
    "tur": "Turkish",
}

CSSc = """
<style>
  table {
    width: 100% !important;
    table-layout: fixed;
    word-wrap: break-word;
  }
  th, td {
    overflow-wrap: break-word;
    white-space: normal !important;
  }
</style>
"""

# Inject <colgroup> for column widths
COLGROUP = """
<colgroup>
  <col style="width:85%">
  <col style="width:15%">
</colgroup>
"""

CSS = """
<style>
  table {
    width: 100% !important;
    table-layout: fixed;
    word-wrap: break-word;    
  }
  td:first-child {
    white-space: normal !important;
    overflow-wrap: break-word;
    word-wrap: break-word;
  }
  td:last-child {
    white-space: nowrap !important;
    max-width: 10%;
  }
</style>
"""
