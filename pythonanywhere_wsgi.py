"""
PythonAnywhere WSGI entry for the Flask app.

Paste the contents of this file into the WSGI configuration file on
PythonAnywhere (Web tab -> Code -> WSGI configuration file).

Change <username> to your PythonAnywhere username.
"""

import sys

PROJECT = "/home/<username>/mysite"
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

from app import app as application  # noqa: E402