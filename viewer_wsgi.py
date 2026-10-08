"""WSGI module for the deployed viewer: ``gunicorn viewer_wsgi:app``.

Kept apart from viewer_main.py so importing that (as the tests do) doesn't
build the app and connect to Firestore.
"""

from viewer_main import create_app

app = create_app()
