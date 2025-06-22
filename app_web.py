# -*- coding: utf-8 -*-
"""
DrAI Web Application - Main Frontend
This Flask application serves the main landing page for the DrAI platform.
"""

from flask import Flask, render_template
import logging

# --- Logger Setup ---
# (Using a simple configuration for the web app)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logger Setup ---

# Initialize Flask App
app = Flask(__name__)

@app.route('/')
def home():
    """
    Renders the main landing page.
    """
    logger.info("Serving the main landing page (index.html).")
    return render_template('index.html')

if __name__ == '__main__':
    logger.info("--- Starting DrAI Web Application ---")
    # Note: For production, use a proper WSGI server like Gunicorn or Waitress.
    app.run(debug=True, port=5000)
    logger.info("--- DrAI Web Application Stopped ---") 