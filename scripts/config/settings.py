# config/settings.py
# Remember to replace the placeholder values with your own Reddit API credentials.
import os
from pathlib import Path


USER_AGENT = "SDS_textanalysis/1.0 (by /u/gwen1126)"
API_BASE_URL = "https://api.reddit.com"
RATE_LIMIT_DELAY = 2
OPENAI_API = "MASKED"
PROJ_PATH = "MASKED" # Replace with directory path to CHINA_ANALYSIS_PROJECT


# PRAW credentials
PRAW_CLIENT = "MASKED"
PRAW_CLIENT_SECRET = "MASKED"
PRAW_USER_AGENT = "MASKED"