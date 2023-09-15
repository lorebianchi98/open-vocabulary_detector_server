import requests
import numpy as np

# URL of the remote server
REMOTE_SERVER_URL = 'http://127.0.0.1:5000/detect'


response = requests.post(REMOTE_SERVER_URL, data="lorenzo")

print(response.text)