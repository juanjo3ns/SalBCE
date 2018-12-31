import requests
import json
import os

assert ("TG_KEY" in os.environ),"Set environment variable telegram token"
token = os.environ['TG_KEY']
url = 'https://api.telegram.org/bot' + token+ '/sendMessage'

def send(message):
    r = requests.post(
        url=url,
        data={'chat_id': 1458951, 'text': message}
        #files = {'media': open('0087.jpg', 'rb')}
    ).json()
