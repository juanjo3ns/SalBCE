import requests
import json
token = ''
url = 'https://api.telegram.org/bot' + token+ '/sendMessage'
def send(message):
    r = requests.post(
        url=url,
        data={'chat_id': 1458951, 'text': message}
        #files = {'media': open('0087.jpg', 'rb')}
    ).json()
