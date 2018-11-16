import requests
import json
token = '658824281:AAETui7gl4muFLRsod1j2cGnuIDox_hj6hY'
url = 'https://api.telegram.org/bot' + token+ '/sendMessage'
def send(message):
    r = requests.post(
        url=url,
        data={'chat_id': 1458951, 'text': message}
        #files = {'media': open('0087.jpg', 'rb')}
    ).json()
