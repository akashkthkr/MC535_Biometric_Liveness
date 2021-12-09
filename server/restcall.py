import requests

url = 'http://127.0.0.1:5000/classification'
headers = {'Authorization' : ‘’, 'Accept' : 'application/json', 'Content-Type' : 'application/json'}
r = requests.post(url, data=open('example.json', 'rb'), headers=headers)