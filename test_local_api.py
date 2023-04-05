import requests
import json



text = input("Enter your line: ")
data = {'sentence': text}
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'} 
r = requests.post("http://127.0.0.1:5000/predict", data=json.dumps(data), headers=headers)  #localhost (127.0.0.1) port 8000

print(r.text)
