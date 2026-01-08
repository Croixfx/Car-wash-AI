import requests
from requests.auth import HTTPBasicAuth

url = "https://api.sms-gate.app/3rdparty/v1/messages"

payload = {
    "deviceId": "LRZwcJMmsqXzcb1YJIVZU",
    "isEncrypted": False,
    "phoneNumbers": [
        "+250787065734",
        "+250739559540",
        "+250791791130"
    ],
    "priority": 0,
    "simNumber": 2,
    "textMessage": {
        "text": "Hello! How are you doing guys"
    },
    "ttl": 86400,
    "withDeliveryReport": True
}


auth = HTTPBasicAuth("AJBJXL", "t1mx0ro2geklk3")

try:
    response = requests.post(url, json=payload, auth=auth)
    print("Status Code:", response.status_code)
    print("Response Text:", response.text)
except Exception as e:
    print("Error:", e)
