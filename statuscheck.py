import requests
from requests.auth import HTTPBasicAuth
import time

auth = HTTPBasicAuth("AJBJXL", "t1mx0ro2geklk3")
message_id = "aZ1z3iRdpUemeyrhLEdtS"

def check_status(message_id):
    url = f"https://api.sms-gate.app/3rdparty/v1/messages/{message_id}/status"
    response = requests.get(url, auth=auth)
    print("Status code:", response.status_code)
    print("Response text:", response.text)  # See what server actually returns
    try:
        return response.json()
    except Exception as e:
        print("Failed to parse JSON:", e)
        return None

