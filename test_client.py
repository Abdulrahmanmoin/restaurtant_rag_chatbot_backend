import requests
import time
import sys

def test_api():
    print("Starting connection test...")
    max_retries = 150
    url = "http://127.0.0.1:8000/health"
    
    for i in range(max_retries):
        try:
            print(f"Attempt {i+1}...")
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"Success! Status: {response.status_code}")
                print(f"Response: {response.json()}")
                
                # Try chat
                chat_url = "http://127.0.0.1:8000/chat"
                chat_payload = {"message": "Hello, are you open?"}
                print(f"Testing chat endpoint: {chat_url}")
                chat_response = requests.post(chat_url, json=chat_payload, timeout=60)
                print(f"Chat Response Code: {chat_response.status_code}")
                print(f"Chat Response: {chat_response.json()}")
                
                with open("client_log.txt", "w") as f:
                    f.write(f"Health Check: {response.json()}\n")
                    f.write(f"Chat Response: {chat_response.json()}\n")
                return
        except Exception as e:
            print(f"Connection failed: {e}")
        
        time.sleep(2)

    print("Failed to connect after retries.")
    with open("client_log.txt", "w") as f:
        f.write("Failed to connect.")

if __name__ == "__main__":
    test_api()
