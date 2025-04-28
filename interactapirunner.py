import requests
import time

# API endpoint
API_URL = "http://localhost:8000"

def send_command(command, wait_time=5000):
    """Send a command to the Interact API"""
    response = requests.post(
        f"{API_URL}/interact",
        json={"command": command, "wait_time": wait_time}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Command: {command}")
        print(f"Success: {result['success']}")
        print(f"Message: {result['message']}")
        print("-" * 50)
        return result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def start_browser():
    """Start the browser explicitly"""
    response = requests.post(f"{API_URL}/start-browser")
    if response.status_code == 200:
        print("Browser started successfully")
        return True
    else:
        print(f"Failed to start browser: {response.text}")
        return False

def close_browser():
    """Close the browser explicitly"""
    response = requests.post(f"{API_URL}/close-browser")
    if response.status_code == 200:
        print("Browser closed successfully")
        return True
    else:
        print(f"Failed to close browser: {response.text}")
        return False

def test_arxiv_transformers_search():
    """Test visiting arXiv, searching for 'transformers', and viewing the first paper"""
    if not start_browser():
        return
    
    try:
        while True:
            command = input("Enter command: ")
            if command.lower() == "exit":
                break
            send_command(command)

    finally:
        close_browser()

if __name__ == "__main__":
    test_arxiv_transformers_search()