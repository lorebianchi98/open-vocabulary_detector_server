import requests
import numpy as np

# URL of the remote server
REMOTE_SERVER_URL = 'http://127.0.0.1:5000/detect'


import cv2
import requests
import numpy as np

# URL of the remote server
REMOTE_SERVER_URL = 'http://remote_server_ip:5000/detect'

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        
        # Display the frame with bounding boxes
        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
if __name__ == '__main__':
    main()

# response = requests.post(REMOTE_SERVER_URL, data="lorenzo")

# print(response.text)