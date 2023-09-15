import json

import cv2
import requests
import numpy as np

import threading

TARGET_SIZE = (224, 224)

COLOR = ((255, 0, 0)) # BGR
# URL of the remote server
REMOTE_SERVER_URL = 'http://127.0.0.1:5000/detect'

LAST_FRAME = None
LAST_RESULTS = None

def draw_bounding_boxes(frame, results, vocabulary):
    # Loop through the results and draw bounding boxes
    for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
        scale_x = frame.shape[1] / TARGET_SIZE[0]  # Width scaling factor
        scale_y = frame.shape[0] / TARGET_SIZE[1]  # Height scaling factor

        x1, y1, x2, y2 =  map(int, box)
        x1 = round(x1 * scale_x)
        y1 = round(y1 * scale_y)
        x2 = round(x2 * scale_x)
        y2 = round(y2 * scale_y)

        # Convert the label to a string
        class_label = vocabulary[label]
        score_str = f"{score:.3f}"

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, 2)

        # Put the label and score on the image
        text = f"{class_label} {score_str}"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
        
    return frame


def send_frame_and_receive_results(frame):
    to_send_frame = cv2.resize(frame, TARGET_SIZE)
    frame_json = json.dumps(to_send_frame.tolist())  # Serialize the numpy array to a JSON string
    response = requests.post(REMOTE_SERVER_URL, data=frame_json)
    results = json.loads(response.text)
    return results

def webcam_capture_and_display():
    global LAST_FRAME, LAST_RESULTS
    
    cap = cv2.VideoCapture(0)
    vocabulary = ['person']

    while True:
        ret, frame = cap.read()
        LAST_FRAME = frame
        global LAST_RESULTS
        if not ret:  # Check if frame capture was successful
            continue  # Skip the current iteration and try again    
        
        if LAST_RESULTS is not None:
            frame = draw_bounding_boxes(frame, LAST_RESULTS, vocabulary)

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
            break

def main():
    capture_thread = threading.Thread(target=webcam_capture_and_display)
    capture_thread.start()
    global LAST_RESULTS
    while True:
        if LAST_FRAME is not None:
            print("Sending request...")
            LAST_RESULTS = send_frame_and_receive_results(LAST_FRAME)
            print("Results received!")

if __name__ == '__main__':
    main()
