import json
import subprocess

import cv2
import requests
import numpy as np

import threading

TARGET_SIZE = (224, 224)
COLOR = (255, 0, 0) # BGR
COLORS = [
    (255, 0, 0),   # Blue
    (0, 255, 0),   # Green
    (0, 0, 255),   # Red
    (0, 165, 255), # Orange
    (0, 255, 255), # Yellow
    (130, 0, 75),  # Indigo
    (128, 0, 128), # Violet
    (147, 20, 255),# Deep Pink
    (0, 128, 0),   # Dark Green
    (128, 128, 128)# Gray
]

# URL of the remote server
REMOTE_SERVER_URL = 'http://127.0.0.1:5000/detect'

# shared variables
LAST_FRAME = None
LAST_RESULTS = None
EXIT_FLAG = False
SCORE_THRESH = 0.1
VOCABOLARY = ['person']

def get_text_entry(default_text, text="Propose a new caption"):
    # Zenity command
    char_width = 7
    width = len(default_text) * char_width + 20
    zenity_cmd = ['zenity', '--entry', '--entry-text', default_text, '--text', text, '--width', str(width)]

    try:
        # Execute Zenity command
        result = subprocess.run(zenity_cmd, capture_output=True, text=True, check=True)

        # Retrieve the entered text
        entered_text = result.stdout.strip()

        # Return the entered text
        return entered_text

    except subprocess.CalledProcessError as e:
        # Handle any errors that occurred during execution
        print(f"An error occurred: {e.stderr}")
        return None


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
        
        color = COLORS[label % len(COLORS)]

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Put the label and score on the image
        text = f"{class_label} {score_str}"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    return frame

def send_frame_and_receive_results(frame, vocabolary, score_thresh):
    to_send_frame = cv2.resize(frame, TARGET_SIZE)
    try:
        response = requests.post(REMOTE_SERVER_URL, 
                                 data=json.dumps({
                                     "frame": to_send_frame.tolist(),
                                     "vocabulary": vocabolary,
                                     "score_thresh": score_thresh
                                 }), 
                                 timeout=5)
        response.raise_for_status()
        results = json.loads(response.text)
    except requests.exceptions.RequestException as e:
        return None
    return results

def webcam_capture_and_display():
    global LAST_FRAME, LAST_RESULTS, EXIT_FLAG, VOCABOLARY, SCORE_THRESH
    
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        # listen to the keyboard
        key = cv2.waitKey(1) & 0xFF
        
        LAST_FRAME = frame
        global LAST_RESULTS
        if not ret:  # Check if frame capture was successful
            continue  # Skip the current iteration and try again    
        
        if LAST_RESULTS is not None:
            frame = draw_bounding_boxes(frame, LAST_RESULTS, VOCABOLARY)
            
        cv2.namedWindow('Open-Vocabulary Object Detection', cv2.WINDOW_NORMAL)
        cv2.imshow('Open-Vocabulary Object Detection', frame)

        if key == ord('s'):
            # Get user input for the input vocabulary
            input = get_text_entry(";".join(VOCABOLARY), text="Write the input vocabulary. Different categories needs to be separeted by a ';'.")
            if input is None:
                continue
            VOCABOLARY = input.split(";")
        
        # Exit on ESC key
        if key == 27:  
            EXIT_FLAG = True
            print("Exiting...")
            break       
        
        # Check for up arrow key
        if key == 82:
            SCORE_THRESH += 0.01
            print("Score threshold: %s" % SCORE_THRESH)
        
        # down arrow
        if key == 84:
            SCORE_THRESH -= 0.01
            print("Score threshold: %s" % SCORE_THRESH)

def main():
    global LAST_RESULTS, EXIT_FLAG, VOCABOLARY, SCORE_THRESH
    
    capture_thread = threading.Thread(target=webcam_capture_and_display)
    capture_thread.start()
    
    while not EXIT_FLAG:
        if LAST_FRAME is not None:
            # print("Sending request...")
            LAST_RESULTS = send_frame_and_receive_results(LAST_FRAME, VOCABOLARY, SCORE_THRESH)
            # print("Results received!")

if __name__ == '__main__':
    main()
