#Importing OpenCV Library for basic image processing functions
import cv2
# Numpy for array related functions
import numpy as np
# Dlib for deep learning based Modules and face landmark detection
import dlib
#face_utils for basic operations of conversion
from imutils import face_utils

import winsound 

import time

night_mode = False
prev_time = 0

#Initializing the camera and taking the instance
cap = cv2.VideoCapture(0)

#Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#status marking for current state
sleep = 0
drowsy = 0
active = 0
status=""
color=(0,0,0)

def compute(ptA,ptB):
	dist = np.linalg.norm(ptA - ptB)
	return dist

def blinked(a,b,c,d,e,f):
	up = compute(b,d) + compute(c,e)
	down = compute(a,f)
	ratio = up/(2.0*down)

	#Checking if it is blinked
	if(ratio>0.25):
		return 2
	elif(ratio>0.21 and ratio<=0.25):
		return 1
	else:
		return 0
     
def mouth_open(mouth):
    A = compute(mouth[2], mouth[10])
    B = compute(mouth[4], mouth[8])
    C = compute(mouth[0], mouth[6])

    mar = (A + B) / (2.0 * C)
    return mar

while True:
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    _, frame = cap.read()
    face_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if night_mode:
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

    faces = detector(gray)
    if len(faces) == 0:
        status = "No Face Detected"
        color = (0, 255, 255)
    #detected face in faces array
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        #The numbers are actually the landmarks which will show eye
        left_blink = blinked(landmarks[36],landmarks[37], 
        	landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42],landmarks[43], 
        	landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        mouth = landmarks[48:68]
        mar = mouth_open(mouth)
        
        #Now judge what to do for the eye blinks
        if(left_blink == 0 and right_blink == 0):
            sleep += 1
            drowsy = 0
            active = 0

            if(sleep > 15):
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                winsound.Beep(1000, 500)  # frequency, duration

        elif(left_blink == 1 or right_blink == 1):
            sleep = 0
            active = 0
            drowsy += 1
            if(drowsy > 6):
                status = "Drowsy !"
                color = (0,0,255)

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if(active > 6):
                status = "Active :)"
                color = (0,255,0)

        if mar > 0.65:
            yawn_count += 1
        else:
            yawn_count = 0

        if yawn_count > 20:
            status = "Yawning!"

        cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"FPS: {int(fps)}", (20,50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        mode_text = "Night Mode ON" if night_mode else "Normal Mode"
        cv2.putText(frame, mode_text, (20,120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        for (x, y) in landmarks:
            cv2.circle(face_frame, (x, y), 1, (255,255,255), -1)
        
        cv2.imshow("Frame", frame)
        cv2.imshow("Result of detector", face_frame)
    
    key = cv2.waitKey(1)
    if key == ord('n'):
        night_mode = not night_mode
    if key == 27:
      	break
cap.release()
cv2.destroyAllWindows()