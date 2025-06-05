import cv2
import pyvirtualcam
import os
os.environ['GLOG_minloglevel'] = '2'

from reactions import Reactions

cap = cv2.VideoCapture(0)
reactions_handler = Reactions()

with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
    print(f'Using virtual camera: {cam.device}')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        reactions_handler.process_frame(frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam.send(rgb_frame)
        cam.sleep_until_next_frame()
