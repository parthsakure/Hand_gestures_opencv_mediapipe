import math
import threading
import cv2
import mediapipe as mp
import queue
import numpy as np
from time import sleep,time
import pyautogui as pgui

from pynput.keyboard import Key, Controller
from pynput.mouse import Controller as ctrlr

mouse = ctrlr()
kb = Controller()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
q = queue.Queue()

cap = cv2.VideoCapture(0)

last_time = 0
pt = ct = 0
fx = fy = 0
last_len = 0
llx = lly = 0
srange = 50
offset = 100
current_pros = -1
# cap.set(10, 0.3)
# cap.set(11, 1)
# cap.set(13, 255)
# cap.set(15, 0.2)

def dist(n1,n2):
    return math.hypot(n1[0]-n2[0],n1[1]-n2[1])

def isinrange(coor):
    return coor[0] < WIDTH - offset and coor[0] > offset and coor[1] < HEIGHT-offset and coor[1] > offset


def check_fingers(landmarks):
    l = [False, False, False, False, False]
    c = 0
    if landmarks[4][0] < landmarks[1][0]:
        l[0] = True
        c += 1
    if landmarks[8][1] < landmarks[5][1]:
        l[1] = True
        c += 1
    if landmarks[12][1] < landmarks[6][1]:
        l[2] = True
        c += 1
    if landmarks[16][1] < landmarks[10][1]:
        l[3] = True
        c += 1
    if landmarks[20][1] < landmarks[13][1]:
        l[4] = True
        c += 1
    return c, l



with mp_hands.Hands(max_num_hands=1, model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        HEIGHT, WIDTH, CHANNELS = img.shape

        cv2.rectangle(img, (offset, offset), (WIDTH-offset, HEIGHT-offset), (255, 255, 0), 3)
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb_img)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                lm = {}
                for id, lms in enumerate(hand_landmarks.landmark):
                    lm[id] = [int(lms.x*WIDTH), int(lms.y*HEIGHT)]

                count, fingers = check_fingers(lm)
                rad = dist(lm[9],lm[0])
                
                # cv2.circle(img, ((lm[8][0]+lm[4][0])//2,(lm[4][1]+lm[8][1])//2),int(rad*0.1),(0,255,0), 2)
                # cv2.circle(img, (lm[9][0], lm[9][1]),int(rad*2),(0,255,0), 2)
                # col = (0,0,255)
                # if dist(lm[12],lm[4]) < rad*0.5 and dist(lm[12],lm[8]) < rad*0.5 and dist(lm[12],lm[16]) < rad*0.5 and dist(lm[12],lm[20]) < rad*0.5:
                #     col = (0,255,0)
                # cv2.circle(img, (lm[12][0], lm[12][1]),int(rad*0.25),col, 2)
                # cv2.circle(img, (lm[8][0], lm[8][1]),int(rad*1),(0,255,0), 2)


                cv2.putText(img, str(count)+" "+str(fingers), (600, 50), cv2.FONT_HERSHEY_PLAIN, 1.2, color=(255, 0, 255))
                if isinrange(lm[8]):

                    if dist(lm[12],lm[4]) < rad*0.35 and dist(lm[12],lm[8]) < rad*0.35 and dist(lm[12],lm[16]) < rad*0.35 and dist(lm[12],lm[20]) < rad*0.35 and fingers[0] and fingers[1]:
                         
                        if time()-last_time > 1:
                            kb.tap(Key.media_play_pause)
                            last_time = time()
                        

                    elif fingers[0] and fingers[1] and fingers[2] and fingers[3] and fingers[4]:
                        if fx == 0 or fy == 0 or current_pros != 0:
                            fx = lm[9][0]
                            fy = lm[9][1]
                            current_pros = 0
                        else:
                            # cv2.line(img, (fx,fy), (lm[9][0],lm[9][1]), (242,220,10), 2)
                            length = fx-lm[9][0]
                            cv2.putText(img, str(length), (WIDTH-offset, 50), cv2.FONT_HERSHEY_PLAIN, 1.3, (242, 220, 10), 2)
                             
                            if last_len-length < -rad*3 and  time()- last_time > 1:
                                pgui.keyDown('ctrl')
                                pgui.press('right')
                                pgui.keyUp('ctrl')
                                last_len = length
                                last_time = time()

                            elif last_len-length > rad*3 and  time()- last_time > 1:
                                pgui.keyDown('ctrl')
                                pgui.press('left')
                                pgui.keyUp('ctrl')
                                last_len = length
                                last_time = time()
                    
                    elif fingers[0] and fingers[1] and not(fingers[2] or fingers[3] or fingers[4]):
                        length = int(dist(lm[4],lm[8]))
                        if length < int(rad*0.25):
                            cx = (lm[4][0] + lm[8][0])//2
                            cy = (lm[4][1] + lm[8][1])//2
                            if fx == 0 or fy ==0 or current_pros != 1:
                                fx = cx
                                fy = cy
                                llx = 0
                                lly = 0
                                current_pros = 1
                            else:
                                # cv2.line(img, (fx,fy), (cx,cy), (46, 248, 255), 2)
                                lx = fx-cx
                                ly = fy-cy
                                cv2.putText(img, "X: "+str(lx)+" Y: "+str(ly), (WIDTH-2*offset, 50), cv2.FONT_HERSHEY_PLAIN, 1.3, (46, 248, 255), 2)

                                ly = int(np.interp(ly, (-rad*2,rad*2), (-5,5)))
                                if ly-lly > 0:
                                    for i in range(ly-lly):
                                        kb.tap(Key.media_volume_up)
                                        sleep(0.1)
                                    lly = ly
                                    
                                else:
                                    for i in range(lly-ly):
                                        kb.tap(Key.media_volume_down)
                                        sleep(0.1)
                                    lly = ly
                                 
                                if lx-llx < -rad and  time()-last_time > 1:
                                    kb.tap(Key.media_next)
                                    llx = lx
                                    last_time = time()
                                
                                elif lx-llx > rad and  time()- last_time > 1:
                                    kb.tap(Key.media_previous)
                                    llx = lx
                                    last_time = time()
                                    

                        else:
                            fx=fy=0
                            llx = 0
                            lly = 0

                        


                    elif fingers[1] and fingers[2] and not(fingers[3] or fingers[4]):
                        if fx == 0 or fy == 0 or current_pros != 2:
                            fx = lm[8][0]
                            fy = lm[8][1]
                            current_pros = 2
                        else:
                            # cv2.line(img, (fx, fy), (lm[8][0], lm[8][1]), (0, 0, 255), 3)
                            length = fy-lm[8][1]
                            length = int(((length+rad)/(2*rad))*(2*srange) - srange)
                            if last_len-length:
                                mouse.scroll(0,last_len-length)
                                last_len = length
                            cv2.putText(img, str(last_len-length),(WIDTH-offset,50), cv2.FONT_HERSHEY_PLAIN, 1.3,(0,0,255),2)
                            cv2.putText(img, str(length),(WIDTH-offset-offset,50), cv2.FONT_HERSHEY_PLAIN, 1.3,(235,9,96),2)

                    else: 
                        fx=fy=0
                        last_len = 0
                        llx = 0
                        lly = 0
                        current_pros = -1

                # cv2.putText(img, "{:.2f}".format(last_time%100),(WIDTH-3*offset,50), cv2.FONT_HERSHEY_PLAIN, 1.3,(0,0,255),2)
                # cv2.putText(img, "{:.2f}".format(time()%100),(WIDTH-3*offset,70), cv2.FONT_HERSHEY_PLAIN, 1.3,(0,0,255),2)
                    

                # mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        ct = time()
        fps = 1/(ct-pt)
        pt = ct

        cv2.putText(img, "FPS: {}".format(int(fps)), (20, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, color=(0, 0, 255))

        cv2.imshow('Hands', img)
        cv2.waitKey(1)

