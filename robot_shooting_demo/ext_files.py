'''
Taken from external source on hand tracking, with helper functions
'''

import cv2
import mediapipe as mp
import numpy as np
import time

def qdist(p1, p2):
    return (p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2
    
class handDetector():
    
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,
                                        self.maxHands,
                                        self.modelComplex,
                                        self.detectionCon,
                                        self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils 
        self.cx = 0
        self.cy = 0

    def findHands(self,img):
        # Send rgb image to hands
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) 

        return img

    def findCentralPosition(self,img):

        self.cx, self.cy = -1, -1
        
        # check wether any landmark was detected
        if self.results.multi_hand_landmarks:
            #Which hand are we talking about
            myHand = self.results.multi_hand_landmarks[0]
            # Get id number and landmark information
            p0 = myHand.landmark[0]
            p1 = myHand.landmark[5]
            p2 = myHand.landmark[17]
            
            cx_ = (p0.x + p1.x + p2.x) / 3
            cy_ = (p0.y + p1.y + p2.y) / 3
            
            h,w,c = img.shape
            
            self.cx, self.cy = int(cx_ * w), int(cy_ * h)

        return self.cx, self.cy
    
    def findstate(self, img):
        
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[0]
            
            p0 = myHand.landmark[0]
            
            p1 = myHand.landmark[4]
            p2 = myHand.landmark[8]
            p3 = myHand.landmark[12]
            p4 = myHand.landmark[16]
            p5 = myHand.landmark[20]
            
            pb1 = myHand.landmark[1]
            pb2 = myHand.landmark[5]
            pb3 = myHand.landmark[9]
            pb4 = myHand.landmark[13]
            pb5 = myHand.landmark[17]
            
            if qdist(p0,p1) > qdist(p0,pb1) and qdist(p0,p2) > qdist(p0,pb2) and qdist(p0,p3) > qdist(p0,pb3) and qdist(p0,p4) > qdist(p0,pb4) and qdist(p0,p5) > qdist(p0,pb5):
                return 1
            return 0
    
    def handdistsq(self, img):
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[0]
            p0 = myHand.landmark[0]
            
            p3 = myHand.landmark[12]
            
            return qdist(p0, p3)