'''
Author: Dhirodaatto Sarkar
Purpose: Club fair demo.
Performance: Less than mediocre (Problems with slow targets, requires target to stop before it shoots, no prediction features)
             Less than ideal for purpose
'''
from multiprocessing import Process, Value, Manager
from multiprocessing.managers import BaseManager
import numpy as np
from robomaster import robot
from robomaster import blaster
import keyboard
import cv2
import threading
import time

from ext_files import handDetector

robot_instance = None
cam_instance = None 
gimbal_instance = None

hd_instance = None
img = None
target_pos_x, target_pos_y = 0, 0
smode = 0

intensionx, intensiony = 640, 360

yaw_controller = None
pitch_controller = None

STOP = Value('i', 0)
DETECTED = Value('i', 0)
x_error = Value('d', 0)
y_error = Value('d', 0)

delt = 0.01

kernel_size = 20
mv_kernel = []

wtf = 0

def push_to_global_list(current):
    global mv_kernel
    global kernel_size
    if len(mv_kernel) == kernel_size:
        mv_kernel.pop(0)
    mv_kernel.append(current)

def compute_moving_avg():
    global mv_kernel
    x, y = np.mean(mv_kernel, axis = 0)
    x = int(x)
    y = int(y)
    return x,y

lock = threading.Lock()

def general_logistic(x, start, eend, growth_rate, mid_point):
    
    return start + (eend - start) / (1 + np.exp(-growth_rate * (x - mid_point)))
    

def plspositiveonly(x):
    if x > 0:
        return x
    else:
        return 0

class PID_Controller():
    '''
    Just a proportional controller
    '''
    e = [0, 0]
    kp = 0
    ki = 0
    kd = 0
    dt = 0
    
    e_int = 0
    e_dot = 0
    
    T = 0
    
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.dt = dt
    
    def getparams(self, error, dt = 0.01):
        e_ = self.e
        self.dt = dt
        e_[1] = error
        
        # self.e_dot = (e_[1] - e_[0])/self.dt
        # self.e_int = self.e_int + e_[1] * self.dt
        
        self.T = self.T + self.dt
        
        return self.kp * e_[1] #+ self.kd * self.e_dot + self.ki * (self.e_int / self.T)
            

def liveview():
    global cam_instance
    global hd_instance
    global img
    global target_pos_x
    global target_pos_y
    global STOP
    global lock
    global delt
    global state
    global wtf
    global smode
    
    while True:
        t1 = time.time()
        img = cam_instance.read_cv2_image()
        
        cv2.circle(img,(target_pos_x, target_pos_y), 5 , (255,0,255), cv2.FILLED)
        cv2.circle(img,(640, 360), 7 , (0,0,255), cv2.FILLED)
        cv2.putText(img, f'error = x : {x_error.value} , y : {y_error.value}',(50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f'state = {state}', (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f'shooting mode = {smode}', (50, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        delt = time.time() - t1
        cv2.imshow("Live View", img)
        
        c = cv2.waitKey(1)
        if c == ord('s'):
            smode = not smode
        elif c == 27:
            break
        # if c == 27:
        #     break
    
    cv2.destroyAllWindows()

def detect():
    global img
    global target_pos_x
    global target_pos_y
    global x_error
    global y_error
    global STOP
    global blaster_instance
    global delt
    global state
    
    global intensionx
    global intensiony
    
    global wtf
    global smode
    
    pitch_influence = 1
    yaw_influence = 1
    
    allowed = 0
    
    ptx = 0
    pty = 0
    count = 0
    
    while True:
        # t1 = time.time()
        if img is None:
            continue
        hd_instance.findHands(img)
        target_pos_x, target_pos_y = hd_instance.findCentralPosition(img)
        state = hd_instance.findstate(img)
        dx = hd_instance.handdistsq(img)
        wtf = dx
        DETECTED.value = 1
        if target_pos_x == -1 or target_pos_y == -1:
            DETECTED.value = 0
            x_error_ = 0
            y_error_ = 0
            blaster_instance.set_led(brightness=0, effect=blaster.LED_OFF)
            gimbal_instance.drive_speed(pitch_speed = 0, yaw_speed = 0)
        else:
            # offset = 80 + 30 * plspositiveonly(np.log(-1 * dx + 100))
            offset = general_logistic(dx, -20, 400, 2, 1)
            # offset = 200
            wtf = offset
            x_error_ = target_pos_x - intensionx 
            y_error_ = intensiony + offset - target_pos_y
            x_error.value = x_error_
            y_error.value = y_error_
            if abs(x_error.value) < 2 and abs(y_error.value) < 2 and smode:
                if state:
                    blaster_instance.set_led(brightness=10, effect=blaster.LED_ON)
                    allowed = allowed + 1
                    if allowed > 2:
                        blaster_instance.fire(times = 1)
                        allowed = 0
                else:
                    blaster_instance.set_led(brightness=0, effect=blaster.LED_OFF)
                # count = count + 1 
                # if count > 4 or count == 100:

                #     count = 0
            gimbal_instance.drive_speed(pitch_speed = pitch_influence * pitch_controller.getparams(y_error.value, delt), yaw_speed = yaw_influence * yaw_controller.getparams(x_error.value, delt))
    
        if STOP.value:
            break
        

def livecam():
    global cam_instance
    global STOP
    cam_instance.start_video_stream(display=False)
    keyboard.wait('esc')
    STOP.value = 1
    cam_instance.stop_video_stream()

if __name__ == "__main__":
    
    hd_instance = handDetector()
    yaw_controller = PID_Controller(.6, 0.1, 0, 0.02)
    pitch_controller = PID_Controller(.4, 0.01, 0, 0.02)
    state = None
    
    shooter = robot.Robot()
    
    shooter.initialize(conn_type = "ap")
    cam_instance = shooter.camera
    gimbal_instance = shooter.gimbal
    blaster_instance = shooter.blaster
    
    gimbal_instance.recenter().wait_for_completed()
    
    cam_thread = threading.Thread(target = livecam, args = ())
    display_thread = threading.Thread(target = liveview, args = ())
    detection_thread = threading.Thread(target = detect, args = ())
    
    robot_instance = shooter
    
    cam_thread.start()
    
    display_thread.start()
    detection_thread.start()
    
    detection_thread.join() 
    display_thread.join()
    
    cam_thread.join() 
    
    blaster_instance.set_led(brightness=0, effect=blaster.LED_OFF)
    gimbal_instance.drive_speed(pitch_speed = 0, yaw_speed = 0)
    gimbal_instance.recenter().wait_for_completed()
    shooter.close()
