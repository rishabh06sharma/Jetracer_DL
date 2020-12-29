## This file is used todrive the car and collect supervised data


import keyboard  # using module keyboard
from jetracer.nvidia_racecar import NvidiaRacecar
import time
car = NvidiaRacecar()


# Importing NvidiaRacecar modeule
# Note: CSICamera object can be created once at a time
from jetcam.csi_camera import CSICamera
camera = CSICamera(width=112, height=112)
camera.running = True
import os



import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
            
car.throttle=0
car.steering=0
car.throttle_gain=-0.5
train=[]
pointer=1


if not os.path.exists("data/"):
    os.makedirs("data/")

print("File_name: ")
file_name=input()
try:
    while(True):
        ## printing trottle | steering
        print("the steering value is {}, the throttle is {}" .format(car.steering,car.throttle))

        ##### Dynamic throttle
        if keyboard.is_pressed('i'):
            if car.throttle<=0.043920292802546:
                car.throttle=0.043920292802546
            if car.throttle>=0.062:
                car.throttle=car.throttle
            else:
                car.throttle=car.throttle+0.001

        if keyboard.is_pressed('i')==False:
            if car.throttle<=0.043920292802546:
                car.throttle=0.043920292802546
            else:
                car.throttle=car.throttle-0.01
                
        ### turn left
        if keyboard.is_pressed('j'):
            if car.steering>=0.6:
                car.steering=car.steering
            else:
                car.steering =car.steering+0.01
        
        ### turn right
        if keyboard.is_pressed('l'):
            if car.steering<=-0.6:
                car.steering=car.steering
            else:
                car.steering =car.steering-0.01


        ### Dynamic steering
        if keyboard.is_pressed('j')==False and keyboard.is_pressed('l')==False:
            if keyboard.is_pressed('j')==False:
                
                if car.steering<=0:
                    car.steering=car.steering
                else:
                    car.steering =car.steering-0.015
            if keyboard.is_pressed('l')==False:
                
                if car.steering>=0:
                    car.steering=car.steering
                else:
                    car.steering =car.steering+0.015
        
        
        # Emergency break 
        if keyboard.is_pressed('space'):
            car.throttle=-100
            time.sleep(0.1)
            car.throttle=0

        # Save the recording labelled data
        if keyboard.is_pressed('s'):
            car.throttle=0
            car.steering=0
            print("saving")
            np.savez("data/"+str(file_name)+".npz", train=np.array(train))
            print("saved")
            camera.running=False
            del camera
            break
        
        ### abort    
        if keyboard.is_pressed('a'):
            car.throttle=0
            car.steering=0
            camera.running=False
            del camera
            break
            
        # append recording    
        train.append([camera.value,car.throttle,car.steering])
except:
    np.savez("data/"+str(file_name)+".npz", train=np.array(train))
    camera.running=False
    del camera
    