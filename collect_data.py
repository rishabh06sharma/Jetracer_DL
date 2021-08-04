import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import time
import keyboard  # using module keyboard
from jetracer.nvidia_racecar import NvidiaRacecar
from jetcam.csi_camera import CSICamera

"""
This file control the vehicle and collects labelled data (Images are labelled with current state)
    Current state: Steering angle and throttle
    Key Bindings:
        I: Accelerate
        J: Turn Left
        L: Turn Right
        S: Save data and stop
        A: Stop (not saved)
        Space: Break
"""


def collect_data(train,throttle,steering,s_gain):

    # Importing NvidiaRacecar modeule and CSICamera
    # Note: CSICamera object can be created once at a time
    camera = CSICamera(width=112, height=112)
    camera.running = True
    car = NvidiaRacecar()

    # initialize parameters: 0
    car.throttle = throttle
    car.steering = steering
    car.throttle_gain = -0.5

    # file name
    if not os.path.exists("data/"):
    os.makedirs("data/")
    file_name=input("File_name: ")


    try:
        while(True):
            # Display steering and Throttle
            print("Steering: {}, Throttle {}" .format(car.steering,car.throttle))

            # Increase throttle on press
            if keyboard.is_pressed('i'):
                if car.throttle<=0.04:
                    car.throttle=0.04
                if car.throttle>=0.062:
                    car.throttle=car.throttle
                else:
                    car.throttle=car.throttle+0.001
            # Decrease throttle on release
            if not keyboard.is_pressed('i'):
                if car.throttle<=0.04:
                    car.throttle=0.04
                else:
                    car.throttle=car.throttle-0.01
                    
            # turn left
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

            # Resease steering back to steering angle of 0 degrees
            if not keyboard.is_pressed('j') and not keyboard.is_pressed('l'):
                if not keyboard.is_pressed('j'):
                    
                    if car.steering<=0:
                        car.steering=car.steering
                    else:
                        car.steering =car.steering-0.015
                if not keyboard.is_pressed('l'):
                    
                    if car.steering>=0:
                        car.steering=car.steering
                    else:
                        car.steering =car.steering+0.015 
            
            # Emergency break 
            if keyboard.is_pressed('space'):
                car.throttle=-100
                time.sleep(0.1)
                car.throttle=0

            # Save the recording labelled data: Press
            if keyboard.is_pressed('s'):
                car.throttle=0
                car.steering=0
                print("saving")
                np.savez("data/"+str(file_name)+".npz", train=np.array(train))
                print("saved")
                camera.running=False
                del camera
                break
            
            # Abort    
            if keyboard.is_pressed('a'):
                car.throttle=0
                car.steering=0
                camera.running=False
                del camera
                break
                
            # append recording    
            train.append([camera.value,car.throttle,car.steering])
    except:
        camera.running=False
        del camera


if __name__ == '__main__':
    # input: train,throttle,steering,s_gain
    collect_data([],0,0,-0.5)
    