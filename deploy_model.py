import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import cv2
import keyboard  # using module keyboard

#### This file is used to deploy the lateral and longituidinal model onto the car

# Use cuda if available
if torch.cuda.is_available():
    device =torch.device('cuda')
    print("Running on the GPU")
else:
    device=torch.device("cpu")
    print("Running on the CPU")


# Importing NvidiaRacecar modeule
from jetracer.nvidia_racecar import NvidiaRacecar
import time
car = NvidiaRacecar()

# Importing NvidiaRacecar modeule
# Note: CSICamera object can be created once at a time
from jetcam.csi_camera import CSICamera
camera = CSICamera(width=224, height=224)
camera.running = True


# Initial throttle and steering values           
car.throttle=0
car.steering=0
car.throttle_gain=-0.5


# Deep neural Nework
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=(224,224))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=3,stride=1)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=3,stride=1)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.drop=nn.Dropout(0.7)

        self.fc1 = nn.Linear(in_features=64*8*8, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=500)
        self.fc3 = nn.Linear(in_features=500, out_features=100)
        self.fc4 = nn.Linear(in_features=100, out_features=10)
        self.out = nn.Linear(in_features=10, out_features=1)

    def forward(self, t):
        t=self.norm(t.float())
        self.nor=t
        
        t = F.elu(self.conv1(t.float()))
        t = F.max_pool2d(t, kernel_size=2)
        self.c1=t
        
        t = F.elu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2)
        self.c2=t
        
        
        t = F.elu(self.conv3(t))
        t = F.max_pool2d(t, kernel_size=2)
        self.c3=t


        t = F.elu(self.conv4(t))
        self.c4=t
        t = F.elu(self.conv5(t))
        self.c5=t
        
        t=self.drop(t)
        t=t.reshape(-1,t.shape[1]*t.shape[2]*t.shape[3])
        t=F.elu(self.fc1(t))
        t=F.elu(self.fc2(t))
        t=F.elu(self.fc3(t))
        t=F.elu(self.fc4(t))
        t=self.out(t)
        return t

# Import trained models
# Lateral model
network = Network().to(device)
network.load_state_dict(torch.load("steer_set3_12ep_100_0.001_rsz_pt4.pth"))

# Longituidnal model
network_1 = Network().to(device)
network_1.load_state_dict(torch.load("acc_set3_18ep_100_0.0001_rsz_pt4.pth"))

# Deploy model
# Pointer>> to control sampling rate
pointer=0
image_sampling_rate=5
throttle_gain=1.2
while True:
    pointer+=1
    if pointer%5==0:
        try:
#             gray=torch.tensor(cv2.resize(cv2.cvtColor(camera.value, cv2.COLOR_BGR2GRAY),dsize = (112,112), interpolation = cv2.INTER_CUBIC)).unsqueeze(1) ## resize image
            gray=torch.tensor(cv2.cvtColor(camera.value, cv2.COLOR_BGR2GRAY)).unsqueeze(1)
            gray=gray.reshape(1,1,gray.shape[2],gray.shape[2]).to(device)

            # Deployment without saving gradiants
            with torch.no_grad():
                car.steering=network(gray).to("cpu").item()
                car.throttle=(network_1(gray).to("cpu").item()) * throttle_gain
                print("the steering angle is {}, the throttle is {}" .format(car.steering,car.throttle))
                
        except:
            car.throttle=0
            car.steering=0
            camera.running=False
            del camera
            break
    if pointer==500: # reset
        pointer=0
                
    if keyboard.is_pressed('a'): ## abort
        car.throttle=-100 ## Break
        car.throttle=0
        car.steering=0
        camera.running=False
        del camera
        break

