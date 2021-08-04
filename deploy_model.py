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


"""
This file takes the lateral and longitudinal .pth file and use it to predict the action
1. Import .pth file
2. Load the netork into Network class as an object
3. Aquire image (with sampling)
4. Predict the action (steering and throttle)
5. Use predicted value to control the vehicle


Key Binding:
    A: Stop the car
"""

# Use cuda if available
if torch.cuda.is_available():
    device =torch.device('cuda')
    print("Running on the GPU")
else:
    device=torch.device("cpu")
    print("Running on the CPU")

# Deep neural Nework
class Network(nn.Module):

    # structure init
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
        # normalization layer
        t = self.norm(t.float())
        
        # 5 convolutional layers
        t = F.elu(self.conv1(t.float()))
        t = F.max_pool2d(t, kernel_size=2)
        
        t = F.elu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2)
        
        t = F.elu(self.conv3(t))
        t = F.max_pool2d(t, kernel_size=2)

        t = F.elu(self.conv4(t))
        t = F.elu(self.conv5(t))
        
        t = self.drop(t)

        # 4 fully connected layer
        t = t.reshape(-1,t.shape[1]*t.shape[2]*t.shape[3])
        t = F.elu(self.fc1(t))
        t = F.elu(self.fc2(t))
        t = F.elu(self.fc3(t))
        t = F.elu(self.fc4(t))
        t = self.out(t)

        return t



def deploy(steering_model_path, throttle_model_path, throttle_gain, image_sampling_rate, throttle, steering, s_gain):

    # Importing NvidiaRacecar modeule and CSICamera
    # Note: CSICamera object can be created once at a time
    camera = CSICamera(width=224, height=224)
    camera.running = True
    car = NvidiaRacecar()

    # initialize parameters: 0
    car.throttle = throttle
    car.steering = steering
    car.throttle_gain = -0.5

    # Import trained models
    # Lateral model
    network_lateral = network_lateral().to(device)
    network_lateral.load_state_dict(torch.load(steering_model_path))

    # Longituidnal model
    network_longitudinal = Network().to(device)
    network_longitudinal.load_state_dict(torch.load(throttle_model_path))

    image_count = 0
    while True:
        image_count += 1

        # Sample image from every 5 (image_sampling_rate) images
        if image_count % image_sampling_rate ==0:

            # Aquire image and reshape as per requirement
            gray = torch.tensor(cv2.cvtColor(camera.value, cv2.COLOR_BGR2GRAY)).unsqueeze(1)
            gray = gray.reshape(1,1,gray.shape[2],gray.shape[2]).to(device)

            # Model Prediction
            with torch.no_grad():
                # throttle and steering prediction
                car.steering = network_lateral(gray).to("cpu").item()
                car.throttle = (network_longitudinal(gray).to("cpu").item()) * throttle_gain
                print("the steering angle is {}, the throttle is {}" .format(car.steering,car.throttle))
                    
        # Avoid overflow
        if image_count == 500:
            image_count = 0

        # Stop Prediction (Press 'A')           
        if keyboard.is_pressed('a'):
            car.throttle = -100 ## Break
            car.throttle, car.steering = 0, 0
            camera.running = False
            del camera
            break


if __name__ == '__main__':
    # input: train,throttle,steering,s_gain
    image_sampling_rate = 5
    throttle_gain = 1.2
    steering_model_path = "steer_set3_12ep_100_0.001_rsz_pt4.pth"
    throttle_model_path = "acc_set3_18ep_100_0.0001_rsz_pt4.pth"
    collect_data(steering_model_path, throttle_model_path, throttle_gain, image_sampling_rate, 0 ,0 , -0.5)