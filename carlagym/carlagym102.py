import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math

# Import environment base class for a wrapper 
from gym import Env

# Temporary
# Import the space shapes for the environment
from gym.spaces import MultiBinary, Box

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Create custom environment 
class CarlaEnv(Env): 
    def __init__(self):
        super().__init__()

        # Specify action space and observation space 
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        
    
    def reset(self):
        # Return the first frame 
        pass
    
    def preprocess(self, observation): 
        pass 
    
    def step(self, action): 
        pass
    
    def render(self, *args, **kwargs):
        pass
        
    def close(self):
        pass

IM_WIDTH = 320
IM_HEIGHT = 240

def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape(IM_HEIGHT, IM_WIDTH,4)
    # print(i2.shape)
    i3 = i2[:,:,:3]
    cv2.imshow("",i3)
    cv2.waitKey(1)
    return i3/255.0

actor_list = []
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(2.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

env = CarlaEnv()
print(f"Observation space {env.observation_space.shape}")

# Spawn car
model_3 = blueprint_library.filter("model3")[0]
spawnpoint = world.get_map().get_spawn_points()
vehicle = world.spawn_actor(model_3,spawnpoint[1])
vehicle.set_autopilot(True)
actor_list.append(vehicle)

################# CAMERA #############################
cam_bp = blueprint_library.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x",f"{IM_WIDTH}")
cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
cam_bp.set_attribute("fov", "110")

spawn_point = carla.Transform(carla.Location(x=2.5,z=0.7))
sensor = world.spawn_actor(cam_bp,spawn_point, attach_to=vehicle)
actor_list.append(sensor)
sensor.listen(lambda data: process_img(data))

time.sleep(10)
for actor in actor_list:
    print(actor)
    actor.destroy()
print("finish")
