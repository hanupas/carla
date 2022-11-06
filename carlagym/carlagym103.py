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

        self.IM_WIDTH = 320
        self.IM_HEIGHT = 240

        self.actor_list = []
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        # Specify action space and observation space 
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        
    
    def reset(self):
        # # Spawn car
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.spawnpoint = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.spawn_actor(self.model_3,self.spawnpoint[1])
        self.vehicle.set_autopilot(True)
        self.actor_list.append(self.vehicle)

        ################# CAMERA #############################
        self.cam_bp = self.blueprint_library.find("sensor.camera.rgb")
        self.cam_bp.set_attribute("image_size_x",f"{self.IM_WIDTH}")
        self.cam_bp.set_attribute("image_size_y", f"{self.IM_HEIGHT}")
        self.cam_bp.set_attribute("fov", "110")

        self.spawn_point = carla.Transform(carla.Location(x=2.5,z=0.7))
        self.sensor = self.world.spawn_actor(self.cam_bp,self.spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.sensor) 
    
    def preprocess(self, observation): 
        pass 
    
    def step(self, action): 
        pass
    
    def render(self, *args, **kwargs):
        pass
        
    def close(self):
        for actor in self.actor_list:
            print(actor)
            actor.destroy()
        print("Program Close")

env = CarlaEnv()
env.reset()
time.sleep(10)
env.close()


