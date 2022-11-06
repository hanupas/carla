import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math

# Import environment base class for a wrapper 
from gym import Env, spaces

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
        self.n_acc = len([-3.0, 0.0, 3.0]),  # discrete value of accelerations
        self.n_steer = len([-0.2, 0.0, 0.2]),  # discrete value of steering angles\
        self.actor_list = []
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.IM_WIDTH, self.IM_HEIGHT, 3), dtype=np.uint8)
        self.action_list = [-0.2, 0.0, 0.2]
        self.action_space = spaces.MultiBinary(len(self.action_list))
    
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
        observation = -1
        reward = 0
        done =  1
        info = 2
        return observation, reward, done, info
    
    def render(self, *args, **kwargs):
        i = np.array(image.raw_data)
        i2 = i.reshape(IM_HEIGHT, IM_WIDTH,4)
        print(i2.shape)
        i3 = i2[:,:,:3]
        cv2.imshow("",i3)
        cv2.waitKey(1)
        
    def close(self):
        for actor in self.actor_list:
            print(actor)
            actor.destroy()
        print("Program Close")

env = CarlaEnv()
env.render()
obs, reward, done, info = env.step(1)
print(f"obs={obs}; reward={reward}; done={done}, info={info}")
env.reset()
time.sleep(10)
env.close()


