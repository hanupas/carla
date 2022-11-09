import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Embedding, Reshape
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Environment settings
EPISODES = 100

# UKURAN FILE
IM_WIDTH = 80
IM_HEIGHT = 80

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 1  # episodes
SHOW_PREVIEW = True

OBSERVATION_SPACE_DIMS = (IM_WIDTH, IM_HEIGHT, 3)
ACTION_SPACE_SIZE = 5

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

		# Overriding init to set initial step and writer (we want one log file for all .fit() calls)
		def __init__(self, **kwargs):
			super().__init__(**kwargs)
			self.step = 1
			self.writer = tf.summary.create_file_writer(self.log_dir)
			self._log_write_dir = self.log_dir

		def set_model(self, model):
			self.model = model

			self._train_dir = os.path.join(self._log_write_dir, 'train')
			self._train_step = self.model._train_counter

			self._val_dir = os.path.join(self._log_write_dir, 'validation')
			self._val_step = self.model._test_counter

			self._should_write_train_graph = False

		def on_epoch_end(self, epoch, logs=None):
			self.update_stats(**logs)

		def on_batch_end(self, batch, logs=None):
			pass

		def on_train_end(self, _):
			pass
		
		def update_stats(self, **stats):
			self._write_logs(stats, self.step)

		def _write_logs(self, logs, index):
			with self.writer.as_default():
				for name, value in logs.items():
					tf.summary.scalar(name, value, step=index)
					self.step += 1
					self.writer.flush()

def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape(IM_HEIGHT, IM_WIDTH,4)
    # print(i2.shape)
    i3 = i2[:,:,:3]
    cv2.imshow("",i3)
    cv2.waitKey(1)
    steer_chosen = random.choice(steer)
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=steer_chosen))
    speed = 3.6 * math.sqrt(vehicle.get_velocity().x**2 + vehicle.get_velocity().y**2 + vehicle.get_velocity().z**2)
    print(f"speed={speed} km/h")


actor_list = []
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(2.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# Spawn car
model_3 = blueprint_library.filter("model3")[0]
spawnpoint = world.get_map().get_spawn_points()
vehicle = world.spawn_actor(model_3,spawnpoint[1])
# vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1.0))
steer = [-1,0,1]
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
