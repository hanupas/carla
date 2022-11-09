import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
import tensorflow as tf
from tqdm import tqdm
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

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
# MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'carla'
MIN_REWARD = 0  # For model save
AVG_REWARD = 0
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 5

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
episode_reward = 0
step = 0
current_state = None
new_state = None

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

class DQNAgent:
	def __init__(self):

			# Main model
			self.model = self.create_model()

			# Target network
			self.target_model = self.create_model()
			self.target_model.set_weights(self.model.get_weights())

			# An array with last n steps for training
			self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

			# Custom tensorboard object
			self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

			# Used to count when to update target network with main network's weights
			self.target_update_counter = 0

	def create_model(self):
			model = Sequential()

			model.add(Conv2D(32, (5, 5), input_shape=OBSERVATION_SPACE_DIMS))  
			model.add(Activation('relu'))
			model.add(MaxPooling2D(pool_size=(2, 2)))
			model.add(Dropout(0.2))

			model.add(Conv2D(64, (3, 3)))
			model.add(Activation('relu'))
			model.add(MaxPooling2D(pool_size=(2, 2)))
			model.add(Dropout(0.2))

			model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
			model.add(Dense(64))

			model.add(Dense(ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
			model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
			return model

	def update_replay_memory(self, transition):
			self.replay_memory.append(transition)

	def get_qs(self, state):
			return self.model.predict(np.array([state]))

	def train(self, terminal_state, step):
			if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
					return
			minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

			current_states = np.array([transition[0] for transition in minibatch])/255
			current_qs_list = self.model.predict(current_states)

			new_current_states = np.array([transition[3] for transition in minibatch])/255
			future_qs_list = self.target_model.predict(new_current_states)

			X = []	#feature sets (iamges from the game)
			y = []	#label sets (actions we can possibly take)

			for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
					if not done:
							max_future_q = np.max(future_qs_list[index])	#highest Q(s', a') value where s' is the next state
							new_q = reward + DISCOUNT * max_future_q	
					else:
							new_q = reward

						#update Q value for given state
					current_qs = current_qs_list[index]    #current Q(s, a) values for current state
					current_qs[action] = new_q	#update how good the given action at the given current state is
					
					X.append(current_state)
					y.append(current_qs)

			self.model.fit(np.array(X)/255, np.array(y), batch_size = MINIBATCH_SIZE, verbose=0, 
							shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

			#updating to determine if we want to update target_model yet
			if terminal_state:
					self.target_update_counter += 1

			if self.target_update_counter > UPDATE_TARGET_EVERY:
					self.target_model.set_weights(self.model.get_weights())
					self.target_update_counter = 0
			


def process_img(image):
	global episode_reward
	global current_state
	global new_state
	global step

	i = np.array(image.raw_data)
	i2 = i.reshape(IM_HEIGHT, IM_WIDTH,4)
	i3 = i2[:,:,:3]
	cv2.imshow("",i3)
	cv2.waitKey(1)

	new_state = i3
	if np.random.random() > epsilon:
		action = np.argmax(agent.get_qs(image))
	else:
		action = np.random.randint(0, len(steer))
	
	vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=steer[action]))
	speed = 3.6 * math.sqrt(vehicle.get_velocity().x**2 + vehicle.get_velocity().y**2 + vehicle.get_velocity().z**2)
	
	reward = speed
	done = False
	info = ""

	agent.update_replay_memory((current_state, action, reward, new_state, done))

	current_state = new_state
	episode_reward += speed

	step += 1
	agent.train(done,step)

	
	


actor_list = []
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(2.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

agent = DQNAgent()

for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episode"):
	agent.tensorboard.step = episode

	episode_reward = 0
	step = 1

	# Spawn car
	model_3 = blueprint_library.filter("model3")[0]
	spawnpoint = world.get_map().get_spawn_points()
	vehicle = world.spawn_actor(model_3,spawnpoint[1])
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

	os.system('cls' if os.name == 'nt' else 'clear')
	print(f"EPISODE = {episode} and EPISODE REWARD = {episode_reward}")

	done = False
	agent.train(done,step)


