import gym
import time
#import theano
from keras.models import Sequential
from keras.layers import Dense
#import tensorflow
import numpy as np
import random


###Taken from blog example code
def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, len(r))):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * 0.99 + r[t]
    discounted_r[t] = running_add
  return discounted_r

###


model = Sequential()
model.add(Dense(units=200, activation='relu', input_shape=(6400,)))
model.add(Dense(units=2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

env = gym.make('Pong-v0')
for i_episode in range(20):
	observation = env.reset()
	frames = []
	rewards = []
	action = env.action_space.sample()
	actions = []
	prev_x = 0
	for t in range(1000):
		#time.sleep(0.1)
		env.render()
		#print(observation)
		
		observation, reward, done, info = env.step(action)

		#Reduce dimensionality
		cur_x = prepro(observation)

		#Get difference framce
		diff_x = cur_x-cur_x if t==0 else cur_x-prev_x

		#Get output from network, and sample from distribution
		classes = model.predict(x=np.matrix(diff_x), batch_size=1)
		print(classes)
		action = 2 if classes[0][0]>random.random() else 3


		#Build dataset for training
		frames.append(diff_x)
		actions.append(action)
		rewards.append(reward)

		print(env.action_space, env.observation_space)
		print(action,reward)

		prev_x = cur_x
		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break


	rewards = discount_rewards(rewards)
	#rewards -= np.mean(rewards)
	#rewards /= np.std(rewards)

	#Build labels. Encourage positive actions, discourage negative.
	labels = []
	for i in range(len(rewards)):
		if rewards[i]<0:
			#opposite_action = 1 if actions[i] == 2 else 0
			label = [0,0]
			label[(actions[i]+1)%2] = -rewards[i]
			label[actions[i]%2] = 1+rewards[i]
			labels.append(label)
		else:
			#positive_action = 0 if actions[i] == 2 else 1
			label = [0,0]
			label[actions[i]%2] = rewards[i]
			label[(actions[i]+1)%2] = 1-rewards[i]
			labels.append(label)

	#Train
	for i in range(len(frames)):
		model.train_on_batch(np.matrix(frames[i]), np.matrix(labels[i]))







