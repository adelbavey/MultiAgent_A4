import gym
import time
#import theano
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD
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
    running_add = running_add * 0.95 + r[t]
    discounted_r[t] = running_add
  return discounted_r

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)

###


model = Sequential()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.add(Dense(units=256, activation='relu',kernel_initializer='random_uniform', input_shape=(1,6400)))
#model.add(Dropout(0.5))
#model.add(Dense(units=200, activation='relu'))
#model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=2, activation='softmax', kernel_initializer='random_uniform'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


env = gym.make('Pong-v0')
frames = []
rewards = []
predProbs = []
actions = []

for i_episode in range(1,2000):
	observation = env.reset()
	prev_x = 0
	action = env.action_space.sample()
	for t in range(10000):
		#time.sleep(0.1)
		env.render()

		#Reduce dimensionality
		cur_x = prepro(observation)

		#Get difference framce
		diff_x = cur_x-cur_x if t==0 else (cur_x-prev_x)
		frames.append(np.matrix(diff_x))

		#Get output from network, and sample from distribution
		classes = model.predict(x=np.array([np.matrix(diff_x)]), batch_size=1)
		print(classes)
		#action = np.random.choice([2,3],1,p=classes[0][0])
		action = np.argmax(classes)+2 if random.random() < 0.5 else np.random.choice([2,3],1,p=classes[0][0])
		y = keras.utils.to_categorical(action-2, num_classes=2)
		predProbs.append(y)

		#print(observation)
		actions.append(action)
		observation, reward, done, info = env.step(action)
		rewards.append(reward)
		
		
		#action = np.argmax(classes)+1
		#action = 2 if classes[0][0][0]> random.random() else 3


		#Build dataset for training
		
		
		
		

		#print(env.action_space, env.observation_space)
		print(action,reward)

		prev_x = cur_x
		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break

	



	#Build labels. Encourage positive actions, discourage negative.
	


		'''
		if rewards[i]<0:
			#opposite_action = 1 if actions[i] == 2 else 0
			label = [0,0]
			label[(actions[i])%2] = -rewards[i]
			label[(actions[i]+1)%2] = 1+rewards[i]
			labels.append(label)
		elif rewards[i]>0:
			#positive_action = 0 if actions[i] == 2 else 1
			label = [0,0]
			label[(actions[i]+1)%2] = rewards[i]
			label[(actions[i])%2] = 1-rewards[i]
			labels.append(label)
		else:
			label = [0.5,0.5]
			labels.append(label)
		'''
		


	#Train
	#print(len(labels))
	#print(len(frames))
	#for i in range(len(frames)):
	if i_episode %50 ==0:
		d_rewards = discount_rewards(rewards)
		#print(d_rewards)
		#d_rewards -= np.mean(d_rewards)
		#d_rewards /= np.std(d_rewards)
		#print (d_rewards)

		labels = []
		for i in range(len(d_rewards)):
			best_action = np.argmax(softmax(np.matrix(predProbs[i]*d_rewards[i])))
			best_action = keras.utils.to_categorical(best_action, num_classes=2)

			labels.append([best_action])
			#print(softmax(np.matrix(predProbs[i]*d_rewards[i])))
			#print(labels[i])
		
		model.train_on_batch(np.array(frames), np.array(labels))
		frames = []
		rewards = []
		predProbs = []
		actions = []
		







