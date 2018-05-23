""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import cPickle as pickle
import gym

#from matplotlib import pyplot as plt

from sklearn.neural_network import MLPRegressor

import time

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[:,:,0] # downsample by factor of 2
  #I[I == 144] = 0 # erase background (background type 1)
  #I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1

  return I.astype(np.float).ravel()
  #return I

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x*5)) # sigmoid "squashing" function to interval [0,1]

def taper_rewards(r):
    gamma = 0.9  # discount factor for reward
    """ take 1D float array of rewards and compute discounted reward """
    tapered_rewards = [0] * len(r)
    running_add = 0
    for t in reversed(xrange(0, len(r))):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]

        # 0,1 form
        #running_add += 2
        #running_add /= 2

        tapered_rewards[t] = running_add

    return tapered_rewards

def get_preferred_action(probabilities):
    cumulative_probabilities = []
    cumulative_probabilities.append(probabilities[0])
    for i in range(1,len(probabilities)):
        cumulative_probabilities.append(probabilities[i] + cumulative_probabilities[i-1])

    r = np.random.uniform()
    action = 0
    for i in range(0, len(probabilities)):
        if r < cumulative_probabilities[i]:
            action = i
            break

    return action


# hyperparameters
# H = 8 # number of hidden layer neurons
# batch_size = 10 # every how many episodes to do a param update?
# learning_rate = 1e-4
# gamma = 0.99 # discount factor for reward
# decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
# resume = False # resume from previous checkpoint?
# render = False

# # model initialization
# D = 20 # input dimensionality: 80x80 grid
# if resume:
#   model = pickle.load(open('save.p', 'rb'))
# else:
#   model = {}
#   model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
#   model['W2'] = np.random.randn(H) / np.sqrt(H)
#
# grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
# rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory

def main_function():
    resume = False

    if resume:
        clf = pickle.load(open('scenario_5_episode_8000', 'rb'))
    else:
        clf = MLPRegressor(solver='sgd', batch_size=10, max_iter=1, verbose=True, warm_start=True, hidden_layer_sizes=(200,))

    env = gym.make("SpaceInvaders-v4")
    observation = env.reset()
    x_vector = []
    reward_vector = []
    action_vector = []
    episode_number = 0
    prev_x = None # used in computing the difference frame
    D = 160 * 160 # input dimensionality: 80x80 grid
    render = False

    time_s = time.time()

    i = 0
    while True:
        if render: env.render()

        # preprocess the observation, set input to network to be difference image
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        # i+= 1
        # if i > 100:
        #     plt.imshow(cur_x, interpolation='nearest')
        #     plt.show()

        # forward the policy network and sample an action from the returned probability
        # aprob, h = policy_forward(x)
        # action = 2 if np.random.uniform() < aprob else 3 # roll the dice!


        if episode_number == 0:
            predict_probabilities = [0.33,0.33,0.33]
        else:
            predict = clf.predict(x.reshape(1,-1))[0]
            predict -= np.mean(predict)

            predict_squashed = sigmoid(predict)

            predict_sum = sum(predict_squashed)
            predict_probabilities = [0,0,0]
            predict_probabilities[0] = predict_squashed[0] / predict_sum
            predict_probabilities[1] = predict_squashed[1] / predict_sum
            predict_probabilities[2] = predict_squashed[2] / predict_sum

        action = get_preferred_action(predict_probabilities)

        action += 1

        # record various intermediates (needed later for backprop)
        x_vector.append(x)  # observation
        action_vector.append(action)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)

        # drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
        reward_vector.append(reward)

        if done:  # an episode finished
            episode_number += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            # epx = np.vstack(xs)
            # epdlogp = np.vstack(dlogps)
            # epr = np.vstack(drs)

            #xs, hs, dlogps, drs = [], [], [], []  # reset array memory

            # compute the discounted reward backwards through time
            tapered_reward_vector = taper_rewards(reward_vector)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            #discounted_epr -= np.mean(discounted_epr)
            #discounted_epr /= np.std(discounted_epr)

            tapered_reward_vector -= np.mean(tapered_reward_vector)
            tapered_reward_vector /= np.std(tapered_reward_vector)

            #epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
            action_labels = []
            for i in range(len(action_vector)):
                action = action_vector[i]

                action_index = action -1

                action_label = [0,0,0]
                action_label[action_index] = tapered_reward_vector[i]

                action_labels.append(action_label)

            x_vector = np.vstack(x_vector)
            action_labels = np.vstack(action_labels)

            clf.fit(x_vector,action_labels)

            x_vector = []
            reward_vector = []
            action_vector = []

            # boring book-keeping
            #running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            #print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
            if episode_number % 100 == 0:
                file_name = "scenario_5_episode_" + str(episode_number)
                pickle.dump(clf, open(file_name, 'wb'))
            #reward_sum = 0
            observation = env.reset()  # reset env
            prev_x = None

            print "Episode time: " + str(time.time() - time_s)
            time_s = time.time()

        if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
            print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')


main_function()
