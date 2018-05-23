""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import cPickle as pickle
import gym

import time

def get_4_dim_state_vector(observation_rgb,state_vec):
    length_limit = 10

    if len(state_vec) == 0:
        for i in range(length_limit):
            state_vec.append([0,0,0,0])
        #state_vec = np.zeros(10,4)

    next_state = get_4_dim_state(observation_rgb,state_vec[0],state_vec[1])

    state_vec.insert(0,next_state)
    del state_vec[-1]

    #print state_vec

    return state_vec


def get_4_dim_state(observation_rgb, prev_state, prev_prev_state):
    [x_dim, y_dim, colors] = observation_rgb.shape

    x_lower_limit = 34
    x_upper_limit = 194
    y_lower_limit = 0
    y_upper_limit = y_dim

    field_dims = [x_lower_limit,x_upper_limit,y_lower_limit,y_upper_limit]

    ball_found = False
    self_found = False
    enemy_found = False

    positions = [0,0,0,0]

    # for x in range(x_lower_limit,x_upper_limit):
    #     for y in range(0,y_dim):
    #
    #         #if not is_background(observation_rgb[x][y]):
    #         #    print ("color: ", observation_rgb[x][y], "position: ", x , y)
    #
    #         if not ball_found and is_ball(observation_rgb[x][y]):
    #             ball_found = True
    #             positions[0] = x
    #             positions[1] = y
    #
    #         if not self_found and is_self(observation_rgb[x][y]):
    #             self_found = True
    #             positions[2] = x
    #
    #         if not enemy_found and is_enemy(observation_rgb[x][y]):
    #             enemy_found = True
    #             positions[3] = x

    for x in range(x_lower_limit,x_upper_limit):
        if not self_found and is_self(observation_rgb[x][140]):
            self_found = True
            positions[2] = x

        if not enemy_found and is_enemy(observation_rgb[x][16]):
            enemy_found = True
            positions[3] = x



    # if prev_state[0] == 0 and prev_state[1] == 0:
    #     prev_state[0] = 115
    #     prev_state[1] = 78

    ball_search_distance = 15
    prev_ball_pos = [prev_state[0],prev_state[1]]

    ball_position = find_ball(prev_ball_pos,ball_search_distance,field_dims,observation_rgb)

    if ball_position == [0,0]:
        ball_position = find_ball([115,80],ball_search_distance,field_dims, observation_rgb)
        if ball_position == [0,0]:
            prev_prev_ball_pos = [prev_prev_state[0], prev_prev_state[1]]

            ball_position = find_ball(prev_prev_ball_pos, ball_search_distance, field_dims, observation_rgb)

    #corr_ball_position = find_ball([0,0],500,field_dims, observation_rgb)

    #dist = math.sqrt( (ball_position[0] - prev_state[0]) * (ball_position[0] - prev_state[0]) + (ball_position[1] - prev_state[1]) * (ball_position[1] - prev_state[1]) )
    #print "Dist: ", dist

    #if (ball_position[0] != corr_ball_position[0] or ball_position[1] != corr_ball_position[1]):
    #    print "incorrect pos"

    positions[0] = ball_position[0]
    positions[1] = ball_position[1]

    #print positions

    return positions

def adapt_positions_to_field_dims(position,search_distance,field_dims):
    x_start = position[0] - search_distance
    if x_start < field_dims[0]: x_start = field_dims[0]

    x_stop = position[0] + search_distance
    if x_stop > field_dims[1]: x_stop = field_dims[1]

    y_start = position[1] - search_distance
    if y_start < field_dims[2]: y_start = field_dims[2]

    y_stop = position[1] + search_distance
    if y_stop > field_dims[3]: y_stop = field_dims[3]

    return [x_start,x_stop,y_start,y_stop]


def find_ball(position_guess,search_distance,field_dims,observation_rgb):
    [x_start,x_stop,y_start,y_stop] = adapt_positions_to_field_dims(position_guess,search_distance,field_dims)

    position = [0, 0]

    for x in range(x_start,x_stop):
        for y in range(y_start,y_stop):
            if is_ball(observation_rgb[x][y]):
                position[0] = x
                position[1] = y

    return position

def is_background(color):
    if color[0] == 144 and color[1] == 72 and color[2] == 17:
        return True

    return False

def is_enemy(color):
    if color[0] == 213 and color[1] == 130 and color[2] == 74:
        return True

    return False


def is_self(color):
    if color[0] == 92 and color[1] == 186 and color[2] == 92:
        return True

    return False


def is_ball(color):
    if color[0] == 236 and color[1] == 236 and color[2] == 236:
        return True

    return False

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

# hyperparameters
H = 8 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False

# model initialization
D = 20 # input dimensionality: 80x80 grid
if resume:
  model = pickle.load(open('scenario_2_episode_8000', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0

state_vec = []
time_s = time.time()

while True:
  if render: env.render()

  # preprocess the observation, set input to network to be difference image
  # cur_x = prepro(observation)
  # x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  # prev_x = cur_x

  state_vec = get_4_dim_state_vector(observation, state_vec)
  x = []
  for i in range(5):
    state_vec_normalized = list(state_vec[i])
    if state_vec_normalized[0] != 0: state_vec_normalized[0] = (state_vec_normalized[0] - 114) / 40.0
    if state_vec_normalized[1] != 0: state_vec_normalized[1] = (state_vec_normalized[1] - 80) / 40.0
    if state_vec_normalized[2] != 0: state_vec_normalized[2] = (state_vec_normalized[2] - 114) / 40.0
    if state_vec_normalized[3] != 0: state_vec_normalized[3] = (state_vec_normalized[3] - 114) / 40.0

    for j in range(len(state_vec_normalized)):
        x.append(state_vec_normalized[j])


  x = np.array(x)

  #time.sleep(1)
  #print x

  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x)
  action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  y = 1 if action == 2 else 0 # a "fake label"
  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  reward_sum += reward

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    xs,hs,dlogps,drs = [],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    grad = policy_backward(eph, epdlogp)
    for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in model.iteritems():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
    if episode_number % 500 == 0:
        file_name = "scenario_2_episode_" + str(episode_number)
        pickle.dump(model, open(file_name, 'wb'))
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None

    print "Episode time: " + str(time.time() - time_s)
    time_s = time.time()

    if episode_number == 8000:
        break


  if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
    print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')


