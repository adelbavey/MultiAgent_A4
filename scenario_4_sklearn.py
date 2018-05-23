""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import cPickle as pickle
import gym

from sklearn.neural_network import MLPRegressor

import time


def get_4_dim_state_vector(observation_rgb, state_vec):
    length_limit = 10

    if len(state_vec) == 0:
        for i in range(length_limit):
            state_vec.append([0, 0, 0, 0])
        # state_vec = np.zeros(10,4)

    next_state = get_4_dim_state(observation_rgb, state_vec[0], state_vec[1])

    state_vec.insert(0, next_state)
    del state_vec[-1]

    # print state_vec

    return state_vec


def get_4_dim_state(observation_rgb, prev_state, prev_prev_state):
    [x_dim, y_dim, colors] = observation_rgb.shape

    x_lower_limit = 34
    x_upper_limit = 194
    y_lower_limit = 0
    y_upper_limit = y_dim

    field_dims = [x_lower_limit, x_upper_limit, y_lower_limit, y_upper_limit]

    ball_found = False
    self_found = False
    enemy_found = False

    positions = [0, 0, 0, 0]

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

    for x in range(x_lower_limit, x_upper_limit):
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
    prev_ball_pos = [prev_state[0], prev_state[1]]

    ball_position = find_ball(prev_ball_pos, ball_search_distance, field_dims, observation_rgb)

    if ball_position == [0, 0]:
        ball_position = find_ball([115, 80], ball_search_distance, field_dims, observation_rgb)
        if ball_position == [0, 0]:
            prev_prev_ball_pos = [prev_prev_state[0], prev_prev_state[1]]

            ball_position = find_ball(prev_prev_ball_pos, ball_search_distance, field_dims, observation_rgb)

    # corr_ball_position = find_ball([0,0],500,field_dims, observation_rgb)

    # dist = math.sqrt( (ball_position[0] - prev_state[0]) * (ball_position[0] - prev_state[0]) + (ball_position[1] - prev_state[1]) * (ball_position[1] - prev_state[1]) )
    # print "Dist: ", dist

    # if (ball_position[0] != corr_ball_position[0] or ball_position[1] != corr_ball_position[1]):
    #    print "incorrect pos"

    positions[0] = ball_position[0]
    positions[1] = ball_position[1]

    # print positions

    return positions


def adapt_positions_to_field_dims(position, search_distance, field_dims):
    x_start = position[0] - search_distance
    if x_start < field_dims[0]: x_start = field_dims[0]

    x_stop = position[0] + search_distance
    if x_stop > field_dims[1]: x_stop = field_dims[1]

    y_start = position[1] - search_distance
    if y_start < field_dims[2]: y_start = field_dims[2]

    y_stop = position[1] + search_distance
    if y_stop > field_dims[3]: y_stop = field_dims[3]

    return [x_start, x_stop, y_start, y_stop]


def find_ball(position_guess, search_distance, field_dims, observation_rgb):
    [x_start, x_stop, y_start, y_stop] = adapt_positions_to_field_dims(position_guess, search_distance, field_dims)

    position = [0, 0]

    for x in range(x_start, x_stop):
        for y in range(y_start, y_stop):
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
  return 1.0 / (1.0 + np.exp(-x*3)) # sigmoid "squashing" function to interval [0,1]

def taper_rewards(r):
    gamma = 0.99  # discount factor for reward
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


def main_function():
    resume = False

    if resume:
        clf = pickle.load(open('scenario_4_episode_8000', 'rb'))
    else:
        clf = MLPRegressor(solver='sgd', batch_size=10, max_iter=1, verbose=True, warm_start=True, hidden_layer_sizes=(10,))

    env = gym.make("Pong-v0")
    observation = env.reset()
    x_vector = []
    reward_vector = []
    action_vector = []
    episode_number = 0
    render = False


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

        var = np.var(x)
        avg = np.mean(x)
        #print var, avg

        # forward the policy network and sample an action from the returned probability
        # aprob, h = policy_forward(x)
        # action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

        # r = np.random.uniform()
        # if episode_number == 0:
        #     if r < 0.5:
        #         action = 2
        #     else:
        #         action = 3
        # else:
        #     predict = clf.predict(x.reshape(1,-1))[0]
        #     predict_squashed = sigmoid(predict)
        #
        #     predict_sum = sum(predict_squashed)
        #     predict_probabilities = [0,0]
        #     predict_probabilities[0] = predict_squashed[0] / predict_sum
        #     predict_probabilities[1] = predict_squashed[1] / predict_sum
        #     if r < predict_probabilities[0]:
        #         action = 2
        #     else:
        #         action = 3

        if episode_number == 0:
            predict_probabilities = [0.5,0.5]
        else:
            predict = clf.predict(x.reshape(1,-1))[0]
            #m = np.mean(predict)
            predict -= np.mean(predict)

            predict_squashed = sigmoid(predict)

            predict_sum = sum(predict_squashed)
            predict_probabilities = [0,0]
            predict_probabilities[0] = predict_squashed[0] / predict_sum
            predict_probabilities[1] = predict_squashed[1] / predict_sum

        action = get_preferred_action(predict_probabilities)

        action += 2

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

                action_index = action - 2

                action_label = [0,0]
                action_label[action_index] = tapered_reward_vector[i]
                #action_label[other_action_index] = -tapered_reward_vector[i]

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
            if episode_number % 500 == 0:
                file_name = "scenario_4_episode_" + str(episode_number)
                pickle.dump(clf, open(file_name, 'wb'))
            #reward_sum = 0
            observation = env.reset()  # reset env
            #prev_x = None

            print "Episode time: " + str(time.time() - time_s)
            time_s = time.time()

        if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
            print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')


main_function()