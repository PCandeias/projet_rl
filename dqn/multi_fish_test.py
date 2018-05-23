from multi_gym_runner import MultiGymRunner
import numpy as np
import gym
from collections import deque
import tensorflow as tf
from keras.backend import tensorflow_backend as K
import utility

scores_dir = "saved_scores/"

class MultiFishermanRunner(MultiGymRunner):
    def _create_environment(self, max_stock=50, initial_stock=50, population=50, growth_rate=2, max_steps=100,
            n_groups=2, g_consumption=None):
        self.env = gym.make('Fisherman-v2')
        self.env.set_environment_variables(max_stock=max_stock, initial_stock=initial_stock, population=population, n_agents=self.n_agents,
                growth_rate=growth_rate, max_steps=max_steps, n_groups=n_groups, g_consumption=g_consumption)

    def get_action_size(self):
        return 2

    def get_observation_size(self):
        return 1 + self.env.n_groups


with tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=16)) as sess:
    K.set_session(sess)

    num_test_iter = 10
    print("Running test with 50 agents, 50 stock and lr=0.00000000025")
    for i in range(num_test_iter):
        print("Training episode %d" % (i,))
        runner = MultiFishermanRunner(n_agents=50, agent_mode='dqn', save_filename=str(i) + '_50_50fisherman',
                                     save_frequency=1000, replay_start_size=20000, gamma=1.0, eps=1.0, eps_decay=0.99995,
                                     eps_min=0.01, alpha=0.00000000025, memory_size=1000000, batch_size=32,
                                     freeze_target_frequency=10000, double_q=True, verbose=False)
        runner._create_environment(max_stock=50, initial_stock=50, population=50, growth_rate=2, max_steps=100,
                n_groups=2)
        train_scores = runner.run(n_episodes=10, train=True, verbose=True, display_frequency=1, eps=0.0001)
        print("Eval episode %d" % (i,))
        eval_scores = runner.run(n_episodes=50, train=False, verbose=True, display_frequency=10, eps=0.0000001)
        np.save(scores_dir + str(i) + "train_50_50fisherman", train_scores)
        np.save(scores_dir + str(i) + "eval_50_50fisherman", eval_scores)

