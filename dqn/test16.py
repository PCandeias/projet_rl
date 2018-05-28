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


def run_test(test_name, n_agents, pop_size, max_stock, initial_stock, lr, growth_rate=2, max_steps=100, n_groups=2,
        n_episodes=20000, g_consumption=None, n_iterations=10, save_model=False):
    print("Running test: %s ## n_agents=%d , pop_size=%d, max_stock=%d, initial_stock=%d, lr=%.15f" % (test_name, n_agents,
        pop_size, max_stock, initial_stock, lr))
    for i in range(n_iterations):
        print("Iteration %d" % i)
        print("Training model")
        runner = MultiFishermanRunner(n_agents=n_agents, agent_mode='dqn', save_filename=str(i)+test_name if save_model else None,
                                     save_frequency=1000, replay_start_size=20000, gamma=1.0, eps=1.0, eps_decay=0.99995,
                                     eps_min=0.01, alpha=lr, memory_size=1000000, batch_size=32,
                                     freeze_target_frequency=10000, double_q=True, verbose=False)
        runner._create_environment(max_stock=max_stock, initial_stock=initial_stock, population=pop_size,
                growth_rate=growth_rate, max_steps=max_steps, n_groups=n_groups, g_consumption=g_consumption)
        runner._create_agents(load_filename=None, gamma=1.0, alpha=lr, eps=1.0, eps_decay=0.99995,
                            eps_min=0.01,memory_size=1000000, batch_size=32, double_q=True,
                            freeze_target_frequency=10000, verbose=False)

        train_scores = runner.run(n_episodes=n_episodes, train=True, verbose=True, display_frequency=2000)
        print("Evaluating model.")
        eval_scores = runner.run(n_episodes=50, train=False, verbose=True, display_frequency=10, eps=0.0000001)
        np.save(scores_dir + str(i) + test_name, train_scores)
        np.save(scores_dir + str(i) + test_name, eval_scores)


with tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=32)) as sess:
    K.set_session(sess)

    run_test("test16", 100, 100, 100, 100, 0.00025)
