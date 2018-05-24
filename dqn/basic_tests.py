from multi_gym_runner import MultiGymRunner
import numpy as np
import gym
from collections import deque
import tensorflow as tf
from keras.backend import tensorflow_backend as K
import utility
import os
from multiprocessing import Pool
from multiprocessing import Process
import threading


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


def run_test(thread_n, test_name, n_agents, pop_size, max_stock, initial_stock, lr, n_groups=2, g_consumption=None,
        growth_rate=2, max_steps=100,n_episodes=20000, n_iterations=10, save_model=False):
        with tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=48, inter_op_parallelism_threads=48)) as sess: 
            K.set_session(sess)
            print("Running test: %s ## n_agents=%d , pop_size=%d, max_stock=%d, initial_stock=%d, lr=%.15f" %
                    (test_name, n_agents,pop_size, max_stock, initial_stock, lr))
            for i in range(n_iterations):
                    print("Iteration %d" % i)
                    print("Training model")
                    runners[thread_n] = MultiFishermanRunner(n_agents=n_agents, agent_mode='dqn', save_filename=str(i)+test_name if save_model else None,
                                                 save_frequency=1000, replay_start_size=20000, gamma=1.0, eps=1.0, eps_decay=0.99995,
                                                 eps_min=0.01, alpha=lr, memory_size=1000000, batch_size=128,
                                                 freeze_target_frequency=10000, double_q=True, verbose=False)
                    runners[thread_n]._create_environment(max_stock=max_stock, initial_stock=initial_stock, population=pop_size,
                            growth_rate=growth_rate, max_steps=max_steps, n_groups=n_groups, g_consumption=g_consumption)
                    runners[thread_n]._create_agents(load_filename=None, gamma=1.0, alpha=lr, eps=1.0, eps_decay=0.99995,
                                        eps_min=0.01,memory_size=1000000, batch_size=32, double_q=True,
                                        freeze_target_frequency=10000, verbose=False)

                    train_scores = runners[thread_n].run(n_episodes=n_episodes, train=True, verbose=True, display_frequency=2000)
                    print("Evaluating model.")
                    eval_scores = runners[thread_n].run(n_episodes=50, train=False, verbose=True, display_frequency=10, eps=0.0000001)
                    np.save(scores_dir + str(i) + test_name, train_scores)
                    np.save(scores_dir + str(i) + test_name, eval_scores)

os.environ['MKL_NUM_THREADS'] = '48'
os.environ['GOTO_NUM_THREADS'] = '48'
os.environ['OMP_NUM_THREADS'] = '48'
os.environ['openmp'] = 'True'


coord = tf.train.Coordinator()
threads = []
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 10, 10, 10, 10, 0.00000000025 )))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 10, 10, 10, 10, 0.000000025 )))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 10, 10, 10, 10, 0.0000025 )))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 10, 10, 10, 10, 0.00025 )))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 10, 10, 6, 6, 0.00000000025 )))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 10, 10, 6, 6, 0.000000025 )))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 10, 10, 6, 6, 0.0000025 )))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 10, 10, 12, 12, 0.00000000025 )))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 10, 10, 12, 12, 0.000000025 )))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 10, 10, 12, 12, 0.0000025 )))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 10, 10, 12, 12, 0.00025 )))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 10, 10, 6, 6, 0.000000025, 4)))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 10, 10, 6, 6, 0.000000025, 8)))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 10, 10, 6, 6, 0.000000025, 16)))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 10, 10, 6, 6, 0.0000025, 4)))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 10, 10, 6, 6, 0.0000025, 8)))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 10, 10, 6, 6, 0.0000025, 16)))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 100, 100, 100, 100, 0.00000000025 )))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 100, 100, 100, 100, 0.000000025 )))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 100, 100, 100, 100, 0.0000025 )))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 100, 100, 100, 100, 0.00025 )))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 100, 100, 60, 60, 0.00000000025 )))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 100, 100, 60, 60, 0.000000025 )))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 100, 100, 60, 60, 0.0000025 )))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 100, 100, 60, 60, 0.000000025, 4)))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 100, 100, 60, 60, 0.000000025, 8)))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 100, 100, 60, 60, 0.000000025, 16)))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 100, 100, 60, 60, 0.0000025, 4)))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 100, 100, 60, 60, 0.0000025, 8)))
threads.append(threading.Thread(target = run_test, args = (len(threads), "test" + str(len(threads)), 100, 100, 60, 60, 0.0000025, 16)))
runners = [None for i in range(len(threads))]

for thread in threads:
    thread.start()

coord.join(threads)


