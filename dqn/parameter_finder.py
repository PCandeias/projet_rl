import numpy as np


def find(runner, v_gamma, v_eps, v_eps_decay, v_eps_min, v_alpha, v_alpha_decay, n_epochs=1, n_episodes=2000,
        verbose=False):
    cur_max = -10000
    cur_params = None
    for gamma in v_gamma:
        for eps in v_eps:
            for eps_decay in v_eps_decay:
                for eps_min in v_eps_min:
                    for alpha in v_alpha:
                        for alpha_decay in v_alpha_decay:
                            if verbose:
                                print("Trying params: Gamma: %f, Eps: %f, Eps_decay: %f, Eps_min: %f, Alpha: %f, Alpha_decay: %f" 
                                        % (gamma, eps, eps_decay, eps_min, alpha, alpha_decay))
                            scores = np.zeros(n_epochs)
                            for e in range(n_epochs):
                                runner.create_agent(gamma=gamma, eps=eps, eps_decay=eps_decay, eps_min=eps_min, 
                                        alpha=alpha, alpha_decay=alpha_decay)
                                scores[e] = runner.run(n_episodes=n_episodes, train=True, verbose=verbose)
                            avg = np.mean(scores)
                            if avg > cur_max:
                                cur_max = avg
                                cur_params = {'gamma': gamma, 'eps': eps, 'eps_decay': eps_decay, 'eps_min': eps_min,
                                        'alpha': alpha, 'alpha_decay': alpha_decay}
    return cur_params


                        
