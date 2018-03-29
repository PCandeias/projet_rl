from dqn_solver import DqnSolver
import numpy as np
from keras.models import clone_model
import utility

a = DqnSolver(action_size=2, observation_size=1, alpha=0.0002)
for i in range(10):
    a.store(np.array([[0]]),np.array(0),np.array(1),np.array([[0]]),True)
    a.store(np.array([[0]]),np.array(1),np.array(0),np.array([[0]]),True)
    a.replay()

b = utility.copy_model(a.model)


print(a.get_values(np.array([[0]])))
print(b.predict(np.array([[0]]), batch_size=1))
for i in range(100):
    a.store(np.array([[0]]),np.array(0),np.array(1),np.array([[0]]),True)
    a.store(np.array([[0]]),np.array(1),np.array(0),np.array([[0]]),True)
    a.replay()
print(a.get_values(np.array([[0]])))
print(b.predict(np.array([[0]]), batch_size=1))
b = utility.copy_model(a.model)
print(b.predict(np.array([[0]]), batch_size=1))
