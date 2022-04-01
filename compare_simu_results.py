# Verify that ./logs/ has the desired files (and run the simulation files again if not; it takes long for IPW)

import numpy as np
import matplotlib.pyplot as plt

fname_ipw = './logs/simulation_IPW.log'
fname_qlearning = './logs/simulation_QLeaning.log'

with open(fname_ipw) as f_ipw:
    f_ipw = f_ipw.readlines()
ipw_vals = []
for line in f_ipw:
    if len(line) >= 10:
        try:
            val = float(line[10:])
        except ValueError:
            continue
        else:
            ipw_vals.append(val)

with open(fname_qlearning) as f_qlearning:
    f_qlearning = f_qlearning.readlines()
qlearning_vals = []
for line in f_qlearning:
    if len(line) >= 10:
        try:
            val = float(line[10:])
        except ValueError:
            continue
        else:
            qlearning_vals.append(val)

assert len(ipw_vals) == len(qlearning_vals) == 50

fig, ax = plt.subplots(1,2, sharey='all')
ax[0].set_title('IPW')
ax[0].boxplot(ipw_vals)
ax[1].set_title('Q-learning')
ax[1].boxplot(qlearning_vals)
plt.show()