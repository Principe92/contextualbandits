import pandas
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from pylab import rcParams


df = pandas.read_csv('rewards_loc10.csv')

ucb,ts,ovr,egr,egr2,agr,agr2,efr,ac,aac,sft = df['ucb'],df['ts'],df['ovr'],\
df['egr'],df['egr2'],df['agr'],df['agr2'],df['efr'],df['ac'],df['aac'],df['sft']

#y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11 = np.mean(ucb), np.mean(ts) \
#,np.mean(ovr), np.mean(egr), np.mean(egr2) \
#,np.mean(agr), np.mean(agr2), np.mean(efr) \
#,np.mean(ac), np.mean(aac), np.mean(sft)

def get_mean_reward(reward_lst):
    mean_rew=list()
    for r in range(len(reward_lst)):
        mean_rew.append(sum(reward_lst[:r+1]) / ((r+1)))
    return mean_rew

y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11 = get_mean_reward(ucb), get_mean_reward(ts) \
,get_mean_reward(ovr), get_mean_reward(egr), get_mean_reward(egr2) \
,get_mean_reward(agr), get_mean_reward(agr2), get_mean_reward(efr) \
,get_mean_reward(ac), get_mean_reward(aac), get_mean_reward(sft)

x1, x2 = [index for index in range(len(ucb))], [index for index in range(len(ts))]
x3, x4 = [index for index in range(len(df['ovr']))], [index for index in range(len(df['egr']))]
x5, x6 = [index for index in range(len(df['egr2']))], [index for index in range(len(df['agr']))]
x7, x8 = [index for index in range(len(df['agr2']))], [index for index in range(len(df['efr']))]
x9, x10 = [index for index in range(len(df['ac']))], [index for index in range(len(df['aac']))]
x11 = [index for index in range(len(df['sft']))]


def CI_model(y, confidence = 0.95):
    std_err_y = st.sem(y)
    n_y = len(y)
    h_y = std_err_y * st.t.ppf((1 + confidence) / 2, n_y - 1)
    return h_y

h_y1, h_y2, h_y3, h_y4, h_y5, h_y6, h_y7, h_y8, h_y9, h_y10, h_y11 = CI_model(ucb), CI_model(ts), CI_model(ovr),\
CI_model(egr), CI_model(egr2), CI_model(agr), CI_model(agr2), CI_model(efr), CI_model(ac), CI_model(aac), CI_model(sft)
plt.errorbar(x1, y1, label='Bootstrapped Upper-Confidence Bound (C.I.=80%)')
plt.errorbar(x2, y2, label='Bootstrapped Thompson Sampling')
plt.errorbar(x3, y3, label='Separate Classifiers + Beta Prior')
plt.errorbar(x4, y4, label='Epsilon-Greedy (p0=20%, decay=0.9999')
#plt.errorbar(x5, y5, yerr= h_y5, label='Epsilon-Greedy (p0=20%, no decay')
#plt.errorbar(x6, y6, label='Adaptive Greedy (decaying threshold)')
#plt.errorbar(x7, y7, yerr= h_y7, label='Adaptive Greedy (p0=30%, decaying percentile)')
#plt.errorbar(x8, y8, yerr= h_y8, label='Explore First (n=1,500)')
#plt.errorbar(x9, y9, yerr= h_y9, label='Active Explorer')
#plt.errorbar(x10, y10, yerr= h_y10, label='Adaptive Active Greedy')
#plt.errorbar(x11, y11, yerr= h_y11, label='Softmax Explorer')
#plt.plot(np.repeat(y.mean(axis=0).max(),len(rewards_sft)),linewidth=4,ls='dashed', label='Overall Best Arm (no context)')

ax = plt.subplot(111)


plt.xlabel('Rounds (models were updated every 50 rounds)', size=10)
plt.ylabel('Cummulative Mean Reward', size=10)
plt.title('Comparison of Online Contextual Bandit Policies in location 10')
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#ax.legend()
plt.savefig("location_10.png", bbox_inches='tight', dpi = 600)
