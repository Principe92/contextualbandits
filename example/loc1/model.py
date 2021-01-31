import pandas as pd, numpy as np, re
import dill
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import SGDClassifier

def parse_data(file_name):
    features = list()
    labels = list()
    with open(file_name, 'rt') as f:
        f.readline()
        for l in f:
            if bool(re.search("^[0-9]", l)):
                g = re.search("^(([0-9]{1,2},?)+)\s(.*)$", l)
                labels.append([int(i) for i in g.group(1).split(",")])
                features.append(eval("{" + re.sub("\s", ",", g.group(3)) + "}"))
            else:
                l = l.strip()
                labels.append([])
                features.append(eval("{" + re.sub("\s", ",", l) + "}"))
    features = pd.DataFrame.from_dict(features).fillna(0).iloc[:,:].values
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)
    return features, y

X, y = parse_data("loc_1.txt")

with open("model_0_loc1.dill", 'rb') as model_file:
    model = dill.load(model_file)

# batch size - algorithms will be refit after N rounds
batch_size=50
nchoices = y.shape[1]


# initial seed - all policies start with the same small random selection of actions/rewards
first_batch = X[:batch_size, :]
action_chosen = np.random.randint(nchoices, size=batch_size)
rewards_received = y[np.arange(batch_size), action_chosen]

lst_a_ucb = action_chosen.copy()
lst_actions = [lst_a_ucb]

# These lists will keep track of the rewards obtained by each policy
rewards_ucb = [list()]

lst_rewards = [rewards_ucb]



# rounds are simulated from the full dataset
def simulate_rounds(model, rewards, actions_hist, X_global, y_global, batch_st, batch_end):
    np.random.seed(batch_st)
    
    ## choosing actions for this batch
    actions_this_batch = model.predict(X_global[batch_st:batch_end, :]).astype('uint8')
    
    # keeping track of the sum of rewards received
    rewards.append(y_global[np.arange(batch_st, batch_end), actions_this_batch].sum())
    
    # adding this batch to the history of selected actions
    new_actions_hist = np.append(actions_hist, actions_this_batch)
    
    # now refitting the algorithms after observing these new rewards
    np.random.seed(batch_st)
    model.fit(X_global[:batch_end, :], new_actions_hist, y_global[np.arange(batch_end), new_actions_hist])
    print(new_actions_hist)
    return new_actions_hist

# now running all the simulation
for i in range(int(np.floor(X.shape[0] / batch_size))):
    batch_st = (i + 1) * batch_size
    batch_end = (i + 2) * batch_size
    batch_end = np.min([batch_end, X.shape[0]])
    
    lst_actions[model] = simulate_rounds([model],
                                             lst_rewards[model],
                                             lst_actions[model],
                                             X, y,
                                             batch_st, batch_end)

        
#plotting

import matplotlib.pyplot as plt
from pylab import rcParams

def get_mean_reward(reward_lst, batch_size=50):
    mean_rew=list()
    for r in range(len(reward_lst)):
        mean_rew.append(sum(reward_lst[:r+1]) * 1.0 / ((r+1)*batch_size))
    return mean_rew


import scipy.stats as st
y1 = get_mean_reward(rewards_ucb)

x1 = [index for index in range(len(rewards_ucb))]

def CI_model(y, confidence = 0.95):
    std_err_y = st.sem(y1)
    n_y = len(y1)
    h_y = std_err_y * st.t.ppf((1 + confidence) / 2, n_y - 1)
    return h_y

h_y1,  = CI_model(y1)
plt.errorbar(x1, y1, yerr= h_y1)

plt.plot(np.repeat(y.mean(axis=0).max(),len(rewards_sft)),linewidth=4,ls='dashed')


plt.xlabel('Rounds (models were updated every 50 rounds)', size=10)
plt.ylabel('Cummulative Mean Reward', size=10)
#plt.title('Comparison of Online Contextual Bandit Policies in location 7\n(Base Algorithm is Logistic Regression with data fit in streams)\n\nDataset\n(159 categories, 1836 attributes)',size=30)
plt.savefig("loc_1_test3.png", bbox_inches='tight', dpi = 600)