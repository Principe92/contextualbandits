import pandas as pd, numpy as np, re
import dill
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from contextualbandits.online import LinUCB, AdaptiveGreedy, \
        SoftmaxExplorer, ActiveExplorer, EpsilonGreedy
from copy import deepcopy

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

X, y = parse_data("all.txt")


nchoices = y.shape[1]
base_sgd = SGDClassifier(random_state=123, loss='log', warm_start=False)
base_ols = LinearRegression(fit_intercept=True)

## Metaheuristic using different base algorithms and configurations
linucb = LinUCB(nchoices = nchoices, alpha = 0.1)
### Important!!! the default hyperparameters for LinUCB in the reference paper
### are very different from what's used in this example
adaptive_active_greedy = AdaptiveGreedy(deepcopy(base_ols), nchoices = nchoices,
                                        smoothing = None, beta_prior = ((3./nchoices,4.), 2),
                                        active_choice = 'weighted', decay_type = 'percentile',
                                        decay = 0.9997, batch_train = True,
                                        )
softmax_explorer = SoftmaxExplorer(deepcopy(base_sgd), nchoices = nchoices,
                                   smoothing = (1,2), beta_prior = None, batch_train = True,
                                   refit_buffer = 50, deep_copy_buffer = False, random_state = 3333)
adaptive_greedy_perc = AdaptiveGreedy(deepcopy(base_ols), nchoices = nchoices,
                                      smoothing = (1,2), beta_prior = None,
                                      decay_type = 'percentile', decay = 0.9997, batch_train = True,
                                      random_state = 4444)
active_explorer = ActiveExplorer(deepcopy(base_sgd), smoothing = None, nchoices = nchoices,
                                 beta_prior = ((3./nchoices, 4.), 2), batch_train = True, refit_buffer = 50,
                                 deep_copy_buffer = False, random_state = 5555)
epsilon_greedy_nodecay = EpsilonGreedy(deepcopy(base_ols), nchoices = nchoices,
                                       smoothing = (1,2), beta_prior = None,
                                       decay = None, batch_train = True,
                                       deep_copy_buffer = False, random_state = 6666)


models = [linucb, adaptive_active_greedy, softmax_explorer, adaptive_greedy_perc,
          active_explorer, epsilon_greedy_nodecay]


# These lists will keep track of the rewards obtained by each policy
rewards_lucb, rewards_aac, rewards_sft, rewards_agr, \
rewards_ac, rewards_egr = [list() for i in range(len(models))]

lst_rewards = [rewards_lucb, rewards_aac, rewards_sft,
               rewards_agr, rewards_ac, rewards_egr]

# batch size - algorithms will be refit after N rounds
batch_size=50

# initial seed - all policies start with the same small random selection of actions/rewards
first_batch = X[:batch_size, :]
np.random.seed(1)
action_chosen = np.random.randint(nchoices, size=batch_size)
rewards_received = y[np.arange(batch_size), action_chosen]

# fitting models for the first time
for model in models:
    model.fit(X=first_batch, a=action_chosen, r=rewards_received)
    
# these lists will keep track of which actions does each policy choose
lst_a_lucb, lst_a_aac, lst_a_sft, lst_a_agr, \
lst_a_ac, lst_a_egr = [action_chosen.copy() for i in range(len(models))]

lst_actions = [lst_a_lucb, lst_a_aac, lst_a_sft,
               lst_a_agr, lst_a_ac, lst_a_egr]

# rounds are simulated from the full dataset
def simulate_rounds_stoch(model, rewards, actions_hist, X_batch, y_batch, rnd_seed):
    np.random.seed(rnd_seed)
    
    ## choosing actions for this batch
    actions_this_batch = model.predict(X_batch).astype('uint8')
    
    # keeping track of the sum of rewards received
    rewards.append(y_batch[np.arange(y_batch.shape[0]), actions_this_batch].sum())
    
    # adding this batch to the history of selected actions
    new_actions_hist = np.append(actions_hist, actions_this_batch)
    
    # rewards obtained now
    rewards_batch = y_batch[np.arange(y_batch.shape[0]), actions_this_batch]
    
    # now refitting the algorithms after observing these new rewards
    np.random.seed(rnd_seed)
    model.partial_fit(X_batch, actions_this_batch, rewards_batch)
    
    return new_actions_hist

# now running all the simulation
for i in range(int(np.floor(X.shape[0] / batch_size))):
    batch_st = (i + 1) * batch_size
    batch_end = (i + 2) * batch_size
    batch_end = np.min([batch_end, X.shape[0]])
    
    X_batch = X[batch_st:batch_end, :]
    y_batch = y[batch_st:batch_end, :]
    
    for model in range(len(models)):
        lst_actions[model] = simulate_rounds_stoch(models[model],
                                                   lst_rewards[model],
                                                   lst_actions[model],
                                                   X_batch, y_batch,
                                                   rnd_seed = batch_st)

    for model in range(len(models)):
    	dill.dump(models[model], open("model_%d_all_v2.dill" % (model), "wb"))

        
#plotting

import matplotlib.pyplot as plt
from pylab import rcParams

def get_mean_reward(reward_lst, batch_size=batch_size):
    mean_rew=list()
    for r in range(len(reward_lst)):
        mean_rew.append(sum(reward_lst[:r+1]) * 1.0 / ((r+1)*batch_size))
    return mean_rew


import scipy.stats as st
y1, y2, y3, y4, y5, y6 = get_mean_reward(rewards_ucb), get_mean_reward(rewards_aac) \
,get_mean_reward(rewards_sft), get_mean_reward(rewards_agr), get_mean_reward(rewards_ac) \
,get_mean_reward(rewards_egr)
x1, x2 = [index for index in range(len(rewards_ucb))], [index for index in range(len(rewards_aac))]
x3, x4 = [index for index in range(len(rewards_sft))], [index for index in range(len(rewards_agr))]
x5, x6 = [index for index in range(len(rewards_ac))], [index for index in range(len(rewards_egr))]


def CI_model(y, confidence = 0.95):
    std_err_y = st.sem(y1)
    n_y = len(y1)
    h_y = std_err_y * st.t.ppf((1 + confidence) / 2, n_y - 1)
    return h_y

h_y1, h_y2, h_y3, h_y4, h_y5, h_y6 = CI_model(y1), CI_model(y2), CI_model(y3),\
CI_model(y4), CI_model(y5), CI_model(y6)
plt.errorbar(x1, y1, yerr= h_y1)
plt.errorbar(x2, y2, yerr= h_y2)
plt.errorbar(x3, y3, yerr= h_y3)
plt.errorbar(x4, y4, yerr= h_y4)
plt.errorbar(x5, y5, yerr= h_y5)
plt.errorbar(x6, y6, yerr= h_y6)

plt.plot(np.repeat(y.mean(axis=0).max(),len(rewards_sft)),linewidth=4,ls='dashed')


plt.xlabel('Rounds (models were updated every 50 rounds)', size=10)
plt.ylabel('Cummulative Mean Reward', size=10)
#plt.title('Comparison of Online Contextual Bandit Policies in location 7\n(Base Algorithm is Logistic Regression with data fit in streams)\n\nDataset\n(159 categories, 1836 attributes)',size=30)
plt.savefig("rewards_v2.png", bbox_inches='tight', dpi = 600)


from numpy import array
ucb, aac, sft, agr, ac, agr, egr = array(rewards_ucb), array(rewards_aac), array(rewards_sft), \
array(rewards_agr), array(rewards_ac), array(rewards_egr)
df = pd.DataFrame({"ucb" : ucb, "aac":aac, "sft":sft, "agr":agr, "ac":ac, "egr":egr})
df.to_csv("rewards_all_V2.csv", index=False)