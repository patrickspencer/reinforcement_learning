import numpy as np
import pandas as pd

# This is a repo for testing reinforcement learning ideas
# Credit:
# Ideas were adapted and modified from this article:
# https://conrmcdonald.medium.com/solving-multiarmed-bandits-a-comparison-of-epsilon-greedy-and-thompson-sampling-d97167ca9a50

class Environment():
    def __init__(self, variants, payouts, n_trials) -> None:
        self.variants = variants
        self.payouts = payouts
        self.n_trials = n_trials
        self.total_reward = 0
        self.num_variants = len(variants)
        self.shape = (self.num_variants, n_trials)


class BaseSampler():
    def __init__(self, env, n_samples=None, n_learning=None, e=0.05) -> None:
        self.env = env
        self.shape = (env.num_variants, n_samples)
        self.variants = env.variants
        self.n_trials = env.n_trials
        self.payouts = env.payouts
        self.ad_i = np.zeros(env.n_trials)
        self.r_i = np.zeros(env.n_trials)
        self.thetas = np.zeros(self.n_trials)
        self.regret_i = np.zeros(env.n_trials)
        self.thetaregret = np.zeros(self.n_trials)
        
        self.a = np.ones(env.num_variants) 
        self.b = np.ones(env.num_variants) 
        self.theta = np.zeros(env.num_variants)
        self.data = None
        self.reward = 0
        self.total_reward = 0
        self.choice = 0
        self.trial_i = 0

    def collect_data(self):
        self.data = pd.DataFrame(dict(ad=self.ad_i, reward=self.r_i, regret=self.thetaregret))


class RandomSampler(BaseSampler):
    def __init__(self, env):
        super().__init__(env)
        
    def choose(self):
        # choose from one of the variants
        self.choice = np.random.choice(self.variants)
        return self.choice
    
    def update(self, trial, choice, reward):
        # nothing to update
        self.ad_i[trial] = choice
        self.r_i[trial] = reward

    def run(self):
        for trial in range(self.env.n_trials):
            choice = self.choose()
            reward = np.random.binomial(1, p=self.env.payouts[choice])
            self.reward = reward
            self.update(trial, choice, reward)
            self.total_reward += reward


class EpsilonGreedySampler(BaseSampler):
    def __init__(self, env, n_learning, e):
        super().__init__(env)
        self.n_learning = n_learning
        self.e = e
        self.ep = np.random.uniform(0, 1, size=env.n_trials)
        self.exploit = (1 - e)
        
    def choose(self):
        # choose from one of the variants
        # explore for n_learning trials
        # otherwise just greedily take the variant with the largest payout so far
        self.choice = np.random.choice(self.variants) if self.trial_i < self.n_learning else np.argmax(self.theta)
        # exploit greedy solution 1-e percent of the time, otherwise explore
        self.choice = np.random.choice(self.variants) if self.ep[self.trial_i] > self.exploit else self.choice
        return self.choice
    
    def update(self, choice, reward):
        self.a[self.choice] += self.reward
        self.b[self.choice] += 1
        self.theta = self.a/self.b # vector

        self.thetas[self.trial_i] = self.theta[self.choice]

        self.ad_i[self.trial_i] = choice
        self.r_i[self.trial_i] = reward
        self.trial_i += 1

    def run(self):
        for i in range(self.env.n_trials):
            choice = self.choose()
            reward = np.random.binomial(1, p=self.env.payouts[choice])
            self.reward = reward
            self.update(choice, reward)
            self.total_reward += reward


class ThompsonSampler(BaseSampler):

    def __init__(self, env):
        super().__init__(env)
        
    def choose(self):
        self.theta = np.random.beta(self.a, self.b)
        self.choice = self.variants[np.argmax(self.theta)]
        return self.choice

    def update(self):
        self.a[self.choice] += self.reward
        self.b[self.choice] += 1 - self.reward

        self.thetas[self.trial_i] += self.theta[self.choice]
        self.thetaregret[self.trial_i] = np.max(self.thetas) - self.theta[self.choice]

        self.ad_i[self.trial_i] = self.choice
        self.r_i[self.trial_i] = self.reward
        self.trial_i += 1

    def run(self):
        for i in range(self.env.n_trials):
            choice = self.choose()
            reward = np.random.binomial(1, p=self.env.payouts[choice])
            self.reward = reward
            self.update()
            self.total_reward += reward

# running the program

np.random.seed(0)
variants = list(range(10))
noise = np.random.normal(0, .04, size=len(variants))

payouts = [0.023, 0.03, 0.029, 0.001, 0.05, 0.06, 0.0234, 0.035, 0.01, 0.11]

en0 = Environment(variants, payouts, n_trials=10000)
rs = RandomSampler(env=en0)
rs.run()
rs.total_reward

# %%
rs = EpsilonGreedySampler(env=en0, n_learning=1000, e=0.05)
rs.run()
rs.total_reward

# %%
rs = ThompsonSampler(env=en0)
rs.run()
rs.total_reward