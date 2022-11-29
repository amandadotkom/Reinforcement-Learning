import random as rd
import numpy as np
import math as meth


class Bandit:

    def __init__(self, arm, sort):
        self.type = sort
        self.arms = arm
        self.reward = 0.0
        self.actions = []
        # As value in dict a list, (total reward, instances) to calculate avg
        self.actions_rewards = dict((k, [float(0), int(0)]) for k in range(self.arms))
        self.policy = None
        self.method = None
        self.e = 0
        self.c = 0

    # Action preferences
    # Initializes preferences of all actions to zero
    def ap(self):
        self.method = "AP"
        for k in range(self.arms):
            self.preferences[k] = 0

    # Policy of action based on Softmax/Boltzmann distribution, used for updating preferences
    def pi(self, action):
        div = 0
        for k in range(self.arms):
            div += np.exp(self.preferences[k])
        return np.exp(self.preferences[action]) / div

    # Updates action preferences for all actions
    def update_preferences(self, reward, action, timestep):
        for a in range(self.arms):
            # Next action taken is the current action, Ht+1(a').
            if a == action:
                new = self.preferences[a] + (1 / self.arms) * (reward - self.reward / (timestep + 1)) * (1 - self.pi(a))
            else:
                new = self.preferences[a] - (1 / self.arms) * (reward - self.reward / (timestep + 1)) * (self.pi(a))
            self.preferences[a] = new

    # Upper Confidence Bound
    def ucb(self):
        self.method = "UCB"
        while True:
            print("Select c (above 0)")  # TRY TO NORMALIZE C
            i = input()
            if i.isnumeric() and float(i) >= 1:
                self.c = int(i)
                break
            print("Invalid input...")

    # Optimistic Initial Values
    def optimistic(self):
        self.method = "OIV"
        for k in self.actions_rewards.values():
            k[0] = 200

    def e_greedy(self):
        self.method = "e-greedy"
        while True:
            print("Greedy (1) or e-Greedy (2)")
            i = input()
            if i.isnumeric() and int(i) in range(1, 3):
                break
            print("Invalid input...")
        if int(i) == 2:
            while True:
                print("Select e (between 1 and 100)")
                i = input()
                if i.isnumeric() and 100 >= float(i) >= 1:
                    self.e = int(i)
                    break
                print("Invalid input...")

    def get_action_greedy(self):
        if self.e >= rd.randint(1, 100) or self.reward == 0:
            return rd.randint(0, self.arms - 1)
        ac = -1
        high = -1
        for action in self.actions_rewards.keys():
            li = self.actions_rewards[action]
            if li[1] > 0 and li[0]/li[1] > high:
                high = li[0]/li[1]
                ac = action
        return ac

    def get_action(self):
        if self.method == "e-greedy":
            return self.get_action_greedy()

    def perform_action(self, action):
        # Gaussian bandit:
        if self.type == "GAU":
            # reward sampled from a normal distribution
            re = np.random.normal(100 + action, 1 + (action / 10))
        # Bernoulli bandit:
        else:
            if (50 + action) > rd.randint(1, 100):
                re = 1
            else:
                re = 0
        self.reward += float(re)
        self.actions_rewards[action][1] += 1  # update number of instances of action
        self.actions_rewards[action][0] += re  # add and store reward of said action
        return re

    def train(self):
        for i in range(100000):
            a = self.get_action()
            self.perform_action(a)
        for se in self.actions_rewards.items():
            if se[1][1] == 0:  # zero instance of action, meaning action never used
                print(se, "na")
            else:
                print(se, se[1][0] / se[1][1])

def run_bandit(arms, sort):
    while True:
        print("(e-)Greedy (1), Optimistic initial values (2), Upper-Confidence Bound (3), or Action Preferences (4)")
        i = input()
        if i.isnumeric() and int(i) in range(1, 5):
            break
        print("Invalid input...")
    bandit = Bandit(arms, sort)
    if int(i) == 1:
        bandit.e_greedy()
        print("e-greedy")
    elif int(i) == 2:
        bandit.optimistic()
        # print("OIV")
    elif int(i) == 3:
        # print("UCB")
        bandit.ucb()
    elif int(i) == 4:
        print("AP")
    bandit.train()
