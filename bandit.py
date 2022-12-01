import csv
import random as rd
import numpy as np


class Bandit:

    def __init__(self, arm, sort):
        self.type = sort
        self.arms = arm
        self.reward = 0.0
        self.actions = []
        self.preferences = [0 for i in range(self.arms)]
        self.probs = [0 for i in range(self.arms)]
        # As value in dict a list, (total reward, instances) to calculate avg
        self.actions_rewards = dict((k, [float(0), int(0)]) for k in range(self.arms))
        self.policy = None
        self.method = None
        self.e = 0
        self.c = 0
        if self.type == "GAU":
            self.means = [np.random.normal(0, 2) for i in range(self.arms)]
        else:
            self.means = [int(np.random.normal(50, self.arms)) for i in range(self.arms)]

    # Action preferences
    # Initializes preferences of all actions to zero
    def ap(self):
        self.method = "AP"
        for k in range(self.arms):
            self.preferences[k] = 0
            self.probs[k] = 1/self.arms

    # Policy of action based on Softmax/Boltzmann distribution, used for updating preferences
    def pi(self):
        ret = [0 for i in range(self.arms)]
        div = 0
        for k in range(self.arms):
            div += np.exp(self.preferences[k])
        for a in range(self.arms):
            ret[a] = np.exp(self.preferences[a]) / div
        return ret

    # Updates action preferences for all actions
    def update_preferences(self, reward, action, timestep):
        self.probs = self.pi()
        for a in range(self.arms):
            # Next action taken is the current action, Ht+1(a').
            if self.actions_rewards[a][1] != 0:
                stepsize = 1 / (self.actions_rewards[a][1])
            else:
                stepsize = 1
            if a == action:
                new = self.preferences[a] + stepsize * (reward - self.reward / (timestep + 1)) * (1 - self.probs[a])
            else:
                new = self.preferences[a] - stepsize * (reward - self.reward / (timestep + 1)) * (self.probs[a])
            self.preferences[a] = new

    # Upper Confidence Bound
    def ucb(self):
        self.method = "UCB"
        self.c = 3
        # while True:
        #     print("Select c (above 0)")  # TRY TO NORMALIZE C
        #     i = input()
        #     if i.isnumeric() and float(i) >= 1:
        #         self.c = int(i)
        #         break
        #     print("Invalid input...")

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

    # Implements the function argmax Gt(a)
    # used for (e)greedy and optimistic initial values,
    def get_max(self):
        ac = -1
        high = -1
        for action in self.actions_rewards.keys():
            li = self.actions_rewards[action]
            if li[1] == 0:
                if li[0] > high:
                    high = li[0]
                    ac = action
            elif li[1] > 0 and li[0] / li[1] > high:
                high = li[0] / li[1]
                ac = action
        return ac

    # Gets action for Greedy
    def get_action_greedy(self):
        if self.e >= rd.randint(1, 100) or self.reward == 0:
            return rd.randint(0, self.arms - 1)
        return self.get_max()

    # Gets action for Optimistic Initial Values
    def get_action_oiv(self):
        if self.reward == 0:  # initially return random action
            return rd.randint(0, self.arms - 1)
        return self.get_max()

    # Gets action for Upper Confidence Bound
    def get_action_ucb(self, i):
        high = -1
        ac = -1
        # find argmax:
        for se in self.actions_rewards.items():
            if se[1][1] == 0:  # get rid of zero division by setting instances (Nt) as 1 initially
                ins = 1
            else:
                ins = se[1][1]  # set instances (Nt), the number of times action has been taken
            qta = se[1][0] / ins  # Qt(a), sum of rewards/instances
            unc = self.c * np.sqrt((np.log(i + 1) / ins))  # uncertainty-part of the UCB equation
            if (qta + unc) > high:
                high = qta + unc
                ac = se[0]
        return ac

    # Action preferences
    def get_action_ap(self):
        arrprob = np.array(self.probs)
        arrout = np.array([i for i in range(self.arms)])
        ac = np.random.choice(arrout, 1, p=arrprob)
        return int(ac[0])

    # Calls the function for the corresponding exploration methods
    def get_action(self, i):
        if self.method == "e-greedy":
            return self.get_action_greedy()
        elif self.method == "OIV":
            return self.get_action_oiv()
        elif self.method == "UCB":
            return self.get_action_ucb(i)
        elif self.method == "AP":
            return self.get_action_ap()

    # Updates the accumulated reward of an action and its instance count
    # Also returns reward (which is useful for the Action Preferences method)
    def perform_action(self, action):
        # Gaussian bandit:
        if self.type == "GAU":
            # reward sampled from a normal distribution
            re = np.random.normal(self.means[action], 1)
        # Bernoulli bandit:
        else:
            if (self.means[action]) > rd.randint(1, 100):
                re = 1
            else:
                re = 0
        self.reward += float(re)
        self.actions_rewards[action][1] += 1  # update number of instances of action
        self.actions_rewards[action][0] += re  # add and store reward of said action
        return re

    def train(self):
        for i in range(100000):
            a = self.get_action(i)
            #  if method is Action Preferences, preference of the action taken must be updated as well
            if self.method == "AP":
                self.update_preferences(self.perform_action(a), a, i)  # param: reward, action, timestep
            else:
                self.perform_action(a)

        file = open('output.csv', 'a', newline='')
        writer = csv.writer(file)
        row = ["Action", "Total Reward", "Instances", "Average Reward", "Mean sampling distribution"]
        # writer.writerow(row)
        for se in self.actions_rewards.items():
            row[0] = se[0]
            row[1] = se[1][0]
            row[2] = se[1][1]
            row[4] = self.means[se[0]]
            if se[1][1] == 0:  # zero instance of action, meaning action never used
                print(se, "na")
                row[3] = "NA"
            else:
                print(se, se[1][0] / se[1][1])
                row[3] = se[1][0] / se[1][1]
            # writer.writerow(row)
        print(self.means)
        writer.writerow([self.reward / (100000*(np.array(self.means).max())) * 100])


def run_bandit(arms, sort):
    while True:
        print("(e-)Greedy (1), Optimistic initial values (2), Upper-Confidence Bound (3), or Action Preferences (4)")
        i = input()
        if i.isnumeric() and int(i) in range(1, 5):
            break
        print("Invalid input...")
    for z in range(10):
        bandit = Bandit(arms, sort)
        if int(i) == 1:
            bandit.e_greedy()
            # print("e-greedy")
        elif int(i) == 2:
            bandit.optimistic()
            # print("OIV")
        elif int(i) == 3:
            # print("UCB")
            bandit.ucb()
        elif int(i) == 4:
            # print("AP")
            bandit.ap()
        bandit.train()
