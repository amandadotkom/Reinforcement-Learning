import random
from copy import copy

import numpy as np
import itertools as it
import csv
import math


class Agent:
    def __init__(self, game, method, epsilon_exploit):
        self.Q_table = {}
        # read in Q-table from csv
        with open('qtable.csv', mode='r') as infile:
            reader = csv.reader(infile)
            self.Q_table = {rows[0]: rows[1] for rows in reader}
            for key in self.Q_table.keys():
                self.Q_table[key] = float(self.Q_table[key])
        self.rewards = {}
        with open('rewards.csv', mode='r') as infile:
            reader = csv.reader(infile)
            self.rewards = {rows[0]: rows[1] for rows in reader}
            for key in self.rewards.keys():
                self.rewards[key] = float(self.rewards[key])
        self.instances = {}
        with open('instances.csv', mode='r') as infile:
            reader = csv.reader(infile)
            self.instances = {rows[0]: rows[1] for rows in reader}
            for key in self.instances.keys():
                self.instances[key] = float(self.instances[key])
        self.game = game
        self.action_selection = method
        self.eps_exp = epsilon_exploit

    # creates Q-table entries   [saved][rolled][action][reward]
    # key: [saved][rolled][action]  value: [reward]
    # [saved][rolled]: state, [action]: action, [reward]: Q-value
    def create_table_entries(self, actions):
        state_str = str(self.game.get_saved())
        state_str += str(self.game.get_rolled())
        for action in actions:
            state_action = state_str + str(action)
            if state_action not in self.Q_table.keys():
                self.Q_table[state_action] = 0
                self.rewards[state_action] = 0
                self.instances[state_action] = 0

    def update_q_table(self, pairs, reward):
        if len(pairs) == 2:
            self.rewards[pairs[0]] += reward * 0.8
            self.rewards[pairs[1]] += reward
            self.instances[pairs[0]] += 1
            self.instances[pairs[1]] += 1
            self.Q_table[pairs[0]] = self.rewards[pairs[0]] / self.instances[pairs[0]]
            self.Q_table[pairs[1]] = self.rewards[pairs[1]] / self.instances[pairs[1]]
        else:
            self.rewards[pairs[0]] += reward
            self.instances[pairs[0]] += 1
            self.Q_table[pairs[0]] = self.rewards[pairs[0]] / self.instances[pairs[0]]


    # returns list of all possible actions from the rolled dices.
    def get_actions(self):
        histo_dices = copy(self.game.get_rolled())
        visual_dices = self.convert_dice_representation(histo_dices)
        actions = []
        for L in range(len(visual_dices) + 1):
            for subset in it.combinations(visual_dices, L):
                actions.append(self.convert_dice_representation(subset))
        return actions

    def convert_dice_representation(self, dices):
        if len(dices) == 6:  # histogram representation to visual representation
            new = np.zeros(int(np.sum(dices)))
            idx = 0
            for x in range(6):
                while dices[x] > 0:
                    new[idx] = x + 1
                    dices[x] -= 1
                    idx += 1
            return new

        else:  # visual representation to histogram representation
            new = np.zeros(6)
            for x in range(len(dices)):
                new[int(dices[x] - 1)] += 1
        return new

    def select_action_softmax(self, actions):
        state_str = str(self.game.get_saved())
        state_str += str(self.game.get_rolled())
        probabilities = np.zeros(len(actions))
        sum_e = 0
        for a in range(len(actions)):
            state_action = state_str + str(actions[a])
            q = self.Q_table.get(state_action)
            probabilities[a] = pow(1+self.eps_exp, q)
            sum_e += pow(1+self.eps_exp, q)
        probabilities /= sum_e
        print(np.amax(probabilities) - np.amin(probabilities))
        x = np.random.choice(range(len(actions)), p=probabilities)
        return actions[x]


    def select_action_e_greedy(self, actions):
        q_values = np.zeros(len(actions))
        state_str = str(self.game.get_saved())
        state_str += str(self.game.get_rolled())
        for x in range(len(actions)):
            state_action = state_str + str(actions[x])
            q_values[x] = self.Q_table.get(state_action)
        if random.uniform(0,1) > self.eps_exp:
            argmax = np.argwhere(q_values == np.amax(q_values))
            argmax = argmax.flatten().tolist()
            return actions[random.choice(argmax)]
        else:
            rand = random.randint(0, len(actions)-1)
            return actions[rand]

    def select_action(self, actions):
        if self.action_selection == 1:
            return self.select_action_e_greedy(actions)
        if self.action_selection == 2:
            return self.select_action_softmax(actions)
    def play_round(self):
        turn = 1
        state_action_pairs = {}
        while turn <= 2 and np.sum(self.game.saved) < 5:
            self.game.roll()
            actions = self.get_actions()
            self.create_table_entries(actions)
            action = self.select_action(actions)
            state_str = str(self.game.get_saved())
            state_str += str(self.game.get_rolled())
            state_action = state_str + str(action)
            state_action_pairs[turn-1] = state_action
            self.game.update(action)
            turn += 1
        if np.sum(self.game.saved) < 5:
            self.game.roll()
            self.game.update(self.game.rolled)
        # print(self.convert_dice_representation(copy(self.game.saved)))
        reward = self.game.get_reward()
        if reward > 0:
            self.update_q_table(state_action_pairs, reward)
        self.game.reset()
        return reward


    # writes Q-table, rewards, and instances to csv
    def store_progress(self):
        with open('qtable.csv', 'w') as f:
            for key in self.Q_table.keys():
                f.write("%s,%s\n" % (key, self.Q_table[key]))
        with open('rewards.csv', 'w') as f:
            for key in self.Q_table.keys():
                f.write("%s,%s\n" % (key, self.rewards[key]))
        with open('instances.csv', 'w') as f:
            for key in self.Q_table.keys():
                f.write("%s,%s\n" % (key, self.instances[key]))
