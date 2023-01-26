import random
import numpy as np
import itertools as it
import csv


class Agent:
    def __init__(self, game):
        self.Q_table = {}
        # read in Q-table from csv
        with open('qtable.csv', mode='r') as infile:
            reader = csv.reader(infile)
            self.Q_table = {rows[0]: rows[1] for rows in reader}
        self.rewards = {}
        with open('rewards.csv', mode='r') as infile:
            reader = csv.reader(infile)
            self.rewards = {rows[0]: rows[1] for rows in reader}
        self.instances = {}
        with open('instances.csv', mode='r') as infile:
            reader = csv.reader(infile)
            self.instances = {rows[0]: rows[1] for rows in reader}
        self.game = game

    # creates Q-table entries   [saved][rolled][action][reward]
    # key: [saved][rolled][action]  value: [reward]
    # [saved][rolled]: state, [action]: action, [reward]: Q-value
    def create_table_entries(self):
        state_str = str(self.game.get_saved())
        state_str += str(self.game.get_rolled())
        print(state_str)
        actions = self.get_actions()
        for action in actions:
            state_action = state_str + str(action)
            if state_action not in self.Q_table.keys():
                self.Q_table[state_action] = 0
                self.rewards[state_action] = 0
                self.instances[state_action] = 0

    # returns list of all possible actions from the rolled dices.
    def get_actions(self):
        histo_dices = self.game.get_rolled()
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

    def select_action_e_greedy(self, actions):
        q_values = np.zeros(len(actions))
        state_str = str(self.game.get_saved())
        state_str += str(self.game.get_rolled())
        for x in actions:
            state_action = state_str + str(actions[x])
            q_values[x] = self.Q_table.get(state_action)
        print(q_values)
        return q_values[0]

    def play_round(self):
        self.game.roll()
        action = self.select_action_e_greedy(self.get_actions())
        self.game.update(action)

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
