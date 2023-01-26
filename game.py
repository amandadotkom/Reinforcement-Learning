import random
import numpy as np


class Game:
    def __init__(self):
        self.saved = np.zeros(6)
        self.rolled = np.zeros(6)

    def get_rolled(self):
        return self.rolled

    def get_saved(self):
        return self.saved

    # simulates rolling the dice
    def roll(self):
        sum_saved = int(np.sum(self.saved))

        for dice in range(5-sum_saved):  # 5 = num of dice
            d = random.randint(1, 6)
            self.rolled[d-1] += 1

    # for when agent already performed action, reset rolled dice
    def update(self, chosen):
        self.rolled = np.zeros(6)
        self.saved = np.add(self.saved, chosen)

    def reset(self):
        self.rolled = np.zeros(6)
        self.saved = np.zeros(6)

    def get_reward(self):
        reward = 0
        if self.check_yahtzee():
            reward = 50
        if self.check_lg_straight():
            if reward < 40:
                reward = 40
        if self.check_sm_straight():
            if reward < 30:
                reward = 30
        if self.check_full_house():
            if reward < 25:
                reward = 25
        if self.check_fr_kind():
            total = 0
            for x in range(len(self.saved)):
                total += self.saved[x] * (x + 1)
            if reward < total:
                reward = total
        if self.check_thr_kind():
            total = 0
            for x in range(len(self.saved)):
                total += self.saved[x] * (x + 1)
            if reward < total:
                reward = total
        return reward

    # 5 of a kind
    def check_yahtzee(self):
        for die in range(len(self.saved)):
            if self.saved[die] == 5:
                print("YATHZEE!!!!!!!!!!!!!!!!!!!!!!")
                return True
        return False

    # sequence of 5
    def check_lg_straight(self):
        cnt = 0
        for die in range(len(self.saved)):
            if self.saved[die] == 0:
                cnt = 0
            else:
                cnt += 1
                if cnt == 5:
                    #print("Large Straight")
                    return True
        return False
    
    # sequence of 4
    def check_sm_straight(self):
        cnt = 0
        for die in range(len(self.saved)):
            if self.saved[die] == 0:
                cnt = 0
            else:
                cnt += 1
                if cnt == 4:
                    #print("Small straight")
                    return True
        return False

    def check_thr_kind(self):
        for die in range(len(self.saved)):
            if self.saved[die] == 3:
                #print("3kind")
                return True
        return False

    def check_fr_kind(self):
        for die in range(len(self.saved)):
            if self.saved[die] == 4:
                #print("4kind")
                return True
        return False

    def check_full_house(self):
        flag = 0
        for die in range(len(self.saved)):
            if self.saved[die] == 2:
                flag = 1

        if flag == 1 and self.check_thr_kind():
            #print("Full House")
            return True
        return False
