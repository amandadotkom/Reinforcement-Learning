import random
import numpy as np

class Game():
    saved = np.zeros(6)
    rolled = np.zeros(6)

    def get_rolled(self):
        return self.rolled

    def get_saved(self):
        return self.saved

    #simulates rolling the dice
    def roll(self):
        sum_saved = np.sum(self.saved)

        for dice in range (5-sum_saved):  # 5 = num of dice
            d = random.randint(1, 6)
            self.rolled[d-1] += 1 
    

    # for when agent alrdy performed action, reset rolled dice
    def update(self, chosen):
        self.rolled = np.zeros(6)
        self.saved = np.add(self.saved, chosen)

    def get_reward(self):
        check_yahtzee(self)
        check_lg_straight(self)
        check_sm_straight(self)
        check_full_house(self)
        check_thr_kind(self)
        check_fr_kind(self)

    # 5 of a kind
    def check_yahtzee(self):
        for die in range(len(self.saved)):
            if self.saved[die] == 5:
                return True
        return False

    # sequence of 5
    def check_lg_straight(self):
        cnt = 0
        for die in range(len(self.saved)):
            if self.saved[die] == 1:
                cnt += 1
        
        if cnt == 5:
            return True
        else:
            return False
    
    # sequence of 4
    def check_sm_straight(self):
        cnt = 0
        for die in range(len(self.saved)):
            if self.saved[die] == 1:
                cnt += 1
        
        if cnt == 4:
            return True

        return False

    def check_thr_kind(self):
        for die in range(len(self.saved)):
            if self.saved[die] == 3:
                return True
        
        return False

    def check_fr_kind(self):
        for die in range(len(self.saved)):
            if self.saved[die] == 4:
                return True
        
        return False

    def check_full_house(self):
        flag = 0
        for die in range(len(self.saved)):
            if self.saved[die] == 2:
                flag = 1

        if flag == 1 and self.check_thr_kind(self) == True:
            return True

        return False

   

    



