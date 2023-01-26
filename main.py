from agent import Agent
from game import Game
import time


if __name__ == "__main__":
    method = 0
    epsilon_exploit = 0
    while True:
        print("Action selection?: e-greedy (1) or softmax (2)")
        method = input()
        if method.isnumeric() and int(method) in range(1, 3):
            break
        print("Invalid input...")

    if int(method) == 1:
        while True:
            print("Enter epsilon: (float between 0 and 1)")
            epsilon_exploit = input()
            try:
                float(epsilon_exploit)
                if 1 >= float(epsilon_exploit) >= 0:
                    break
            except ValueError:
                print("Invalid input...")
            print("Invalid input...")

    if int(method) == 2:
        while True:
            print("Enter exploit rate: (float between 0 (random) and 4)")
            epsilon_exploit = input()
            try:
                float(epsilon_exploit)
                if 4 >= float(epsilon_exploit) >= 0:
                    break
            except ValueError:
                print("Invalid input...")
            print("Invalid input...")


    game = Game()
    agent = Agent(game, int(method), float(epsilon_exploit))
    while True:
        time_to_run = input("How many minutes?")
        if time_to_run.isdigit():
            break
        print("Enter Integer")

    t_end = time.time() + 60 * int(time_to_run)
    avg_reward = 0
    rounds = 0
    while time.time() < t_end:
        avg_reward += agent.play_round()
        rounds += 1


    if rounds != 0:
        print("Avg reward over {} minutes is {}".format(time_to_run, avg_reward/rounds))

    agent.store_progress()
