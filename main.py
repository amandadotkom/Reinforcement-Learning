from agent import Agent
from game import Game


if __name__ == "__main__":
    game = Game()
    agent = Agent(game)
    agent.play_round()



    agent.store_progress()
