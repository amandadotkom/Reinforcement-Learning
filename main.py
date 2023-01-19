from agent import Agent
from game import Game


if __name__ == "__main__":
    game = Game()
    agent = Agent(game)
    game.roll()
    agent.create_table_entries()



    agent.store_progress()
