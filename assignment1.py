from bandit import run_bandit


def run(bandit_type, arms):
    if bandit_type == 1:
        run_bandit(arms, "GAU")
    if bandit_type == 2:
        run_bandit(arms, "BER")


if __name__ == "__main__":
    while True:
        print("Gaussian Bandit (1) or Bernoulli Bandit (2)")
        i = input()
        if i.isnumeric() and int(i) in range(1, 3):
            break
        print("Invalid input...")
    while True:
        print("Choose amount of arms (between 2 and 10)")
        a = input()
        if a.isnumeric() and int(a) in range(2, 11):
            break
        print("Invalid input...")

    run(int(i), int(a))
