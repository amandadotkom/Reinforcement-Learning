from bandit import run_bandit


def run(i, a):
    if i == 1:
        run_bandit(a, "GAU")
    if i == 2:
        run_bandit(a, "BER")


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
