import matplotlib.pyplot as plt
import numpy as np


def load_plot(file_path):
    list_train_xent = list()
    list_valid_xent = list()

    file_load = open(file_path, "r")

    lines = file_load.readlines()

    for value in lines[0].split(","):
        list_train_xent.append(float(value))

    for value in lines[1].split(","):
        list_valid_xent.append(float(value))

    return list_train_xent, list_valid_xent


list_xent_train, list_xent_valid = load_plot(
    "/Users/ntdat/Downloads/PreSummBaseline/save_plot_Baseline_8000.txt")


arr_train = np.arange(start=0, stop=len(list_xent_train) * 100, step=100)
arr_valid = np.arange(start=0, stop=len(list_xent_valid) * 100, step=100)

plt.clf()
plt.title("Loss Function")
plt.plot(arr_valid, list_xent_valid, color="red", label="Validate")
plt.plot(arr_train, list_xent_train, color="blue", label="Train")
plt.ylabel("Cross Entropy(Xent)")
plt.xlabel("Step")
plt.legend()
plt.savefig(str(8000) + ".jpg")
