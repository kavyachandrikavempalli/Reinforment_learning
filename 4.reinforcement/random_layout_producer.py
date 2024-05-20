import matplotlib.pyplot as plt
import seaborn as sns
# from datetime import datetime
import numpy as np
import os

def createDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def layoutSave(folder, filename, lay):
    createDir(folder)
    path = os.path.join(folder, filename)
    with open(path, 'w') as file:
        for row in lay:
            file.write(''.join(row) + '\n')

def plotKernelDensityEstimate(data, title, xlabel):
    # sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data, bw_adjust=0.5, fill=True, color='red')
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Density Estimate", fontsize=14)  # Updated ylabel
    plt.show()

def layoutNew(size):
    lay = [[' ' for _ in range(size)] for _ in range(size)]
    
    # Place outer walls
    for i in range(size):
        lay[0][i] = '%'
        lay[size-1][i] = '%'
        lay[i][0] = '%'
        lay[i][size-1] = '%'
    
    # Place Pacman and ghosts
    lay[1][1] = 'P'  # Pacman starting position
    ghostCap = np.random.randint(1,4)
    ghostCount = 0

    # Randomly place pellets
    for i in range(1, size-1):
        for j in range(1, size-1):
            if lay[i][j] == ' ' and np.random.rand() < 0.2:  # 20% chance to place a pellet
                lay[i][j] = '.'
            elif lay[i][j] == ' ' and np.random.rand() < 0.05 and ghostCount <= ghostCap: # 5% chance to place ghost if under cap
                lay[i][j] = 'G'
                ghostCount += 1
    if ghostCount < 1:  # default in case ghost is not placed
        lay[size-2][1] = 'G'

    # Randomly place internal walls
    WallProb = 0.1  # Probability of placing a wall in an empty space
    for i in range(2, size-2):  # Avoid placing walls too close to the border
        for j in range(2, size-2):
            if lay[i][j] == ' ' and np.random.rand() < WallProb:
                lay[i][j] = '%'

    return lay

def main():
    num_files = 10
    folder = "layouts"
    size_data = []
    size_count = {}

    # Generate a test sample to check distribution
    test_sizes = np.random.normal(16, 2, 10000)
    plotKernelDensityEstimate(test_sizes, "Grid Layout Distribution", "Grid Size")

    for i in range(num_files):
        size = int(np.clip(np.random.normal(16, 2), 7, 25))
        lay = layoutNew(size)
        size_data.append(size)

        # Increment the count for the current size
        if size not in size_count:
            size_count[size] = 0
        size_count[size] += 1

        filename = f"lay{size}_num{size_count[size]}.lay"
        layoutSave(folder, filename, lay)
        print(f"Layout with grid size {size}, count {size_count[size]} saved to {filename}")

if __name__ == "__main__":
    main()
