from enum import Enum
from random import choices
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Direction(Enum):
    Up = 1
    Right = 2
    Down = 3
    Left = 4


class Position():
    x = 0
    y = 0
    direction = Direction.Right

    def __init__(self, x, y, direction = Direction.Right):
        self.x = x
        self.y = y
        self.direction = direction


class Step():
    direction = Direction.Right
    sensedDistance = 0
    realPosition = Position(0, 0)

    def __init__(self, direction, sensedDistance, position):
        self.direction = direction
        self.sensedDistance = sensedDistance
        self.realPosition = position

N = 10

map = []

epsilon = 1/(N*N*N)

probabilities = np.full((N, N), 1 / (N * N))

# x = row
# y = column
currentPosition = Position(5, 2)

steps = [Step(direction=Direction.Right, sensedDistance=0, position=Position(5, 6)),
         Step(direction=Direction.Up, sensedDistance=0, position=Position(0, 6)),
         Step(direction=Direction.Left, sensedDistance=0, position=Position(0, 0)),
         Step(direction=Direction.Down, sensedDistance=5, position=Position(4, 0)),  # robot gets kidnapped here
         Step(direction=Direction.Right, sensedDistance=4, position=Position(4, 5)),
         Step(direction=Direction.Down, sensedDistance=1, position=Position(8, 5)),
         Step(direction=Direction.Left, sensedDistance=0, position=Position(8, 3))]

def normalize(matrix):
    retVal = matrix.copy()

    retVal = retVal - retVal.mean()
    retVal = retVal / np.abs(retVal).max()

    return retVal


def plotData():
    global probabilities, realPos, epsilon, map, currentPosition

    fig = plt.figure(dpi=500)
    # fig.tight_layout()

    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.07])

    # =======
    # Map
    # =======

    ax = fig.add_subplot(gs[0])

    ax.matshow(map, vmin=0, vmax=1, cmap='Greys')
    ticks = np.arange(0, N, 1)

    plt.grid(which='major', axis='both', linestyle=':', color='black')

    for i in range(len(map)):
        for j in range(len(map)):
            if currentPosition.x == i and currentPosition.y == j:
                ax.text(j, i, "R", ha="center", va="center", color="red", weight='bold')

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels(range(0, N))
    ax.set_yticklabels(range(0, N))

    # =======
    # Correlation Matrix
    # =======

    ax = fig.add_subplot(gs[1])

    data = normalize(probabilities)

    im = ax.matshow(data, vmin=np.min(data), vmax=np.max(data))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im, cax=cax)

    ticks = np.arange(0, N, 1)

    plt.grid(which='major', axis='both', linestyle=':', color='black')

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels(range(0, N))
    ax.set_yticklabels(range(0, N))


    for i in range(len(data)):
        for j in range(len(data)):
            if data[i, j] > epsilon:
                if currentPosition.x == i and currentPosition.y == j:
                    ax.text(j, i, round(data[i, j], 2), ha="center", va="center", color="red", weight='bold')
                else:
                    ax.text(j, i, round(data[i, j], 2), ha="center", va="center", color="w")

    plt.tight_layout()

    plt.show()


def calcPrior(direction):
    global probabilities, N

    possibleStepSizes = [3, 4, 5, 6, 7]
    stepProbability = [0, 0, 0, 0.1, 0.2, 0.4, 0.2, 0.1]

    newProbabilities = np.full((N, N), 0.0)

    for i in range(0, N):
        for j in range(0, N):
            for stepSize in possibleStepSizes:

                if direction == Direction.Right:
                    if j - stepSize >= 0:
                        newProbabilities[i, j] += probabilities[i, j - stepSize] * stepProbability[stepSize]

                if direction == Direction.Left:
                    if j + stepSize < N:
                        newProbabilities[i, j] += probabilities[i, j + stepSize] * stepProbability[stepSize]

                if direction == Direction.Up:
                    if i + stepSize < N:
                        newProbabilities[i, j] += probabilities[i + stepSize, j] * stepProbability[stepSize]

                if direction == Direction.Down:
                    if i - stepSize >= 0:
                        newProbabilities[i, j] += probabilities[i - stepSize, j] * stepProbability[stepSize]

    return newProbabilities


def calcPosterior(sensorValue, direction):
    global probabilities, N

    sensorProbability = {
                            sensorValue - 2: 0.1,
                            sensorValue - 1: 0.2,
                            sensorValue: 0.4,
                            sensorValue + 1: 0.2,
                            sensorValue + 2: 0.1
                        }

    newProbabilities = np.full((N, N), 0.0)

    for i in range(0, N):
        for j in range(0, N):

            if direction == Direction.Up:
                for k in range(max(0, sensorValue - 2), sensorValue + 2 + 1):
                    if 0 <= k < N:
                        newProbabilities[k, j] = probabilities[k, j] * sensorProbability[k]

            elif direction == Direction.Right:
                for k in range(sensorValue - 2, sensorValue + 2 + 1):
                    if N > N - k - 1 >= 0:
                        newProbabilities[i, N - k - 1] = probabilities[i, N - k - 1] * sensorProbability[k]

            elif direction == Direction.Down:
                for k in range(sensorValue - 2, sensorValue + 2 + 1):
                    if N > N - k - 1 >= 0:
                        newProbabilities[N - k - 1, j] = probabilities[N - k - 1, j] * sensorProbability[k]

            elif direction == Direction.Left:
                for k in range(sensorValue - 2, sensorValue + 2 + 1):
                    if 0 <= k < N:
                        newProbabilities[i, k] = probabilities[i, k] * sensorProbability[k]

    newProbabilities = newProbabilities / np.sum(newProbabilities)

    return newProbabilities


def main():
    global probabilities, map, currentPosition

    map = np.empty((N, N))
    map[:] = 0

    map[0, 8] = 1
    map[0, 9] = 1
    map[1, 1] = 1
    map[1, 8] = 1
    map[1, 9] = 1
    map[3, 4] = 1
    map[5, 7] = 1
    map[6, 8] = 1
    map[7, 2] = 1
    map[7, 3] = 1
    map[7, 7] = 1

    plotData()

    for step in steps:

        currentPosition = step.realPosition
        # 1. calulate prior
        probabilities = calcPrior(step.direction)

        # 2. get sensor values
        distance = step.sensedDistance

        # 3. calulate posterior
        probabilities = calcPosterior(distance, step.direction)

        plotData()


if __name__ == "__main__":
    main()
