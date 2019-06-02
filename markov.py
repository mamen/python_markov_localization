from enum import Enum
import numpy as np
import matplotlib.pyplot as plt


class Direction(Enum):
    Up = 1
    Right = 2
    Down = 3
    Left = 4


class Position:
    x = 0
    y = 0
    direction = Direction.Right

    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.direction = direction


class Step:
    direction = Direction.Right
    sensedDistance = 0
    size = 0

    def __init__(self, direction, sensedDistance, size):
        self.direction = direction
        self.sensedDistance = sensedDistance
        self.size = size


N = 10

epsilon = 1/(N*N*N)

probabilities = np.full((N, N), 1 / (N * N))

steps = [Step(direction=Direction.Right, sensedDistance=4, size=3),
         Step(direction=Direction.Up, sensedDistance=2, size=3),
         Step(direction=Direction.Left, sensedDistance=1, size=4),
         Step(direction=Direction.Down, sensedDistance=3, size=4)]

# steps = [Step(direction=Direction.Right, sensedDistance=80, size=3),
#          Step(direction=Direction.Up, sensedDistance=50, size=3),
#          Step(direction=Direction.Up, sensedDistance=45, size=3)]


def normalize(matrix):
    retVal = matrix.copy()

    retVal = retVal - retVal.mean()
    retVal = retVal / np.abs(retVal).max()

    return retVal


def plotData():
    global probabilities, N, epsilon

    # plot correlation matrix
    fig = plt.figure(dpi=500)
    ax = fig.add_subplot(111)

    # data = normalize(probabilities)
    data = probabilities

    cax = ax.matshow(data, vmin=np.min(data), vmax=np.max(data))
    fig.colorbar(cax)
    ticks = np.arange(0, N, 1)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels(range(0, N))
    ax.set_yticklabels(range(0, N))

    for i in range(len(data)):
        for j in range(len(data)):
            if data[i, j] > epsilon:
                ax.text(j, i, round(data[i, j], 2), ha="center", va="center", color="w")

    ax.set_title("Occupancy Grid")

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
    global probabilities, steps

    plotData()

    for step in steps:
        # 1. calulate prior
        probabilities = calcPrior(step.direction)

        # 2. get sensor values
        distance = step.sensedDistance

        # 3. calulate posterior
        probabilities = calcPosterior(distance, step.direction)

        plotData()


if __name__ == "__main__":
    main()
