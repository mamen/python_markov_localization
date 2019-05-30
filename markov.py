from enum import Enum
from random import choices
import numpy as np
import matplotlib.pyplot as plt
import random

class Direction(Enum):
    Up = 1
    Right = 2
    Down = 3
    Left = 4


class Position():
    x = 0
    y = 0
    direction = Direction.Right

    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.direction = direction


class Step():
    direction = Direction.Right
    sensedDistance = 0

    def __init__(self, direction, sensedDistance):
        self.direction = direction
        self.sensedDistance = sensedDistance

N = 10

map =  np.full((N, N), 0)

epsilon = 1/(N*N*N)

probabilities = np.full((N, N), 1 / (N * N))

realPos = Position(x=2, y=5, direction=Direction.Right)

steps = [Step(direction=Direction.Right, sensedDistance=4),
         Step(direction=Direction.Up, sensedDistance=2),
         Step(direction=Direction.Left, sensedDistance=1),
         Step(direction=Direction.Down, sensedDistance=3)]


def normalize(matrix):
    retVal = matrix.copy()

    retVal = retVal - retVal.mean()
    retVal = retVal / np.abs(retVal).max()

    return retVal


def plotData():
    global probabilities

    # plot correlation matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)

    data = normalize(probabilities)

    cax = ax.matshow(data, vmin=np.min(data), vmax=np.max(data))
    fig.colorbar(cax)
    ticks = np.arange(0, N, 1)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels(range(1, N))
    ax.set_yticklabels(range(1, N))

    for i in range(len(map)):
        for j in range(len(map)):
            text = ax.text(j, i, np.round(data[i, j], 2), ha="center", va="center", color="w")

    print(np.sum(map))

    #ax.set_title("Harvest of local farmers (in tons/year)")
    plt.show()


def calcPrior(direction):
    global probabilities, epsilon

    steps = [3, 4, 5, 6, 7]
    stepProbability = [0, 0, 0, 0.1, 0.2, 0.4, 0.2, 0.1]

    newProbabilities = np.full((N, N), 0.0)

    for i in range(0, N):
        for j in range(0, N):
            for stepSize in steps:
                if j + stepSize < N:
                    newProbabilities[i, j + stepSize] += probabilities[i, j] * stepProbability[stepSize]

    newProbabilities[newProbabilities < epsilon] = epsilon

    return newProbabilities


def calcPosterior(sensorValue, direction):
    global probabilities, epsilon

    sensorValues = [sensorValue - 2, sensorValue - 1, sensorValue, sensorValue + 1, sensorValue + 2]

    sensorProbability = [0.1, 0.2, 0.4, 0.2, 0.1]

    newProbabilities = np.full((N, N), 0.0)


    for i in range(0, N):
        for j in range(0, N):

            if direction == Direction.Up:
                return 0

            elif direction == Direction.Right:
                for k in range(0, N - sensorValue - 2):
                    newProbabilities[i, k] = epsilon

                for k in range(sensorValue - 2, sensorValue + 2):
                    # passt noch nicht
                    newProbabilities[i, k] = probabilities[i, N - k] * sensorProbability[k - len(sensorValues) - 1]

                for k in range(sensorValue + 2, N):
                    newProbabilities[i, k] = epsilon

            elif direction == Direction.Down:
                return 0

            elif direction == Direction.Left:
                return 0

    return 0


def getSensorDerivation():
    # z_t
    population = [-2, -1, 0, 1, 2]
    weights = [0.1, 0.2, 0.4, 0.2, 0.1]

    return choices(population, weights)[0]


def getStepSize():
    # x_t
    population = [3, 4, 5, 6, 7]
    weights = [0.1, 0.2, 0.4, 0.2, 0.1]

    return choices(population, weights)[0]


def doStep(direction):

    stepSize = getStepSize()

    print(f'Moving {stepSize} in Direction {direction}')

    if direction == Direction.Up:

        if realPos.y - stepSize < 0:
            stepSize = realPos.y
            print(f'robot hit upper wall, moved only {stepSize}')

        realPos.y = realPos.y - stepSize

    elif direction == Direction.Right:

        if realPos.x + stepSize > N:
            stepSize = N - realPos.x
            print(f'robot hit right wall, moved only {stepSize}')

        realPos.x = realPos.x + stepSize

    elif direction == Direction.Down:

        if realPos.y + stepSize > N:
            stepSize = N - realPos.y
            print(f'robot hit lower wall, moved only {stepSize}')

        realPos.y = realPos.y + stepSize

    elif direction == Direction.Left:

        if realPos.x - stepSize < 0:
            stepSize = realPos.x
            print(f'robot hit left wall, moved only {stepSize}')

        realPos.x = realPos.x - stepSize


def senseDistance(direction):
    global realPos

    distance = 0

    if direction == Direction.Up:

        for i in range(1, realPos.y+1):

            if realPos.y - i < 0:
                break

            if map[realPos.x - i,  realPos.y] == 0:
                distance += 1
            else:
                break

    elif direction == Direction.Right:

        for i in range(1, N - realPos.x):

            if realPos.x + i > N:
                break

            if map[realPos.x,  realPos.y + i] == 0:
                distance += 1
            else:
                break

    elif direction == Direction.Down:

        for i in range(1, N - realPos.y):

            if realPos.y + i > N:
                break

            if map[realPos.x,  realPos.y + i] == 0:
                distance += 1
            else:
                break

    elif direction == Direction.Left:
        for i in range(1, realPos.x + 1):

            if realPos.x - i < 0:
                break

            if map[realPos.x - i, realPos.y] == 0:
                distance += 1
            else:
                break

    return distance


def main():
    global probabilities

    # print(realPos.x, realPos.y)
    plotData()

    for step in steps:
        # 1. take step
        doStep(step.direction)

        # 2. calulate prior
        probabilities = calcPrior(step.direction)

        # 4. plot new position
        plotData()

        # distance = senseDistance(step.direction)

        distance = step.sensedDistance

        # 3. calulate posterior
        calcPosterior(distance, step.direction)

        # print(realPos.x, realPos.y)


if __name__ == "__main__":
    main()
