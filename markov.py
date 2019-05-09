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


class Position(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self


N = 100

board = np.full((N, N),  1 / (N*N))

realPos = Position(x=10, y=10, dir=Direction.Right)

steps = [Direction.Right, Direction.Up, Direction.Up]


def plotMap():
    plt.imshow(board, cmap='hot')
    plt.show()


def calcPrior(direction):
    global board

    # new bel at position x = sum over all old positions x { prob(new position x | sensordata now, pos old) * believe(old position)

    stepSize = getStepSize()

    p = np.full((N, N),  1)

    for i in range(1, N):
        for j in range(1, N):
            # board[i, j] =  * board[i, j]
            print('TODO')

    return 0


def calcPosterior(sensorValue, direction):
    return 0


def getStepSize():
    population = [3, 4, 5, 6, 7]
    weights = [0.1, 0.2, 0.4, 0.2, 0.1]

    return choices(population, weights)[0]


def getSensorDerivation():
    population = [-2, -1, 0, 1, 2]
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

def main():

    print(realPos.x, realPos.y)
    plotMap()

    for step in steps:
        # 1. take step
        doStep(step)

        # 2. calulate prior
        calcPrior(step)

        # 3. calulate posterior
        calcPosterior(0, 0)

        # 4. plot new position
        plotMap()

        print(realPos.x, realPos.y)


if __name__ == "__main__":
    main()
