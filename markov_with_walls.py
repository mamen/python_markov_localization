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

epsilon = 1/(N*N*N*N)

probabilities = np.full((N, N), 1 / (N * N))

# x = row
# y = column
currentPosition = Position(5, 2)

# normal
# steps = [Step(direction=Direction.Right, sensedDistance=1, position=Position(5, 5)),
#          Step(direction=Direction.Up, sensedDistance=2, position=Position(2, 5)),
#          Step(direction=Direction.Left, sensedDistance=1, position=Position(2, 1)),
#          Step(direction=Direction.Down, sensedDistance=1, position=Position(8, 1)),
#          Step(direction=Direction.Right, sensedDistance=3, position=Position(8, 6)),
#          Step(direction=Direction.Up, sensedDistance=0, position=Position(0, 6))]

# kidnapped
steps = [Step(direction=Direction.Right, sensedDistance=0, position=Position(5, 6)),
         Step(direction=Direction.Up, sensedDistance=0, position=Position(0, 6)),
         Step(direction=Direction.Left, sensedDistance=0, position=Position(0, 0)),
         Step(direction=Direction.Down, sensedDistance=5, position=Position(4, 0)),  # robot gets kidnapped here
         Step(direction=Direction.Right, sensedDistance=4, position=Position(4, 5)),
         Step(direction=Direction.Down, sensedDistance=1, position=Position(8, 5)),
         Step(direction=Direction.Right, sensedDistance=0, position=Position(8, 9)),
         Step(direction=Direction.Up, sensedDistance=3, position=Position(5, 9))]


def normalize(matrix):
    retVal = matrix.copy()

    retVal = retVal - retVal.mean()
    retVal = retVal / np.abs(retVal).max()

    return retVal


def plotData():
    global probabilities, epsilon, map, currentPosition

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
    global probabilities, epsilon, map

    steps = [3, 4, 5, 6, 7]
    stepProbability = [0, 0, 0, 0.1, 0.2, 0.4, 0.2, 0.1]

    newProbabilities = np.full((N, N), 0.0)

    # x = row = i
    # y = column = j

    # for each row
    for i in range(0, N):
        # for each column
        for j in range(0, N):

            # horizontal movement
            if direction == Direction.Right or direction == Direction.Left:
                # if this row contains a wall
                if map[i, :].__contains__(1):

                    # if robot moves to the right
                    if direction == Direction.Right:
                        # start from the left side and calc prior up to a wall
                        for stepSize in steps:

                            # wall found, restart right from the wall
                            if map[i, j] == 1:
                                break
                            else:

                                if j - stepSize >= 0:
                                    skip = False
                                    for t in range(0, stepSize + 1):
                                        if map[i, j - t] == 1:
                                            skip = True
                                            break
                                    if not skip:
                                        newProbabilities[i, j] += probabilities[i, j - stepSize] * stepProbability[stepSize]
                                    else:
                                        break
                    # robot moves to the left
                    else:
                        # start from the left side and calc prior up to a wall
                        for stepSize in steps:

                            # wall found, restart right from the wall
                            if map[i, j] == 1:
                                break
                            else:
                                if j + stepSize < N:
                                    skip = False
                                    for t in range(0, stepSize + 1):
                                        if map[i, j + t] == 1:
                                            skip = True
                                            break
                                    if not skip:
                                        newProbabilities[i, j] += probabilities[i, j + stepSize] * stepProbability[stepSize]
                                    else:
                                        break
                # no wall in this row
                else:
                    # normal calculation
                    for stepSize in steps:
                        if direction == Direction.Right:
                            if j - stepSize >= 0:
                                newProbabilities[i, j] += probabilities[i, j - stepSize] * stepProbability[stepSize]
                        if direction == Direction.Left:
                            if j + stepSize < N:
                                newProbabilities[i, j] += probabilities[i, j + stepSize] * stepProbability[stepSize]

            # vertical movement
            elif direction == Direction.Down or direction == Direction.Up:
                # if this row contains a wall
                if map[:, j].__contains__(1):

                    # robot moves up
                    if direction == Direction.Up:
                        # start from the left side and calc prior up to a wall
                        for stepSize in steps:

                            # wall found, restart right from the wall
                            if map[i, j] == 1:
                                break
                            else:

                                if i + stepSize < N:
                                    skip = False
                                    for t in range(0, stepSize + 1):
                                        if map[i + t, j] == 1:
                                            skip = True
                                            break
                                    if not skip:
                                        newProbabilities[i, j] += probabilities[i + stepSize, j] * stepProbability[stepSize]
                                    else:
                                        break
                    # robot moves down
                    else:
                        # start from the left side and calc prior up to a wall
                        for stepSize in steps:

                            # wall found, restart right from the wall
                            if map[i, j] == 1:
                                break
                            else:

                                if i - stepSize >= 0:
                                    skip = False
                                    for t in range(0, stepSize + 1):
                                        if map[i - t, j] == 1:
                                            skip = True
                                            break
                                    if not skip:
                                        newProbabilities[i, j] += probabilities[i - stepSize, j] * stepProbability[stepSize]
                                    else:
                                        break
                # no wall in this row
                else:
                    # normal calculation
                    for stepSize in steps:
                        if direction == Direction.Up:
                            if i + stepSize < N:
                                newProbabilities[i, j] += probabilities[i + stepSize, j] * stepProbability[stepSize]

                        if direction == Direction.Down:
                            if i - stepSize >= 0:
                                newProbabilities[i, j] += probabilities[i - stepSize, j] * stepProbability[stepSize]

    return newProbabilities


def calcPosterior(sensorValue, direction):
    global probabilities, epsilon, map

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

            # horizontal movement
            if direction == Direction.Right or direction == Direction.Left:
                # if this row contains a wall
                if map[i, :].__contains__(1):

                    # if robot moves to the right
                    if direction == Direction.Right:

                        # wall found, restart right from the next field
                        if map[i, N - j - 1] == 1:
                            continue
                        else:
                            # is there a wall in range?
                            for k in range(max(0, sensorValue - 2), sensorValue + 2 + 1):
                                if j + k + 1 >= N or map[i, j + k + 1] == 1:
                                    newProbabilities[i, j] = probabilities[i, j] * sensorProbability[k]
                    # left
                    elif direction == Direction.Left:
                        # wall found, restart right from the next field
                        if map[i, j] == 1:
                            continue
                        else:
                            # is there a wall in range?
                            for k in range(max(0, sensorValue - 2), sensorValue + 2 + 1):
                                if j - k - 1 < 0 or map[i, j - k - 1] == 1:
                                    newProbabilities[i, j] = probabilities[i, j] * sensorProbability[k]

                # no wall in this row
                else:
                    # normal calculation
                    if direction == Direction.Right:
                        for k in range(sensorValue - 2, sensorValue + 2 + 1):
                            if N > N - k - 1 >= 0:
                                newProbabilities[i, N - k - 1] = probabilities[i, N - k - 1] * sensorProbability[k]
                    if direction == Direction.Left:
                        for k in range(sensorValue - 2, sensorValue + 2 + 1):
                            if 0 <= k < N:
                                newProbabilities[i, k] = probabilities[i, k] * sensorProbability[k]

            #vertical movement
            elif direction == Direction.Down or direction == Direction.Up:
                # if this column contains a wall
                if map[:, j].__contains__(1):

                    # robot moves up
                    if direction == Direction.Up:

                        # wall found, restart right from the next field
                        if map[i, j] == 1:
                            continue
                        else:
                            # is there a wall in range?
                            for k in range(max(0, sensorValue - 2), sensorValue + 2 + 1):
                                if i - k - 1 < 0 or map[i - k - 1, j] == 1:
                                    newProbabilities[i, j] = probabilities[i, j] * sensorProbability[k]

                    # robot moves down
                    else:

                        # wall found, restart right from the next field
                        if map[i, j] == 1:
                            continue
                        else:
                            # is there a wall in range?
                            for k in range(max(0, sensorValue - 2), sensorValue + 2 + 1):
                                if i + k + 1 >= N or map[i + k + 1, j] == 1:
                                    newProbabilities[i, j] = probabilities[i, j] * sensorProbability[k]


                # no wall in this column
                else:
                    # normal calculation
                    if direction == Direction.Up:
                        for k in range(max(0, sensorValue - 2), sensorValue + 2 + 1):
                            if 0 <= k < N:
                                newProbabilities[k, j] = probabilities[k, j] * sensorProbability[k]

                    if direction == Direction.Down:
                        for k in range(sensorValue - 2, sensorValue + 2 + 1):
                            if N > N - k - 1 >= 0:
                                newProbabilities[N - k - 1, j] = probabilities[N - k - 1, j] * sensorProbability[k]


    newProbabilities[newProbabilities < epsilon] = epsilon

    newProbabilities = newProbabilities / np.sum(newProbabilities)

    return newProbabilities


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
    global realPos

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
        # 1. take random step
        # doStep(step.direction)

        currentPosition = step.realPosition

        # 2. calulate prior
        probabilities = calcPrior(step.direction)

        # 3. get sensor values
        # distance = senseDistance(step.direction) + getSensorDerivation()
        distance = step.sensedDistance

        # 4. calulate posterior
        probabilities = calcPosterior(distance, step.direction)

        plotData()


if __name__ == "__main__":
    main()
