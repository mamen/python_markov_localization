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

# x = row
# y = column
currentPosition = Position(5, 2)

# steps = [Step(direction=Direction.Up, sensedDistance=8, position=Position(5, 5))]

# normal
# steps = [Step(direction=Direction.Right, sensedDistance=0, position=Position(5, 6)),
#          Step(direction=Direction.Up, sensedDistance=0, position=Position(0, 6)),
#          Step(direction=Direction.Left, sensedDistance=0, position=Position(0, 0)),
#          Step(direction=Direction.Down, sensedDistance=5, position=Position(4, 0)),  # robot gets kidnapped here
#          Step(direction=Direction.Right, sensedDistance=4, position=Position(4, 6)),
#          Step(direction=Direction.Down, sensedDistance=1, position=Position(8, 5)),
#          Step(direction=Direction.Right, sensedDistance=0, position=Position(8, 9)),
#          Step(direction=Direction.Up, sensedDistance=1, position=Position(4, 9))]

# kidnapped
steps = [Step(direction=Direction.Right, sensedDistance=0, position=Position(5, 6)),
         Step(direction=Direction.Up, sensedDistance=0, position=Position(0, 6)),
         Step(direction=Direction.Left, sensedDistance=0, position=Position(0, 0)),  # robot gets kidnapped here
         Step(direction=Direction.Down, sensedDistance=0, position=Position(9, 4)),
         Step(direction=Direction.Right, sensedDistance=0, position=Position(9, 9)),
         Step(direction=Direction.Down, sensedDistance=1, position=Position(4, 9)),
         Step(direction=Direction.Left, sensedDistance=5, position=Position(4, 5)),
         Step(direction=Direction.Down, sensedDistance=1, position=Position(8, 5))]


def normalize(matrix):
    retVal = matrix.copy()

    retVal = retVal - retVal.mean()
    retVal = retVal / np.abs(retVal).max()

    return retVal


def plotData(step, prior, posterior):
    global epsilon, map, currentPosition

    fig = plt.figure(dpi=500)
    # fig.tight_layout()

    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.07, 1.07])

    # =======
    # Map
    # =======

    ax = fig.add_subplot(gs[0])

    ax.matshow(map, vmin=0, vmax=1, cmap='Greys')

    plt.title('Map', x=0.5, y=1.2)

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
    # Prior
    # =======

    ax = fig.add_subplot(gs[1])

    data = normalize(prior)

    im = ax.matshow(data, vmin=np.min(data), vmax=np.max(data))

    plt.title('Prior', x=0.5, y=1.2)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im, cax=cax)

    ticks = np.arange(0, N, 1)

    plt.grid(which='major', axis='both', linestyle=':', color='black')

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels(range(0, N))
    ax.set_yticklabels(range(0, N))

    # =======
    # Posterior
    # =======

    ax = fig.add_subplot(gs[2])

    data = normalize(posterior)

    im = ax.matshow(data, vmin=np.min(data), vmax=np.max(data))

    plt.title('Posterior', x=0.5, y=1.2)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im, cax=cax)

    ticks = np.arange(0, N, 1)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels(range(0, N))
    ax.set_yticklabels(range(0, N))

    # plt.tight_layout()

    plt.show()

    fig.savefig(f"step{step}.png")


def calcPrior(oldPosterior, direction):
    global epsilon, map

    steps = [3, 4, 5, 6, 7]
    stepProbability = [0, 0, 0, 0.1, 0.2, 0.4, 0.2, 0.1]

    prior = np.full((N, N), 0.0)

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
                                        if j - t >= 0 and map[i, j - t] == 1:
                                            skip = True
                                            break
                                    if not skip:
                                        if j - stepSize >= 0:
                                            prior[i, j] += oldPosterior[i, j - stepSize] * stepProbability[stepSize]
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
                                        if j + t < N and map[i, j + t] == 1:
                                            skip = True
                                            break
                                    if not skip:
                                        if j + stepSize < N:
                                            prior[i, j] += oldPosterior[i, j + stepSize] * stepProbability[stepSize]
                                    else:
                                        break
                # no wall in this row
                else:
                    # normal calculation
                    for stepSize in steps:
                        if direction == Direction.Right:
                            if j - stepSize >= 0:
                                prior[i, j] += oldPosterior[i, j - stepSize] * stepProbability[stepSize]
                        if direction == Direction.Left:
                            if j + stepSize < N:
                                prior[i, j] += oldPosterior[i, j + stepSize] * stepProbability[stepSize]

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
                                        if i + t < N and map[i + t, j] == 1:
                                            skip = True
                                            break
                                    if not skip:
                                        if i + stepSize < N:
                                            prior[i, j] += oldPosterior[i + stepSize, j] * stepProbability[stepSize]
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
                                        if 0 <= i - t and map[i - t, j] == 1:
                                            skip = True
                                            break
                                    if not skip:
                                        if i - stepSize > 0:
                                            prior[i, j] += oldPosterior[i - stepSize, j] * stepProbability[
                                            stepSize]
                                    else:
                                        break
                # no wall in this row
                else:
                    # normal calculation
                    for stepSize in steps:
                        if direction == Direction.Up:
                            if i + stepSize < N:
                                prior[i, j] += oldPosterior[i + stepSize, j] * stepProbability[stepSize]

                        if direction == Direction.Down:
                            if i - stepSize >= 0:
                                prior[i, j] += oldPosterior[i - stepSize, j] * stepProbability[stepSize]

    return prior


def calcPosterior(sensorValue, direction, prior):
    global epsilon, map

    sensorProbability = {
                            sensorValue - 2: 0.1,
                            sensorValue - 1: 0.2,
                            sensorValue: 0.4,
                            sensorValue + 1: 0.2,
                            sensorValue + 2: 0.1
                        }

    posterior = np.full((N, N), 0.0)

    for i in range(0, N):
        for j in range(0, N):

            # horizontal movement
            if direction == Direction.Right or direction == Direction.Left:
                # if this row contains a wall
                if map[i, :].__contains__(1):

                    # if robot moves to the right
                    if direction == Direction.Right:

                        # wall found, restart right from the next field
                        if map[i, j] == 1:
                            continue
                        else:
                            # is there a wall in range?
                            for k in range(max(0, sensorValue - 2), sensorValue + 2 + 1):
                                if j + k + 1 == N or (j + k + 1 < N and map[i, j + k + 1] == 1):
                                    posterior[i, j] = prior[i, j] * sensorProbability[k]
                    # left
                    elif direction == Direction.Left:
                        # wall found, restart right from the next field
                        if map[i, j] == 1:
                            continue
                        else:
                            # is there a wall in range?
                            for k in range(max(0, sensorValue - 2), sensorValue + 2 + 1):
                                if j - k - 1 == -1 or (j - k - 1 >= 0 and map[i, j - k - 1] == 1):
                                    posterior[i, j] = prior[i, j] * sensorProbability[k]

                # no wall in this row
                else:
                    # normal calculation
                    if direction == Direction.Right:
                        for k in range(sensorValue - 2, sensorValue + 2 + 1):
                            if N > N - k - 1 >= 0:
                                posterior[i, N - k - 1] = prior[i, N - k - 1] * sensorProbability[k]
                    if direction == Direction.Left:
                        for k in range(sensorValue - 2, sensorValue + 2 + 1):
                            if 0 <= k < N:
                                posterior[i, k] = prior[i, k] * sensorProbability[k]

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
                                if i - k - 1 == -1 or (i - k - 1 >= 0 and map[i - k - 1, j] == 1):
                                    posterior[i, j] = prior[i, j] * sensorProbability[k]

                    # robot moves down
                    else:
                        # wall found, restart right from the next field
                        if map[i, j] == 1:
                            continue
                        else:
                            # is there a wall in range?
                            for k in range(max(0, sensorValue - 2), sensorValue + 2 + 1):
                                if i + k + 1 == N or (i + k + 1 < N and map[i + k + 1, j] == 1):
                                    posterior[i, j] = prior[i, j] * sensorProbability[k]

                # no wall in this column
                else:
                    # normal calculation
                    if direction == Direction.Up:
                        for k in range(max(0, sensorValue - 2), sensorValue + 2 + 1):
                            if 0 <= k < N:
                                posterior[k, j] = prior[k, j] * sensorProbability[k]

                    if direction == Direction.Down:
                        for k in range(sensorValue - 2, sensorValue + 2 + 1):
                            if N > N - k - 1 >= 0:
                                posterior[N - k - 1, j] = prior[N - k - 1, j] * sensorProbability[k]


    posterior[posterior < epsilon] = epsilon

    posterior = posterior / np.sum(posterior)

    return posterior


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
    global map, currentPosition

    map = np.empty((N, N))
    map[:] = 0

    map[1, 1] = 1
    map[1, 8] = 1
    map[2, 9] = 1
    map[3, 4] = 1
    map[5, 7] = 1
    map[6, 8] = 1
    map[7, 2] = 1
    map[7, 3] = 1
    map[7, 7] = 1

    probabilities = np.full((N, N), 1 / (N * N - np.sum(map)))

    i = 0

    plotData(i, probabilities, probabilities)



    for step in steps:
        i += 1
        # 1. take random step
        # doStep(step.direction)

        currentPosition = step.realPosition

        # 2. calulate prior
        prior = calcPrior(probabilities, step.direction)

        # 3. get sensor values
        # distance = senseDistance(step.direction) + getSensorDerivation()
        distance = step.sensedDistance

        # 4. calulate posterior
        posterior = calcPosterior(distance, step.direction, prior)

        # probabilities[map == 1] = 0

        plotData(i, prior, posterior)

        probabilities = posterior


if __name__ == "__main__":
    main()
