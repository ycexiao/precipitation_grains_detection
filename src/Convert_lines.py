'''

本文使用的是修饰后的广度优先搜索算法，每次遇到分支会产生一次新的搜索，目的是减少晶粒生成的骨架中的不同直线的数量，便于后续的霍夫转换分析。

建议先阅读广度优先搜索算法，以及霍夫转换相关资料。

'''

import copy
from skimage import morphology
import numpy as np

class LineBud():
    def __init__(self, i, j):
        self.start_position = (i,j)
        self.current_position = (i, j)
        self.memory = []  # recording history of steps
        self.status = 'open'  # 'open' if (the LineBud can still grow) else 'close'

    def go(self, i=0, j=0):
        self.memory.append(list(self.current_position))
        self.current_position[0] += i
        self.current_position[1] += j

    def moveto(self, x, y):
        self.memory.append(list(self.current_position))
        self.current_position = (x, y)

def SearchLines(skeleton):
    skeleton = copy.deepcopy(skeleton)
    def Look_around(linebud, skeleton):
        i, j = linebud.current_position
        m, n = skeleton.shape

        # in avoidance the 3*3 template exceed skeleton
        if i == 0:
            rows = [i, i + 1]
        elif i == m - 1:
            rows = [i - 1, i]
        else:
            rows = [i - 1, i, i + 1]

        if j == 0:
            cols = [j, j + 1]
        elif j == n - 1:
            cols = [j - 1, j]
        else:
            cols = [j - 1, j, j + 1]

        # find directions to search along the skeleton
        coordinates = [[i, j] for i in rows for j in cols]
        coordinates.remove(list(linebud.current_position))  # current position is not recorded
        coordinates = np.array(coordinates)
        solid_points = np.argwhere(np.array([skeleton[i, j] for i, j in coordinates]) == 255)

        # reorganize the dimension of the output arrays
        out = []
        if len(solid_points) != 0:
            for i in range(len(solid_points)):
                out.append(coordinates[solid_points[i][0]])
            return out
        else:
            return []

    # Initialize
    solid_index = np.argwhere(skeleton == 255)
    corner = []
    corner.extend([solid_index[i] for i in [func(solid_index[:, 1]) for func in [np.argmin, np.argmax]]])  # left and right
    corner.extend([solid_index[i] for i in [func(solid_index[:, 0]) for func in [np.argmin, np.argmax]]])  # top and bottom
    # make sure the initial point is at either ends of the skeleton
    for i in range(len(corner)):
        start_point = corner[i]
        linebud = LineBud(*start_point)
        if len(Look_around(linebud, skeleton)) == 1:
            break

    # the pixel possessed by the lines is assigned 0 in case it will be searched again by other LineBud
    skeleton[start_point[0], start_point[1]] = 0
    total_buds = [linebud]
    lines = []
    maximum_iteration = 500
    flag = 0

    # Iteration
    while total_buds != []:
        if flag > maximum_iteration:
            break
        flag += 1
        for j in range(len(total_buds)):
            current_position = total_buds[j].current_position  # save the value in case it is changed by updating
            solid_coordinates = Look_around(total_buds[j], skeleton)  # solid pixels around the tip of current LineBud

            if len(solid_coordinates) == 0:  # there are no solid pixels around
                total_buds[j].memory.append(list(total_buds[j].current_position))
                total_buds[j].status = 'close'
            else:
                for k in range(len(solid_coordinates)):
                    if k == 0:  # searching along the first direction should not create new LineBud to bifurcate
                        total_buds[j].moveto(*solid_coordinates[k])
                    else:
                        new_bud = LineBud(*current_position)  # bifurcate
                        new_bud.moveto(*solid_coordinates[k])
                        total_buds.append(new_bud)
                    skeleton[solid_coordinates[k][0], solid_coordinates[k][1]] = 0

        lines.extend(list(filter(lambda x: x.status == 'close', total_buds)))  # lineBuds that stop grow
        total_buds = list(filter(lambda x: x.status == 'open', total_buds))
    return lines


def Skeletonize(bi_grain):
    bi_grain = (bi_grain >= 125).astype('uint8')
    skeleton = morphology.skeletonize(bi_grain) * 255
    skeleton = skeleton.astype('uint8')
    return skeleton
