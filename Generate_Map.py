import random
import numpy as np
import matplotlib.pyplot as plt
import math


def create_random_map(height, width, core_perc, core_grow, N_wounded, seed):
    # set seed
    random.seed(seed)

    # range of the map
    map_grid = [[10 for j in range(0, width)] for i in range(0, height)]
    map_grid = np.array(map_grid)
    # O
    heightO = 99
    widthO = 0
    map_grid[heightO, widthO] = 10
    # D
    heightD = height - 1
    widthD = width - 1
    map_grid[heightD, widthD] = 10

    # generate block core:
    core_notOD = 0
    map_grid_flat = list(range(0, height * width))
    while (core_notOD != 1):
        core_notOD = 1
        core_flat = random.sample(map_grid_flat, int(height * width * core_perc))
        core_location = []
        for i in core_flat:
            H = int(i / width)
            W = i % width
            flag = False
            if (heightO - 4 < H < heightO + 4 and widthO - 4 < W < widthO + 4) \
                    or (heightD - 4 < H < heightD + 4 and widthD - 4 < W < widthD + 4):
                flag = True
            if flag:
                core_notOD = 0
                break
            core_location.append([H, W])

    # grow the core
    grow = list(range(1, core_grow))
    for i in core_location:
        map_grid[i[0]][i[1]] = 0
        for j in grow:
            growthreshold = (core_grow - j) / core_grow
            num = random.random()
            if num < (growthreshold):
                try:
                    map_grid[i[0] + j][i[1]] = 0
                except:
                    pass
            num = random.random()
            if num < (growthreshold):
                try:
                    map_grid[i[0]][i[1] + j] = 0
                except:
                    pass
            num = random.random()
            if num < (growthreshold):
                try:
                    map_grid[i[0] + j][i[1] + j] = 0
                except:
                    pass
            num = random.random()
            if num < (growthreshold):
                try:
                    map_grid[i[0] - j][i[1]] = 0
                except:
                    pass
            num = random.random()
            if num < (growthreshold):
                try:
                    map_grid[i[0]][i[1] - j] = 0
                except:
                    pass
            num = random.random()
            if num < (growthreshold):
                try:
                    map_grid[i[0] - j][i[1] - j] = 0
                except:
                    pass
            num = random.random()
            if num < (growthreshold):
                try:
                    map_grid[i[0] + j][i[1] - j] = 0
                except:
                    pass
            num = random.random()
            if num < (growthreshold):
                try:
                    map_grid[i[0] - j][i[1] + j] = 0
                except:
                    pass

    # generate food(wounded people)
    location_wounded = map_grid_flat
    for i in map_grid_flat:
        if map_grid[int(i / width)][i % width] == 0:
            location_wounded.remove(i)
    location_wounded = random.sample(location_wounded, N_wounded)
    for i in location_wounded:
        map_grid[int(i / width)][i % width] = 7

    return map_grid


def search(loc_h,loc_w,search_range,map_searched,map_grid_true):
    for i in list(range(-search_range,search_range+1)):
        for j in list(range(-search_range,search_range+1)):
            if map_searched[loc_h+i][loc_w+j]==-1:
                map_searched[loc_h+i][loc_w+j]=map_grid_true[loc_h+i][loc_w+j]
    return map_searched


def make_block_map(map_grid):
    map_grid=map_grid.tolist()
    for q in range(0,len(map_grid)):
        for k in range(0,len(map_grid[0])):
            if map_grid[q][k]==0:
                map_grid[q][k]=1
            else:
                map_grid[q][k]=0
    return map_grid
