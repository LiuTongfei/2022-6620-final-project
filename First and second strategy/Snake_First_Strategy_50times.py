import pygame
import time
import numpy as np
import Generate_Map
from Generate_Map import create_random_map
import JPS
from queue import PriorityQueue
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Game Setting
WIDTH = 900  # total width of screen
ROWS = 100  # rows = columns
SUPPLIES = np.inf
BLOCK_SIZE = WIDTH // ROWS
# pygame.init()
# WIN = pygame.display.set_mode((WIDTH, WIDTH))
# pygame.display.set_caption("Snake with Supplies")
# fps = pygame.time.Clock()
WIN = None
fps = None

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)
BROWN = (160, 82, 45)

# Initializing references and maps.
# maps is an index array indicating barriers and victims, grid is a node array for visualize maps in pygame
map_refer = {'null': 10, 'barrier': 0, 'victim': 7, 'unexplored': np.inf}
# maps = create_random_map(100, 100, 0.03, 2, 30)
# explored = np.full((ROWS, ROWS), np.inf)
SEARCH_RANGE = [10, 10]


def draw_line_dashed(surface, color, start_pos, end_pos, width = 1, dash_length = 10, exclude_corners = True):
    # convert tuples to numpy arrays
    start_pos = np.array(start_pos)
    end_pos = np.array(end_pos)

    # get euclidian distance between start_pos and end_pos
    length = np.linalg.norm(end_pos - start_pos)

    # get amount of pieces that line will be split up in (half of it are amount of dashes)
    dash_amount = int(length / dash_length)

    # x-y-value-pairs of where dashes start (and on next, will end)
    dash_knots = np.array([np.linspace(start_pos[i], end_pos[i], dash_amount) for i in range(2)]).transpose()

    return [pygame.draw.line(surface, color, tuple(dash_knots[n]), tuple(dash_knots[n+1]), width)
            for n in range(int(exclude_corners), dash_amount - int(exclude_corners), 2)]

# Create node object in grid
class Node(object):
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.y = row * width  # ***
        self.x = col * width  # ***
        self.color = WHITE  # default color
        self.neighbors = {}  # a dict: position: dist
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col  # position in map

    def is_barrier(self):
        return self.color == BROWN

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == RED

    def is_unexplored(self):
        return self.color == BLACK  # ***

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = ORANGE

    def make_barrier(self):
        self.color = BROWN

    def make_end(self):  # make victim
        self.color = RED

    def make_unexplored(self):
        self.color = BLACK  # ***

    def draw(self, win):
        # when drawing, first position in a screen is x, then y.
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = {}
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():
            # DOWN
            self.neighbors['DOWN'] = grid[self.row + 1][self.col]

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            # UP
            self.neighbors['UP'] = grid[self.row - 1][self.col]

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():
            # RIGHT
            self.neighbors['RIGHT'] = grid[self.row][self.col + 1]

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            # LEFT
            self.neighbors['LEFT'] = grid[self.row][self.col - 1]

        if self.row < self.total_rows - 1 and self.col > 0 and not grid[self.row + 1][self.col - 1].is_barrier():
            # DOWN_LEFT
            self.neighbors['DOWN_LEFT'] = grid[self.row + 1][self.col - 1]

        if self.row < self.total_rows - 1 and self.col < self.total_rows - 1 and not grid[self.row + 1][
            self.col + 1].is_barrier():
            # DOWN_RIGHT
            self.neighbors['DOWN_RIGHT'] = grid[self.row + 1][self.col + 1]

        if self.row > 0 and self.col > 0 and not grid[self.row - 1][self.col - 1].is_barrier():
            # UP_LEFT
            self.neighbors['UP_LEFT'] = grid[self.row - 1][self.col - 1]

        if self.row > 0 and self.col < self.total_rows - 1 and not grid[self.row - 1][self.col + 1].is_barrier():
            # UP_RIGHT
            self.neighbors['UP_RIGHT'] = grid[self.row - 1][self.col + 1]

    def surround_by_barriers(self):
        return self.neighbors == {}

    def __lt__(self, other):
        return False


class SnakeGame(object):
    def __init__(self, win, fps, total_width, total_rows, block_size, supplies, search_range):
        self.win = win
        self.fps = fps
        self.total_width = total_width
        self.total_rows = total_rows
        self.block_size = block_size  # width of each rectangle in screen
        self.supplies = supplies  # the mission for a snake delivery
        self.remaining = self.supplies
        self.saved = 0
        self.search_range = search_range
        self.step = 0

    def make_nodes_in_grid(self, maps, map_reference):
        """prepare a grid of node_objects for visualizing in pygame"""
        barrier_idx = map_reference['barrier']
        victim_idx = map_reference['victim']
        unexplored_idx = map_reference['unexplored']
        grid = []
        # gap = total_width // rows  # width of each rect
        for i in range(self.total_rows):
            grid.append([])
            for j in range(self.total_rows):
                spot = Node(i, j, self.block_size, self.total_rows)
                if maps[i][j] == barrier_idx:
                    spot.make_barrier()
                elif maps[i][j] == victim_idx:
                    spot.make_end()
                elif maps[i][j] == unexplored_idx:  # including unexplored parts
                    spot.make_unexplored()
                grid[i].append(spot)
        return grid

    def draw_grid(self):
        """draw vertical and horizontal lines to make maps look like a grid"""
        # for each y
        for i in range(self.total_rows):
            pygame.draw.line(self.win, GREY, (0, i * self.block_size), (self.total_width, i * self.block_size))
            # for each x
            for j in range(self.total_rows):
                pygame.draw.line(self.win, GREY, (j * self.block_size, 0), (j * self.block_size, self.total_width))

    def draw_patrol_route(self):
        color = BLACK
        dash_length = 10
        draw_line_dashed(self.win, color, (10.5 * self.block_size, 89.5 * self.block_size),
                         (10.5 * self.block_size, 10.5 * self.block_size), width=3, dash_length=dash_length)
        draw_line_dashed(self.win, color, (10.5 * self.block_size, 10.5 * self.block_size),
                         (31.5 * self.block_size, 10.5 * self.block_size), width=3, dash_length=dash_length)
        draw_line_dashed(self.win, color, (31.5 * self.block_size, 10.5 * self.block_size),
                         (31.5 * self.block_size, 89.5 * self.block_size), width=3, dash_length=dash_length)
        draw_line_dashed(self.win, color, (31.5 * self.block_size, 89.5 * self.block_size),
                         (52.5 * self.block_size, 89.5 * self.block_size), width=3, dash_length=dash_length)
        draw_line_dashed(self.win, color, (52.5 * self.block_size, 89.5 * self.block_size),
                         (52.5 * self.block_size, 10.5 * self.block_size), width=3, dash_length=dash_length)
        draw_line_dashed(self.win, color, (52.5 * self.block_size, 10.5 * self.block_size),
                         (73.5 * self.block_size, 10.5 * self.block_size), width=3, dash_length=dash_length)
        draw_line_dashed(self.win, color, (73.5 * self.block_size, 10.5 * self.block_size),
                         (73.5 * self.block_size, 89.5 * self.block_size), width=3, dash_length=dash_length)
        draw_line_dashed(self.win, color, (73.5 * self.block_size, 89.5 * self.block_size),
                         (92.5 * self.block_size, 89.5 * self.block_size), width=3, dash_length=dash_length)
        draw_line_dashed(self.win, color, (92.5 * self.block_size, 89.5 * self.block_size),
                         (92.5 * self.block_size, 10.5 * self.block_size), width=3, dash_length=dash_length)

    def draw_one_frame(self, grid):
        # draw screen
        self.win.fill(WHITE)
        # draw explored parts based on grid on screen
        for row in grid:
            for node in row:
                if not node.is_unexplored():
                    node.draw(self.win)
        # draw patrol route
        self.draw_patrol_route()
        # draw grid
        self.draw_grid()
        # draw unexplored part on top of explored nodes and grids
        for row in grid:
            for node in row:
                if node.is_unexplored():
                    node.draw(self.win)
        # pygame.display.update()

    def update_explored(self, pos, search_range, rows, explored_map, true_map):
        # pos is the snake position in maps (row, col)
        start_row = pos[0] - search_range[0]
        end_row = pos[0] + search_range[0]
        start_col = pos[1] - search_range[1]
        end_col = pos[1] + search_range[1]
        if start_row < 0:
            start_row = 0
        if end_row >= rows:
            end_row = rows - 1
        if start_col < 0:
            start_col = 0
        if end_col >= rows:
            end_col = rows - 1
        # update 'explored'
        explored_map[start_row:end_row + 1, start_col:end_col + 1] = true_map[start_row:end_row + 1, start_col:end_col + 1]

    def show_supply(self):
        score_font = pygame.font.SysFont('times new roman', 20)
        score_surface = score_font.render('Remaining Supplies: ' + str(self.remaining), True, RED)
        score_rect = score_surface.get_rect()
        self.win.blit(score_surface, score_rect)

    def show_step(self):
        score_font = pygame.font.SysFont('times new roman', 20)
        score_surface = score_font.render('Steps: ' + str(self.step), True, RED)
        score_rect = score_surface.get_rect()
        score_rect.midtop = (self.total_width // 1.05, 0)
        self.win.blit(score_surface, score_rect)

    def game_over(self):
        # my_font = pygame.font.SysFont('times new roman', 30)
        # game_over_surface = my_font.render('Saved victims: ' + str(self.saved), True, RED)
        # game_over_rect = game_over_surface.get_rect()
        # game_over_rect.midtop = (self.total_width / 2, self.total_width / 4)
        # self.win.blit(game_over_surface, game_over_rect)
        print("total step is", self.step)
        # pygame.display.flip()
        # time.sleep(1)
        pygame.quit()
        # quit()
        return self.step

    def get_patrol_route(self, true_map, search_range):
        """
        Snake position is defaulted to start from the left down corner of the map
        """
        col_range = 2 * search_range[1] + 1
        path = []
        vertical_routes = true_map.shape[1] // col_range
        if vertical_routes == 0:  # when col_range is greater than the matrix dimension
            vertical_routes = 1
        turn = 0
        for route in range(vertical_routes):
            col = col_range * route + search_range[1]
            if route % 2 == 0:
                # from bottom to top in one vertical route
                for row in range(true_map.shape[0] - 1 - search_range[0], search_range[0] - 1, -1):
                    path.append([row, col])
                turn += 1
            else:
                # from top to bottom
                for row in range(search_range[0], true_map.shape[0] - search_range[0]):
                    path.append([row, col])
                turn += 1
            # create horizontal route
            if route < (vertical_routes - 1):
                last_row = path[-1][0]
                last_col = path[-1][1]
                next_col = col_range * (route + 1) + search_range[1]
                for hor in range(last_col + 1, next_col):
                    path.append([last_row, hor])
        remains = true_map[:, vertical_routes * col_range:]
        if remains.size != 0:
            col = vertical_routes * col_range + (true_map.shape[0] - vertical_routes * col_range) // 2
            # create horizontal route first!
            last_row = path[-1][0]
            last_col = path[-1][1]
            for hor in range(last_col + 1, col):
                path.append([last_row, hor])

            if turn % 2 == 0:
                # from bottom to top
                for row in range(true_map.shape[0] - 1 - search_range[0], search_range[0] - 1, -1):
                    path.append([row, col])
            else:
                # from top to bottom
                for row in range(search_range[0], true_map.shape[0] - search_range[0]):
                    path.append([row, col])
        return path  # a list of nodes_coordinates that need to be traverse for a snake

    def h(self, p1, p2):
        """a heuristic function for A star algorithm"""
        x1, y1 = p1
        x2, y2 = p2
        return abs(x1 - x2) + abs(y1 - y2)

    def reconstruct_path(self, came_from, start, end):
        path = [end]
        curr = end
        while curr != start:
            prev = came_from[curr]
            path.append(prev)
            curr = prev
        return list(reversed(path))

    def A_star(self, grid, start, end):
        count = 0
        open_set = PriorityQueue()
        open_set.put((0, count, start))
        came_from = {}
        g_score = {spot: float("inf") for row in grid for spot in row}
        g_score[start] = 0
        f_score = {spot: float("inf") for row in grid for spot in row}
        f_score[start] = 0 + self.h(start.get_pos(), end.get_pos())
        open_set_hash = {start}
        fours = {'UP', 'DOWN', 'LEFT', 'RIGHT'}

        while not open_set.empty():
            current = open_set.get()[2]
            open_set_hash.remove(current)

            if current == end:
                shortest_path = self.reconstruct_path(came_from, start, end)
                return True, shortest_path

            for pos, neighbor in current.neighbors.items():
                if pos in fours:
                    temp_g_score = g_score[current] + 1
                else:
                    temp_g_score = g_score[current] + 1.4

                if temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + self.h(neighbor.get_pos(), end.get_pos())
                    if neighbor not in open_set_hash:
                        count += 1
                        open_set.put((f_score[neighbor], count, neighbor))
                        open_set_hash.add(neighbor)

        return False, None

    def explore(self, snake_map_position, explore_position, explored_grid, explored_map):
        start = explored_grid[snake_map_position[0]][snake_map_position[1]]
        end = explored_grid[explore_position[0]][explore_position[1]]
        flag = False
        move_one_block = None
        # *** if all neighbors around end point are not barriers
        if not end.surround_by_barriers():
            flag, path = self.A_star(explored_grid, start, end)
            if flag:
                move_one_block = path[1]
        return flag, move_one_block

    def move_one_step(self, snake_position, end, explored_grid):
        """For A Star"""
        start_row = snake_position[1] // self.block_size
        start_col = snake_position[0] // self.block_size
        move_position = None
        for pos, neighbor in explored_grid[start_row][start_col].neighbors.items():
            if neighbor == end:
                move_position = pos
                break
        # Moving the snake
        if move_position == 'UP':
            snake_position[1] -= self.block_size
        elif move_position == 'DOWN':
            snake_position[1] += self.block_size
        elif move_position == 'LEFT':
            snake_position[0] -= self.block_size
        elif move_position == 'RIGHT':
            snake_position[0] += self.block_size
        elif move_position == 'UP_LEFT':
            snake_position[1] -= self.block_size
            snake_position[0] -= self.block_size
        elif move_position == 'UP_RIGHT':
            snake_position[1] -= self.block_size
            snake_position[0] += self.block_size
        elif move_position == 'DOWN_LEFT':
            snake_position[1] += self.block_size
            snake_position[0] -= self.block_size
        elif move_position == 'DOWN_RIGHT':
            snake_position[1] += self.block_size
            snake_position[0] += self.block_size
        return snake_position

    def get_vic_in_view(self, snake_map_pos, search_range, explored_grid):
        pos = snake_map_pos
        start_row = pos[0] - search_range[0]
        end_row = pos[0] + search_range[0]
        start_col = pos[1] - search_range[1]
        end_col = pos[1] + search_range[1]
        if start_row < 0:
            start_row = 0
        if end_row >= self.total_rows:
            end_row = self.total_rows - 1
        if start_col < 0:
            start_col = 0
        if end_col >= self.total_rows:
            end_col = self.total_rows - 1
        victims = []
        for i in range(start_row, end_row + 1):
            for j in range(start_col, end_col + 1):
                if explored_grid[i][j].is_end():
                    victims.append([i, j])
        return victims  # a list of victims indices in grid

    def get_unexplored_part(self, search_range, explore_position, explored_grid):
        pos = explore_position
        start_row = pos[0] - search_range[0]
        end_row = pos[0] + search_range[0]
        start_col = pos[1] - search_range[1]
        end_col = pos[1] + search_range[1]
        if start_row < 0:
            start_row = 0
        if end_row >= self.total_rows:
            end_row = self.total_rows - 1
        if start_col < 0:
            start_col = 0
        if end_col >= self.total_rows:
            end_col = self.total_rows - 1
        unexplored = []
        for i in range(start_row, end_row + 1):
            for j in range(start_col, end_col + 1):
                if explored_grid[i][j].is_unexplored():
                    unexplored.append([i, j])
        return unexplored

    def find_nearest_patrol_site(self, path, search_range, explored_grid, explore_position):
        # corner case 1: near end of patrol route, how to return while saving steps
        if len(path) >= search_range[0]:
            idx_max = search_range[0] - 1
        else:
            idx_max = len(path) - 1
        next_pos = path[idx_max]  # initialize a next position to go
        curr_idx = idx_max

        # corner case 2: when turn direction, directly go to the turning point
        change_row = abs(explore_position[0] - next_pos[0])
        change_col = abs(explore_position[1] - next_pos[1])
        pos_idx = 0
        if change_row > 0 and change_col > 0:
            # there is a turning point, then we behave conservatively
            while (pos_idx + 1) <= idx_max:
                curr = path[pos_idx]
                if pos_idx == 0:
                    prev = explore_position
                else:
                    prev = path[pos_idx - 1]
                next = path[pos_idx + 1]
                change_1_row = abs(curr[0] - prev[0])
                change_1_col = abs(curr[1] - prev[1])
                change_2_row = abs(next[0] - curr[0])
                change_2_col = abs(next[1] - curr[1])
                if (change_1_row > 0 and change_1_col == 0 and change_2_row == 0 and change_2_col > 0) or (
                        change_1_row == 0 and change_1_col > 0 and change_2_row > 0 and change_2_col == 0
                ):
                    next_pos = curr
                    curr_idx = pos_idx
                    break
                else:
                    pos_idx += 1

        flag = True
        while flag:
            if (not explored_grid[path[curr_idx][0]][path[curr_idx][1]].is_barrier()) or (curr_idx == 0):
                flag = False
            else:
                curr_idx -= 1
                next_pos = path[curr_idx]

        return next_pos

    def play_game(self, map_reference, explored_map, true_map):
        # Initializing a snake
        snake_position = [0 * self.block_size, (true_map.shape[0] - 1) * self.block_size]  # (x, y) in the screen
        snake_body = [snake_position]
        search_range = self.search_range
        # Initializing exploring and saving strategy
        path = self.get_patrol_route(true_map, search_range)  # path for exploring map
        explore_position = path.pop(0)  # first exploring target site
        been_there = False  # whether we have reached the current explore position
        victims = []
        victim = None
        unexplored_parts = []
        unexplored_part = None
        flag = True
        while flag:
            # update explored and draw it
            self.update_explored([snake_position[1] // self.block_size, snake_position[0] // self.block_size],
                                 search_range, self.total_rows, explored_map, true_map)
            explored_grid = self.make_nodes_in_grid(explored_map, map_reference)
            # self.draw_one_frame(explored_grid)
            # update neighbors for all nodes in grid
            for row in explored_grid:
                for node in row:
                    if not node.is_barrier():  # ***
                        node.update_neighbors(explored_grid)

            # get current location of snake in map coordinate
            snake_col = snake_position[0] // self.block_size  # x is col in a grid
            snake_row = snake_position[1] // self.block_size  # y is row in a grid
            snake_map_pos = [snake_row, snake_col]

            # Initialize displaying
            # for pos in snake_body:
            #     pygame.draw.rect(self.win, GREEN, pygame.Rect(pos[0], pos[1], self.block_size, self.block_size))
            # self.show_supply()
            # pygame.display.update()
            # # Frame Per Second / Refresh Rate
            # snake_speed = 15
            # self.fps.tick(snake_speed)

            # exploring and saving strategy one
            if (not victims) and (not victim) and (
                    not explored_grid[explore_position[0]][explore_position[1]].is_barrier()) and (not been_there):
                if snake_map_pos != explore_position:
                    f, move_one_block = self.explore(snake_map_pos, explore_position, explored_grid, explored_map)
                    if f:
                        snake_position = self.move_one_step(snake_position, move_one_block, explored_grid)
                    else:
                        been_there = True
                        # *** cannot reach a patrol position
                        if not path:
                            flag = False
                        else:
                            explore_position = path.pop(0)
                else:
                    # reach an explore_position and then start to save victims nearby
                    been_there = True
                    victims = self.get_vic_in_view(snake_map_pos, search_range, explored_grid)
            elif victim or victims:
                if not victim:
                    victim = victims.pop(0)
                if snake_map_pos == victim:
                    victim = None
                    # check if there are other victims nearby and append into current victims list
                    victims_nearby = self.get_vic_in_view(snake_map_pos, [2, 2], explored_grid)
                    for victim_nearby in victims_nearby:
                        if victim_nearby not in victims:
                            victims.append(victim_nearby)
                    # if not victims:
                    if not victims:
                        if path:
                            patrol_pos = self.find_nearest_patrol_site(path, search_range, explored_grid, explore_position)
                            explore_position = patrol_pos
                            patrol_pos_idx = path.index(patrol_pos)
                            path = path[patrol_pos_idx + 1:]
                else:
                    f, move_one_block = self.explore(snake_map_pos, victim, explored_grid, explored_map)
                    if f:
                        snake_position = self.move_one_step(snake_position, move_one_block, explored_grid)
                    else:
                        # if we cannot save the current victim
                        victim = None
                        # if not victims:
                        if not victims:
                            if path:
                                patrol_pos = self.find_nearest_patrol_site(path, search_range, explored_grid, explore_position)
                                explore_position = patrol_pos
                                patrol_pos_idx = path.index(patrol_pos)
                                path = path[patrol_pos_idx + 1:]

            elif explored_grid[explore_position[0]][explore_position[1]].is_barrier() and (
                    (not been_there) or unexplored_part or unexplored_parts):
                # when there is a barrier in an explore site, check whether there are unexplored parts around the site
                if not been_there:
                    been_there = True
                    unexplored_parts = self.get_unexplored_part(search_range, explore_position, explored_grid)
                else:
                    if not unexplored_part:
                        unexplored_part = unexplored_parts.pop(0)
                    if (snake_map_pos == unexplored_part) or (
                            not explored_grid[unexplored_part[0]][unexplored_part[1]].is_unexplored()):
                        unexplored_part = None
                        # when all explored, check victims around
                        if not unexplored_parts:
                            victims = self.get_vic_in_view(explore_position, search_range, explored_grid)
                    else:
                        f, move_one_block = self.explore(snake_map_pos, unexplored_part, explored_grid, explored_map)
                        if f:
                            snake_position = self.move_one_step(snake_position, move_one_block, explored_grid)
                        else:
                            # there is no way we can reach the current unexplored part
                            unexplored_part = None
                            # when all explored, check victims around
                            if not unexplored_parts:
                                victims = self.get_vic_in_view(explore_position, search_range, explored_grid)
            else:
                pop = True
                while pop:
                    if self.get_unexplored_part(search_range, explore_position, explored_grid):
                        pop = False
                    else:
                        if not path:
                            flag = False
                            pop = False
                        else:
                            explore_position = path.pop(0)
                    been_there = False

            # update snake position
            snake_col = snake_position[0] // self.block_size
            snake_row = snake_position[1] // self.block_size
            if snake_map_pos != [snake_row, snake_col]:
                self.step += 1

            # Game Over conditions
            if self.remaining == 0:
                self.game_over()
                flag = False
            elif snake_position[0] < 0 or snake_position[0] >= WIDTH:
                self.game_over()
                flag = False
            elif snake_position[1] < 0 or snake_position[1] >= WIDTH:
                self.game_over()
                flag = False
            elif explored_grid[snake_row][snake_col].is_barrier():  # ***
                self.game_over()
                flag = False

            # Movement of snake in one frame if not reaching barrier
            snake_body.insert(0, list(snake_position))  # insert list(snake_position) at index 0
            snake_body.pop()
            # for pos in snake_body:
            #     pygame.draw.rect(self.win, GREEN, pygame.Rect(pos[0], pos[1], self.block_size, self.block_size))

            # Reaching a victim to distribute supplies
            # Note: always in snake's view
            if explored_grid[snake_row][snake_col].is_end():
                explored_grid[snake_row][snake_col].reset()
                self.remaining -= 1
                self.saved += 1
                # Note: need to update victims in maps and explored
                true_map[snake_row][snake_col] = map_reference['null']
                explored_map[snake_row][snake_col] = map_reference['null']

            # displaying
            # self.show_supply()
            # self.show_step()
            # pygame.display.update()
        end_time = time.time()
        print('total computation time is {}.'.format(end_time - start_time))
        self.game_over()
        return self.step


if __name__ == "__main__":
    all_steps = []
    # map_range = range(0, 1000)
    # sample_num = 50
    # random.seed(99)
    # map_indices = random.sample(map_range, sample_num)  # all seeds for generating maps
    map_indices = [413, 389, 204, 613, 183, 235, 254, 136, 778, 88, 257, 746, 392, 543, 700, 717, 551, 91, 960, 638,
                   501, 431, 739, 627, 602, 221, 811, 923, 807, 382, 842, 397, 882, 698, 217, 155, 905, 676, 870, 580,
                   346, 84, 969, 733, 541, 693, 793, 89, 480, 323]
    # print(map_indices)

    iteration = 1

    for map_idx in map_indices:
        start_time = time.time()
        print("current map index: ", map_idx)
        maps = create_random_map(100, 100, 0.03, 2, 30, seed=map_idx)
        explored = np.full((ROWS, ROWS), np.inf)

        snake_game = SnakeGame(WIN, fps, WIDTH, ROWS, BLOCK_SIZE, SUPPLIES, SEARCH_RANGE)
        steps = snake_game.play_game(map_refer, explored, maps)
        all_steps.append(steps)
        print('{}th finished'.format(iteration))
        print('-------------')
        iteration += 1

    mean_step = np.mean(all_steps)
    std_step = np.std(all_steps)
    print('mean of steps:{}'.format(mean_step))
    print('std of steps:{}'.format(std_step))

    with open(r"1st_Strategy_Summary.txt", 'w') as f:
        f.write("Mean: ")
        f.write(str(mean_step))
        f.write("\r\n")
        f.write("Std: ")
        f.write(str(std_step))
        f.write("\r\n")
        f.write("Map_Indices: ")
        f.write(str(map_indices))

    sns.set_style('whitegrid')
    ax = sns.kdeplot(all_steps)
    ax.set(xlabel='Steps')
    plt.show()

    plt.hist(all_steps, bins=20)
    plt.gca().set(title='Frequency Histogram for exploring maps', xlabel='Steps', ylabel='Frequency')
    plt.show()

    sns.set_style('whitegrid')
    ax = sns.distplot(all_steps)
    ax.set(xlabel='Steps')
    plt.show()
