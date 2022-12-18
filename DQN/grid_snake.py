import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math

# pygame.init()
# font = pygame.font.SysFont(None, 25)


# Reset
# Reward
# Play(action) -> Direction
# Game_Iteration
# is_collision


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    RU = 5
    RD = 6
    LU = 7
    LD = 8


Point = namedtuple('Point', 'x , y')

BLOCK_SIZE = 5
SPEED = 9999
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

class SnakeGameAI:
    def __init__(self, true_map):
        self.w = len(true_map[0])
        self.h = len(true_map)
        # init display
        self.display = pygame.display.set_mode((self.w * BLOCK_SIZE, self.h * BLOCK_SIZE))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        # init game state
        self.true_map = true_map
        self.reset()


    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(0, 0)
        self.snake = [self.head]
        self.score = 0

        self.food = []
        for i in range(0,self.h):
            for j in range(0,self.w):
                if self.true_map[i][j] ==7:
                    self.food.append(Point(i,j))
        self.block = []
        for i in range(0,self.h):
            for j in range(0,self.w):
                if self.true_map[i][j] == 0:
                    self.block.append(Point(i,j))
        self.frame_iteration = 0

    def remove_food(self, point):
        self.food.remove(point)

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. Collect the user input
        for event in pygame.event.get():
            if (event.type == pygame.QUIT):
                pygame.quit()
                quit()

        # 2. Move
        self._move(action)
        self.snake.insert(0, self.head)
        # 3. Check if game Over
        reward = -1  # eat food: +300 , game over: -500 , else: -5
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * (self.score + 1) or not self.food:
            game_over = True
            reward = -100
            return reward, game_over, self.score

        # 4. Place new Food or just move
        for i in self.food:
            if self.head == i:
                self.score += 1
                reward = 300
                self.remove_food(i)

        self.snake.pop()

        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. Return game Over and Display Score

        return reward, game_over, self.score

    def _update_ui(self):
        self.display.fill(WHITE)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        for i in self.food:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(i.x * BLOCK_SIZE, i.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        for i in self.block:
            pygame.draw.rect(self.display, RED, pygame.Rect(i.x * BLOCK_SIZE, i.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        # text = font.render("Score: " + str(self.score), True, WHITE)
        # self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        x = self.head.x
        y = self.head.y

        if np.array_equal(action, [1, 0, 0, 0, 0, 0, 0, 0]):
            new_dir = Direction.RIGHT
        elif np.array_equal(action, [0, 1, 0, 0, 0, 0, 0, 0]):
            new_dir = Direction.DOWN
        elif np.array_equal(action, [0, 0, 1, 0, 0, 0, 0, 0]):
            new_dir = Direction.LEFT
        elif np.array_equal(action, [0, 0, 0, 1, 0, 0, 0, 0]):
            new_dir = Direction.UP
        elif np.array_equal(action, [0, 0, 0, 0, 1, 0, 0, 0]):
            new_dir = Direction.LD
        elif np.array_equal(action, [0, 0, 0, 0, 0, 1, 0, 0]):
            new_dir = Direction.LU
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 1, 0]):
            new_dir = Direction.RD
        else:
            new_dir = Direction.RU
        self.direction = new_dir


        if (self.direction == Direction.RIGHT):
            x += 1
        elif (self.direction == Direction.LEFT):
            x -= 1
        elif (self.direction == Direction.DOWN):
            y += 1
        elif (self.direction == Direction.UP):
            y -= 1
        elif (self.direction == Direction.LD):
            x -= 1
            y += 1
        elif (self.direction == Direction.LU):
            x -= 1
            y -= 1
        elif (self.direction == Direction.RD):
            x += 1
            y += 1
        else:
            x += 1
            y -= 1

        self.head = Point(x, y)

    def is_collision(self, pt=None):
        if (pt is None):
            pt = self.head
        # hit boundary
        if (pt.x > self.w - 1 or pt.x < 0 or pt.y > self.h - 1 or pt.y < 0):
            return True
        # hit block
        for i in self.block:
            if pt == i:
                return True
        return False
