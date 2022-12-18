import torch
import random
import numpy as np
from collections import deque
from grid_snake import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from Helper import plot
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from Generate_Map import create_random_map
import math

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.005


class Agent:
    def __init__(self, searched_map):
        self.n_game = 0
        self.epsilon = 0  # Randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(27, 200, 8)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.searched_map = searched_map

        # for n,p in self.model.named_parameters():
        #     print(p.device,'',n)
        # self.model.to('cuda')
        # for n,p in self.model.named_parameters():
        #     print(p.device,'',n)
        # TODO: model,trainer

    # state (11 Values)
    # [ danger straight, danger right, danger left,
    #
    # direction left, direction right,
    # direction up, direction down
    #
    # food left,food right,
    # food up, food down]



    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y - 1)
        point_d = Point(head.x, head.y + 1)
        point_ru = Point(head.x + 1, head.y - 1)
        point_rd = Point(head.x + 1, head.y + 1)
        point_lu = Point(head.x - 1, head.y - 1)
        point_ld = Point(head.x - 1, head.y + 1)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        dir_ru = game.direction == Direction.RU
        dir_rd = game.direction == Direction.RD
        dir_lu = game.direction == Direction.LU
        dir_ld = game.direction == Direction.LD

        food_x, food_y, distance = 0, 0, 999
        closest_food = game.food[0]
        for i in game.food:
            food_x = head.x - i.x
            food_y = head.y - i.y
            if distance > math.sqrt(abs(food_x) ** 2 + abs(food_y) ** 2):
                distance = math.sqrt(abs(food_x) ** 2 + abs(food_y) ** 2)
                closest_food = i
        food_x = head.x - closest_food.x
        food_y = head.y - closest_food.y
        distance = math.sqrt(abs(food_x) ** 2 + abs(food_y) ** 2)

        state = [
            game.is_collision(point_l),
            game.is_collision(point_r),
            game.is_collision(point_u),
            game.is_collision(point_d),
            game.is_collision(point_ru),
            game.is_collision(point_rd),
            game.is_collision(point_lu),
            game.is_collision(point_ld),

            dir_l,
            dir_r,
            dir_u,
            dir_d,
            dir_ru,
            dir_rd,
            dir_lu,
            dir_ld,

            # Food Location
            food_x,
            food_y,
            distance
        ]

        # blocks around the food
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i != 0 or j != 0:
                    try:
                        if game.true_map[game.food.x + i][game.food.y + j] == 0:
                            state.append(1)
                        else:
                            state.append(0)
                    except:
                        state.append(0)

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if memory exceed

    def train_long_memory(self):
        if (len(self.memory) > BATCH_SIZE):
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_game
        final_move = [0, 0, 0, 0, 0, 0, 0, 0]
        if (random.randint(0, 200) < self.epsilon):
            move = random.randint(0, 7)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).cuda()
            prediction = self.model(state0).cuda()  # prediction by model
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train(maxiter,true_map):
    plot_scores = []
    plot_mean_scores = []
    plot_step_length = []
    total_score = 0
    record = 0
    agent = Agent(true_map)
    game = SnakeGameAI(true_map)
    iter = 0
    while iter <= maxiter:
        # Get Old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory,plot result
            t = game.frame_iteration
            game.reset()
            agent.n_game += 1
            agent.train_long_memory()
            if (score > reward):  # new High score
                reward = score
                agent.model.save()
            print('Game:', agent.n_game, 'Score:', score, 'Record:', record, 'step_length:', t)

            plot_scores.append(score)
            plot_step_length.append(t)
            total_score += score
            mean_score = total_score / agent.n_game
            plot_mean_scores.append(mean_score)
            if iter % 50 == 0:
                # plot(plot_scores, plot_mean_scores, plot_step_length)
                plot(plot_scores, plot_mean_scores)
            iter += 1

if (__name__ == "__main__"):
    true_map = create_random_map(100, 100, 0.01, 2, 30, 620)
    train(10000,true_map)