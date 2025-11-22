
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math

pygame.font.init()
font = pygame.font.Font(None, 24)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple("Point", "x y")

WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 155, 0)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20

class SnakeGame:
    """A self-contained Snake environment that draws onto its own surface.

    It does NOT manage the main display or event loop.
    The trainer is responsible for:
      - handling pygame events
      - creating the main window
      - blitting each env.surface into a grid
    """
    def __init__(self, w=320, h=240):
        self.w = w
        self.h = h
        self.surface = pygame.Surface((self.w, self.h))
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y),
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _distance_to_food(self):
        return math.dist((self.head.x, self.head.y), (self.food.x, self.food.y))

    def get_state(self):
        head = self.snake[0]

        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        # distances to walls (normalized)
        dist_left_wall = head.x / self.w
        dist_right_wall = (self.w - head.x) / self.w
        dist_top_wall = head.y / self.h
        dist_bottom_wall = (self.h - head.y) / self.h

        # normalized food delta
        dist_x = (self.food.x - head.x) / self.w
        dist_y = (self.food.y - head.y) / self.h

        state = [
            int(self.is_collision(point_r)),
            int(self.is_collision(point_l)),
            int(self.is_collision(point_u)),
            int(self.is_collision(point_d)),
            int(dir_l),
            int(dir_r),
            int(dir_u),
            int(dir_d),
            dist_x,
            dist_y,
            dist_left_wall,
            dist_right_wall,
            dist_top_wall,
            dist_bottom_wall,
            len(self.snake) / 100.0,
        ]

        return np.array(state, dtype=float)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x < 0 or pt.x > self.w - BLOCK_SIZE:
            return True
        if pt.y < 0 or pt.y > self.h - BLOCK_SIZE:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def step(self, action_onehot):
        """Takes one step in the environment.

        action_onehot: np.array length 3 -> [straight, right, left]
        Returns: reward, done, score
        """
        self.frame_iteration += 1
        old_distance = self._distance_to_food()

        # Move
        self._move(action_onehot)
        self.snake.insert(0, self.head)

        reward = 0.0

        # collision or too long without progress
        if self.is_collision() or self.frame_iteration > 150 * len(self.snake):
            reward = -20.0
            done = True
            return reward, done, self.score

        if self.head == self.food:
            self.score += 1
            reward = 30.0
            self._place_food()
            self.frame_iteration = 0
        else:
            self.snake.pop()
            new_distance = self._distance_to_food()
            if new_distance < old_distance:
                reward += 1.0
            else:
                reward -= 0.8

        done = False
        return reward, done, self.score

    def _move(self, action_onehot):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        move_idx = int(np.argmax(action_onehot))
        if move_idx == 0:  # straight
            new_dir = clock_wise[idx]
        elif move_idx == 1:  # right
            new_dir = clock_wise[(idx + 1) % 4]
        else:  # left
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE

        self.head = Point(x, y)

    def render(self):
        """Draws the snake and food onto self.surface."""
        self.surface.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.surface, GREEN1, (pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.surface, GREEN2, (pt.x + 4, pt.y + 4, BLOCK_SIZE - 8, BLOCK_SIZE - 8))
        pygame.draw.rect(self.surface, RED, (self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Optional: tiny score in corner
        text = font.render(str(self.score), True, WHITE)
        self.surface.blit(text, (4, 4))

    def get_surface(self):
        return self.surface
