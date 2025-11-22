
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math

pygame.init()
font = pygame.font.Font(None, 25)

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
SPEED = 40

class SnakeGameAI:
    def __init__(self, w=640, h=480, render=True):
        self.w = w
        self.h = h
        self.render = render

        if render:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("RL Snake")
        else:
            self.display = None

        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)]

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

        # Improved state: includes distance to food
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
            len(self.snake) / 100.0
        ]

        return np.array(state, dtype=float)

    def play_step(self, action):
        self.frame_iteration += 1
        old_distance = self._distance_to_food()

        # Quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # Move
        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            reward = -10
            return reward, True, self.score

        if self.head == self.food:
            self.score += 1
            reward = +10
            self._place_food()
        else:
            self.snake.pop()
            new_distance = self._distance_to_food()
            if new_distance < old_distance:
                reward = +1
            else:
                reward = -0.5

        if self.render:
            self._update_ui()
            self.clock.tick(SPEED)

        return reward, False, self.score

    def _distance_to_food(self):
        return math.dist((self.head.x, self.head.y), (self.food.x, self.food.y))

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

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN1, (pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2, (pt.x+4, pt.y+4, BLOCK_SIZE-8, BLOCK_SIZE-8))

        pygame.draw.rect(self.display, RED, (self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(score_text, (0, 0))
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT: x += BLOCK_SIZE
        elif self.direction == Direction.LEFT: x -= BLOCK_SIZE
        elif self.direction == Direction.UP: y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN: y += BLOCK_SIZE

        self.head = Point(x, y)
