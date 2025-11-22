
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font(None, 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple("Point", "x y")

# RGB colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 155, 0)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40  # game speed (frames per second)

class SnakeGameAI:
    def __init__(self, w=640, h=480, render=True):
        self.w = w
        self.h = h
        self.render = render

        if self.render:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("RL Snake")
        else:
            self.display = None

        self.clock = pygame.time.Clock()
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

        # Danger straight
        if dir_r:
            danger_straight = self.is_collision(point_r)
            danger_right = self.is_collision(point_d)
            danger_left = self.is_collision(point_u)
        elif dir_l:
            danger_straight = self.is_collision(point_l)
            danger_right = self.is_collision(point_u)
            danger_left = self.is_collision(point_d)
        elif dir_u:
            danger_straight = self.is_collision(point_u)
            danger_right = self.is_collision(point_r)
            danger_left = self.is_collision(point_l)
        else:  # dir_d
            danger_straight = self.is_collision(point_d)
            danger_right = self.is_collision(point_l)
            danger_left = self.is_collision(point_r)

        # Food location
        food_left = self.food.x < head.x
        food_right = self.food.x > head.x
        food_up = self.food.y < head.y
        food_down = self.food.y > head.y

        state = [
            # Danger
            int(danger_straight),
            int(danger_right),
            int(danger_left),

            # Direction
            int(dir_l),
            int(dir_r),
            int(dir_u),
            int(dir_d),

            # Food location
            int(food_left),
            int(food_right),
            int(food_up),
            int(food_down),
        ]

        return np.array(state, dtype=int)

    def play_step(self, action):
        self.frame_iteration += 1
        reward = 0

        # 1. Collect user input (just to allow quit)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # 2. Move based on action
        self._move(action)  # update head
        self.snake.insert(0, self.head)

        # 3. Check if game over
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            reward = -10
            game_over = True
            return reward, game_over, self.score

        # 4. Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. Update UI and clock
        if self.render:
            self._update_ui()
            self.clock.tick(SPEED)

        # 6. Return game_over and score
        game_over = False
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x + 4, pt.y + 4, BLOCK_SIZE - 8, BLOCK_SIZE - 8))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # action is [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)


if __name__ == "__main__":
    # Manual test of environment (random actions)
    game = SnakeGameAI(render=True)
    while True:
        state = game.get_state()
        # random action
        action = np.zeros(3, dtype=int)
        action[random.randint(0, 2)] = 1
        reward, done, score = game.play_step(action)
        if done:
            print("Game over! Score:", score)
            game.reset()
