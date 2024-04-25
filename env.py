import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import random
import time
from collections import deque

####################
# Global Variables
####################

# Screen width and height
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Frame name
pygame.display.set_caption('Flappy Bird')

# Screen width and height
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600

# Velocity and physics
SPEED = 20
GRAVITY = 2.5
GAME_SPEED = 20

# Ground
GROUND_WIDTH = SCREEN_WIDTH
GROUND_HEIGHT = 100

# Pipes
PIPE_WIDTH = 80
PIPE_HEIGHT = 500
PIPE_GAP = 150

# Observation Shape
N_CHANNELS = 3
HEIGHT = SCREEN_HEIGHT
WIDTH = SCREEN_WIDTH
pygame.init()

# Background image
BACKGROUND = pygame.image.load('assets/sprites/background-night.png')
BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDTH, SCREEN_HEIGHT))
BEGIN_IMAGE = pygame.image.load('assets/sprites/message.png').convert_alpha()

# Audio for wing and dead
# Leave commented when training
# Really annoying 12 flapping birds
# wing = 'assets/audio/wing.wav'
# hit = 'assets/audio/hit.wav'


class FlappyBird(gym.Env):
    """
    Flappy Bird environment, based on the OpenAI Gym interface.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Initialize flappy bird game, make sure all self values exists
        Here we also define our observation space and action space.
        """
        super(FlappyBird, self).__init__()

        # Define action space, there are only 2 actions: flap and not flap
        self.action_space = spaces.Discrete(2)

        # Set screen, clock, bird and pipe
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # Variables for pipes
        self.pipe_lower_y = None
        self.pipe_top_y = None

        # Deque for storing previous actions
        self.previous_actions = deque(maxlen=10)

        # Set observation space with 5 values: distance to ceiling, distance to floor, vertical distance to lower pipe, horizontal distance to center of the gap, and velocity
        self.observation_space = spaces.Box(low=0, high=255, shape=(5,), dtype=np.int32)

    def _get_observation(self):
        """
        Get the observation of the current state of the game.

        Returns:
            observation (np.array): The observation of the current state of the game. Contains the following: vertical distance to lower pipe, horizontal distance to center of the gap, velocity, distance to floor, distance to ceiling.
        """
        # Get current bird position
        bird_x = self.bird.rect[0]

        # Calculate top of bottom pipe and calculate bottom of top pipe
        if self.pipe_group:
            upper_pipe = self.pipe_group.sprites()[1]
            lower_pipe = self.pipe_group.sprites()[0]
            self.pipe_top_y = upper_pipe.rect[1] + upper_pipe.rect[3]
            self.pipe_lower_y = lower_pipe.rect[1]

        # Get velocity of the bird
        velocity = self.bird.speed

        # Calculate distance to ceiling and floor
        vertical_distance_to_ceiling = self.bird.rect.top
        vertical_distance_to_floor = SCREEN_HEIGHT - GROUND_HEIGHT - self.bird.rect.bottom

        # Calculate vertical distance to lower pipe
        self.vertical_distance_lower_pipe = self.pipe_lower_y - self.bird.rect.bottom

        # Calculate x distance to center gap
        pipe_gap_middle_x = (self.pipe_group.sprites()[0].rect.right + self.pipe_group.sprites()[1].rect.left) / 2

        # Calculate horizontal distance to gap
        horizontal_distance_to_gap = pipe_gap_middle_x - bird_x

        # Combining all observations in one array
        observation = np.array([self.vertical_distance_lower_pipe, horizontal_distance_to_gap,velocity, vertical_distance_to_floor, vertical_distance_to_ceiling])

        return observation

    def step(self, action):
        """
        Take a step in the environment based on the action given.

        Args:
            action (int): The action to take in the environment.

        Returns:
            obs (np.array): The observation of the current state of the game.
            reward (float): The reward for the action taken.
            terminated (bool): Whether the game is terminated.
            truncated (bool): Whether the game is truncated.
            info (dict): Additional information about the game.
        """
        self.render()
        obs = self._get_observation()

        # Get the reward and check if the game is done
        reward, terminated, truncated = self._get_reward_value_and_done_state()

        # Store info (not used in this environment)
        info = {}

        # Store every action the agent has taken, this is used for rendering
        self.previous_actions.append(action)

        # load in ground, pipes, and bird mechanics etc,
        self.bird.begin()
        self.bird.update()
        self.ground_group.update()
        self.pipe_group.update()

        # Jump if action is 1
        if action == 1:
            self.bird.bump()
            # pygame.mixer.music.load(wing)
            # pygame.mixer.music.play()

        # Keep adding pipes on screen
        if self._is_off_screen(self.pipe_group.sprites()[0]):
            self.pipe_group.remove(self.pipe_group.sprites()[0])
            self.pipe_group.remove(self.pipe_group.sprites()[0])
            pipes = get_random_pipes(SCREEN_WIDTH * 2)
            self.pipe_group.add(pipes[0])
            self.pipe_group.add(pipes[1])

        if self._is_off_screen(self.ground_group.sprites()[0]):
            self.ground_group.remove(self.ground_group.sprites()[0])
            new_ground = Ground(self.ground_group.sprites()[-1].rect.right)
            self.ground_group.add(new_ground)
        
        # Return observation, reward, terminated, truncated, and info
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state. This is called when the environment is initialized and when the game is terminated.

        Returns:
            obs (np.array): The observation of the current state of the game.
            info (dict): Additional information about the game.
        """

        # Set variables to initial values
        self.info = {}
        self.previous_actions.clear()
        self.bird = Bird()
        self.pipe_group = pygame.sprite.Group()
        self.ground_group = pygame.sprite.Group()
        self.pipes_passed = 0
        self.previous_actions = deque(maxlen=10)

        # Add ground and pipes to the game
        for i in range(2):
            ground = Ground(i * GROUND_WIDTH)
            self.ground_group.add(ground)

        for i in range(2):
            pipes = get_random_pipes(SCREEN_WIDTH * i + 800)
            self.pipe_group.add(pipes[0])
            self.pipe_group.add(pipes[1])

        # Return observation and info
        return self._get_observation(), self.info

    def render(self, mode='human'):
        """
        Render the game in the environment.

        """

        # Render background
        self.screen.fill((255, 255, 255))
        self.screen.blit(BACKGROUND, (0, 0))

        # Load in begin (welcome) screen if no actions have been taken yet
        if len(self.previous_actions) == 0:
            self.screen.blit(BEGIN_IMAGE, (120, 150))

        # Load in bird, pipes and ground
        self.screen.blit(self.bird.image, self.bird.rect.topleft)
        self.pipe_group.draw(self.screen)
        self.ground_group.draw(self.screen)

        # Set game ticks per second. 100 is fine for playing, 1000 for training
        pygame.display.update()
        self.clock.tick(100)

    def _get_reward_value_and_done_state(self):
        """
        Get the reward value and check if the game is done.

        Returns:
            reward (float): The reward for the action taken.
            terminated (bool): Whether the game is terminated.
            truncated (bool): Whether the game is truncated.
        """
        reward = 0
        terminated = False
        truncated = False

        # Collision with pipes
        if pygame.sprite.spritecollideany(self.bird, self.pipe_group, pygame.sprite.collide_mask):
            reward -= 1000
            terminated = True
            truncated = True

        # Hit the top (out of bounds)
        if self.bird.rect.top <= 0:
            reward -= 1000
            terminated = True
            truncated = True

        # Hit the ground
        if self.bird.rect.bottom >= SCREEN_HEIGHT - GROUND_HEIGHT:
            reward -= 1000
            terminated = True
            truncated = True

        # Penalize if bird goes higher than upper pipe
        if self.bird.rect.top < self.pipe_top_y:
            reward -= 30

        # Penalize if bird goes lower than lower pipe
        if self.bird.rect.bottom > self.pipe_lower_y:
            reward -= 30

        # Reward if bird is in between pipes
        if self.bird.rect.top > self.pipe_top_y and self.bird.rect.bottom < self.pipe_lower_y:
            reward += 3

        # Reward for passing pipes
        for pipe in self.pipe_group:
            if pipe.rect.right < self.bird.rect.left:
                self.pipes_passed += 1
                reward += self.pipes_passed * 10 + 40

        return reward, terminated, truncated

    def _is_off_screen(self, sprite):
        """
        Check if the sprite is off the screen.

        Args:
            sprite (pygame.sprite.Sprite): The sprite to check.

        Returns:
            bool: Whether the sprite is off the screen.
        """
        return sprite.rect[0] < -(sprite.rect[2])


########################################################################################
# Code below is the same as the original flappy bird code. It is not developed further.#
########################################################################################
class Bird(pygame.sprite.Sprite):

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)

        self.images = [pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
                       pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
                       pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha()]

        self.speed = SPEED

        self.current_image = 0
        self.image = pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect[0] = SCREEN_WIDTH / 6
        self.rect[1] = SCREEN_HEIGHT / 2

    def update(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        self.speed += GRAVITY

        # Update bird's position based on speed
        self.rect[1] += self.speed

        # Ensure bird doesn't go below ground level
        if self.rect.bottom >= SCREEN_HEIGHT - GROUND_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT - GROUND_HEIGHT
            self.speed = 0  # Stop the bird from falling further

    def bump(self):
        self.speed = -SPEED

    def begin(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]


class Pipe(pygame.sprite.Sprite):

    def __init__(self, inverted, xpos, ysize):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDTH, PIPE_HEIGHT))

        self.rect = self.image.get_rect()
        self.rect[0] = xpos

        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = - (self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_HEIGHT - ysize

        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        self.rect[0] -= GAME_SPEED


class Ground(pygame.sprite.Sprite):

    def __init__(self, xpos):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDTH, GROUND_HEIGHT))

        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self):
        self.rect[0] -= GAME_SPEED


# Pipe randomization....
# Generating various heights for the pipe
# why is it here all the way in the back?
# don't ask me
def get_random_pipes(xpos):
    size = random.randint(100, 300)
    pipe = Pipe(False, xpos, size)
    pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
    return pipe, pipe_inverted


