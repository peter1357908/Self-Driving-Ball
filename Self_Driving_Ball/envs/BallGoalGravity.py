# By Shengsong Gao on 10/5/2019
# implements OpenAI Gym interface
# the ball drives itself in a 2D space with downward gravity, trying to reach the goal.


import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

# parameters for rendering
WINDOW_WIDTH = 250
WINDOW_HEIGHT = WINDOW_WIDTH
FRAME_RATE = 30
TIME_STEP = 1.0 / FRAME_RATE  # how much time passes between 2 frames
METER_PER_PIXEL = 0.5  # every pixel is this many meters on the canvas
RENDERED_RADIUS = WINDOW_WIDTH * 0.1  # radius of the rendered ball and goal, in pixels

# parameters for physical simulation
GRAVITATIONAL_ACCELERATION = 9.8  # y-direction only; remember that the y-axis is inverted
MAX_ACCELERATION = 100
MIN_ACCELERATION = -MAX_ACCELERATION
MAX_VELOCITY = 0.5 * MAX_ACCELERATION
MIN_VELOCITY = -MAX_VELOCITY
WIDTH = int(WINDOW_WIDTH * METER_PER_PIXEL)  # "actual" width, in meters; converted to int so we can use it to build arrays
HEIGHT = int(WINDOW_HEIGHT * METER_PER_PIXEL)
GOAL_RADIUS = WIDTH * 0.1  # radius of the goal, in meters
MAX_TIME = 15  # how many seconds allowed in each episode

# learning environment setting
MAX_FRAMES = FRAME_RATE * MAX_TIME
FRAME_PENALTY = 1
GOAL_REWARD = MAX_FRAMES


class BallGoalGravityEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],  # rgb_array for video production
        'video.frames_per_second': FRAME_RATE
    }

    def __init__(self):
        self.gravity = GRAVITATIONAL_ACCELERATION
        self.action_space = spaces.Box(
            low=np.array([MIN_ACCELERATION, MIN_ACCELERATION]), high=np.array([MAX_ACCELERATION, MAX_ACCELERATION]), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([-WIDTH, -HEIGHT, MIN_VELOCITY, MIN_VELOCITY]), high=np.array([WIDTH, HEIGHT, MAX_VELOCITY, MAX_VELOCITY]), dtype=np.float32
        )
        self.frame_count = 0

        # for rendering
        self.viewer = None

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        ax, ay = action
        # take the action
        self.x += self.vx * TIME_STEP
        self.y += self.vy * TIME_STEP

        # keep the values in bounds
        self.x = np.clip(self.x, 0, WIDTH)
        self.y = np.clip(self.y, 0, HEIGHT)
        self.vx += ax * TIME_STEP
        self.vy += (ay + GRAVITATIONAL_ACCELERATION) * TIME_STEP
        self.vx = np.clip(self.vx, MIN_VELOCITY, MAX_VELOCITY)
        self.vy = np.clip(self.vy, MIN_VELOCITY, MAX_VELOCITY)

        # check if the goal state is reached (ball touched goal)
        x_diff = self.goal_x - self.x
        y_diff = self.goal_y - self.y
        if (x_diff ** 2 + y_diff ** 2) <= GOAL_RADIUS ** 2:
            reward = GOAL_REWARD
            done = True
        else:
            reward = -FRAME_PENALTY
            done = False

        self.frame_count += 1
        if self.frame_count >= MAX_FRAMES:
            done = True

        # state, reward, done, info
        return np.asarray((x_diff, y_diff, self.vx, self.vy)), reward, done, {}

    def reset(self):
        self.goal_x = np.random.uniform(0, WIDTH)
        self.goal_y = np.random.uniform(0, HEIGHT)
        self.x = np.random.uniform(0, WIDTH)
        self.y = np.random.uniform(0, HEIGHT)
        self.vx = 0
        self.vy = 0
        self.frame_count = 0

        x_diff = self.goal_x - self.x
        y_diff = self.goal_y - self.y

        return np.asarray((x_diff, y_diff, self.vx, self.vy))

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_WIDTH, WINDOW_HEIGHT)
            ball = rendering.make_circle(radius=RENDERED_RADIUS)
            goal = rendering.make_circle(radius=RENDERED_RADIUS)
            ball.set_color(0, 0, 1)  # blue
            goal.set_color(0, 1, 0)  # green
            self.ball_trans = rendering.Transform()
            ball.add_attr(self.ball_trans)
            self.goal_trans = rendering.Transform()
            goal.add_attr(self.goal_trans)
            self.viewer.add_geom(goal)
            self.viewer.add_geom(ball)

        self.ball_trans.set_translation(self.x / METER_PER_PIXEL, self.y / METER_PER_PIXEL)
        self.goal_trans.set_translation(self.goal_x / METER_PER_PIXEL, self.goal_y / METER_PER_PIXEL)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
