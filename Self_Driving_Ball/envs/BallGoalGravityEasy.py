# By Shengsong Gao on 10/5/2019
# implements OpenAI Gym interface
# the ball drives itself in a 2D space with downward gravity, trying to reach the goal
# this easy version rewards every attempt at getting closer to the goal (per-frame reward for good acceleration)
# also it makes the wall less confusing to the agent by keeping the goal in the center (otherwise it interferes with the agent's plan learned when the wall is in different relative position)
# to increase difficulty, the agent is reward for near-good acceleration as well (acceleration in the same relative quadrant as the desired acceleration)


from Self_Driving_Ball.envs.BallGoalGravity import *
import numpy as np

APPROACHING_REWARD = FRAME_PENALTY
RENDERED_RADIUS = RENDERED_RADIUS * 2 * 0.5  # *2 because the environment is now four times in size; *0.5 for increased difficulty
GOAL_RADIUS = GOAL_RADIUS * 2 * 0.5


class BallGoalGravityEasyEnv(BallGoalGravityEnv):

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        ax, ay = action
        # take the action
        self.x += self.vx * TIME_STEP
        self.y += self.vy * TIME_STEP

        # keep the values in bounds
        self.x = np.clip(self.x, 0, WIDTH * 2)
        self.y = np.clip(self.y, 0, HEIGHT * 2)
        self.vx += ax * TIME_STEP
        self.vy += (ay + GRAVITATIONAL_ACCELERATION) * TIME_STEP
        self.vx = np.clip(self.vx, MIN_VELOCITY, MAX_VELOCITY)
        self.vy = np.clip(self.vy, MIN_VELOCITY, MAX_VELOCITY)

        done = False

        # check if the goal state is reached (ball stops in goal)
        x_diff = WIDTH - self.x
        y_diff = HEIGHT - self.y
        if (x_diff ** 2 + y_diff ** 2) <= GOAL_RADIUS ** 2:
            reward = GOAL_REWARD
            done = True
        # reward is positive if the ball is trying to reach the goal
        elif np.sign(x_diff) == np.sign(ax) and np.sign(y_diff) == np.sign(ay):
            reward = APPROACHING_REWARD
        else:
            reward = -FRAME_PENALTY

        self.frame_count += 1
        if self.frame_count >= MAX_FRAMES:
            done = True

        # state, reward, done, info
        return np.asarray((x_diff, y_diff, self.vx, self.vy)), reward, done, {}

    def reset(self):
        self.x = np.random.uniform(0, WIDTH * 2)
        self.y = np.random.uniform(0, HEIGHT * 2)
        self.vx = 0
        self.vy = 0
        self.frame_count = 0

        return np.asarray((WIDTH - self.x, HEIGHT - self.y, self.vx, self.vy))

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_WIDTH * 2, WINDOW_HEIGHT * 2)
            ball = rendering.make_circle(radius=RENDERED_RADIUS)
            goal = rendering.make_circle(radius=RENDERED_RADIUS)
            ball.set_color(0, 0, 1)  # blue
            goal.set_color(0, 1, 0)  # green
            self.ball_trans = rendering.Transform()
            ball.add_attr(self.ball_trans)
            goal.add_attr(rendering.Transform(translation=(WINDOW_WIDTH, WINDOW_HEIGHT)))
            self.viewer.add_geom(ball)
            self.viewer.add_geom(goal)

        self.ball_trans.set_translation(self.x / METER_PER_PIXEL, self.y / METER_PER_PIXEL)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
