# By Shengsong Gao on 10/5/2019
# implements OpenAI Gym interface
# the ball drives itself in a 2D space with downward gravity, trying to reach and stop in the goal.
# is the condition too harsh? The vicinity simplification does not lower the complexity since we'd still have to reach an exact speed (instead of being in the speed vicinity)
# maybe the goal vicinity does simplify things because it's harder to achieve than exact 0 velocity?


from Self_Driving_Ball.envs.BallGoalGravity import *
import numpy as np


class BallGoalGravityStopEnv(BallGoalGravityEnv):
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

        # check if the goal state is reached (ball stops in goal)
        x_diff = self.goal_x - self.x
        y_diff = self.goal_y - self.y
        if (x_diff ** 2 + y_diff ** 2) <= GOAL_RADIUS ** 2 and self.vx == self.vy == 0:
            reward = GOAL_REWARD
            done = True
        else:
            reward = -FRAME_PENALTY
            done = False
        self.frame_count += 1
        if self.frame_count >= MAX_FRAMES: done = True

        # state, reward, done, info
        return np.asarray((x_diff, y_diff, self.vx, self.vy)), reward, done, {}
