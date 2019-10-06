from gym.envs.registration import register

register(
    id='BallGoalGravity-v0',
    entry_point='Self_Driving_Ball.envs:BallGoalGravityEnv',
)
register(
    id='BallGoalGravityStop-v0',
    entry_point='Self_Driving_Ball.envs:BallGoalGravityStopEnv',
)
register(
    id='BallGoalGravityEasy-v0',
    entry_point='Self_Driving_Ball.envs:BallGoalGravityEasyEnv',
)