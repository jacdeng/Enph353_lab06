from gym_gazebo.envs.gazebo_lab06.gazebo_env_lab06 import Gazebo_Lab06_Env

register(
    id='Gazebo_Lab06-v0',
    entry_point='gym_gazebo.envs.gazebo_lab06:Gazebo_Lab06_Env',
    max_episode_steps=3000,
)
