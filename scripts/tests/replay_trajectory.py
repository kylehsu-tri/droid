from droid.robot_env import RobotEnv
from droid.trajectory_utils.misc import replay_trajectory

# pick up can
# trajectory_folderpath = "/home/ashwinbalakrishna/kylehsu/code/fix-droid/droid/data/success/2024-07-01/Mon_Jul__1_23:33:27_2024"

# pick up teaspoon (by bowl)
# trajectory_folderpath = "/home/ashwinbalakrishna/kylehsu/code/fix-droid/droid/data/success/2024-07-02/Tue_Jul__2_12:59:19_2024"

# pick up teaspoon (by handle)
trajectory_folderpath = "/home/ashwinbalakrishna/kylehsu/code/fix-droid/droid/data/success/2024-07-02/Tue_Jul__2_12:59:49_2024"

# action_space = "joint_position"
# action_space = "cartesian_position"
# action_space = "joint_velocity"
action_space = "cartesian_velocity"

# Make the robot env
env = RobotEnv(action_space=action_space)

# Replay Trajectory #
h5_filepath = trajectory_folderpath + "/trajectory.h5"
replay_trajectory(env, filepath=h5_filepath)
