import ipdb
import os

import requests
import json_numpy

json_numpy.patch()
import numpy as np

from droid.controllers.oculus_controller import VRPolicy
from droid.evaluation.policy_wrapper import PolicyWrapper
from droid.robot_env import RobotEnv
from droid.user_interface.data_collector import DataCollecter
from droid.user_interface.gui import RobotGUI
from droid.data_processing.timestep_processing import TimestepProcesser

policy_action_space = "cartesian_velocity"
policy_camera_kwargs = {}

dir_path = os.path.dirname(os.path.realpath(__file__))
exp_name = "debug_vla_put_marker_in_cup"
save_data = False
log_dir = os.path.join(dir_path, "../../evaluation_logs", exp_name)

env = RobotEnv(action_space=policy_action_space, camera_kwargs=policy_camera_kwargs)
controller = VRPolicy()


class VLAPolicy:
    def __init__(
        self,
        timestep_filtering_kwargs=dict(),
        image_transform_kwargs=dict(remove_alpha=True, bgr_to_rgb=True, augment=False, to_tensor=False),
        eval_mode=True
    ):
        self.timestep_processor = TimestepProcesser(
            ignore_action=True, gripper_action_space="gripper_position", **timestep_filtering_kwargs,
            image_transform_kwargs=image_transform_kwargs
        )

    def forward(self, obs):
        processed_timestep = self.timestep_processor.forward({"observation": obs})

        obs = {
            "camera/image/hand_camera_left_image":      processed_timestep["observation"]["camera"]["image"][
                                                            "hand_camera"][0],
            "camera/image/hand_camera_right_image":     processed_timestep["observation"]["camera"]["image"][
                                                            "hand_camera"][1],
            "camera/image/varied_camera_1_left_image":  processed_timestep["observation"]["camera"]["image"][
                                                            "varied_camera"][0],
            "camera/image/varied_camera_1_right_image": processed_timestep["observation"]["camera"]["image"][
                                                            "varied_camera"][1],
            "camera/image/varied_camera_2_left_image":  processed_timestep["observation"]["camera"]["image"][
                                                            "varied_camera"][2],
            "camera/image/varied_camera_2_right_image": processed_timestep["observation"]["camera"]["image"][
                                                            "varied_camera"][3],
        }

        image = obs['camera/image/varied_camera_1_left_image']

        action = requests.post(
            "http://0.0.0.0:8000/act",
            json={"image": image, "instruction": "pick up can"}
        ).json()

        cartesian_velocity = action[:6]
        gripper_position = 1 - action[6:7]
        return np.concatenate([cartesian_velocity, gripper_position])


policy = VLAPolicy()

data_collector = DataCollecter(
    env=env,
    controller=controller,
    policy=policy,
    save_traj_dir=log_dir,
    save_data=save_data
)
RobotGUI(robot=data_collector)
