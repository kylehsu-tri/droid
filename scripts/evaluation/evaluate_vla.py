import ipdb
import os
import argparse
import pathlib

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


class VLAPolicy:
    def __init__(
        self,
        action_chunk,
        timestep_filtering_kwargs=dict(),
        image_transform_kwargs=dict(remove_alpha=True, bgr_to_rgb=True, augment=False, to_tensor=False),
        eval_mode=True
    ):
        self.action_chunk = action_chunk
        self.timestep_processor = TimestepProcesser(
            ignore_action=True, gripper_action_space="gripper_position", **timestep_filtering_kwargs,
            image_transform_kwargs=image_transform_kwargs
        )
        self.action_cache = []

    def forward(self, obs):
        if len(self.action_cache) > 0:
            print(f"Cache size {len(self.action_cache)}")
            action = self.action_cache.pop(0)
            print(f"gripper position: {action[-1]}")
            return action

        proprio = np.array(obs["robot_state"]["cartesian_position"] + [obs["robot_state"]["gripper_position"]])
        processed_timestep = self.timestep_processor.forward({"observation": obs})
        image_dict = processed_timestep["observation"]["camera"]["image"]
        image_dict = {
            "camera/image/hand_camera_left_image": image_dict["hand_camera"][0],
            "camera/image/hand_camera_right_image": image_dict["hand_camera"][1],
            "camera/image/varied_camera_1_left_image": image_dict["varied_camera"][0],
            "camera/image/varied_camera_1_right_image": image_dict["varied_camera"][1],
            "camera/image/varied_camera_2_left_image": image_dict["varied_camera"][2],
            "camera/image/varied_camera_2_right_image": image_dict["varied_camera"][3],
        }

        images = {
            "primary": image_dict["camera/image/varied_camera_1_left_image"],
            "secondary": image_dict["camera/image/varied_camera_2_left_image"],
            "wrist": image_dict["camera/image/hand_camera_left_image"],
        }

        actions = requests.post(
            "http://0.0.0.0:8000/act",
            json={
                "images": images,
                "proprio": proprio,
                "instruction": "put marker in cup",
                "action_chunk": self.action_chunk,
                "unnorm_key": "droid_put_marker_in_cup"
            }
        ).json()
        assert actions.ndim == 2

        cartesian_velocities = actions[:, :6]
        gripper_positions = 1 - actions[:, 6:7]
        actions = np.concatenate([cartesian_velocities, gripper_positions], axis=1)
        actions = np.clip(actions, -1, 1)       # asserted by RobotEnv
        self.action_cache = [a for a in actions]
        action = self.action_cache.pop(0)
        # print(f"gripper position: {action[-1]}")
        return action


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_uid", type=str)
    parser.add_argument("--action_chunk", type=int, default=1)
    parser.add_argument("--policy_action_space", type=str, default="cartesian_velocity")
    parser.add_argument("--gripper_action_space", type=str, default="position")
    parser.add_argument("--random_reset", type=bool, default=False)
    args = parser.parse_args()

    import pprint
    pprint.pprint(args)

    policy_camera_kwargs = {}

    save_data = True
    log_dir = pathlib.Path("/home/ashwinbalakrishna/kylehsu/evals") / args.eval_uid

    env = RobotEnv(
        action_space=args.policy_action_space,
        gripper_action_space=args.gripper_action_space,
        camera_kwargs=policy_camera_kwargs,
        random_reset=args.random_reset
    )
    controller = VRPolicy()

    policy = VLAPolicy(action_chunk=args.action_chunk)

    data_collector = DataCollecter(
        env=env,
        controller=controller,
        policy=policy,
        save_traj_dir=log_dir,
        save_data=save_data
    )
    RobotGUI(robot=data_collector)
