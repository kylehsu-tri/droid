import ipdb
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import requests
import json_numpy
json_numpy.patch()
import numpy as np

dataset_name = 'droid_pick_up_can_target'
instruction = 'pick up can'
ds = tfds.load(dataset_name, split='train')
all_l1 = []
for i, episode in enumerate(ds.take(5)):
    images_combined = []
    actions = []
    for step in episode['steps']:
        ipdb.set_trace()
        images_combined.append(step['observation']['combined_exterior_wrist_image_left'].numpy())
        action = tf.concat(
            [
                step['action_dict']['cartesian_velocity'][:3],
                step['action_dict']['cartesian_velocity'][3:6],
                1 - step['action_dict']['gripper_position']
            ],
            axis=-1
        )
        actions.append(action.numpy())
    actions = np.stack(actions, axis=0)

    predicted_actions = []
    for image in images_combined:
        action = requests.post(
            "http://0.0.0.0:8000/act",
            json={"image": image, "instruction": "pick up can"}
        ).json()
        predicted_actions.append(action)
    predicted_actions = np.stack(predicted_actions, axis=0)

    l1 = np.abs(actions - predicted_actions).sum(axis=-1)
    all_l1.append(l1)
all_l1 = np.concatenate(all_l1, axis=0)
ipdb.set_trace()