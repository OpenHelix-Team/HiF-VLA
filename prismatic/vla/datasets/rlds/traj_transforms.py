"""
traj_transforms.py

Contains trajectory transforms used in the orca data pipeline. Trajectory transforms operate on a dictionary
that represents a single trajectory, meaning each tensor has the same leading dimension (the trajectory length).
"""

import logging
from typing import Dict

import tensorflow as tf
from PIL import Image
from io import BytesIO
import imageio
import re
from mvextractor.videocap import VideoCap
import torch
import os
import subprocess
import numpy as np
import json
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)

def chunk_act_obs(traj: Dict, window_size: int, future_action_window_size: int = 0, history_length: int = 8) -> Dict:
    """
    Chunks actions and observations into the given window_size.给定窗口切片动作和观察值

    "observation" keys are given a new axis (at index 1) of size `window_size` containing `window_size - 1`
    observations from the past and the current observation. "action" is given a new axis (at index 1) of size
    `window_size + future_action_window_size` containing `window_size - 1` actions from the past, the current
    action, and `future_action_window_size` actions from the future. "pad_mask" is added to "observation" and
    indicates whether an observation should be considered padding (i.e. if it had come from a timestep
    before the start of the trajectory).
    """
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]
    effective_traj_len = traj_len - future_action_window_size-1
    chunk_indices = tf.broadcast_to(tf.range(-window_size + 1, 1), [effective_traj_len, window_size]) + tf.broadcast_to(
        tf.range(effective_traj_len)[:, None], [effective_traj_len, window_size]
    )

    action_chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1 + future_action_window_size),
        [effective_traj_len, window_size + future_action_window_size],
    ) + tf.broadcast_to(
        tf.range(effective_traj_len)[:, None],
        [effective_traj_len, window_size + future_action_window_size],
    )

    floored_chunk_indices = tf.maximum(chunk_indices, 0)

    goal_timestep = tf.fill([effective_traj_len], traj_len - 1)#
    floored_action_chunk_indices = tf.minimum(tf.maximum(action_chunk_indices, 0), goal_timestep[:, None])

    traj["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, floored_chunk_indices), traj["observation"])
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)

    # indicates whether an entire observation is padding
    traj["observation"]["pad_mask"] = chunk_indices >= 0

    if "mv_history" in traj:

        f_mv_mat = action_chunk_indices + 1
        f_mv = tf.gather(traj["mv_history"],f_mv_mat)
        traj["mv_future"] = f_mv

        traj_len = tf.cast(traj_len, tf.int32)

        row = tf.broadcast_to(tf.range(-history_length + 1, 1, dtype=tf.int32), [traj_len, history_length])      # 每行: [-K+1, ..., 0]
        col = tf.broadcast_to(tf.range(traj_len, dtype=tf.int32)[:, None], [traj_len, history_length])  # 每列: [0..T-1]^T
        
        mat = tf.maximum(1, col + 1 + row)
        
        n_keep = tf.clip_by_value(traj_len - tf.cast(NUM_ACTIONS_CHUNK+1, tf.int32), 0, traj_len)
        mv_chunk_indices = mat[:n_keep]
        obs_mv = tf.gather(traj["mv_history"], mv_chunk_indices) 
        zero_first = tf.zeros_like(obs_mv[0])
        obs_mv = tf.concat([zero_first[None, ...], obs_mv[:]], axis=0)
        traj["mv_history"] = obs_mv

    # Truncate other elements of the trajectory dict
    traj["task"] = tf.nest.map_structure(lambda x: tf.gather(x, tf.range(effective_traj_len)), traj["task"])
    traj["dataset_name"] = tf.gather(traj["dataset_name"], tf.range(effective_traj_len))
    traj["absolute_action_mask"] = tf.gather(traj["absolute_action_mask"], tf.range(effective_traj_len))
    traj["episode_idx"] = tf.gather(traj["episode_idx"], tf.range(effective_traj_len))
    traj["motion_path"] = tf.gather(traj["motion_path"], tf.range(effective_traj_len))
    traj["video_path"] = tf.gather(traj["video_path"], tf.range(effective_traj_len))

    return traj


def subsample(traj: Dict, subsample_length: int) -> Dict:
    """Subsamples trajectories to the given length."""
    traj_len = tf.shape(traj["action"])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)

    return traj


def add_pad_mask_dict(traj: Dict) -> Dict:
    """
    Adds a dictionary indicating which elements of the observation/task should be treated as padding.
        =>> traj["observation"|"task"]["pad_mask_dict"] = {k: traj["observation"|"task"][k] is not padding}
    """
    traj_len = tf.shape(traj["action"])[0]

    for key in ["observation", "task"]:
        pad_mask_dict = {}
        for subkey in traj[key]:
            # Handles "language_instruction", "image_*", and "depth_*"
            if traj[key][subkey].dtype == tf.string:
                pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0

            # All other keys should not be treated as padding
            else:
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)

        traj[key]["pad_mask_dict"] = pad_mask_dict

    return traj


def extract_motion(traj: Dict) -> Dict:   
    task_description = str(traj["task"]["language_instruction"][0])

    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")
    video_name = f"temp_video_task={processed_task_description}"  

    dataset_name = traj["dataset_name"][0]
    temp_dir = f"./tmp1/{dataset_name}/videos"
    os.makedirs(temp_dir, exist_ok=True)
    video_path = os.path.join(temp_dir, f"{video_name}.mp4")
    video_writer = imageio.get_writer(video_path, fps=10)
    images=[]
    for step in traj["observation"]["image_primary"]:
        img=np.array(Image.open(BytesIO(step.numpy())).convert('RGB'))
        images.append(img)
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {video_path}")
    traj["observation"]["video_path"] = video_path  
    
    g_len=len(traj["task"]["timestep"])
    tmp_video_path = os.path.join(temp_dir, f"{video_name}_6fps.mp4")
    ffmpeg_cmd = f'ffmpeg -threads 8 -loglevel error -i {video_path} -filter:v fps={10} -c:v mpeg4 -g {g_len} -f rawvideo {tmp_video_path}'
    subprocess.run(args=ffmpeg_cmd,shell=True,timeout=120)

    mean = np.array([[0.0, 0.0]], dtype=np.float64)
    std =  np.array([[0.0993703, 0.1130276]], dtype=np.float64)


    cap = VideoCap()
    ret = cap.open(tmp_video_path)
    frames, motions, frame_types, motion_vector = [], [], [], []
    while True:
        ret, frame, motion_vectors, frame_type, timestamp = cap.read()
        if not ret:
            break
        
        motion_vector.append(motion_vectors)
    motion_data_list = []
    for mv in motion_vector:
        motion_data = {
            "motion": mv.tolist()  # (H, W, 2) -> list
        }
        motion_data_list.append(motion_data)

    tmp_motion_dir = f"./tmp1/{dataset_name}/motion"
    os.makedirs(tmp_motion_dir, exist_ok=True)
    motion_path = os.path.join(tmp_motion_dir, f"motion_types_{processed_task_description}.json")
    with open(motion_path, "w") as f:
        json.dump(motion_data_list, f)
    traj["observation"]["motion_path"] = motion_path  

    return traj


def extract_motion_tf(traj: Dict) -> Dict:
    """TensorFlow compatible version of extract_motion. Returns updated traj with path strings only."""
    traj_len = tf.shape(traj["action"])[0]

    episode_id=traj["episode_idx"][0]
    task_description = traj["task"]["language_instruction"][0]
    task_description = tf.strings.lower(task_description)
    task_description = tf.strings.regex_replace(task_description, r"[ \n\.]", "_")


    dataset_name = traj["dataset_name"][0]


    video_path = tf.strings.format("./tmp/{}/videos/temp_video_{}_task={}.mp4",
                                   (dataset_name, episode_id, task_description))
    motion_path = tf.strings.format("./tmp/{}/motion/motion_{}_types_{}.json",
                                    (dataset_name, episode_id, task_description))
    video_path = tf.strings.regex_replace(video_path, '"', '')
    motion_path = tf.strings.regex_replace(motion_path, '"', '')

    traj["motion_path"] = tf.repeat(motion_path, traj_len)
    traj["video_path"] = tf.repeat(video_path, traj_len)

    return traj

def read_mv_from_json(traj: Dict) -> Dict:
    """Reads motion vectors from a JSON file."""
    
    # motion_path = traj["motion_path"][0].numpy().decode("utf-8")
    # motion_path = traj["motion_path"][0]
    def _load(path):
        motion_vectors = []
        path = path.numpy().decode("utf-8")
        with open(path, "r") as f:
            motion_data = json.load(f)
        for motion_dict in motion_data:
            for key, value in motion_dict.items():
                if key.startswith('motion_'):
                    # 将嵌套列表转换为 numpy 数组
                    motion_array = np.array(value)
                    motion_vectors.append(motion_array)
        merged_vector = np.stack(motion_vectors, axis=0)
        
        return merged_vector.astype(np.float32)

    motion_data = tf.py_function(
        func=_load,
        inp=[traj["motion_path"][0]],
        Tout=tf.float32,
        )
    

    traj["mv_history"] = motion_data
    return traj