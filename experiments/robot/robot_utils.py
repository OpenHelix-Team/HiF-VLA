"""Utils for evaluating robot policies in various environments."""

import os
import random
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import subprocess
import av
from prismatic.vla.datasets.rlds.utils.motion_utils import MotionVectorProcessor

from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
)
import cv2

# Initialize important constants
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Configure NumPy print settings
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# Model image size configuration
MODEL_IMAGE_SIZES = {
    "openvla": 224,
    # Add other models as needed
}


def set_seed_everywhere(seed: int) -> None:
    """
    Set random seed for all random number generators for reproducibility.

    Args:
        seed: The random seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_model(cfg: Any, wrap_diffusion_policy_for_droid: bool = False) -> torch.nn.Module:
    """
    Load and initialize model for evaluation based on configuration.

    Args:
        cfg: Configuration object with model parameters
        wrap_diffusion_policy_for_droid: Whether to wrap diffusion policy for DROID

    Returns:
        torch.nn.Module: The loaded model

    Raises:
        ValueError: If model family is not supported
    """
    if cfg.model_family == "openvla":
        model = get_vla(cfg)
    else:
        raise ValueError(f"Unsupported model family: {cfg.model_family}")

    print(f"Loaded model: {type(model)}")
    return model


def get_image_resize_size(cfg: Any) -> Union[int, tuple]:
    """
    Get image resize dimensions for a specific model.

    If returned value is an int, the resized image will be a square.
    If returned value is a tuple, the resized image will be a rectangle.

    Args:
        cfg: Configuration object with model parameters

    Returns:
        Union[int, tuple]: Image resize dimensions

    Raises:
        ValueError: If model family is not supported
    """
    if cfg.model_family not in MODEL_IMAGE_SIZES:
        raise ValueError(f"Unsupported model family: {cfg.model_family}")

    return MODEL_IMAGE_SIZES[cfg.model_family]


def get_action(
    cfg: Any,
    model: torch.nn.Module,#motion_decoder: Optional[torch.nn.Module],#%%%%motion_decoder
    obs: Dict[str, Any],
    task_label: str,
    processor: Optional[Any] = None,
    action_head: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
    noisy_action_projector: Optional[torch.nn.Module] = None,
    motion_token: Optional[Any] = None,
    motion_encoder: Optional[torch.nn.Module] = None,
    his_motion_seq: Optional[Any] = None,
    use_film: bool = False,
) -> Union[List[np.ndarray], np.ndarray]:
    """
    Query the model to get action predictions.

    Args:
        cfg: Configuration object with model parameters
        model: The loaded model
        obs: Observation dictionary
        task_label: Text description of the task
        processor: Model processor for inputs
        action_head: Optional action head for continuous actions
        proprio_projector: Optional proprioception projector
        noisy_action_projector: Optional noisy action projector for diffusion
        use_film: Whether to use FiLM

    Returns:
        Union[List[np.ndarray], np.ndarray]: Predicted actions

    Raises:
        ValueError: If model family is not supported
    """
    with torch.no_grad():
        if cfg.model_family == "openvla":
            action = get_vla_action(
                cfg=cfg,
                vla=model,#motion_decoder=motion_decoder,#%%%%
                processor=processor,
                obs=obs,
                task_label=task_label,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                motion_token=motion_token,
                motion_encoder=motion_encoder,
                his_motion_seq=his_motion_seq,
                use_film=use_film,
            )
        else:
            raise ValueError(f"Unsupported model family: {cfg.model_family}")

    return action


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize gripper action from [0,1] to [-1,+1] range.

    This is necessary for some environments because the dataset wrapper
    standardizes gripper actions to [0,1]. Note that unlike the other action
    dimensions, the gripper action is not normalized to [-1,+1] by default.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

    Args:
        action: Action array with gripper action in the last dimension
        binarize: Whether to binarize gripper action to -1 or +1

    Returns:
        np.ndarray: Action array with normalized gripper action
    """
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    # Normalize the last action dimension to [-1,+1]
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])

    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Flip the sign of the gripper action (last dimension of action vector).

    This is necessary for environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.

    Args:
        action: Action array with gripper action in the last dimension

    Returns:
        np.ndarray: Action array with inverted gripper action
    """
    # Create a copy to avoid modifying the original
    inverted_action = action.copy()

    # Invert the gripper action
    inverted_action[..., -1] *= -1.0

    return inverted_action


def extract_motion_vectors_from_images(img_list, fps=6, num_frames=40, target_size=(256, 256)):

    if img_list[0].shape[0] != target_size[0] or img_list[0].shape[1] != target_size[1]:
        H, W = target_size
        resized_img_list = []
        for img in img_list:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            resized_img_list.append(img)
        img_list = resized_img_list

    h, w, _ = img_list[0].shape

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "mpeg4",   
        "-bf", "0",
        "-g", str(num_frames),
        "-f", "matroska",
        "pipe:1"             
    ]

    # The motion vector mean and std
    mean = np.array([[0.0, 0.0]], dtype=np.float64)
    std =  np.array([[0.0993703, 0.1130276]], dtype=np.float64)

    # start = time.time()
    for i in range(1):
        ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,stderr=subprocess.PIPE)

        for frame in img_list:
            ffmpeg.stdin.write(frame.astype(np.uint8).tobytes())
        ffmpeg.stdin.close()

        container = av.open(ffmpeg.stdout, format='matroska')
        vstream = container.streams.video[0]

        vstream.codec_context.options = {"flags2": "+export_mvs"}

        mv_per_frame = []  
        motion_vector_list = []
        mvs_visual = []
        decoded = 0
        for packet in container.demux(vstream):
            for frame in packet.decode():

                found = False
                for sd in frame.side_data:
                    
                    if str(sd.type).endswith("MOTION_VECTORS"):
                        arr = sd.to_ndarray()  
                        mv_per_frame.append(arr)
                        found = True
                        break
                if not found:
                    mv_per_frame.append(np.zeros(0, dtype=np.dtype([])))
                decoded += 1
                if decoded >= num_frames:
                    break
        
            if decoded >= num_frames:
                break

        container.close()
        ffmpeg.stderr.read()
        ffmpeg.wait()


        for i in range(len(mv_per_frame)-1):

            motion_list = np.array(mv_per_frame[i+1].tolist())  

            # frame_save = np.zeros((256,256,3),dtype=np.uint8)
            # frame_save = draw_motion_vectors(frame_save, motion_list)
            # mvs_visual.append(frame_save)

            # motion_list = torch.stack(frame_mv_tensor[i+1])
            h, w = 256, 256
            mv = np.ones((h,w,2)) * -10000   # The
            position = motion_list[:,5:7].clip((0,0),(w-1,h-1))#当前帧矢量原点的位置
            mvs = motion_list[:,0:1] * motion_list[:,8:10] / motion_list[:, 10:]

            # Normalize the motion vector with resoultion
            mvs[:, 0] = mvs[:, 0] / w
            mvs[:, 1] = mvs[:, 1] / h
            # Normalize the motion vector
            mvs = (mvs - mean) / std

            mv[position[:,1],position[:,0]] = mvs

            motion_vector_list.append(mv)
    

    clip_motions = [torch.from_numpy(motion_vector_list[i].transpose((2,0,1))) for i in range(num_frames-1)]
    clip_motions = torch.stack(clip_motions).float()

    motion_trans = MotionVectorProcessor(clip_motions, width=16, height=16)
    # end = time.time()

    # print(f"Extracted motion vectors for {decoded} frames in {end - start:.2f} seconds")

    return motion_trans