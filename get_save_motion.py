"""
get_save_motion.py

Standalone script: Loads the RLDS dataset, extracts motion vectors, and saves them to a JSON file.
Usage:
    python get_save_motion.py --data_root_dir /path/to/rlds --dataset_name my_dataset --output_dir /path/to/rlds
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import tensorflow as tf
import tensorflow_datasets as tfds
import draccus
from dataclasses import dataclass
import shutil

from prismatic.vla.datasets.rlds import traj_transforms
from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds
from prismatic.vla.datasets.rlds.oxe.materialize import make_oxe_dataset_kwargs  
from prismatic.vla.datasets.rlds.oxe.materialize import ACTION_PROPRIO_NORMALIZATION_TYPE
from prismatic.vla.datasets.rlds.utils.motion_utils import motion_save


@dataclass
class MotionSaveConfig:
    """Configuration for motion vector extraction and saving."""
    # Dataset settings
    data_root_dir: str = ""
    dataset_name: str = "libero_goal_no_noops"  

def get_oxe_dataset_kwargs_and_weights(
    cfg: MotionSaveConfig,
    data_root_dir: Path,
    load_camera_views: Tuple[str] = ("primary",),
    load_depth: bool = False,      
    load_proprio: bool = True,
    load_language: bool = True,
    action_proprio_normalization_type = ACTION_PROPRIO_NORMALIZATION_TYPE,
) -> Tuple[Dict[str, Any], List[float]]:
    dataset_kwargs_list = make_oxe_dataset_kwargs(
                    cfg.dataset_name,
                    data_root_dir,
                    load_camera_views,
                    load_depth,
                    load_proprio,
                    load_language,
                    action_proprio_normalization_type,
                )
    return dataset_kwargs_list


@draccus.wrap()
def main(cfg: MotionSaveConfig) -> None:
    """Main entry point for motion vector extraction."""
    print(f"=== Motion Vector Extraction ===")
    print(f"Dataset: {cfg.dataset_name}")
    print(f"Data root: {cfg.data_root_dir}")
    

    # fmt: off
    if "aloha" in cfg.dataset_name:
        load_camera_views = ("primary", "left_wrist", "right_wrist")
    else:
        load_camera_views = ("primary", "wrist")

    # Get dataset configurations
    dataset_kwargs = get_oxe_dataset_kwargs_and_weights(cfg, cfg.data_root_dir, load_camera_views)

    # rlds_config = dict(
    #         dataset_kwargs_list=dataset_kwargs,
    #         shuffle_buffer_size=100_000,
    #         sample_weights=1.0,
    #         balance_weights=True,
    #         traj_transform_threads=1,
    #         traj_read_threads=1,
    #         train=True,
    #     )
    
    print(f"\n--- Processing: {dataset_kwargs['name']} ---")


    _, dataset_statistics = make_dataset_from_rlds(**dataset_kwargs, train=True)

    dataset, _ = make_dataset_from_rlds(
            **dataset_kwargs,
            train=True,
            num_parallel_calls=1,
            num_parallel_reads=1,
            dataset_statistics=dataset_statistics,
        )
    
    dataset = dataset.traj_map(traj_transforms.extract_motion_tf,num_parallel_calls=8)
    # Save motion vectors
    motion_save(dataset)
    
    # del video
    p = Path("./tmp") / cfg.dataset_name / "video"
    if p.exists() and p.is_dir():
        shutil.rmtree(p)
    
    print(f"✓ Completed: {dataset_kwargs['name']}")

    print("\n=== All datasets processed ===")


if __name__ == "__main__":
    main()
