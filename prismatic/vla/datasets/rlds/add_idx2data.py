
import rlds
import uuid
import os
import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.rlds.rlds_decode import decode as rlds_decode


def add_episode_id(traj):
    """为每个轨迹添加唯一 episode_id"""
    episode_id = str(uuid.uuid4())  # 生成唯一标识符
    traj["episode_id"] = tf.constant(episode_id, dtype=tf.string)
    return traj


def save_dlimp_dataset_with_same_name(dataset, original_path, new_dir):
    filename = os.path.basename(original_path)
    new_path = os.path.join(new_dir, filename)
    os.makedirs(new_dir, exist_ok=True)

    serialized_dataset = dataset.map(rlds.serialize)
    writer = tf.data.experimental.TFRecordWriter(new_path)
    writer.write(serialized_dataset)
    print(f"已保存：{new_path}")

def add_episode_id_and_save(name, original_dir, new_dir):

    # custom_data_dir = "/my/folder"
    # ds_test = tfds.load("my_dataset/single_number:1.0.0", split="test", data_dir=custom_data_dir)
    

    tfrecord_files = sorted([
        os.path.join(original_dir, f)
        for f in os.listdir(original_dir)
        if f.endswith(".tfrecord") or ".tfrecord-" in f
    ])
    
    for path in tfrecord_files:
        print(f"正在处理：{path}")
        # ds = tfds.load(name,data_dir=path, split='train')
        raw_ds = tf.data.TFRecordDataset(path).map(rlds.deserialize)
        
        episode_id=0
        ds_aa=[]
        for episode in raw_ds:
            episode["episode_id"]=episode_id
            episode_id=+1
            ds_aa.append(episode)

    # 构造新的 TFRecord 文件名，保留原文件名
    file_name = os.path.basename(path)
    target_path = os.path.join(new_dir, file_name)

    # print(f"保存到：{target_path}")
    # with tf.io.TFRecordWriter(target_path) as writer:
    #     for ep in ds_aa:
    #         serialized = rlds.serialize(ep)
    #         writer.write(serialized.numpy())
            

    #     dataset = dataset.traj_map(add_episode_id)
    #     save_dlimp_dataset_with_same_name(dataset, path, new_dir)
    
    # builder = tfds.builder(name, data_dir=data_dir)
    # dataset = dl.DLataset.from_rlds(builder, split=split, shuffle=True)
    # dataset = dataset.map(add_episode_id)

    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # writer = tf.data.experimental.TFRecordWriter(output_path)
    # writer.write(serialized_dataset)
    # dataset.save(output_path)



split = "train"
name ='libero_spatial_no_noops'


original_dir = "/liujiacheng/linminghui/datasets/modified_libero_rlds/libero_spatial_no_noops/1.0.0"
new_dir = "/liujiacheng/linminghui/datasets/reepisode_libero/libero_spatial_no_noops/1.0.0"
os.makedirs(os.path.dirname(new_dir), exist_ok=True)

add_episode_id_and_save(name, original_dir, new_dir)