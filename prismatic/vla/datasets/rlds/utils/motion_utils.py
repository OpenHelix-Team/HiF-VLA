import torch
import os
import json
import subprocess
import numpy as np
import tensorflow as tf
from mvextractor.videocap import VideoCap
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import math
import concurrent.futures
from tqdm import tqdm
from joblib import Parallel, delayed
import cv2

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def MotionVectorProcessor(motions,width, height):
        transform_list = [
            transforms.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        ]
        motion_transform = transforms.Compose(transform_list)

        h, w = motions.shape[-2:]
        pad_h_size = math.ceil(h / 16) * 16
        pad_w_size = math.ceil(w / 16) * 16

        padding_h = pad_h_size - h
        padding_w = pad_w_size - w

        pad = (0,padding_w,0,padding_h)
        motions = torch.nn.functional.pad(motions, pad, mode="constant", value=-10000)
        motions = torch.nn.functional.max_pool2d(motions, kernel_size=16, stride=16)
        motions[motions < -1000] = 0.0

        # # interpolate the 13-th frame, which is I frame, don't have a motion
        # motions_to_inter = torch.cat([motions[11:12], motions[13:14]])
        # motions_to_inter = motions_to_inter.permute(1, 0, 2, 3).unsqueeze(0)
        # motions_to_inter = torch.nn.functional.interpolate(motions_to_inter, scale_factor=(1.5, 1.0, 1.0), mode='trilinear')[0]
        # motions_to_inter = motions_to_inter.permute(1, 0, 2, 3)
        # motions[12] = motions_to_inter[1]

        motion_vectors = motion_transform(motions)   # [T, C, H, W]
        # motion_vectors = motion_vectors.permute(1, 0, 2, 3)  # From [T, C, H, W] => [C, T, H, W]

        return motion_vectors#t,2,32,32
    

def real_motion_processing_1(traj):
        
        # 读取路径
        video_path = traj["video_path"][0].decode("utf-8")
        motion_path = traj["motion_path"][0].decode("utf-8")

        if os.path.exists(motion_path) and os.path.exists(video_path):
            print(f" Skip existing motion and video file: {motion_path}")
            return

        # 创建视频目录
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        # # Step 1: 保存图像为视频（10 fps）
        # video_writer = imageio.get_writer(video_path, fps=10)#, codec="ffv1"
        # for step in traj["observation"]["image_primary"]:
        #     img = np.array(Image.open(BytesIO(step.numpy())).convert("RGB"))
        #     video_writer.append_data(img)
        # video_writer.close()
        # print(f"Saved video to: {video_path}")

        # Write raw jpgs
        img_dir = video_path.replace(".mp4", "_imgs")
        os.makedirs(img_dir, exist_ok=True)
        if b"libero" in traj["dataset_name"][0]:
            for i, step in enumerate(traj["observation"]["image_primary"]):
                with open(f"{img_dir}/{i:06d}.jpg", "wb") as f:
                    f.write(step)
        elif b"calvin" in traj["dataset_name"][0]:
            target_size = (256, 256)  # (H, W)
            for i, step in enumerate(traj["observation"]["image_primary"]):
                # 获取原始字节
                if isinstance(step, (bytes, bytearray)):
                    step_bytes = step
                elif tf.is_tensor(step):
                    step_bytes = step.numpy()
                else:
                    step_bytes = bytes(step)
                out_path = f"{img_dir}/{i:06d}.png"
                arr = np.frombuffer(step_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
                if img is None:
                    return False
                if img.shape[0:2] != (target_size[0], target_size[1]):
                    img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
                cv2.imwrite(out_path, img)

        # Step 2: 使用 ffmpeg 重新编码为特定 GOP 的视频（mpeg4）
        tmp_video_path = video_path.replace(".mp4", "_fps.mp4")
        g_len = len(traj["task"]["language_instruction"])  # GOP length based on task language instruction length
        # ffmpeg_cmd = f'ffmpeg -threads 8 -loglevel error -i {video_path} -filter:v fps={10} -c:v mpeg4 -g {g_len} -f rawvideo {tmp_video_path}'
        if b"libero" in traj["dataset_name"][0]:
            ffmpeg_cmd = f'ffmpeg -y -threads 8 -framerate {6} -i {img_dir}/%06d.jpg -c:v mpeg4 -g {g_len} -preset veryfast {tmp_video_path}'
        else:
            ffmpeg_cmd = f'ffmpeg -y -threads 8 -framerate {6} -i {img_dir}/%06d.png -c:v mpeg4 -g {g_len} -preset veryfast {tmp_video_path}'            
        subprocess.run(ffmpeg_cmd, shell=True, timeout=300)
        print(f"Re-encoded video to: {tmp_video_path}")

        # The motion vector mean and std
        mean = np.array([[0.0, 0.0]], dtype=np.float64)
        std =  np.array([[0.0993703, 0.1130276]], dtype=np.float64)

        
        # Step 3: 使用 mvextractor 提取运动向量
        cap = VideoCap()
        cap.open(tmp_video_path)
        motion_vector_list = []
        while True:
            ret, frame, motion_vectors, _, _ = cap.read()
            
            if not ret:
                break
            h, w = frame.shape[:2]
            mv = np.ones((h,w,2)) * -10000   # The
            position = motion_vectors[:,5:7].clip((0,0),(w-1,h-1))#当前帧矢量原点的位置
            mvs = motion_vectors[:,0:1] * motion_vectors[:,7:9] / motion_vectors[:, 9:]

            # Normalize the motion vector with resoultion
            mvs[:, 0] = mvs[:, 0] / w
            mvs[:, 1] = mvs[:, 1] / h
            # Normalize the motion vector
            mvs = (mvs - mean) / std

            mv[position[:,1],position[:,0]] = mvs

            motion_vector_list.append(mv)
        cap.release()
        
        traj_len = int(tf.shape(traj["action"])[0])
        clip_motions = [torch.from_numpy(motion_vector_list[i].transpose((2,0,1))) for i in range(traj_len)]
        clip_motions = torch.stack(clip_motions).float()
        
        motion_trans = MotionVectorProcessor(clip_motions, width=16, height=16)

        
        # Step 4: 保存运动向量为 JSON
        os.makedirs(os.path.dirname(motion_path), exist_ok=True)
        # motion_data_list = [{"motion": mv.tolist()} for mv in motion_vector_list]
        motion_data_list = [{f"motion_{i}": motion_trans[i].tolist()} for i in range(motion_trans.shape[0])]

        with open(motion_path, "w") as f:
            json.dump(motion_data_list, f)
        print(f"Saved motion vectors to: {motion_path}")

        # Step 5: 清理临时文件和视频文件
        try:
            # 删除临时重编码视频
            if os.path.exists(tmp_video_path):
                os.remove(tmp_video_path)
                print(f"Removed temporary video: {tmp_video_path}")
            
            # 删除原始视频文件（可选）
            # if os.path.exists(video_path):
            #     os.remove(video_path)
            #     print(f"Removed video: {video_path}")
            
            # 删除临时图片目录
            if os.path.exists(img_dir):
                import shutil
                shutil.rmtree(img_dir)
                print(f"Removed image directory: {img_dir}")
        except Exception as e:
            print(f"Warning: Failed to clean up temporary files: {e}")


def traj_to_numpy(traj):
    return tf.nest.map_structure(lambda x: x.numpy() if tf.is_tensor(x) else x, traj)

def motion_save(dataset):
    traj_list = [traj_to_numpy(traj) for traj in dataset.as_numpy_iterator()]
    real_motion_processing_1(traj_list[0])  # Test the function with the first trajectory
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(real_motion_processing_1, traj_list)))
    # Parallel(n_jobs=8, backend="loky")(
    #     delayed(real_motion_processing_1)(traj) for traj in traj_list
    #     )