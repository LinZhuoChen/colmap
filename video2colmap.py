import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from decord import VideoReader
import imageio
pose_path = "/nas3/zsz/RealEstate10k_v2/pose_files/159e4575ac8d8465.txt"
clip_path = "/nas3/zsz/RealEstate10k_v2/video_clips/00ccbtp2aSQ/159e4575ac8d8465.mp4"
output_dir = "/data0/zsz/mast3recon/data/test"
sparse_dir = output_dir + '/sparse'
dense_dir = output_dir + '/mvs'
os.makedirs(dense_dir, exist_ok=True)
os.makedirs(sparse_dir, exist_ok=True)
# os.makedirs(sparse_dir + '/0', exist_ok=True)

with open(pose_path, 'r') as f:
    poses = f.readlines()

poses = [pose.strip().split(' ') for pose in poses[1:]]
with open(pose_path, 'r') as f:
    poses = f.readlines()
poses = [pose.strip().split(' ') for pose in poses[1:]]
camera_params = [[float(x) for x in pose] for pose in poses]

video_reader = VideoReader(clip_path)
H, W = video_reader[0].shape[:2]

frame_id = 1
cameras = []
images = []

for cam_param, frame in zip(camera_params, video_reader):
    fx, fy, cx, cy = cam_param[1:5]
    w2c_mat = np.array(cam_param[7:]).reshape(3, 4).astype(np.float32)
    K = np.asarray([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], dtype=np.float32)
    fx = fx * W
    fy = fy * H
    cx = cx * W
    cy = cy * H
    image_name = f'frame_{frame_id:05d}.png'
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    imageio.imwrite(os.path.join(output_dir, 'images', image_name), frame.asnumpy())

    R_matrix = w2c_mat[:, :3]
    t = w2c_mat[:, 3]

    r = R.from_matrix(R_matrix)
    q = r.as_quat()

    cameras.append((H, W, fx, fy, cx, cy))
    images.append((frame_id, q, t, frame_id, image_name))

    frame_id += 1

with open(os.path.join(sparse_dir,  'cameras.txt'), 'w') as f:
    for idx, (H, W, fx, fy, cx, cy) in enumerate(cameras, start=1):
        f.write(f"{idx} PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")

with open(os.path.join(sparse_dir,  'images.txt'), 'w') as f:
    for img_id, q, t, cam_id, image_name in images:
        qx, qy, qz, qw = q
        tx, ty, tz = t
        f.write(f"{img_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {cam_id} {image_name}\n\n")

with open(os.path.join(sparse_dir, 'points3D.txt'), 'w') as f:
    pass

os.system(f"colmap feature_extractor \
    --database_path {output_dir}/database.db \
    --image_path {output_dir}/images")

os.system(f"colmap exhaustive_matcher --database_path {output_dir}/database.db")


os.system(f"colmap point_triangulator \
    --database_path {output_dir}/database.db\
    --image_path {output_dir}/images --input_path {sparse_dir} \
    --output_path {sparse_dir}")

os.system(f"colmap image_undistorter \
    --image_path {output_dir}/images \
    --input_path {sparse_dir} \
    --output_path {dense_dir}")

os.system(f"colmap patch_match_stereo \
    --workspace_path {dense_dir}")

os.system(f"colmap stereo_fusion \
    --workspace_path  {dense_dir} \
    --output_path {dense_dir}/fused.ply")

