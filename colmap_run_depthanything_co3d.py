import multiprocessing
import os
import time
import argparse
import glob
from pathlib import Path
import os
import argparse
import json
import logging
import os
import os.path as osp
import shutil
import tempfile
from pathlib import Path
import numpy as np
import glob
nas_path = 'input1'
local_path = '/root/code/zsz/DL3DV'
# collect_data_path = f"/{nas_path}/datasets/DL3DV_dust3r"
ROOT='/input_ssd/zsz/co3d_processed_full.zip/co3d_subset_processed_full'
import torch
# import mmcv
import cv2
from colmap.depth_anything_v2.dpt import DepthAnythingV2

import os
import cv2
import numpy as np

import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

degree = 1
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
ransac = RANSACRegressor(max_trials=1000)
linear_model = make_pipeline(poly_features, ransac)


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """ Open an image or a depthmap with opencv-python.
    """
    if path.endswith(('.exr', 'EXR')):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f'Could not load image={path} with {options=}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def scale_shift_align_depth(src, tar, mask, disp=True, fit_type="ransac"):
    '''
    src: HxW, 需要align的单目depth
    tar: HxW, 目标深度图, 一般是metric depth,或者sfm depth
    mask: HxW, bool mask,用于选择目标深度图中的有效区域
    disp: 是否是disparity,depth anything/midas是disparity model,需要设置为True, depth pro, metric3d v2是depth,需要设置为False
    fit_type: 拟合类型,可选'poly'或'ransac'
    '''
    tar_val = tar[mask].astype(np.float32)
    src_val = src[mask].astype(np.float32)
    if disp:
        tar_val = np.clip(tar_val, 1e-4, None)
        tar_val = 1 / tar_val

    # if fit_type == "poly":
    #     a, b = np.polyfit(src_val, tar_val, deg=1)
    if fit_type == "ransac":
        linear_model.fit(src_val[:, None], tar_val[:, None])
        a = linear_model.named_steps["ransacregressor"].estimator_.coef_
        b = linear_model.named_steps["ransacregressor"].estimator_.intercept_
        a, b = a.item(), b.item()
    else:
        # Log.debug("Unknown fit type")
        a, b = 1, 0
    # Log.debug(f"Fit {fit_type}: scale: {a}, shift: {b}")
    if a < 0:
        # Log.warn("Negative scale detected")
        return False, None
    src_ = a * src + b
    if disp:
        src_ = 1 / np.clip(src_, 1e-2, None)  # max 100 meters
    return True, src_


def run_stereo(nK, save_mvs_path, model):
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id), 'images'
    # 这里是你运行场景的代码
    filenames = glob.glob(os.path.join(save_mvs_path, 'images', '*.jpg'))
    depth_path = os.path.join(save_mvs_path, 'depth_anything')#rgs.img_path.replace('images', 'depth_anything')
    depth_aligned_path = os.path.join(save_mvs_path, 'depth_anything_aligned')#rgs.img_path.replace('images', 'depth_anything')

    # depth_colmap_path = os.path.join(save_mvs_path, 'depth')#rgs.img_path.replace('images', 'depth_anything')
    os.makedirs(depth_path, exist_ok=True)
    os.makedirs(depth_aligned_path, exist_ok=True)
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        raw_image = cv2.imread(filename)
        save_name = os.path.join(depth_path, os.path.basename(filename).replace('png', 'npy'))
        if os.path.exists(save_name) == False:
            depth = depth_anything.infer_image(raw_image)
            np.save(save_name, depth)
        else:
            depth = np.load(save_name)
        view_idx = int(os.path.basename(filename).split('.')[0].replace('frame', ''))
        depth_colmap_path = osp.join(save_mvs_path, 'depths', f'frame{view_idx:06n}.jpg.geometric.png')
        metadata_path = osp.join(save_mvs_path, 'images', f'frame{view_idx:06n}.npz')
        input_metadata = np.load(metadata_path)
        colmap_depthmap = imread_cv2(depth_colmap_path, cv2.IMREAD_UNCHANGED)
        colmap_depthmap = (colmap_depthmap.astype(np.float32) / 65535) * np.nan_to_num(input_metadata['maximum_depth'])
        # colmap_depthmap = np.load(filename.replace('images', 'depth').replace('.png', '.npy').replace('frame_', ''))
        if (colmap_depthmap > 0).sum() > 100:
            _, depth_align = scale_shift_align_depth(depth, colmap_depthmap, colmap_depthmap > 0, disp=True, fit_type="ransac")
            save_name_aligned = os.path.join(depth_aligned_path, os.path.basename(filename).replace('png', 'npy'))
            np.save(save_name_aligned, depth_align)
        else:
            save_name_aligned = os.path.join(depth_aligned_path, os.path.basename(filename).replace('png', 'npy'))
            np.save(save_name_aligned, np.zeros_like(colmap_depthmap))


def get_gpu_count():
    num_gpus = torch.cuda.device_count()
    return num_gpus

import psutil



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes_file', type=str, required=True, help='Path to the scenes .txt file')
    parser.add_argument('--nK', type=str, required=True, help='Number of something (nK parameter)')
    args = parser.parse_args()
    nK = args.nK
    scenes_file = args.scenes_file
    os.system(f'ossutil64 cp -r oss://antsys-vilab/zsz/depth_anything_v2_vitl.pth ./ -u')
    os.system(f'ossutil64 cp -r oss://antsys-vilab/zsz/co3d_txt_depth/{nK}/{scenes_file} ./ -u')
    with open(scenes_file, 'r') as f:
        scenes = [line.strip().split('/')[-2] for line in f.readlines()]
    # ROOT = '/input_ssd/zsz/co3d_processed_full.zip/co3d_subset_processed_full'
    # with open(osp.join(ROOT, f'selected_seqs_{self.split}.json'), 'r') as f:
    #     scenes = json.load(f)
    #     scenes = {k: v for k, v in self.scenes.items() if len(v) > 0}
    #     scenes = {(k, k2): v2 for k, v in self.scenes.items()
    #                 for k2, v2 in v.items()}
    # # 剩余待处理的场景列表
    # scene_list = list(scenes.keys())
    scenes_left = scenes.copy()
    # 用于跟踪每个GPU上的运行进程
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    depth_anything = DepthAnythingV2(**model_configs['vitl'])
    depth_anything.load_state_dict(torch.load(f'depth_anything_v2_vitl.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    while scenes_left:
        obj, instance = scenes_left.pop(0).split('-')
        save_mvs_path = osp.join(ROOT, obj, instance)
        run_stereo(nK, save_mvs_path, depth_anything)
        os.system('touch ' + f'{obj}_{instance}.txt')
        os.system(f'ossutil64 cp -r {obj}_{instance}.txt oss://antsys-vilab/zsz/co3d_mvs_depth_results/{obj}-{instance}/ -u')
