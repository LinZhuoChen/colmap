import pycolmap
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
import torch

import submitit
    
import numpy as np
from hloc import (
    extract_features,
    handler,
    logger,
    match_features,
    pairs_from_exhaustive,
    reconstruction,
    pairs_from_retrieval,
)


import numpy as np
import cv2
from scipy import ndimage
import torch
import torchvision.transforms as T
from PIL import Image
# import matplotlib.pyplot as plt

# from torchvision.models.segmentation import deeplabv3_resnet50

from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv


def segment_sky(image_path, segmodel):

    seg_result = inference_model(segmodel, image_path)
    sky_idx = segmodel.dataset_meta['classes'].index('sky')
    seg_mask = seg_result.pred_sem_seg.data
    seg_mask[seg_mask!=sky_idx] = 1
    seg_mask[seg_mask==sky_idx] = 0

    seg_mask_np = (255*seg_mask[0].cpu().numpy()).astype(np.uint8)
    # cv2.imwrite('seg_mask.png', (255*seg_mask[0].cpu().numpy()).astype(np.uint8))

    mask_dir = os.path.dirname(image_path).replace("mvs/images", "mvs/masks")
    os.makedirs(mask_dir, exist_ok=True)
    
    mask_filename = os.path.join(mask_dir, os.path.basename(image_path))
    
    # Save the mask to a new file
    # mask_filename = 'mask_' + image_path.split('/')[-1]
    cv2.imwrite(mask_filename +".png", seg_mask_np)
    


def segment_sky_wrapper(image_dir, segmodel):
    images = glob.glob(str(image_dir) + "/*")
    for image in images:
        segment_sky(image, segmodel)
    return True


def run_stereo(scene_name):
    
    print(scene_name)
    ori_colmap = scene_name.replace("images", "colmap") + "/colmap/sparse/0"
    
    colmap_path = Path(ori_colmap)
    output_path = Path(scene_name + "/undistort_and_depth_debug0")
    output_path.mkdir(exist_ok=True)

    image_dir = Path(scene_name + "/images_4")

    mvs_path = output_path / "mvs"


    config_file = 'deeplabv3plus_r101-d16-mg124_4xb2-80k_cityscapes-512x1024.py'
    checkpoint_file = 'deeplabv3plus_r101-d16-mg124_512x1024_80k_cityscapes_20200908_005644-ee6158e0.pth'

    segmodel = init_model(config_file, checkpoint_file, device='cuda:0')
    segmodel.eval()  # Set the model to inference mode

    # if os.path.exists(ori_colmap + "/cameras.bin"):
    #     reconstruction = pycolmap.Reconstruction(ori_colmap)
    #     cameras = reconstruction.cameras
        
    #     for camera_id in cameras:
    #         params = cameras[camera_id].params
    #         params[:4] /= 4

    #         cameras[camera_id].width = cameras[camera_id].width // 4
    #         cameras[camera_id].height = cameras[camera_id].height // 4
    #     reconstruction.write(output_path)


    #     pycolmap.undistort_images(mvs_path, output_path, image_dir)


    segment_sky_wrapper(mvs_path / "images", segmodel)
        
        # mvsoptions = pycolmap.PatchMatchOptions()
        # mvsoptions.cache_size = 256

        # pycolmap.patch_match_stereo(mvs_path, options=mvsoptions)
        
    fusionoptions = pycolmap.StereoFusionOptions()
    fusionoptions.cache_size =  256
    
    fusionoptions.mask_path = str(mvs_path / "masks")
    
        # print(f"running stereo fusion with {str(mvs_path / "masks")}")
    pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path, options=fusionoptions)
    # shutil.rmtree(str(mvs_path) + "/stereo/normal_maps")






def run_stereo_wrapper(scenes):
    for scene_name in scenes:
        run_stereo(scene_name)
