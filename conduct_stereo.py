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

    
import numpy as np
# from hloc import (
#     extract_features,
#     handler,
#     logger,
#     match_features,
#     pairs_from_exhaustive,
#     reconstruction,
#     pairs_from_retrieval,
# )



def run_sfm(output_dir):
    images = Path(output_dir)
    outputs = Path(osp.join(output_dir, "output"))
    sfm_pairs = outputs / "pairs-sfm.txt"
    sfm_dir = outputs / "sfm"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"
    

    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_inloc"]
    matcher_conf = match_features.confs["superglue"]
    retrieval_path = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=25)
    feature_path = extract_features.main(feature_conf, images, outputs)
    match_path = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )
    model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path)
    return model


def run_stereo(scene):
    # scenes = glob.glob("~/DL3DV/images/7K/*")
    # scenes = scenes[mpIdx*move: (mpIdx+1)*move]
    # for scene_name in scenes:
    print(scene)
    scene_name = f"/ossfs/workspace/{scene}"
    ori_colmap = f'/ossfs/workspace/DL3DV_colmap/DL3DV_colmap/{scene}/colmap/sparse/0'
    colmap_path = Path(ori_colmap)
    output_path = Path(scene_name + "/undistort_and_depth_debug0")
    output_path.mkdir(exist_ok=True)


    image_dir = Path(scene_name + "/images_4")

    mvs_path = output_path / "mvs"

    if os.path.exists(ori_colmap + "/cameras.bin"):
        # if False:
        #     with tempfile.TemporaryDirectory() as tmpdir:
        #         images_dir = os.path.join(tmpdir, "mapping")
        #         # imgpaths = glob.glob(scene_name + "/images_4/*")
                
        #         shutil.copytree(scene_name + "/images_4", images_dir)

        #         reconstruction = run_sfm(tmpdir)      
        #         import pdb;pdb.set_trace()          
        #         reconstruction.write(output_path)
        # else:
        reconstruction = pycolmap.Reconstruction(ori_colmap)
        cameras = reconstruction.cameras
        for camera_id in cameras:
            params = cameras[camera_id].params
            params[:4] /= 4
            cameras[camera_id].width = cameras[camera_id].width // 4
            cameras[camera_id].height = cameras[camera_id].height // 4

        reconstruction.write(output_path)
        pycolmap.undistort_images(mvs_path, output_path, image_dir)
        # mvsoptions = pycolmap.PatchMatchOptions()
        # mvsoptions.cache_size = 256
        os.system(f'/usr/local/bin/colmap patch_match_stereo \
        --workspace_path {mvs_path} \
        --workspace_format COLMAP \
        --PatchMatchStereo.geom_consistency true')
        # pycolmap.patch_match_stereo(mvs_path)
        fusionoptions = pycolmap.StereoFusionOptions()
        # fusionoptions.cache_size =  256
        pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)
        shutil.rmtree(str(mvs_path) + "/stereo/normal_maps")

for scene in scenes:
    run_stereo(scene)
# for i in range(16, 33):
#     job = submit_job(i)
#     jobs.append(job)
