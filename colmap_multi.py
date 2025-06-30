import multiprocessing
import os
import time
import argparse
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
import numpy as np
import glob
nas_path = 'input_my_v2'
local_path = '/root/code/zsz/DL3DV'
collect_flag_path = f"/{nas_path}/zsz/DL3DV/flag"
def run_stereo(nK, scene, gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # 这里是你运行场景的代码
    scene_name = f"/{nas_path}/zsz/DL3DV/{nK}/{scene}"
    ori_colmap = f'{local_path}/zsz/DL3DV/{nK}/{scene}/colmap/sparse/0'
    colmap_path = Path(ori_colmap)
    output_path = Path(f"{local_path}/{nK}_processed/{scene}" + "/undistort_and_depth_debug0")
    os.system('mkdir -p ' + str(output_path))
    os.system(f'ossutil64 cp -r oss://antsys-vilab/zsz/scene_images/{scene} {local_path}/{nK}/ -j 20 -u')
    print(f'ossutil64 cp -r oss://antsys-vilab/zsz/scene_images/{scene} {local_path}/{nK}/ -j 20 -u')
    image_dir = Path(f'{local_path}/{nK}/{scene}/' + "images_4")
    mvs_path = output_path / "mvs"
    save_mvs_path = Path(f"/{nas_path}/zsz/DL3DV/{nK}_processed/{scene}" + "/undistort_and_depth_debug0")
    os.system('mkdir -p ' + str(save_mvs_path))
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
        print(str(mvs_path) + "/stereo/normal_maps")
        print(mvs_path)
        if os.path.exists(collect_flag_path + f"/{scene}/success.txt") == False:
            os.system(f'colmap patch_match_stereo \
            --workspace_path {mvs_path} \
            --workspace_format COLMAP \
            --PatchMatchStereo.geom_consistency true')
            # os.system(f'cp -r {mvs_path} {save_mvs_path}')
            output_root_path = f"{local_path}/{nK}_processed/{scene}/" 
            os.system(f'ossutil64 cp -r {output_root_path} oss://antsys-vilab/zsz/DL3DV_mvs_results/{nK}_processed/{scene}/ -u -j 8')
            print(f'ossutil64 cp -r {output_root_path} oss://antsys-vilab/zsz/DL3DV_mvs_results/{nK}_processed/{scene}/ -u -j 8')
            print(mvs_path / "stereo/depth_maps/frame_00001.png.geometric.bin")
            if os.path.exists(mvs_path / "stereo/depth_maps/frame_00001.png.geometric.bin"):
                os.system(f"mkdir -p {collect_flag_path}/{nK}/{scene}")
            # print(f'cp -r {mvs_path} {save_mvs_path}/')
            print(output_root_path)
            os.system(f'rm -rf {output_root_path}')

        # pycolmap.patch_match_stereo(mvs_path)
        # fusionoptions = pycolmap.StereoFusionOptions()
        # # fusionoptions.cache_size =  256
        # pycolmap.stereo_fusion(save_mvs_path / "dense.ply", mvs_path)
        # # shutil.rmtree(str(mvs_path) + "/stereo/normal_maps")
        # shutil.rmtree(scene_name)
        # os.system(f'python colmap2mvsnet.py --dense_folder {}')
    else:
        print(f'No cameras.bin in {ori_colmap}')
    
    time.sleep(1)
import torch

def get_gpu_count():
    num_gpus = torch.cuda.device_count()
    return num_gpus

import psutil



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes_file', type=str, required=True, help='Path to the scenes .txt file')
    parser.add_argument('--nK', type=str, required=True, help='Number of something (nK parameter)')
    os.system(f'mkdir /{nas_path} && chmod 777 -R /{nas_path} &&  mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport alipayshnas-004-gkn38.cn-shanghai-eu13-a01.nas.aliyuncs.com:/ /{nas_path}')
    os.system(f'mkdir -p {collect_flag_path}')
    args = parser.parse_args()
    # 你的场景列表
    scenes_file = args.scenes_file
    num_gpus = get_gpu_count()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'Found {num_gpus} GPUs with {gpu_memory:.1f} GB memory each')
    total_cpus = psutil.cpu_count(logical=True)  # 获取逻辑CPU的数量
    cpus_per_gpu = total_cpus // num_gpus
    max_processes_per_gpu = 1
    print(f'Running {max_processes_per_gpu} processes per GPU')
    nK = args.nK
    os.system(f'ossutil64 cp -r oss://antsys-vilab/zsz/DL3DV_txt/{nK}/{scenes_file} ./')
    with open(scenes_file, 'r') as f:
        scenes = [line.strip().split('/')[-2] for line in f.readlines()]
    
    # 剩余待处理的场景列表
    scenes_left = scenes.copy()
    # 用于跟踪每个GPU上的运行进程
    running_processes = {gpu_id: [] for gpu_id in range(num_gpus)}

    while scenes_left or any(running_processes[gpu_id] for gpu_id in range(num_gpus)):
        for gpu_id in range(num_gpus):
            # 检查并移除已完成的进程
            for p in running_processes[gpu_id][:]:
                if not p.is_alive():
                    p.join()
                    running_processes[gpu_id].remove(p)

            # 如果GPU上的进程少于10个，且有剩余的场景，分配新的场景
            while len(running_processes[gpu_id]) < max_processes_per_gpu and scenes_left:
                scene = scenes_left.pop(0)
            
                os.system(f'ossutil64 cp -r oss://antsys-vilab/zsz/DL3DV_mvs_results/{nK}_processed/{scene}/ /{local_path}/{nK}_processed/{scene}/ -u -j 8')
                nas_mvs_path = Path(f"/{nas_path}/zsz/DL3DV/{nK}_processed/{scene}" + "/undistort_and_depth_debug0")
                local_mvs_path = Path(f"{local_path}/{nK}_processed/{scene}" + "/undistort_and_depth_debug0")

                nas_mvs_last_bin= f"/{nas_path}/zsz/DL3DV/{nK}_processed/{scene}" + "/undistort_and_depth_debug0/mvs/stereo/depth_maps/*.bin"
                local_mvs_last_bin = f"{local_path}/{nK}_processed/{scene}" + "/undistort_and_depth_debug0/mvs/stereo/depth_maps/*.bin"
                nas_mvs_last= f"/{nas_path}/zsz/DL3DV/{nK}_processed/{scene}" + "/undistort_and_depth_debug0/mvs/images/*.png"
                local_mvs_last = f"{local_path}/{nK}_processed/{scene}" + "/undistort_and_depth_debug0/mvs/images/*.png"

                nas_mvs_last_num = len(glob.glob(nas_mvs_last_bin))//2
                local_mvs_last_num = len(glob.glob(local_mvs_last_bin))//2
                nas_png_list = len(glob.glob(nas_mvs_last))
                local_png_list = len(glob.glob(local_mvs_last))
                print(nas_mvs_last + " " + str(nas_png_list))
                print(local_mvs_last + " " + str(local_png_list))
                print(local_mvs_last_bin + " " + str(local_mvs_last_num))
                print(nas_mvs_last_bin + " " + str(nas_mvs_last_num))
                nas_valid = (nas_mvs_last_num == nas_png_list) and (nas_mvs_last_num > 0)
                local_valid =  (local_mvs_last_num == local_png_list) and (local_mvs_last_num > 0)
                save_mvs_path = Path(f"/{nas_path}/zsz/DL3DV/{nK}_processed/{scene}" + "/undistort_and_depth_debug0")
                nas_complete = os.path.exists(str(save_mvs_path) + "/success.txt")
                local_complete = os.path.exists(str(local_mvs_path) + "/success.txt")
                
                local_flag = local_valid and local_complete
                nas_flag = nas_valid and nas_complete

                colmap_dir = f"/{nas_path}/zsz/DL3DV/{nK}/{scene}"
                if local_flag:
                    # os.system(f'ossutil64 cp -r /{local_path}/{nK}_processed/{scene}/ oss://antsys-vilab/zsz/DL3DV_mvs_results/{nK}_processed/{scene}/ -u -j 8')
                    os.system(f'rm -rf {local_path}/{nK}_processed/{scene}/')
                    os.system(f'rm -rf /{nas_path}/zsz/DL3DV/{nK}_processed/{scene}/')
                    os.system(f'rm -rf {colmap_dir}')
                    os.system(f"mkdir -p {collect_flag_path}/{nK}/{scene}")
                    print(f"local_flag! mkdir -p {collect_flag_path}/{nK}/{scene}")
                    continue
                elif nas_flag:
                    os.system(f'ossutil64 cp -r /{nas_path}/zsz/DL3DV/{nK}_processed/{scene}/ oss://antsys-vilab/zsz/DL3DV_mvs_results/{nK}_processed/{scene}/ -u -j 8')
                    os.system(f'rm -rf {local_path}/{nK}_processed/{scene}/')
                    os.system(f'rm -rf /{nas_path}/zsz/DL3DV/{nK}_processed/{scene}/')
                    os.system(f'rm -rf {colmap_dir}')
                    os.system(f"mkdir -p {collect_flag_path}/{nK}/{scene}")
                    print(f"nas_flag! mkdir -p {collect_flag_path}/{nK}/{scene}")
                    continue
                else:
                    # print(scene)
                    # continue
                    os.system(f'ossutil64 cp -r oss://antsys-vilab/zsz/DL3DV/{nK}/{scene} {local_path}/zsz/DL3DV/{nK}/ -u -j 8')
                    p = multiprocessing.Process(target=run_stereo, args=(nK, scene, gpu_id))
                    p.start()
                    running_processes[gpu_id].append(p)
                    
        # 等待一段时间再检查（避免过于频繁的检查）
        time.sleep(0.1)

    # 确保所有进程都已完成
    for gpu_id in range(num_gpus):
        for p in running_processes[gpu_id]:
            p.join()