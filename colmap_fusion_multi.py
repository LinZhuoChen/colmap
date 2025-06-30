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
collect_flag_path = f"/{nas_path}/zsz/DL3DV/flag_fusion"
import torch
from mmseg.apis import inference_model, init_model, show_result_pyplot
# import mmcv
import cv2

import os
import cv2
import numpy as np
# 获取 PyTorch 的 lib 路径


def calculate_true_ratio(image_path):
    # 使用cv2读取图像，并将其转换为灰度图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 将图像转换为布尔数组，假设非零的像素值为True
    img_bool = img > 0
    
    # 计算True值的比例
    true_ratio = np.sum(img_bool) / img_bool.size
    return true_ratio

def process_directory(mvs_path):
    mask_path = os.path.join(mvs_path, 'filter/mask')
    ratios = []
    
    for filename in os.listdir(mask_path):
        if filename.endswith('_final.png'):
            image_path = os.path.join(mask_path, filename)
            true_ratio = calculate_true_ratio(image_path)
            ratios.append(true_ratio)
    
    if ratios:
        average_ratio = sum(ratios) / len(ratios)
        return average_ratio
    else:
        return 0

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



def run_stereo(nK, scene, gpu_id):
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # 这里是你运行场景的代码
    print(f'ossutil64 cp -r oss://antsys-vilab/zsz/DL3DV/{nK}/{scene} {nas_path}/zsz/DL3DV/{nK}/ -u')
    scene_name = f"/{nas_path}/zsz/DL3DV/{nK}/{scene}"
    ori_colmap = f'{local_path}/zsz/DL3DV/{nK}/{scene}/colmap/sparse/0'
    image_dir = Path(f'/{nas_path}/datasets/DL3DV-10K/scene_images/{scene}/' + "/images_4")
    output_path = Path(f"{local_path}/{nK}_processed/{scene}" + "/undistort_and_depth_debug0")
    mvs_path = output_path / "mvs"
    
    config_file = 'deeplabv3plus_r101-d16-mg124_4xb2-80k_cityscapes-512x1024.py'
    checkpoint_file = 'deeplabv3plus_r101-d16-mg124_512x1024_80k_cityscapes_20200908_005644-ee6158e0.pth'

    segmodel = init_model(config_file, checkpoint_file, device='cuda')
    segmodel.eval()
    segment_sky_wrapper(mvs_path / "images", segmodel)
        
    # fusionoptions = pycolmap.StereoFusionOptions()
    # fusionoptions.cache_size =  256
    
    # fusionoptions.mask_path = str(mvs_path / "masks")
    
    # pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path, options=fusionoptions)
    os.system(f'python colmap/colmap2mvsnet.py --dense_folder {mvs_path}')
    os.environ['MKL_SERVICE_FORCE_INTEL'] = 'True'
    os.system(f'python colmap/fusion.py {mvs_path}')
    os.system(f'rm -rf {mvs_path}/images_renamed')
    os.system(f'rm -rf {mvs_path}/sparse')
    os.system(f'rm -rf {mvs_path}/stereo')
    os.system(f'ossutil64 cp -r {mvs_path} oss://antsys-vilab/zsz/DL3DV_mvs_fusion_results/{nK}_processed/{scene}/ -u -j 200')
    os.system(f'touch {collect_flag_path}/{scene}.txt')
    ratio = process_directory(mvs_path)
    if ratio > 0.05:
        os.system(f'touch {collect_flag_path}/{scene}_valid.txt')
    os.system(f'rm -rf {mvs_path}')
    time.sleep(1)

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
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    # 获取当前的 LD_LIBRARY_PATH
    current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")

    # 更新 LD_LIBRARY_PATH
    new_ld_library_path = f"{current_ld_library_path}:{torch_lib_path}" if current_ld_library_path else torch_lib_path

    # 设置新的 LD_LIBRARY_PATH
    os.environ["LD_LIBRARY_PATH"] = new_ld_library_path

    # 打印 LD_LIBRARY_PATH，确认是否设置成功
    print(os.environ["LD_LIBRARY_PATH"])
    scenes_file = args.scenes_file
    num_gpus = get_gpu_count()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'Found {num_gpus} GPUs with {gpu_memory:.1f} GB memory each')
    total_cpus = psutil.cpu_count(logical=True)  # 获取逻辑CPU的数量
    cpus_per_gpu = total_cpus // num_gpus
    max_processes_per_gpu = 1
    print(f'Running {max_processes_per_gpu} processes per GPU')
    nK = args.nK
    os.system(f'ossutil64 cp -r oss://antsys-vilab/zsz/DL3DV_txt_fusion/{nK}_fusion/{scenes_file} ./ -u')
    os.system(f'ossutil64 cp -r oss://antsys-vilab/zsz/deeplab/ ./ -j 20 -u')
    with open(scenes_file, 'r') as f:
        scenes = [line.strip().split('/')[-2] for line in f.readlines()]
    
    # 剩余待处理的场景列表
    scenes_left = scenes.copy()
    # 用于跟踪每个GPU上的运行进程
    running_processes = {gpu_id: [] for gpu_id in range(num_gpus)}

    while scenes_left:
        # for gpu_id in range(num_gpus):
            # 检查并移除已完成的进程
            # for p in running_processes[gpu_id][:]:
            #     if not p.is_alive():
            #         p.join()
            #         running_processes[gpu_id].remove(p)

            # 如果GPU上的进程少于10个，且有剩余的场景，分配新的场景
        scene = scenes_left.pop(0)
        os.system(f'ossutil64 cp -r oss://antsys-vilab/zsz/DL3DV_mvs_results/{nK}_processed/{scene}/ {local_path}/{nK}_processed/{scene}/ -u -j 200')
        local_mvs_path = Path(f"{local_path}/{nK}_processed/{scene}" + "/undistort_and_depth_debug0")
        local_mvs_last_bin = f"{local_path}/{nK}_processed/{scene}" + "/undistort_and_depth_debug0/mvs/stereo/depth_maps/*.bin"
        local_mvs_last = f"{local_path}/{nK}_processed/{scene}" + "/undistort_and_depth_debug0/mvs/images/*.png"
        local_mvs_last_num = len(glob.glob(local_mvs_last_bin))//2
        local_png_list = len(glob.glob(local_mvs_last))

        local_valid =  abs(local_mvs_last_num - local_png_list) < 2 and (local_mvs_last_num > 0)
        save_mvs_path = Path(f"/{nas_path}/zsz/DL3DV/{nK}_processed/{scene}" + "/undistort_and_depth_debug0")
        # local_complete = os.path.exists(str(local_mvs_path) + "/success.txt")
        
        local_flag = local_valid #and local_complete
        finish_flag = f'{collect_flag_path}/{scene}.txt'
        if local_flag and not os.path.exists(finish_flag):
            try:
                run_stereo(nK, scene, 0)
            except:
                print(f"Scene {scene} failed, skipping",flush = True)
        elif not local_flag:
            time.sleep(0.1)
            print(f"Scene {scene} is not valid, skipping",flush = True)
            print(f"Not valid: {local_mvs_last_num} {local_png_list}!!!!!!!!!!!!!!!!!!!!",flush = True)
            os.system(f'rm -rf {local_path}/{nK}_processed/{scene}')
        else:
            time.sleep(0.1)
            print(f"Scene {scene} is already processed, skipping",flush = True)
            os.system(f'rm -rf {local_path}/{nK}_processed/{scene}')
            # p = multiprocessing.Process(target=run_stereo, args=(nK, scene, gpu_id))
            # p.start()
            # running_processes[gpu_id].append(p)
                    
            # 等待一段时间再检查（避免过于频繁的检查）

    # 确保所有进程都已完成
    for gpu_id in range(num_gpus):
        for p in running_processes[gpu_id]:
            p.join()