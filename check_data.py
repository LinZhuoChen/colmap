import shutil
import glob
import os
import argparse
from multiprocessing import Pool,Process
import json
import torch

nas_path = 'input1_my'
local_path = '/root/code/zsz/DL3DV'
collect_flag_path = f"/{nas_path}/zsz/DL3DV/json/"
collect_data_path = f"/{nas_path}/datasets/DL3DV_dust3r"

def move_single_file(file_src_dst):
    src_file, dest_dir = file_src_dst
    # 确保目标目录存在
    if not os.path.exists(dest_dir):
        os.system(f'mkdir -p {dest_dir}')
    # 移动文件到目标目录
    target_file = os.path.join(dest_dir, os.path.basename(src_file))
    if not os.path.exists(target_file):
        shutil.move(src_file, dest_dir)

def gpu_task():
    # 确保使用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建大矩阵以确保计算负载足够大（约 60 GB）
    matrix_size = 86848  # 这个大小可以占用约 60GB 内存
    A = torch.randn(matrix_size, matrix_size, device=device)
    B = torch.randn(matrix_size, matrix_size, device=device)

    # 无限循环执行矩阵乘法，增加GPU利用率
    while True:
        # 执行矩阵乘法
        C = torch.matmul(A, B)

        # 可选：为了防止内存溢出，可以同步 GPU 操作
        torch.cuda.synchronize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes_file', type=str, required=True, help='Path to the scenes .txt file')
    parser.add_argument('--nK', type=str, required=True, help='Number of something (nK parameter)')
    os.system(f'mkdir /{nas_path} && chmod 777 -R /{nas_path} &&  mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport alipayshnas-044-kak35.cn-shanghai-eu13-a01.nas.aliyuncs.com:/ /{nas_path}')
    os.system(f'mkdir -p {collect_flag_path}')
    args = parser.parse_args()
    scenes_file = args.scenes_file
    nK = args.nK

    scenes_left = scenes.copy()
    while scenes_left:
        scene = scenes_left.pop(0)
        complete_flag = f'{collect_flag_path}/{scene}.json'
        save_path = f'{collect_data_path}/{scene}/'
        os.makedirs(os.path.dirname(complete_flag), exist_ok=True)
        if os.path.exists(complete_flag) == False:
            scene_local_path = f'{local_path}/{nK}_processed/{scene}/'
            os.system(f'ossutil64 cp -r oss://antsys-vilab/zsz/DL3DV_mvs_fusion_results/{nK}_processed/{scene}/ {scene_local_path} -u -j 200')
            local_image_path = sorted(glob.glob(os.path.join(scene_local_path, "images", "*.png")))
            local_depth_path = sorted(glob.glob(os.path.join(scene_local_path, "filter", "depth", "*.npy")))
            local_mask_path = sorted(glob.glob(os.path.join(scene_local_path, "filter", "mask", "*_final.png")))
            local_cam_path = sorted(glob.glob(os.path.join(scene_local_path, "cams", "*.txt")))
            # 定义文件的源路径和目标路径对
            if len(local_image_path) == len(local_depth_path) == len(local_mask_path) == len(local_cam_path):
                print(f"Scene {scene} has {len(local_image_path)} images")
            else:
                print(f"Scene {scene} has different number of images, depth, mask and cam files")
                continue
            file_paths = [
                (local_image_path, os.path.join(save_path, "images")),
                (local_depth_path, os.path.join(save_path, "depth")),
                (local_mask_path, os.path.join(save_path, "mask")),
                (local_cam_path, os.path.join(save_path, "cams"))
            ]
            files_to_move = []
            for file_list, dest_dir in file_paths:
                for src_file in file_list:
                    files_to_move.append((src_file, dest_dir))


            # 使用多进程移动文件
            with Pool() as pool:
                pool.map(move_single_file, files_to_move)
            local_image_path = [os.path.basename(p) for p in local_image_path]
            local_depth_path = [os.path.basename(p) for p in local_depth_path]
            local_mask_path = [os.path.basename(p) for p in local_mask_path]
            local_cam_path = [os.path.basename(p) for p in local_cam_path]
            complete_json = {'image': local_image_path, 'depth': local_depth_path, 'mask': local_mask_path, 'cams': local_cam_path}
            
            with open(complete_flag, 'w') as f:
                json.dump(complete_json, f)
            os.system(f'ossutil64 cp -r {complete_flag} oss://antsys-vilab/zsz/DL3DV_mvs_final_results/{nK}_processed/{scene}/')
            os.system(f'rm -rf {scene_local_path}')
