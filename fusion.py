import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time

import sys
import cv2
from plyfile import PlyData, PlyElement
from PIL import Image
import multiprocessing

from multiprocessing import Pool
import cv2
from functools import partial

cudnn.benchmark = True

# parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse. May be different from the original implementation')
# parser.add_argument('--model', default='mvsnet', help='select model')

# parser.add_argument('--dataset', default='dtu_yao_eval', help='select dataset')
# parser.add_argument('--testpath', default='/data1/wzz/tnt/',help='testing data path')
# parser.add_argument('--testlist', default='/data1/wzz/test_tnt.txt',help='testing scan list')

# parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
# parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
# parser.add_argument('--interval_scale', type=float, default=1.06, help='the depth interval scale')

# parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
# parser.add_argument('--outdir', default='/data1/wzz/outputs_tnt_1101', help='output dir')
# parser.add_argument('--display', action='store_true', help='display depth images and masks')

# parser.add_argument('--test_dataset', default='tanks', help='which dataset to evaluate')

# parse arguments and check
# args = parser.parse_args()
# print("argv:", sys.argv[1:])
# print_args(args)


# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size

    # intrinsics[:2, :] *= scale

    # if (flag==0):
    #     intrinsics[0,2]-=index
    # else:
    #     intrinsics[1,2]-=index
  
    return intrinsics, extrinsics


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    # assert mask.dtype == np.bool_
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            src_views = src_views[:10]
            data.append((ref_view, src_views))
            assert ref_view != 0
    return data

def read_score_file(filename):
    data=[]
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            scores = [float(x) for x in f.readline().rstrip().split()[2::2]]
            data.append(scores)
    return data

# run MVS model to save depth maps and confidence maps
def save_depth():
    # dataset, dataloader

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
    model.eval()

    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            sample_cuda = tocuda(sample)
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
            outputs = tensor2numpy(outputs)
            del sample_cuda
            print('Iter {}/{}'.format(batch_idx, len(TestImgLoader)))
            filenames = sample["filename"]

            # save depth maps and confidence maps
            for filename, depth_est, photometric_confidence in zip(filenames, outputs["depth"],
                                                                   outputs["photometric_confidence"]):
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                # save depth maps
                save_pfm(depth_filename, depth_est)
                # save confidence maps
                save_pfm(confidence_filename, photometric_confidence)


def depth_to_visual(depth_est_averaged, final_mask):
    depth_float = depth_est_averaged.astype(np.float32) * final_mask.astype(np.uint8)
    depth_float = np.log(depth_float + 1)
    depth_normalized = cv2.normalize(depth_float, None, 0, 255, cv2.NORM_MINMAX)
    depth_color = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    return depth_color


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_NEAREST)
    # sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_and_photometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, img_ref,
                      depth_src, intrinsics_src, extrinsics_src, img_src,
                      dist_threshold=1, depth_diff_threshold=0.0075, color_diff_threshold=0.1):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    
    # Reproject depth and calculate new coordinates
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(
        depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src)
    
    # Reproject pixels using depth_src, x2d_src, y2d_src
    map_x, map_y = x2d_src.astype(np.float32), y2d_src.astype(np.float32)
    reprojected_img = cv2.remap(img_src, map_x, map_y, cv2.INTER_LINEAR)
    
    # cv2.imwrite("reprojected_img.png", (reprojected_img*255).astype(np.uint8))
    
    # Geometric consistency checks
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref
    geo_mask = np.logical_and(dist < dist_threshold, relative_depth_diff < depth_diff_threshold)
    
    if geo_mask.mean() < 0.1:
        geo_mask = np.logical_and(dist < dist_threshold, relative_depth_diff < np.quantile(relative_depth_diff[relative_depth_diff!=0], 0.2))

    # Photometric consistency checks
    color_diff = np.linalg.norm(reprojected_img - img_ref, axis=2)  # Assuming img_ref and reprojected_img are normalized
    color_mask = color_diff < color_diff_threshold
    
    # Combine geometric and photometric masks
    combined_mask = np.logical_and(geo_mask, color_mask)
    depth_reprojected[~combined_mask] = 0
    
    return combined_mask, geo_mask, color_mask, depth_reprojected, x2d_src, y2d_src


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def process_view(pair, scan_folder, out_folder):
    ref_view, src_views = pair
    ref_view_colmap, src_views_colmap = ref_view , [x for x in src_views]
    
    # ct2 += 1
    # load the reference image
    ref_img = read_img(os.path.join(scan_folder, 'images/frame_{:0>5}.png'.format(ref_view_colmap)))
    # load the estimated depth of the reference view
    ref_depth_est = read_array(os.path.join(scan_folder, 'stereo/depth_maps/frame_{:0>5}.png.geometric.bin'.format(ref_view_colmap)))
    # load the photometric mask of the reference view
    # confidence = read_array(os.path.join(scan_folder, 'stereo/depth_maps/frame_{:0>5}.png.geometric.bin'.format(ref_view_colmap)))
    # load the camera parameters
    ref_intrinsics, ref_extrinsics = read_camera_parameters(
        os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
    
    # photo_mask = ref_depth_est>0 #confidence > np.quantile(confidence, 0.2) #photo_threshold
    
    
    all_srcview_depth_ests = []
    all_srcview_x = []
    all_srcview_y = []
    all_srcview_geomask = []
    # compute the geometric mask
    geo_mask_sum = 0
    geo_mask_sums=[]
    
    color_mask_sum = 0
    color_mask_sums = []
    
    combined_mask_sum = 0
    combined_mask_sums = []
    
    n=1
    for src_view in src_views:
        n+=1
    ct = 0
    
    for src_view, src_view_colmap in zip(src_views, src_views_colmap):
            ct = ct + 1
            # camera parameters of the source view
            # NOTE!!!!!!!!!!!!!!!!
            # NOTE!!!!!!!!!!!!!!!!
            # NOTE!!!!!!!!!!!!!!!!
            # NOTE!!!!!!!!!!!!!!!!
            # NOTE!!!!!!!!!!!!!!!!
            # NOTE!!!!!!!!!!!!!!!!
            # NOTE!!!!!!!!!!!!!!!!
            # NOTE!!!!!!!!!!!!!!!!
            # Here should be src_view instead of src_view_colmap
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            # NOTE!!!!!!!!!!!!!!!!
            # NOTE!!!!!!!!!!!!!!!!
            # NOTE!!!!!!!!!!!!!!!!
            # NOTE!!!!!!!!!!!!!!!!
            # NOTE!!!!!!!!!!!!!!!!
            
            
            
            # the estimated depth of the source view
            src_depth_est = read_array(os.path.join(scan_folder, 'stereo/depth_maps/frame_{:0>5}.png.geometric.bin'.format(src_view_colmap)))
            src_img = read_img(os.path.join(scan_folder, 'images/frame_{:0>5}.png'.format(src_view_colmap)))

            #src_depth_est=cv2.pyrUp(src_depth_est)
            # src_depth_est=cv2.pyrUp(src_depth_est)

            src_confidence = read_array(os.path.join(scan_folder, 'stereo/depth_maps/frame_{:0>5}.png.geometric.bin'.format(src_view_colmap)))
            combined_mask, geo_mask, color_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_and_photometric_consistency(ref_depth_est,       
                                                                                        ref_intrinsics,
                                                                                        ref_extrinsics,
                                                                                        ref_img,
                                                                                        src_depth_est,
                                                                                        src_intrinsics, 
                                                                                        src_extrinsics,
                                                                                        src_img)
            combined_mask_sum+=combined_mask.astype(np.int32)
            geo_mask_sum+=geo_mask.astype(np.int32)
            color_mask_sum+=color_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)


    geo_mask=geo_mask_sum>=len(src_views)//2
    color_mask=color_mask_sum>=len(src_views)//2
    combined_mask=combined_mask_sum>=len(src_views)//2
    
    # for i in range (2,n):
    #     geo_mask=np.logical_and(geo_mask,geo_mask_sums[i-2]>=i)
    #     print(geo_mask.mean())

    depth_est_averaged_window = (sum(all_srcview_depth_ests) + ref_depth_est) / (combined_mask_sum + 1)
    
    if False:
        combined_mask = combined_mask & (depth_est_averaged<np.quantile(depth_est_averaged, 0.95))
        combined_mask = cv2.erode(combined_mask.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1)
    
    if (not isinstance(geo_mask, bool)):
        
        sky_mask_path = os.path.join(scan_folder, "masks/frame_{:0>5}.png.png".format(ref_view_colmap))
        if os.path.exists(sky_mask_path):
            sky_mask = read_mask(sky_mask_path)
            final_mask = combined_mask & sky_mask
        else:
            final_mask = combined_mask
        
        
        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        os.makedirs(os.path.join(out_folder, "depth"), exist_ok=True)
        # os.makedirs(os.path.join(out_folder, "depth_window_avg"), exist_ok=True)


        np.save(os.path.join(out_folder, "depth/{:0>5}.npy".format(ref_view_colmap)), ref_depth_est.astype(np.float32))
        depth_color = depth_to_visual(ref_depth_est, final_mask)
        cv2.imwrite(os.path.join(out_folder, "depth/{:0>5}.png".format(ref_view_colmap)), depth_color)

        # np.save(os.path.join(out_folder, "depth_window_avg/{:0>5}.npy".format(ref_view_colmap)), depth_est_averaged_window.astype(np.float32))        
        # depth_window_avg_color = depth_to_visual(depth_est_averaged_window, final_mask)
        # cv2.imwrite(os.path.join(out_folder, "depth_window_avg/{:0>5}.png".format(ref_view_colmap)), depth_window_avg_color)

        save_mask(os.path.join(out_folder, "mask/{:0>5}_raw.png".format(ref_view_colmap)), ref_depth_est>0)
        save_mask(os.path.join(out_folder, "mask/{:0>5}_geo.png".format(ref_view_colmap)),  geo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>5}_color.png".format(ref_view_colmap)), color_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>5}_final.png".format(ref_view_colmap)), final_mask)

        print("processing {}, ref-view{:0>2}, geo/color/final-mask:/{}/{}/{}".format(scan_folder, ref_view,
                                                                                    geo_mask.mean(),
                                                                                    color_mask.mean(),
                                                                                    final_mask.mean()))


    # if False:
    height, width = ref_depth_est.shape[:2]
    x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
    # valid_points = np.logical_and(final_mask, ~used_mask[ref_view])
    valid_points = final_mask
    print("valid_points", valid_points.mean())
    x, y, depth = x[valid_points], y[valid_points], ref_depth_est[valid_points]
    color = ref_img[:, :, :][valid_points]  # hardcoded for DTU dataset
    xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                        np.vstack((x, y, np.ones_like(x))) * depth)
    xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                            np.vstack((xyz_ref, np.ones_like(x))))[:3]
    
    return (xyz_world.transpose((1, 0)), (color * 255).astype(np.uint8), ref_depth_est)
    
    # return True


def filter_depth(scan_folder, out_folder):
    # the pair file
    pair_file = os.path.join(scan_folder, "pair.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)
    score_data = read_score_file(pair_file)

    nviews = len(pair_data)

    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Map the processing function to the data points
        # results = pool.map(process_view, pair_data, scan_folder, out_folder)
        process_view_partial = partial(process_view, scan_folder=scan_folder, out_folder=out_folder)
        # Map the processing function to the data points
        results = pool.map(process_view_partial, pair_data)

    # process_view(pair_data[-1], scan_folder, out_folder)

    
    # for i in range(len(pair_data)):
    #     process_view(pair_data[i])
    
    
    # with Pool(processes=os.cpu_count()) as pool: 
    #     results = pool.map(process_view, pair_data)
    # process_view(pair_data[0])
    # for ref_view, src_views in pair_data:
    vertexs = []
    vertex_colors = []
    depths = []
    for (xyz_world, color, depth_est_averaged) in results:
        vertexs.append(xyz_world)
        vertex_colors.append(color)
        depths.append(depth_est_averaged)
    vertexs = np.concatenate(vertexs, axis=0)[::16]
    # vertexs_norm = np.linalg.norm(vertexs, axis=1)
    # mask = vertexs_norm < np.quantile(vertexs_norm, 0.9)
    vertex_colors = np.concatenate(vertex_colors, axis=0)[::16]
    vertexs = torch.from_numpy(vertexs).float()
    vertex_colors = torch.from_numpy(vertex_colors).float()
    # import ipdb; ipdb.set_trace()
    import nerfvis.scene as scene_vis
    scene_vis.set_title("My Scene")
    scene_vis.set_opencv()
    scene_vis.add_points("points", vertexs, vert_color=vertex_colors/255.)
    # scene_vis.display(port=8180)
    scene_vis.export(scan_folder, embed_output = True)

    return True

if __name__ == '__main__':
    # step1. save all the depth maps and the masks in outputs directory
    # save_depth()

    # scan_folder = args.testpath
    # out_folder = args.outdir
    
    scan_folder = sys.argv[1]
    #"/data/home/jianyuan/datasets/DL3DV/debug_sky/maskfusion/undistort_and_depth_debug0/mvs"
    out_folder = scan_folder + "/filter"
    filter_depth(scan_folder, out_folder)
    
    # step2. filter saved depout_folderth maps with photometric confidence maps and geometric constraints

    # photo_threshold=0.1
    # filter_depth(scan_folder, out_folder, os.path.join(out_folder, 'out.ply'), photo_threshold)
    
    
    
    