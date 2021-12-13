from __future__ import print_function

import os
import time
import glob
import random
import math
import re
import sys

import cv2
import numpy as np
import scipy.io
import urllib
import matplotlib.pyplot as plt

from PIL import Image
from utils.sampling_utils import *

TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None
    
    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()

def read_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]

    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4))

    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.zeros([4, 4])
    intrinsics_upper = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsics_upper = intrinsics_upper.reshape((3, 3))
    intrinsics[:3, :3] = intrinsics_upper
    intrinsics[3, 3] = 1.

    # # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0])
    return intrinsics, extrinsics

def read_dynamic_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    nr = len(lines)
    mat = np.fromstring(' '.join(lines), dtype=np.float32, sep=' ')
    mat = mat.reshape((nr, -1))
    return mat

def resize_image(img, h, w):
    """ resize image """
    image = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    return image

def center_image(img):
    """ normalize image input """
    return (img.astype(np.float32) / 255.) * 2. - 1.

def uncenter_image(img):
    """ normalize image input """
    return ((img + 1.) / 2.) * 255.

def save_image(img, filepath, uncenter=True):
    img = img.detach().cpu().numpy()
    if uncenter:
        img = uncenter_image(img)
    img = Image.fromarray(img.astype(np.uint8))
    img.save(filepath)

def scale_camera(cam, scale=1):
    """ resize input in order to produce sampled depth map """
    new_cam = np.copy(cam)
    # focal:
    new_cam[0][0] = cam[0][0] * scale
    new_cam[1][1] = cam[1][1] * scale
    # principle point:
    new_cam[0][2] = cam[0][2] * scale
    new_cam[1][2] = cam[1][2] * scale
    return new_cam

def scale_mvs_camera(cams, scale=1):
    """ resize input in order to produce sampled depth map """
    for view in range(FLAGS.view_num):
        cams[view] = scale_camera(cams[view], scale=scale)
    return cams

def scale_image(image, scale=1, interpolation=cv2.INTER_AREA):
    """ resize image using cv2 """
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)

def scale_mvs_input(images, cams, view_num, depths=None, scale=1):
    """ resize input to fit into the memory """
    for view in range(view_num):
        images[view] = scale_image(images[view], scale=scale)

        if not depths is None:
            depths[view] = scale_image(depths[view], scale=scale, interpolation=cv2.INTER_NEAREST)

        cams[view] = scale_camera(cams[view], scale=scale)

    if not depths is None:
        return images, cams, depths
    else:
        return images, cams

def crop_mvs_input(
        images, cams, view_num, depths=None, max_w=0, max_h=0, base_image_size=8
        ):
    """ resize images and cameras to fit the network (can be divided by base image size) """

    # crop images and cameras
    for view in range(view_num):
        h, w = images[view].shape[0:2]
        new_h = h
        new_w = w

        if new_h > max_h:
            new_h = max_h
        else:
            new_h = int(math.ceil(h / base_image_size) * base_image_size)
        if new_w > max_w:
            new_w = max_w
        else:
            new_w = int(math.ceil(w / base_image_size) * base_image_size)

        if max_w > 0:
            new_w = max_w
        if max_h > 0:
            new_h = max_h

        start_h = int(math.ceil((h - new_h) / 2))
        start_w = int(math.ceil((w - new_w) / 2))
        finish_h = start_h + new_h
        finish_w = start_w + new_w
        images[view] = images[view][start_h:finish_h, start_w:finish_w]

        if not depths is None:
            depths[view] = depths[view][start_h:finish_h, start_w:finish_w]

        cams[view][0][2] = cams[view][0][2] - start_w
        cams[view][1][2] = cams[view][1][2] - start_h

    if not depths is None:
        return images, cams, depths
    else:
        return images, cams

def mask_depth_image(depth_image, min_depth, max_depth):
    """ mask out-of-range pixel to zero """
    # print ('mask min max', min_depth, max_depth)
    ret, depth_image = cv2.threshold(depth_image, min_depth, 100000, cv2.THRESH_TOZERO)
    ret, depth_image = cv2.threshold(depth_image, max_depth, 100000, cv2.THRESH_TOZERO_INV)
    depth_image = np.expand_dims(depth_image, 2)
    return depth_image

def load_cam(file, interval_scale=1):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))
    words = file.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = FLAGS.max_d
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = words[30]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

    return cam

def write_cam(file, cam):
    # f = open(file, "w")
    f = file_io.FileIO(file, "w")

    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()

def write_pfm(file, image, scale=1):
    file = file_io.FileIO(file, mode='wb')
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image_string = image.tostring()
    file.write(image_string)

    file.close()

def gen_dtu_resized_path(dtu_data_folder, mode='training'):
    """ generate data paths for dtu dataset """
    sample_list = []

    # parse camera pairs
    cluster_file_path = dtu_data_folder + '/Cameras/pair.txt'

    # cluster_list = open(cluster_file_path).read().split()
    cluster_list = file_io.FileIO(cluster_file_path, mode='r').read().split()

    # 3 sets
    training_set = [2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
                    45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
                    74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                    101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
                    121, 122, 123, 124, 125, 126, 127, 128]
    validation_set = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]

    data_set = []
    if mode == 'training':
        data_set = training_set
    elif mode == 'validation':
        data_set = validation_set

    # for each dataset
    for i in data_set:

        image_folder = os.path.join(dtu_data_folder, ('Rectified/scan%d_train' % i))
        cam_folder = os.path.join(dtu_data_folder, 'Cameras/train')
        depth_folder = os.path.join(dtu_data_folder, ('Depths/scan%d_train' % i))

        if mode == 'training':
            # for each lighting
            for j in range(0, 7):
                # for each reference image
                for p in range(0, int(cluster_list[0])):
                    paths = []
                    # ref image
                    ref_index = int(cluster_list[22 * p + 1])
                    ref_image_path = os.path.join(
                        image_folder, ('rect_%03d_%d_r5000.png' % ((ref_index + 1), j)))
                    ref_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % ref_index))
                    paths.append(ref_image_path)
                    paths.append(ref_cam_path)
                    # view images
                    for view in range(FLAGS.view_num - 1):
                        view_index = int(cluster_list[22 * p + 2 * view + 3])
                        view_image_path = os.path.join(
                            image_folder, ('rect_%03d_%d_r5000.png' % ((view_index + 1), j)))
                        view_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % view_index))
                        paths.append(view_image_path)
                        paths.append(view_cam_path)
                    # depth path
                    depth_image_path = os.path.join(depth_folder, ('depth_map_%04d.pfm' % ref_index))
                    paths.append(depth_image_path)
                    sample_list.append(paths)
        elif mode == 'validation':
            j = 3
            # for each reference image
            for p in range(0, int(cluster_list[0])):
                paths = []
                # ref image
                ref_index = int(cluster_list[22 * p + 1])
                ref_image_path = os.path.join(
                    image_folder, ('rect_%03d_%d_r5000.png' % ((ref_index + 1), j)))
                ref_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % ref_index))
                paths.append(ref_image_path)
                paths.append(ref_cam_path)
                # view images
                for view in range(FLAGS.view_num - 1):
                    view_index = int(cluster_list[22 * p + 2 * view + 3])
                    view_image_path = os.path.join(
                        image_folder, ('rect_%03d_%d_r5000.png' % ((view_index + 1), j)))
                    view_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % view_index))
                    paths.append(view_image_path)
                    paths.append(view_cam_path)
                # depth path
                depth_image_path = os.path.join(depth_folder, ('depth_map_%04d.pfm' % ref_index))
                paths.append(depth_image_path)
                sample_list.append(paths)

    return sample_list

def gen_dtu_mvs_path(dtu_data_folder, mode='training'):
    """ generate data paths for dtu dataset """
    sample_list = []

    # parse camera pairs
    cluster_file_path = dtu_data_folder + '/Cameras/pair.txt'
    cluster_list = open(cluster_file_path).read().split()

    # 3 sets
    training_set = [2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
                    45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
                    74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                    101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
                    121, 122, 123, 124, 125, 126, 127, 128]
    validation_set = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]
    evaluation_set = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77,
                      110, 114, 118]

    # for each dataset
    data_set = []
    if mode == 'training':
        data_set = training_set
    elif mode == 'validation':
        data_set = validation_set
    elif mode == 'evaluation':
        data_set = evaluation_set

    # for each dataset
    for i in data_set:

        image_folder = os.path.join(dtu_data_folder, ('Rectified/scan%d' % i))
        cam_folder = os.path.join(dtu_data_folder, 'Cameras')
        depth_folder = os.path.join(dtu_data_folder, ('Depths/scan%d' % i))

        if mode == 'training':
            # for each lighting
            for j in range(0, 7):
                # for each reference image
                for p in range(0, int(cluster_list[0])):
                    paths = []
                    # ref image
                    ref_index = int(cluster_list[22 * p + 1])
                    ref_image_path = os.path.join(
                        image_folder, ('rect_%03d_%d_r5000.png' % ((ref_index + 1), j)))
                    ref_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % ref_index))
                    paths.append(ref_image_path)
                    paths.append(ref_cam_path)
                    # view images
                    for view in range(FLAGS.view_num - 1):
                        view_index = int(cluster_list[22 * p + 2 * view + 3])
                        view_image_path = os.path.join(
                            image_folder, ('rect_%03d_%d_r5000.png' % ((view_index + 1), j)))
                        view_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % view_index))
                        paths.append(view_image_path)
                        paths.append(view_cam_path)
                    # depth path
                    depth_image_path = os.path.join(depth_folder, ('depth_map_%04d.pfm' % ref_index))
                    paths.append(depth_image_path)
                    sample_list.append(paths)
        else:
            # for each reference image
            j = 5
            for p in range(0, int(cluster_list[0])):
                paths = []
                # ref image
                ref_index = int(cluster_list[22 * p + 1])
                ref_image_path = os.path.join(
                    image_folder, ('rect_%03d_%d_r5000.png' % ((ref_index + 1), j)))
                ref_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % ref_index))
                paths.append(ref_image_path)
                paths.append(ref_cam_path)
                # view images
                for view in range(FLAGS.view_num - 1):
                    view_index = int(cluster_list[22 * p + 2 * view + 3])
                    view_image_path = os.path.join(
                        image_folder, ('rect_%03d_%d_r5000.png' % ((view_index + 1), j)))
                    view_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % view_index))
                    paths.append(view_image_path)
                    paths.append(view_cam_path)
                # depth path
                depth_image_path = os.path.join(depth_folder, ('depth_map_%04d.pfm' % ref_index))
                paths.append(depth_image_path)
                sample_list.append(paths)

    return sample_list

def gen_blendedmvs_path(blendedmvs_data_folder, mode='training_mvs'):
    """ generate data paths for blendedmvs dataset """

    # read data list
    if mode == 'training_mvs':
        proj_list = open(os.path.join(blendedmvs_data_folder, 'BlendedMVS_training.txt')).read().splitlines()
    elif mode == 'training_mvg':
        proj_list = open(os.path.join(blendedmvs_data_folder, 'BlendedMVG_training.txt')).read().splitlines()
    elif mode == 'validation':
        proj_list = open(os.path.join(blendedmvs_data_folder, 'validation_list.txt')).read().splitlines()

    # parse all data
    mvs_input_list = []
    for data_name in proj_list:

        dataset_folder = os.path.join(blendedmvs_data_folder, data_name)

        # read cluster
        cluster_path = os.path.join(dataset_folder, 'cams', 'pair.txt')
        cluster_lines = open(cluster_path).read().splitlines()
        image_num = int(cluster_lines[0])

        # get per-image info
        for idx in range(0, image_num):
            ref_idx = int(cluster_lines[2 * idx + 1])
            cluster_info =  cluster_lines[2 * idx + 2].split()
            total_view_num = int(cluster_info[0])

            if total_view_num < FLAGS.view_num - 1:
                continue

            paths = []

            ref_image_path = os.path.join(dataset_folder, 'blended_images', '%08d_masked.jpg' % ref_idx)
            ref_depth_path = os.path.join(dataset_folder, 'rendered_depth_maps', '%08d.pfm' % ref_idx)
            ref_cam_path = os.path.join(dataset_folder, 'cams', '%08d_cam.txt' % ref_idx)
            paths.append(ref_image_path)
            paths.append(ref_cam_path)
            paths.append(ref_depth_path)

            for cidx in range(0, FLAGS.view_num - 1):
                view_idx = int(cluster_info[2 * cidx + 1])
                view_image_path = os.path.join(dataset_folder, 'blended_images', '%08d_masked.jpg' % view_idx)
                views_depth_path = os.path.join(dataset_folder, 'rendered_depth_maps', '%08d.pfm' % view_idx)
                view_cam_path = os.path.join(dataset_folder, 'cams', '%08d_cam.txt' % view_idx)
                paths.append(view_image_path)
                paths.append(view_cam_path)
                paths.append(view_depth_path)

            mvs_input_list.append(paths)

    return mvs_input_list

def gen_eth3d_path(eth3d_data_folder, mode='training'):
    """ generate data paths for eth3d dataset """

    sample_list = []

    data_names = []
    if mode == 'training':
        data_names = ['delivery_area', 'electro', 'forest']
    elif mode == 'validation':
        data_names = ['playground', 'terrains']

    for data_name in data_names:

        data_folder = os.path.join(eth3d_data_folder, data_name)

        image_folder = os.path.join(data_folder, 'images')
        depth_folder = os.path.join(data_folder, 'depths')
        cam_folder = os.path.join(data_folder, 'cams')

        # index to image name
        index2name = dict()
        dict_file = os.path.join(cam_folder,'index2prefix.txt')
        dict_list = file_io.FileIO(dict_file, mode='r').read().split()
        dict_size = int(dict_list[0])
        for i in range(0, dict_size):
            index = int(dict_list[2 * i + 1])
            name = str(dict_list[2 * i + 2])
            index2name[index] = name

        # image name to depth name 
        name2depth = dict()
        name2depth['images_rig_cam4_undistorted'] = 'images_rig_cam4'
        name2depth['images_rig_cam5_undistorted'] = 'images_rig_cam5'
        name2depth['images_rig_cam6_undistorted'] = 'images_rig_cam6'
        name2depth['images_rig_cam7_undistorted'] = 'images_rig_cam7'

        # cluster
        cluster_file = os.path.join(cam_folder,'pair.txt')
        cluster_list = file_io.FileIO(cluster_file, mode='r').read().split()
        for p in range(0, int(cluster_list[0])):
            paths = []

            # ref image
            ref_index = int(cluster_list[22 * p + 1])
            ref_image_name = index2name[ref_index]
            ref_image_path = os.path.join(image_folder, ref_image_name)
            ref_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % ref_index))
            paths.append(ref_image_path)
            paths.append(ref_cam_path)

            # view images
            for view in range(FLAGS.view_num - 1):
                view_index = int(cluster_list[22 * p + 2 * view + 3])
                view_image_name = index2name[view_index]
                view_image_path = os.path.join(image_folder, view_image_name)
                view_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % view_index))
                paths.append(view_image_path)
                paths.append(view_cam_path)

            # depth path
            image_prefix = os.path.split(ref_image_name)[1]
            depth_sub_folder = name2depth[os.path.split(ref_image_name)[0]]
            ref_depth_name = os.path.join(depth_sub_folder, image_prefix)
            ref_depth_name = os.path.splitext(ref_depth_name)[0] + '.pfm'
            depth_image_path = os.path.join(depth_folder, ref_depth_name)
            paths.append(depth_image_path)
            sample_list.append(paths)

    return sample_list

def gen_mvs_list(mode='training'):
    """output paths in a list: [[I1_path1,  C1_path, I2_path, C2_path, ...(, D1_path)], [...], ...]"""
    sample_list = []

    if FLAGS.train_dtu:
        dtu_sample_list = gen_dtu_mvs_path(FLAGS.dtu_data_root, mode=mode)
        sample_list = sample_list + dtu_sample_list

    return sample_list

# for testing
def gen_pipeline_mvs_list(dense_folder):
    """ mvs input path list """
    image_folder = os.path.join(dense_folder, 'images')
    cam_folder = os.path.join(dense_folder, 'cams')
    cluster_list_path = os.path.join(dense_folder, 'pair.txt')
    cluster_list = open(cluster_list_path).read().split()

    # for each dataset
    mvs_list = []
    pos = 1
    for i in range(int(cluster_list[0])):
        paths = []
        # ref image
        ref_index = int(cluster_list[pos])
        pos += 1
        ref_image_path = os.path.join(image_folder, ('%08d.jpg' % ref_index))
        ref_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % ref_index))
        paths.append(ref_image_path)
        paths.append(ref_cam_path)
        # view images
        all_view_num = int(cluster_list[pos])
        pos += 1
        check_view_num = min(FLAGS.view_num - 1, all_view_num)
        for view in range(check_view_num):
            view_index = int(cluster_list[pos + 2 * view])
            view_image_path = os.path.join(image_folder, ('%08d.jpg' % view_index))
            view_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % view_index))
            paths.append(view_image_path)
            paths.append(view_cam_path)
        pos += 2 * all_view_num
        # depth path
        mvs_list.append(paths)
    return mvs_list

def get_extrinsics(extrinsics_file, args, default_exts=None):
    if args.fix_view:
        return np.stack(
            [np.eye(4) for i in range(args.total_num_views)],
            axis=0
            )

    # Default if path does not exist
    if default_exts is None:
        default_exts = np.stack(
            [np.eye(4) for i in range(args.total_num_views)],
            axis=0
            )

    # Extrinsics
    extrinsics_file = os.path.join(extrinsics_file)

    if os.path.exists(extrinsics_file):
        extrinsics_data = np.load(extrinsics_file)
    else:
        extrinsics_data = default_exts

    return extrinsics_data

def get_camera_params(intrinsics_file, extrinsics_file, args, default_exts=None):
    if '.mat' in intrinsics_file:
        K = scipy.io.loadmat(intrinsics_file)['K']
    else:
        K = np.load(intrinsics_file)

    return K, get_extrinsics(extrinsics_file, args, default_exts)

def reprojection_test(
    image,
    ray_gen_fn,
    project_fn,
    relative_pose,
    distance
    ):
    rays_o, rays_d = ray_gen_fn(np.eye(4))
    rays_o = tf.cast(rays_o, tf.float32)
    rays_d = tf.math.l2_normalize(tf.cast(rays_d, tf.float32), axis=-1)
    relative_pose = tf.cast(relative_pose, tf.float32)

    points = rays_o + rays_d * distance
    new_distance = tf.linalg.norm(points, axis=-1)
    pixels = project_fn(points, relative_pose)

    interp_image = interpolate_image(
        image[None],
        pixels[None],
        shifts_x=[0],
        shifts_y=[0]
        )
    
    return interp_image[0], new_distance

def recenter_poses(poses):
    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses, np.linalg.inv(c2w)