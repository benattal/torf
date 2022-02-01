import numpy as np
import cv2

from PIL import Image

from datasets.tof_dataset import *

from utils.utils import rgb2gray, resize_all_images

class RealDataset(ToFDataset):
    def __init__(
        self,
        args,
        file_endings={
            'tof': 'npy',
            'color': 'npy',
            'depth': 'npy',
            'cams': 'mat'
            },
        ):
        super().__init__(args, file_endings)

    def _read_color(self, color_filename):
        return np.load(color_filename)

    def _read_depth(self, depth_filename):
        return np.load(depth_filename)

    def _read_tof(self, tof_filename):
        return np.load(tof_filename)

    def _process_camera_params(self, args):
        super()._process_camera_params(args)

        # Relative pose
        relative_R_path = os.path.join(args.datadir, f'{args.scan}/cams/relative_R.mat')
        relative_T_path = os.path.join(args.datadir, f'{args.scan}/cams/relative_T.mat')

        if os.path.exists(relative_R_path) and os.path.exists(relative_T_path):
            R = scipy.io.loadmat(relative_R_path)['R']
            T = np.squeeze(scipy.io.loadmat(relative_T_path)['T'])

            E = np.eye(4)
            E[:3, :3] = R
            E[:3, -1] = T
            E = np.linalg.inv(E)

            twist = np.array(se3_vee(E))
            twist[0] = -twist[0]
            twist[1] = -twist[1]
            twist[2] = -twist[2]
            E = np.array(se3_hat(twist))

            self.dataset['relative_pose'] = E

        # Phase offset
        phase_offset_path = os.path.join(args.datadir, f'{args.scan}/cams/phase_offset.mat')

        if os.path.exists(phase_offset_path):
            phase_offset = scipy.io.loadmat(phase_offset_path)['P']
            self.dataset['phase_offset'] = phase_offset

        # Depth range
        depth_range_path = os.path.join(args.datadir, f'{args.scan}/cams/depth_range.mat')

        if os.path.exists(depth_range_path) and args.depth_range < 0:
            depth_range = scipy.io.loadmat(depth_range_path)['depth_range']
            self.dataset['depth_range'] = np.array(depth_range).astype(np.float32)
        else:
            self.dataset['depth_range'] = np.array(args.depth_range).astype(np.float32)