import numpy as np
import cv2

from PIL import Image

from datasets.tof_dataset import *
from utils.utils import rgb2gray, resize_all_images

class IOSDataset(TOFDataset):
    def __init__(
        self,
        args,
        file_endings={
            'tof': 'npy',
            'color': 'npy',
            'depth': 'npy',
            'cams': 'npy'
            },
        ):
        super().__init__(args, file_endings)

    def _read_tof(self, tof_filename):
        return np.zeros([self.args.tof_image_height, self.args.tof_image_width, 3], dtype=np.float32)

    def _read_color(self, color_filename):
        return np.load(color_filename)

    def _get_depth_filename(self, frame_id):
        depth_filename = os.path.join(
                self.args.datadir,
                f'{self.args.scan}/distance/{frame_id:04d}.{self.file_endings["depth"]}' 
                )
        
        return depth_filename

    def _read_depth(self, depth_filename):
        depth_im = np.squeeze(np.load(depth_filename))
        return depth_im

    def _post_process_dataset(self, args):
        #self.dataset['color_intrinsics'][:2, :3] *= 2
        #self.dataset['tof_intrinsics'][:2, :3] *= 2

        gray = rgb2gray(self.dataset['color_images'])
        gray = resize_all_images(
            gray,
            self.dataset['tof_images'].shape[2],
            self.dataset['tof_images'].shape[1]
            )
        gray = np.ones_like(gray)
        self.dataset['depth_images'] = resize_all_images(
            self.dataset['depth_images'],
            self.dataset['tof_images'].shape[2],
            self.dataset['tof_images'].shape[1],
            cv2.INTER_NEAREST
            )

        self.dataset['tof_images'] = tof_from_depth(
            self.dataset['depth_images'], gray, self.dataset['depth_range']
            ).astype(np.float32)
        
        self.dataset['tof_images'] = normalize_im_max(self.dataset['tof_images'])

        self.dataset['tof_depth_images'] = self.dataset['depth_images']