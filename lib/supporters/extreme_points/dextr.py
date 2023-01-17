import numpy as np
import cv2
import yaml
import os
from collections import OrderedDict

import torch
from torch.nn.functional import upsample

from ..general import COLOR
from .networks import deeplab_resnet as resnet
from . import helpers


class DEXTR:
    def __init__(self, img_size, **kwargs):
        self.img_size = img_size

        self.point_color = COLOR.YELLOW
        self.point_border_color = COLOR.BLACK
        self.select_color = COLOR.GREEN  # color for current foreground

    def mouse_cb(self,event,x,y,flags,**kwargs):
        # Draw points
        self.img_vis = self.img.copy()
        for point in self.current_points_list:
            cv2.circle(self.img_vis, tuple(point), 6, self.point_border_color, -1)
            cv2.circle(self.img_vis, tuple(point), 5, self.point_color, -1)

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_num_points < 4:  # if do not have enough 4 extreme points
                self.current_num_points += 1  # add one more point
                self.current_points_list.append([x, y])

    def get_segments_map_from_extreme_points(self,):
        extreme_points_ori = np.array(self.current_points_list)

        # Crop image to the bounding box from the extreme points and resize
        bbox = helpers.get_bbox(self.img, points=extreme_points_ori, pad=self.pad, zero_pad=True)
        crop_image = helpers.crop_from_bbox(self.img, bbox, zero_pad=True)
        resize_image = helpers.fixed_resize(crop_image, (512, 512)).astype(np.float32)

        #  Generate extreme point heat map normalized to image values
        extreme_points = extreme_points_ori - [np.min(extreme_points_ori[:, 0]), np.min(extreme_points_ori[:, 1])] + [self.pad,self.pad]
        extreme_points = (512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(int)
        extreme_heatmap = helpers.make_gt(resize_image, extreme_points, sigma=10)
        extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)

        # Concatenate inputs and convert to tensor
        input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)
        inputs = torch.from_numpy(input_dextr.transpose((2, 0, 1))[np.newaxis, ...])

        # Run a forward pass
        inputs = inputs.to(self.device)
        outputs = self.net.forward(inputs)
        outputs = upsample(outputs, size=(512, 512), mode='bilinear', align_corners=True)
        outputs = outputs.to(torch.device('cpu'))

        pred = np.transpose(outputs.data.numpy()[0, ...], (1, 2, 0))
        pred = 1 / (1 + np.exp(-pred))
        pred = np.squeeze(pred)

        result = helpers.crop2fullmask(pred, bbox, im_size=self.img.shape[:2], zero_pad=True, relax=self.pad) > self.thres

        self.objects_map.append(result)

        self.points_list.append(self.current_points_list)
        # Clear setup
        self.current_num_points = 0
        self.current_points_list = []

    def update_segments_cur(self, new_object_map):
        self.segments_out_cur = (self.segments_out_cur + new_object_map).clip(0, 1)
        self.segments_color_out_cur = (np.stack([self.segments_out_cur]*3, -1) * self.select_color).astype(np.uint8)

    def process(self, in_button):
        if in_button == 8:  # Delete lastest point
            if self.current_num_points > 0:
                self.current_num_points -= 1
                self.current_points_list.pop()

        elif in_button == ord('n'):  # Execute dextr algorithm
            if self.current_num_points == 4:  # if already have enough 4 extreme points
                self.get_segments_map_from_extreme_points()
                self.update_segments_cur(self.objects_map[-1])

    def reset_env_activate(self, img, segments_out, segments_color_out, model_config):
        self.img = img
        self.img_vis = img.copy()
        self.segments_out_cur = segments_out.copy()  # current segment for blob
        self.segments_color_out_cur = segments_color_out.copy()  # current segment for blob

        self.current_num_points = 0  # Current number of points for current object
        self.current_points_list = []  # Current points list for current object
        self.points_list = []  # Points list for all object
        self.objects_map = []  # list of object segment results

        self.get_model(model_config)

    def get_model(self, model_config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(model_config, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.net = resnet.resnet101(1, nInputChannels=4, classifier=cfg['classifier'])
        print("Initializing weights from: {}".format(os.path.join(cfg['models_dir'], cfg['model_name'] + '.pth')))
        state_dict_checkpoint = torch.load(os.path.join(cfg['models_dir'], cfg['model_name'] + '.pth'),
                                   map_location=lambda storage, loc: storage)

        # Remove the prefix .module from the model when it is trained using DataParallel
        if 'module.' in list(state_dict_checkpoint.keys())[0]:
            new_state_dict = OrderedDict()
            for k, v in state_dict_checkpoint.items():
                name = k[7:]  # remove 'module.' from multi-gpu training
                new_state_dict[name] = v
        else:
            new_state_dict = state_dict_checkpoint

        self.net.load_state_dict(new_state_dict)
        self.net.eval()
        self.net.to(self.device)

        self.pad = cfg['pad']

        self.thres = cfg['thres']
        self.min_thres = cfg['min_thres']
        self.max_thres = cfg['max_thres']

        del cfg

    def get_instruction_image(self, ins_image = None):
        """For visualize instruction"""
        if ins_image is None:
            ins_image = np.zeros((round(self.img_size[0]), round(self.img_size[1]*0.7)) + (3,))
            
        # General options
        cv2.putText(ins_image, "=== Technique: Extreme points ===", (round(self.img_size[1]*0.01), round(self.img_size[0]*0.86)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR.CYAN, 2)
        cv2.putText(ins_image, f"Note: Have to have 4 points to execute dextr algorithm !!", (round(self.img_size[1]*0.01), round(self.img_size[0]*0.90)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, f"Current points: {self.current_points_list}", (round(self.img_size[1]*0.01), round(self.img_size[0]*0.95)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "- [backspace] - Remove the lastest point", (round(self.img_size[1]*0.05), round(self.img_size[0]*1.00)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "- [n] - Start extreme points algorithm (after 4 points)", (round(self.img_size[1]*0.05), round(self.img_size[0]*1.05)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        return ins_image