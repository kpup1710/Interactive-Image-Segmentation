import numpy as np
import cv2
import sys
import os
import time
import matplotlib.pyplot as plt
import argparse

from lib.annotator import ImageSegmentAnnotator


""""
Fast tutorial:
    Usage: python interactive_segment.py --img_path {path/to/image} --img_size {image size}

    Change the iamge file in img_list
    Change image size input in img_size var
    
    == General options ==
    Key 'q' - Decrease the mouse's radius
    Key 'e' - Increase the mouse's radius
    Key 's' - Save segment output
    Key 'm' - Merge current mask with final output
    Key 'r' - Replace final output with current mask
    Key 'p' - Clear current mask
    Key esc - Exit the app

    == Technique selections ==
    Key '1' - Scribble
    Key '2' - Blobs
    Key '3' - Grabcut

    == Scribble options ==
    Key '`' - Change from box drawing to scribble (or otherwise)

    == Blob options ==
    Key 'a' - Change level for blob function (down)
    Key 'd' - Change level for blob function (up)

    == Grabcut options ==
    Key '`' - Change from box drawing to scribble (or otherwise)
    Key 'n' - Start grabcut algorithm (after draw rectangle)
    Key 'z' - background drawing
    Key 'x' - foreground drawing
    Key 'c' - probable background drawing
    Key 'v' - probable foreground drawing

    == Extreme points options ==
    Key 'backspace' - Remove lastest point
    Key 'n' - Start extreme points algorithm (after 4 points)
    
"""


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    ## Init setup
    img_segment_annotator = ImageSegmentAnnotator(args)
    img_segment_annotator.input_img(args.img_path)
    img_segment_annotator.reset_env_activate()
    
    while(1):
        k = cv2.waitKey(1)

        img_segment_annotator.process(k)


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive Segmentation Options")
    parser.add_argument('--img_path', help='image file path', required=True)
    parser.add_argument('--gt_path', help='groundtruth file path')
    parser.add_argument('--img_size', type=int, nargs='+', default=[640, 640], help='image size for algorithm input')
    parser.add_argument('--save_dir', type=str, default='results', help='annotation save dir')
    # Window setup
    parser.add_argument('--win_input_name', type=str, default='input', help='window input name')
    parser.add_argument('--win_input_position', type=float, nargs='+', default=[0.0005, 0], help='window input position ratio')
    parser.add_argument('--win_output_name', type=str, default='final output', help='window output name')
    parser.add_argument('--win_output_position', type=float, nargs='+', default=[0.374, 0], help='window output position ratio')
    parser.add_argument('--win_current_result_name', type=str, default='current mask', help='window current_result name')
    parser.add_argument('--win_current_result_position', type=float, nargs='+', default=[0.374, 0.5], help='window current_result position ratio')
    parser.add_argument('--win_instruction_name', type=str, default='instruction', help='window instruction name')
    parser.add_argument('--win_instruction_position', type=float, nargs='+', default=[0.9, 0], help='window instruction position ratio')
    # Mouse setup
    parser.add_argument('--mouse_radius', type=int, default=3, help='mouse radius')
    parser.add_argument('--mouse_max_radius', type=int, default=40, help='mouse max radius')
    
    parser.add_argument('--checkpoint', type=str, default='coco_lvis_h18_itermask.pth', help='weights')
    parser.add_argument('--device', type=str, default='cpu', help='weights')

    # Extreme points options
    parser.add_argument('--extreme_points_config', type=str, 
                    default='lib/supporters/extreme_points/dextr_config.yaml', help='config file for extreme points technique')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)