import numpy as np
import cv2
import os
import shutil
import scipy
import pyautogui
from rdp import rdp

from .supporters import COLOR, Watershed, Grabcut, Scribble, DEXTR, Pointwise
from .supporters.pointwise.utils import utils

def on_mouse(event, x, y, flags, param):
    param.mouse_cb(event, x, y, flags)


def nothing(x):
    pass


class ImageSegmentAnnotator:
    def __init__(self, args):
        ## Global variables settings
        self.args = args
        self.img_size = tuple(args.img_size)

        # Window setup
        window_size = pyautogui.size()
        window_width = window_size[0]
        window_height = window_size[1]
        self.win_input_name = args.win_input_name
        self.win_input_position = [round(args.win_input_position[0] * window_width), round(args.win_input_position[1] * window_height)]
        self.win_output_name = args.win_output_name
        self.win_output_position = [round(args.win_output_position[0] * window_width), round(args.win_output_position[1] * window_height)]
        self.win_current_result_name = args.win_current_result_name
        self.win_current_result_position = [round(args.win_current_result_position[0] * window_width), round(args.win_current_result_position[1] * window_height)]
        self.win_instruction_name = args.win_instruction_name
        self.win_instruction_position = [round(args.win_instruction_position[0] * window_width), round(args.win_instruction_position[1] * window_height)]

        self.save_dir = args.save_dir

        self.class2color = {
            'bg': COLOR.BLACK,
            'fg': COLOR.GREEN,
        }  # Not important yet
        self.class2id = {
            'bg':0,
            'fg':1,
        }  # Not important yet
        self.id2class = {v:k for k, v in self.class2id.items()}

        self.segments_color_out = np.zeros(self.img_size + (3,))  # color segments map (follow class color mapping)
        self.segments_out = np.zeros(self.img_size)  # segment map (follow class2id mapping)
        self.img_seg = np.zeros(self.img_size + (3,))  # img blend with segment
        self.reset_env = False

        # Segment support options
        self.interactive_options = ['scribble', 'blob', 'grabcut', 'extreme_points', 'pointwise']
        self.interactive_options_lib = {
            'scribble': Scribble,
            'blob': Watershed,
            'grabcut': Grabcut,
            'extreme_points': DEXTR,
            'pointwise': Pointwise,
        }
        self.cur_option = 0
        self.eps = 3

        # Mouse setup
        self.cur_mouse = (0, 0)  # initial mouse position
        self.radius = args.mouse_radius  # initial current mouse radius
        self.max_radius = args.mouse_max_radius  # max mouse radius
        self.mouse_radius_map = np.zeros(self.img_size, dtype=np.float64)
        cv2.circle(self.mouse_radius_map, self.cur_mouse, self.radius, 1., -1)

        self.polypoints = []

        self.select_color = COLOR.GREEN  # TODO: Change latter

        # Setup windows
        cv2.namedWindow(self.win_input_name)
        cv2.moveWindow(self.win_input_name, *self.win_input_position)
        cv2.namedWindow(self.win_output_name)
        cv2.moveWindow(self.win_output_name, *self.win_output_position)
        cv2.namedWindow(self.win_instruction_name)
        cv2.moveWindow(self.win_instruction_name, *self.win_instruction_position)
        cv2.namedWindow(self.win_current_result_name)
        cv2.moveWindow(self.win_current_result_name, *self.win_current_result_position)

        cv2.setMouseCallback(self.win_input_name, on_mouse, self)
        cv2.createTrackbar('brush size', self.win_input_name, self.radius, self.max_radius, nothing)
        
        #Pointwise setup
        checkpoint_path = utils.find_checkpoint("./lib/supporters/pointwise/model/weights", args.checkpoint)
        self.model = utils.load_is_model(checkpoint_path, args.device, cpu_dist_maps=True)
        self.prob_thresh = 0.5
        self.device = 'cpu'
        self.predictor_params = {'brs_mode': 'NoBRS'}
        
        self.supporter = self.interactive_options_lib[self.interactive_options[self.cur_option]](img_size=self.img_size,net=self.model,\
                predictor_params=self.predictor_params, prob_thresh=self.prob_thresh, device='cpu')
        
        
    def mouse_cb(self, event, x, y, flags):
        self.cur_mouse = (x, y)
        self.img_seg = cv2.addWeighted(self.img/255, 0.7, self.segments_color_out/255, 0.3, 0.0)
        
        if self.interactive_options[self.cur_option] == 'blob':
            self.supporter.mouse_cb(event, x, y, flags,
                            mouse_radius_map=self.mouse_radius_map,
                            mouse_radius=self.radius)
        elif self.interactive_options[self.cur_option] == 'grabcut':
            self.supporter.mouse_cb(event, x, y, flags,
                            mouse_radius=self.radius)
        elif self.interactive_options[self.cur_option] == 'scribble':
            self.supporter.mouse_cb(event, x, y, flags,
                            mouse_radius_map=self.mouse_radius_map)
        elif self.interactive_options[self.cur_option] == 'extreme_points':
            self.supporter.mouse_cb(event, x, y, flags)
        elif self.interactive_options[self.cur_option] == 'pointwise':
            self.supporter.mouse_cb(event, x, y, flags,
                        mouse_radius_map=self.mouse_radius_map)
        else:
            raise

    def input_img(self, img_path):
        self.img_name = img_path.split(os.sep)[-1].split('.')[0]
        self.img = cv2.imread(img_path)
        self.img = cv2.resize(self.img, self.img_size)

    def reset_env_activate(self):
        """Use for regenerate blob map or change options"""
        self.img_seg = cv2.addWeighted(self.img/255, 0.7, self.segments_color_out/255, 0.3, 0.0)

        if self.interactive_options[self.cur_option] == 'blob':
            self.supporter.reset_env_activate(self.img, self.segments_out, self.segments_color_out)
        elif self.interactive_options[self.cur_option] == 'grabcut':
            self.supporter.reset_env_activate(self.img, self.segments_out, self.segments_color_out)
        elif self.interactive_options[self.cur_option] == 'scribble':
            self.supporter.reset_env_activate(self.img, self.segments_out, self.segments_color_out)
        elif self.interactive_options[self.cur_option] == 'extreme_points':
            self.supporter.reset_env_activate(self.img, self.segments_out, self.segments_color_out, self.args.extreme_points_config)
        elif self.interactive_options[self.cur_option] == 'pointwise':
            self.supporter.reset_env_activate(self.img, self.segments_out, self.segments_color_out, predictor_params=self.predictor_params)
        else:
            raise 

    def get_instruction_image(self, ins_image = None):
        """For visualize instruction"""
        if ins_image is None:
            ins_image = np.zeros((round(self.img_size[0]*1.5), round(self.img_size[1]*0.8)) + (3,))
            
        # General options
        cv2.putText(ins_image, "=== General ===", (round(self.img_size[1]*0.01), round(self.img_size[0]*0.06)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR.CYAN, 2)
        cv2.putText(ins_image, f"Current mouse's radius: {self.radius}", (round(self.img_size[1]*0.01), round(self.img_size[0]*0.1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "- [q] - Decrease the mouse's radius", (round(self.img_size[1]*0.05), round(self.img_size[0]*0.15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "- [e] - Increase the mouse's radius", (round(self.img_size[1]*0.05), round(self.img_size[0]*0.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "- [s] - Save segment output", (round(self.img_size[1]*0.05), round(self.img_size[0]*0.25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "- [m] - Merge current mask with final output", (round(self.img_size[1]*0.05), round(self.img_size[0]*0.3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "- [r] - Replace final output with current mask", (round(self.img_size[1]*0.05), round(self.img_size[0]*0.35)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "- [p] - Clear current mask", (round(self.img_size[1]*0.05), round(self.img_size[0]*0.4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "- [esc] - Exit the app", (round(self.img_size[1]*0.05), round(self.img_size[0]*0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "___ Select technique selections ___", (round(self.img_size[1]*0.01), round(self.img_size[0]*0.55)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR.WHITE, 2)
        cv2.putText(ins_image, "- [1] - Scribble", (round(self.img_size[1]*0.05), round(self.img_size[0]*0.60)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "- [2] - Blobs", (round(self.img_size[1]*0.05), round(self.img_size[0]*0.65)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "- [3] - Grabcut", (round(self.img_size[1]*0.05), round(self.img_size[0]*0.70)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "- [4] - Extreme points", (round(self.img_size[1]*0.05), round(self.img_size[0]*0.75)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "- [5] - Pointwise", (round(self.img_size[1]*0.05), round(self.img_size[0]*0.80)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        ins_image = self.supporter.get_instruction_image(ins_image)
        return ins_image

    def process(self, in_button):
        
        if in_button == 27:
            exit()

        elif in_button == ord('q'):  # decrease the mouse radius
            diff_k = int(np.clip(self.radius * 0.4, 1, 5))
            self.radius -= diff_k
            self.radius = np.clip(self.radius, 1, self.max_radius)
            cv2.setTrackbarPos('brush size', self.win_input_name, self.radius)

        elif in_button == ord('e'):  # increase the mouse radius
            diff_k = int(np.clip(self.radius * 0.4, 1, 5))
            self.radius += diff_k
            self.radius = np.clip(self.radius, 1, self.max_radius)
            cv2.setTrackbarPos('brush size', self.win_input_name, self.radius)

        elif in_button == ord('m'):  # Merge current mask with final mask
            self.segments_out = (self.segments_out + self.supporter.segments_out_cur).clip(0, 1)
            self.segments_color_out = (np.stack([self.segments_out]*3, -1) * self.select_color).astype(np.uint8)
            self.supporter.segments_out_cur = self.segments_out.copy()
            self.supporter.segments_color_out_cur = self.segments_color_out.copy()

        elif in_button == ord('r'):  # Replace final mask with current mask
            self.segments_out = self.supporter.segments_out_cur.copy()
            self.segments_color_out = (np.stack([self.segments_out]*3, -1) * self.select_color).astype(np.uint8)

        elif in_button == ord('p'):  # Clear current mask
            self.supporter.segments_out_cur = np.zeros_like(self.segments_out)  # reset current segment
            self.supporter.segments_color_out_cur = np.zeros_like(self.segments_color_out)  # reset current color segment
        
        elif in_button == ord('v') and self.interactive_options[self.cur_option] == 'extreme_points':
            self.eps += 1
            if self.eps >= 5:
                self.eps = 5
            img = cv2.cvtColor(self.supporter.segments_color_out_cur, cv2.COLOR_BGR2GRAY)
            contours,_= cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            pts = max(contours, key=len)
            pts_2 = rdp(pts, epsilon=self.eps)
            self.polypoints = pts_2
            
        elif in_button == ord('b') and self.interactive_options[self.cur_option] == 'extreme_points':
            # self.supporter.segments_color_out_cur = self.supporter.segments_for_polygon.copy()
            self.eps -= 1
            if self.eps <= 1:
                self.eps = 1
            img = cv2.cvtColor(self.supporter.segments_color_out_cur, cv2.COLOR_BGR2GRAY)
            contours,_= cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            pts = max(contours, key=len)
            pts_2 = rdp(pts, epsilon=self.eps)
            self.polypoints = pts_2
            
        # Change options
        elif 49 <= in_button and in_button <= 57: # is numeric number (1 - 9)
            self.polypoints = []
            # 49: 1  -->  57: 9
            if in_button - 49 < len(self.interactive_options) and in_button - 49 != self.cur_option:  # if in range of options and is changed
                self.cur_option = in_button - 49  # for range 0 - 8
                self.supporter = self.interactive_options_lib[self.interactive_options[self.cur_option]](img_size=self.img_size,net=self.model,\
                predictor_params=self.predictor_params, prob_thresh=self.prob_thresh, device='cpu')
                self.reset_env_activate()

        else:
            self.supporter.process(in_button)

        self.radius = cv2.getTrackbarPos('brush size', self.win_input_name)
        self.mouse_radius_map = np.zeros(self.img_size, dtype=np.float64)
        cv2.circle(self.mouse_radius_map, self.cur_mouse, self.radius, 1., -1)

        segments_out_visual = (self.segments_out + self.supporter.segments_out_cur).clip(0, 1)
        segments_color_out_visual = (np.stack([segments_out_visual]*3, -1) * self.select_color).astype(np.uint8)
        segments_out_visual = np.stack([segments_out_visual]*3,-1)

        if self.interactive_options[self.cur_option] == 'blob':
            img_vis = self.supporter.orin_img_blob * ((1 - segments_out_visual) + segments_out_visual*0.7) \
                        + segments_color_out_visual / 255 * segments_out_visual*0.3
            img_vis = cv2.addWeighted(img_vis, 0.8, self.supporter.blob_seg, 0.2, 0.0)
        elif self.interactive_options[self.cur_option] == 'grabcut':
            img_vis = self.supporter.img_vis/255 * ((1 - segments_out_visual) + segments_out_visual*0.7) \
                        + segments_color_out_visual / 255 * segments_out_visual*0.3
        elif self.interactive_options[self.cur_option] == 'scribble':
            img_vis = self.img/255 * ((1 - segments_out_visual) + segments_out_visual*0.7) \
                        + segments_color_out_visual / 255 * segments_out_visual*0.3
        elif self.interactive_options[self.cur_option] == 'extreme_points':
            img_vis = self.supporter.img_vis/255 * ((1 - segments_out_visual) + segments_out_visual*0.7) \
                        + segments_color_out_visual / 255 * segments_out_visual*0.3
            self.segment_for_polygon = self.supporter.segments_color_out_cur
        elif self.interactive_options[self.cur_option] == 'pointwise':
            img_vis = self.supporter.img_seg_blob.copy()
        else:
            raise

        cv2.circle(img_vis, self.cur_mouse, self.radius, (200, 200, 200), 1)
        cv2.imshow(self.win_input_name, img_vis)

        cv2.imshow(self.win_output_name, self.segments_color_out)
        cv2.imshow(self.win_instruction_name, self.get_instruction_image())
        if len(self.polypoints) == 0:
            cv2.imshow(self.win_current_result_name, self.supporter.segments_color_out_cur)
        else:
            img = self.supporter.segments_color_out_cur.copy()
            for pt in self.polypoints:
                cv2.circle(img, pt[0], 3, (255, 0, 0), -1)
            # cv2.drawContours(img,self.polypoints[-1],-1,(0,0,255),1)
            cv2.imshow(self.win_current_result_name, img)
        
        

        if in_button == ord('s'):  # Save annotations
            if self.args.gt_path is not None:
                gt_path = self.args.gt_path
                if gt_path.split('.')[-1].lower() == 'mat':
                    seg_map = scipy.io.loadmat(gt_path)['groundTruth'][0][0][1]
                    seg_map = cv2.resize(seg_map, self.img_size, cv2.INTER_CUBIC)
                    seg_map = seg_map * 255
                    cv2.imwrite(os.path.join(self.save_dir, 'segments', f'{self.img_name}.png'), seg_map)
                elif gt_path.split('.')[-1].lower() == 'png':
                    seg_map = cv2.imread(gt_path)
                    if seg_map.shape[2] == 1:
                        seg_map = cv2.resize(seg_map, self.img_size, cv2.INTER_CUBIC)
                        seg_map = seg_map * 255
                        cv2.imwrite(os.path.join(self.save_dir, 'segments', f'{self.img_name}.png'), seg_map)
                    else:
                        # [0, 0, 254]  hair
                        # [0, 85, 254]  shirt
                        # [220, 169, 51]  left hand
                        # [254, 254, 0]  right hand
                        # [85, 85, 0]  # trouser
                        # [254, 0, 0]  # face
                        goal = (0, 0, 254)
                        # seg_map = (seg_map == goal).astype(np.uint8)
                        seg_map = np.all(seg_map == goal, axis=-1).astype(np.uint8)
                        seg_map = cv2.resize(seg_map, self.img_size, cv2.INTER_CUBIC)
                        seg_map = seg_map * 255
                        cv2.imwrite(os.path.join(self.save_dir, 'segments', f'{self.img_name}.png'), seg_map)

            shutil.copy(self.args.img_path, os.path.join(self.save_dir, 'images', f'{self.img_name}.jpg'))
            cv2.imwrite(os.path.join(self.save_dir, self.interactive_options[self.cur_option], f'{self.img_name}.jpg'), self.segments_color_out)

            # cv2.imwrite(os.path.join(self.save_dir, f'{self.img_name}_color.jpg'), self.segments_color_out)
            # cv2.imwrite(os.path.join(self.save_dir, f'{self.img_name}.jpg'), self.segments_out)
            # print(f"- Image annotation has been saved to {os.path.join(self.save_dir, f'{self.img_name}_color.jpg')} "
            #     f"and {os.path.join(self.save_dir, f'{self.img_name}.jpg')}")

            print()