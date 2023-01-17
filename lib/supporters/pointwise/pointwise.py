import time
import numpy as np
import cv2
import numpy as np
from lib.supporters.general import COLOR

from .utils import clicker
from .predictors import get_predictor
from .utils.vis import draw_with_blend_and_clicks
import torch


class Pointwise:
    def __init__(self, img_size, **kwargs):
        self.img_size = img_size
        self.net = kwargs['net']
        self.prob_thresh = kwargs['prob_thresh']
        self.clicker = clicker.Clicker()
        self.states = []
        self.probs_history = []
        self.object_count = 0
        self._result_mask = None
        self._init_mask = None
        self.init_prob_thresh = kwargs['prob_thresh']
        
        self.image = None
        self.predictor = None
        self.device = kwargs['device']
        self.predictor_params = kwargs['predictor_params']
        self.reset_predictor()
        print('reset prdictor done')
        # self.file = Path(img_path).nameq
        self.img_seg_blob = np.zeros(img_size + (3,))
        self.clicks = np.empty([0,3],dtype=np.int64)
        self.pred = np.zeros(img_size,dtype=np.uint8)
        self.segments_blob = np.zeros(img_size)
        self.select_color = COLOR.GREEN
        self.firstclick = True
        
    def set_image(self, image):  #get input image
        self.image = image
        self._result_mask = np.zeros(image.shape[:2], dtype=np.uint16)
        self.object_count = 0
        self.reset_last_object(update_image=False)
        
    def set_mask(self, mask):
        # raise self.image.shape[:2] == mask.shape[:2]
        
        if len(self.probs_history) > 0: # if you use mask from outside you should delete other segmentation masks
            self.reset_last_object()
        
        self._init_mask = mask.astype(np.float32)
        self.probs_history.append((np.zeros_like(self._init_mask), self._init_mask)) # why need a tuple of zeros mask and mask
        self._init_mask = torch.Tensor(self._init_mask, device=self.device).unsqueeze(0).unsqueeze(0) # why increase number of dim? Maybe for creating batch?
        self.clicker.click_indx_offset = 1
        
    def reset_predictor(self, predictor_params=None):
        if predictor_params is not None:
            self.predictor_params = predictor_params
        self.predictor = get_predictor(self.net, device=self.device, **self.predictor_params)
        if self.image is not None:
            self.predictor.set_input_image(self.image)
    
    def reset_init_mask(self):
        self._init_mask = None
        self.clicker.click_indx_offset = 0
    
    def reset_last_object(self, update_image=False):
        self.state = []
        self.probs_history = []
        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        if update_image:
            self.__update()

    def reset_env_activate(self, img, segments_out, segments_color_out, predictor_params, prob_thresh=0.5):
        self.set_image(img)
        self.reset_predictor(predictor_params=predictor_params)
        print('reset predictor done')
        self.reset_init_mask()
        self.reset_last_object()
        self.prob_thresh = prob_thresh
        self.img_seg_blob = self.image.copy()
        self.set_mask(segments_out)
        # self._result_mask = segments_out.copy()
        self.segments_out_cur = segments_out.copy()
        self.segments_color_out_cur = segments_color_out.copy()
        self.__update()

    def __update(self):
        # cv2.imshow('Annotator',self.merge[..., ::-1])
        result = self.get_visualization(alpha_blend=.7, click_radius=2)
        self.img_seg_blob= result
        # self.fig.canvas.draw()

        
    def process(self, in_button):
        if in_button == ord('h'):  # up level
            self.prob_thresh += 0.01
            if self.prob_thresh >= 0.95:
                self.prob_thresh = 0.95                
            self._update_prob_thresh()
            # self.prob_thresh = self.init_prob_thresh
        elif in_button == ord('n'):  # down level
            self.prob_thresh -= 0.01
            if self.prob_thresh <= 0.1:
                self.prob_thresh = 0.1
            self._update_prob_thresh()
            # self.prob_thresh = self.init_prob_thresh
        elif in_button == ord('u'):
            self.undo_click()
        elif in_button == ord('f'):
            self.finish_object()
    
    
    def add_click(self, x, y, is_positive):
        self.states.append({
            'clicker': self.clicker.get_state(),
            'predictor': self.predictor.get_states()
        })
        
        click = clicker.Click(is_positive=is_positive, coords=(y, x))
        self.clicker.add_click(click)
        pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)
        if self._init_mask is not None and len(self.clicker) == 1:
            pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)
            
        torch.cuda.empty_cache()
        
        if self.probs_history:
            self.probs_history.append((self.probs_history[-1][0], pred))
        else:
            self.probs_history.append((np.zeros_like(pred), pred))
            
        self.__update()
        
    def undo_click(self):
        if not self.states:
            return
        prev_state = self.states.pop()
        self.clicker.set_state(prev_state['clicker'])
        self.predictor.set_states(prev_state['predictor'])
        if not self.probs_history: #after pop if the probs_history become empty
            self.reset_init_mask()
        self.probs_history.pop()
        self.__update()
    
    def _update_prob_thresh(self):
        if self.is_incomplete_mask:
            # self.prob_thresh = value
            self.__update()
        
    def mouse_cb(self, event, x,y, flags, **kwargs):        
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            if event == cv2.EVENT_LBUTTONDOWN:
                button = 1
            if event == cv2.EVENT_RBUTTONDOWN:
                button = 3
            self.add_click(x, y,is_positive=(button == 1))

            
    def get_visualization(self, alpha_blend, click_radius):
        if self.image is None:
            return None

        results_mask_for_vis = self.result_mask
        print(results_mask_for_vis[results_mask_for_vis > 0])
        self.segments_color_out_cur[self.segments_color_out_cur >0] = 0
        self.segments_out_cur[self.segments_out_cur > 0] = 0
        self.segments_out_cur[results_mask_for_vis >0] = 1
        self.segments_color_out_cur[results_mask_for_vis >0] = self.select_color
        vis = draw_with_blend_and_clicks(self.image, mask=results_mask_for_vis, alpha=alpha_blend,
                                         clicks_list=self.clicker.clicks_list, radius=click_radius, pos_color=self.select_color)
        if self.probs_history:
            total_mask = self.probs_history[-1][0] > self.prob_thresh
            results_mask_for_vis[np.logical_not(total_mask)] = 0
            vis = draw_with_blend_and_clicks(vis, mask=results_mask_for_vis, alpha=alpha_blend, pos_color=self.select_color)
        return vis
    
    def finish_object(self):
        if self.current_object_prob is None:
            return

        self._result_mask = self.result_mask
        self.object_count += 1
        self.reset_last_object()
    
    @property
    def current_object_prob(self):
        if self.probs_history:
            current_prob_total, current_prob_additive = self.probs_history[-1]
            return np.maximum(current_prob_total, current_prob_additive)
        else:
            return None
        
    @property
    def is_incomplete_mask(self):
        return len(self.probs_history) > 0
    
    @property
    def result_mask(self):
        result_mask = self._result_mask.copy()
        if self.probs_history:
            result_mask[self.current_object_prob > self.prob_thresh] = self.object_count + 1
        return result_mask
    
    def get_instruction_image(self, ins_image = None):
        """For visualize instruction"""
        if ins_image is None:
            ins_image = np.zeros((round(self.img_size[0]), round(self.img_size[1]*0.7)) + (3,))
            
        # General options
        cv2.putText(ins_image, "=== Technique: Pointwise ===", (round(self.img_size[1]*0.01), round(self.img_size[0]*0.86)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR.CYAN, 2)
        cv2.putText(ins_image, f"Current prob_threshold  {self.prob_thresh}", (round(self.img_size[1]*0.01), round(self.img_size[0]*0.90)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "- [h] - Increase the prob_thresh", (round(self.img_size[1]*0.05), round(self.img_size[0]*0.95)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "- [n] - Decrease the prob_thresh", (round(self.img_size[1]*0.05), round(self.img_size[0]*1.00)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "- [f] - Finish object", (round(self.img_size[1]*0.05), round(self.img_size[0]*1.05)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "- [u] - Undo click", (round(self.img_size[1]*0.05), round(self.img_size[0]*1.1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        return ins_image