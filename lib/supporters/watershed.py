import time
import numpy as np
import cv2

from .general import COLOR

from skimage.filters import sobel
from skimage.color import rgb2gray
from skimage.segmentation import watershed
from skimage.segmentation import mark_boundaries


class Watershed:
    def __init__(self, img_size, **kwargs):
        self.img_size = img_size
        # Blobs setup
        self.blob_levels = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
        self.cur_blob_level = 0

        self.segments_blob = np.zeros(img_size)  # segments map from blob technique (watershed)
        self.move_color = COLOR.SILVER  # color region when move the mouse
        self.select_color = COLOR.GREEN  # TODO: Replace class color mapping later
        self.on_lbutton_down = False  # if the left mouse button is being down
        self.trigger_segment = True  # True if segment bg to fg, False otherwise, trigger by the first interacted region
        self.img_seg_blob = np.zeros(img_size + (3,))  # img blend with segment plus blob boundary

    def mouse_cb(self,event,x,y,flags,**kwargs):
        mouse_radius_map = kwargs['mouse_radius_map']

        overlap_values = np.unique(self.segments_blob[mouse_radius_map > 0]) # Get overlap value if segments with mouse radius map

        if event == cv2.EVENT_LBUTTONDOWN:
            self.on_lbutton_down = True
            if self.segments_out_cur[y,x] == 0:  # is background
                self.trigger_segment = True
                for valid_v in overlap_values:
                    self.segments_out_cur[self.segments_blob == valid_v] = 1  # TODO: modify
                    self.segments_color_out_cur[self.segments_blob == valid_v] = self.segments_color_out_cur[self.segments_blob == valid_v] * 0 + self.select_color # TODO
            else:  # is foreground
                self.trigger_segment = False
                for valid_v in overlap_values:
                    self.segments_out_cur[self.segments_blob == valid_v] = 0  # TODO: modify
                    self.segments_color_out_cur[self.segments_blob == valid_v] = self.segments_color_out_cur[self.segments_blob == valid_v] *0 # TODO

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.on_lbutton_down:  # When left button mouse down
                if self.trigger_segment == True:
                    for valid_v in overlap_values:
                        self.segments_out_cur[self.segments_blob == valid_v] = 1  # TODO: modify
                        self.segments_color_out_cur[self.segments_blob == valid_v] = self.segments_color_out_cur[self.segments_blob == valid_v] * 0 + self.select_color # TODO
                else:
                    for valid_v in overlap_values:
                        self.segments_out_cur[self.segments_blob == valid_v] = 0  # TODO: modify
                        self.segments_color_out_cur[self.segments_blob == valid_v] = self.segments_color_out_cur[self.segments_blob == valid_v] *0 # TODO

            

        elif event == cv2.EVENT_LBUTTONUP:
            self.on_lbutton_down = False

        blob_seg = np.zeros_like(self.orin_img_blob)
        for valid_v in overlap_values:
            blob_seg[self.segments_blob == valid_v] = blob_seg[self.segments_blob == valid_v] + self.move_color
        self.blob_seg = blob_seg / 255
        # self.img_seg_blob = cv2.addWeighted(self.orin_img_blob, 0.7, self.segments_color_out_cur / 255, 0.3, 0.0)
        # self.img_seg_blob = cv2.addWeighted(self.img_seg_blob, 0.7, cur_seg, 0.3, 0.0)

    def process(self, in_button):
        if in_button == ord('d'):  # up level
            self.cur_blob_level += 1
            self.cur_blob_level = self.cur_blob_level % len(self.blob_levels)
            self.reset_env_activate(self.img, self.segments_out_cur, self.segments_color_out_cur)

        elif in_button == ord('a'):  # down level
            self.cur_blob_level -= 1
            self.cur_blob_level = len(self.blob_levels) - 1 if self.cur_blob_level < 0 else self.cur_blob_level
            self.reset_env_activate(self.img, self.segments_out_cur, self.segments_color_out_cur)

    def reset_env_activate(self, img, segments_out, segments_color_out):
        self.segments_out_cur = segments_out.copy()  # current segment for blob
        self.segments_color_out_cur = segments_color_out.copy()  # current segment for blob

        self.img = img
        start = time.time()

        gradient = sobel(rgb2gray(img))
        segments_ws = watershed(gradient, markers=self.blob_levels[self.cur_blob_level],
                                compactness=0.001)
        self.segments_blob = segments_ws.copy()
        self.orin_img_blob = mark_boundaries(img, segments_ws, (1, 0, 0))
        self.blob_seg = np.zeros_like(self.orin_img_blob)

        self.img_seg_blob = self.orin_img_blob.copy()

        end = time.time()

        print(f"Watershed Number of segments: {len(np.unique(segments_ws))}")
        print(f"Watershed Time per image: {end - start:.3f}s")
        print()

    def get_instruction_image(self, ins_image = None):
        """For visualize instruction"""
        if ins_image is None:
            ins_image = np.zeros((round(self.img_size[0]), round(self.img_size[1]*0.7)) + (3,))
            
        # General options
        cv2.putText(ins_image, "=== Technique: Blobs ===", (round(self.img_size[1]*0.01), round(self.img_size[0]*0.86)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR.CYAN, 2)
        cv2.putText(ins_image, f"Current blob levels: {self.cur_blob_level}", (round(self.img_size[1]*0.01), round(self.img_size[0]*0.90)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "- [a] - Change level for blob function (down)", (round(self.img_size[1]*0.05), round(self.img_size[0]*0.95)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "- [d] - Change level for blob function (up)", (round(self.img_size[1]*0.05), round(self.img_size[0]*1.00)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        return ins_image