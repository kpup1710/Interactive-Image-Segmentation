import numpy as np
import cv2

from .general import COLOR


class Scribble:
    def __init__(self, img_size, **kwargs):
        self.img_size = img_size

        self.draw_positive = True  # Draw positive or not
        self.is_drawing = False  # current drawing or not
        self.select_color = COLOR.GREEN  # color for current foreground

    def mouse_cb(self,event,x,y,flags,**kwargs):
        mouse_radius_map = kwargs['mouse_radius_map']

        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            if self.draw_positive:
                self.segments_out_cur = (self.segments_out_cur + mouse_radius_map).clip(0, 1)
                self.segments_color_out_cur = (np.stack([self.segments_out_cur]*3, -1) * self.select_color).astype(np.uint8)
            else:
                self.segments_out_cur = (self.segments_out_cur - mouse_radius_map).clip(0, 1)
                self.segments_color_out_cur = (np.stack([self.segments_out_cur]*3, -1) * self.select_color).astype(np.uint8)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing:
                if self.draw_positive:
                    self.segments_out_cur = (self.segments_out_cur + mouse_radius_map).clip(0, 1)
                    self.segments_color_out_cur = (np.stack([self.segments_out_cur]*3, -1) * self.select_color).astype(np.uint8)
                else:
                    self.segments_out_cur = (self.segments_out_cur - mouse_radius_map).clip(0, 1)
                    self.segments_color_out_cur = (np.stack([self.segments_out_cur]*3, -1) * self.select_color).astype(np.uint8)

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False

    def process(self, in_button):
        if in_button == ord('`'):  # Change draw negative/positive
            self.draw_positive = False if self.draw_positive else True

    def reset_env_activate(self, img, segments_out, segments_color_out):
        self.segments_out_cur = segments_out.copy()  # current segment for blob
        self.segments_color_out_cur = segments_color_out.copy()  # current segment for blob

    def get_instruction_image(self, ins_image = None):
        """For visualize instruction"""
        if ins_image is None:
            ins_image = np.zeros((round(self.img_size[0]), round(self.img_size[1]*0.7)) + (3,))
            
        # General options
        cv2.putText(ins_image, "=== Technique: Scribble ===", (round(self.img_size[1]*0.01), round(self.img_size[0]*0.86)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR.CYAN, 2)
        cv2.putText(ins_image, f"Current draw option: {'Positive' if self.draw_positive else 'Negative'}", (round(self.img_size[1]*0.01), round(self.img_size[0]*0.90)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "- [`] - Change from box drawing to scribble (or otherwise)", (round(self.img_size[1]*0.05), round(self.img_size[0]*0.95)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        return ins_image