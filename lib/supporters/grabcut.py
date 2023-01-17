import time
import numpy as np
import cv2

from .general import COLOR


class Grabcut:
    def __init__(self, img_size, **kwargs):
        self.img_size = img_size
        # Grabcut setup
        self.rectangle = False
        self.rect_over = False  # first draw rectangle or not
        self.rect_or_mask = 100
        self.rect = (0,0,1,1)
        self.grabcut_mask = np.zeros(img_size, dtype=np.uint8)
        self.BBOX_COLOR = COLOR.YELLOW
        self.BBOX_LINE_COLOR = COLOR.BLACK
        self.select_color = COLOR.GREEN
        self.x1 = 0  # top left
        self.y1 = 0  # top left
        self.x2 = 1  # bottom right
        self.y2 = 1  # bottom right

        self.draw_box = True  # draw box or not

        # Scribbles setup
        # Colors list
        self.scribbles_color_options = {
            'DRAW_BG' : {'color': COLOR.BLACK, 'val': 0},
            'DRAW_FG' : {'color': COLOR.GREEN, 'val': 1},
            'DRAW_PR_BG' : {'color': COLOR.BROWN, 'val': 2},
            'DRAW_PR_FG' : {'color': COLOR.WHITE, 'val': 3},
        }
        self.scribbles_chosen = 'DRAW_FG'
        self.scribbles_drawing = False  # Flag for drawing curves
        self.scribbles_color = self.scribbles_color_options[self.scribbles_chosen]  # color of scribbles
        self.scribbles_thicknesss = 3  # brush thickness
        self.skip_learn_GMMS = False  # shether to skip learning GMM parameters

    def mouse_cb(self,event,x,y,flags,**kwargs):
        self.scribbles_thicknesss = kwargs['mouse_radius']

        if self.draw_box:  # Draw rectangle
            if event == cv2.EVENT_LBUTTONDOWN:
                self.rectangle = True
                self.x1 = x
                self.y1 = y

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.rectangle == True:
                    self.img_vis = self.img.copy()
                    cv2.rectangle(self.img_vis, (self.x1, self.y1), (x, y), self.BBOX_LINE_COLOR, 3)
                    cv2.rectangle(self.img_vis, (self.x1, self.y1), (x, y), self.BBOX_COLOR, 2)
                    self.rect = (min(self.x1,x),min(self.y1,y),abs(self.x1-x),abs(self.y1-y))
                    self.rect_or_mask = 0

            elif event == cv2.EVENT_LBUTTONUP:
                self.rectangle = False
                self.rect_over = True  # First rectangle has been drawn
                self.x2 = x
                self.y2 = y
                self.img_vis = self.img.copy()
                cv2.rectangle(self.img_vis, (self.x1, self.y1), (x, y), self.BBOX_LINE_COLOR, 3)
                cv2.rectangle(self.img_vis, (self.x1, self.y1), (x, y), self.BBOX_COLOR, 2)
                self.rect = (min(self.x1,x),min(self.y1,y),abs(self.x1-x),abs(self.y1-y))
                self.rect_or_mask = 0
                print(" Now press the key 'n' to segment (can press several time for better result) \n")

        else:  # Draw scribbles
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.rect_over == False:
                    print("Please draw rectangle first! \n")
                else:
                    self.scribbles_drawing = True
                    # cv2.circle(self.img_vis, (x, y), self.scribbles_thicknesss, self.scribbles_color['color'], -1)
                    cv2.circle(self.grabcut_mask, (x, y), self.scribbles_thicknesss, self.scribbles_color['val'], -1)

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.scribbles_drawing == True:
                    # cv2.circle(self.img_vis, (x, y), self.scribbles_thicknesss, self.scribbles_color['color'], -1)
                    cv2.circle(self.grabcut_mask, (x, y), self.scribbles_thicknesss, self.scribbles_color['val'], -1)
            
            elif event == cv2.EVENT_LBUTTONUP:
                if self.scribbles_drawing == True:
                    self.scribbles_drawing = False
                    # cv2.circle(self.img_vis, (x, y), self.scribbles_thicknesss, self.scribbles_color['color'], -1)
                    cv2.circle(self.grabcut_mask, (x, y), self.scribbles_thicknesss, self.scribbles_color['val'], -1)
                    self.skip_learn_GMMS = True

    def process(self, in_button):
        if in_button == ord('`'):  # Change from box drawing to scribble (or otherwise)
            self.draw_box = False if self.draw_box else True

        elif in_button == ord('z'):  # BG Drawing
            print(" mark background regions with left mouse button \n")
            self.scribbles_chosen = 'DRAW_BG'
            self.scribbles_color = self.scribbles_color_options['DRAW_BG']
        elif in_button == ord('x'):  # FG Drawing
            print(" mark foreground regions with left mouse button \n")
            self.scribbles_chosen = 'DRAW_FG'
            self.scribbles_color = self.scribbles_color_options['DRAW_FG']
        elif in_button == ord('c'):  # PR BG Drawing
            print(" mark probable background regions with left mouse button \n")
            self.scribbles_chosen = 'DRAW_PR_BG'
            self.scribbles_color = self.scribbles_color_options['DRAW_PR_BG']
        elif in_button == ord('v'):  # PR FG Drawing
            print(" mark probable foreground regions with left mouse button \n")
            self.scribbles_chosen = 'DRAW_PR_FG'
            self.scribbles_color = self.scribbles_color_options['DRAW_PR_FG']

        elif in_button == ord('n'):  # Begin grabcut
            self.change_mask = True
            if self.rect_or_mask == 0:  # grabcut with rect
                self.bgdmodel = np.zeros((1,65),np.float64)
                self.fgdmodel = np.zeros((1,65),np.float64)
                cv2.grabCut(self.img,self.grabcut_mask,self.rect,self.bgdmodel,self.fgdmodel,1,cv2.GC_INIT_WITH_RECT)
                self.rect_or_mask = 1
            elif self.rect_or_mask == 1:  # grabcut with mask
                self.grabcut_mask, self.bgfmodel, self.fgdmodel = cv2.grabCut(self.img,self.grabcut_mask,self.rect,self.bgdmodel,self.fgdmodel,1,cv2.GC_INIT_WITH_MASK)

        object_segment_map = np.where((self.grabcut_mask == 1) + (self.grabcut_mask == 3), 1, 0).astype('uint8')
        # self.segments_out = (self.segments_out + object_segment_map).clip(0, 1)
        # self.segments_color_out[object_segment_map > 0] = self.segments_color_out[object_segment_map > 0] * 0 + self.select_color
        self.segments_out_cur = object_segment_map.clip(0, 1)
        self.segments_color_out_cur = (np.stack([self.segments_out_cur]*3, -1) * self.select_color).astype(np.uint8)

    def reset_env_activate(self, img, segments_out, segments_color_out):
        self.img = img
        self.img_vis = self.img.copy()  # Used for visualize

        self.segments_out_cur = segments_out.copy()  # current segment for grabcut
        self.segments_color_out_cur = segments_color_out.copy()  # current segment for grabcut

    def get_instruction_image(self, ins_image = None):
        """For visualize instruction"""
        if ins_image is None:
            ins_image = np.zeros((round(self.img_size[0]), round(self.img_size[1]*0.7)) + (3,))
            
        # General options
        cv2.putText(ins_image, "=== Technique: Grabcut ===", (round(self.img_size[1]*0.01), round(self.img_size[0]*0.86)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR.CYAN, 2)
        cv2.putText(ins_image, f"Current drawing option: {'rectangle' if self.draw_box else 'scribbles'}", (round(self.img_size[1]*0.01), round(self.img_size[0]*0.90)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "- [`] - Change from box drawing to scribble (or otherwise)", (round(self.img_size[1]*0.05), round(self.img_size[0]*0.95)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "- [n] - Start grabcut algorithm (after drawing)", (round(self.img_size[1]*0.05), round(self.img_size[0]*1.00)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "- Scribbles drawing options (Need drawing rectangle first to apply):", (round(self.img_size[1]*0.05), round(self.img_size[0]*1.05)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.WHITE, 2)
        cv2.putText(ins_image, "+ [z] - background drawing", (round(self.img_size[1]*0.08), round(self.img_size[0]*1.08)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.GREEN if self.scribbles_chosen=='DRAW_BG' else COLOR.WHITE, 1)
        cv2.putText(ins_image, "+ [x] - foreground drawing", (round(self.img_size[1]*0.08), round(self.img_size[0]*1.11)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.GREEN if self.scribbles_chosen=='DRAW_FG' else COLOR.WHITE, 1)
        cv2.putText(ins_image, "+ [c] - probable background drawing", (round(self.img_size[1]*0.08), round(self.img_size[0]*1.14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.GREEN if self.scribbles_chosen=='DRAW_PR_BG' else COLOR.WHITE, 1)
        cv2.putText(ins_image, "+ [v] - probable foreground drawing", (round(self.img_size[1]*0.08), round(self.img_size[0]*1.17)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR.GREEN if self.scribbles_chosen=='DRAW_PR_FG' else COLOR.WHITE, 1)
        return ins_image