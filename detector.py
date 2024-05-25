import torch, detectron2
import numpy as np
import os
import cv2
import sys
import time
from PIL import Image as IP

current_directory = os.path.abspath('')
DETIC_PATH = current_directory + '/Detic'
print("current path:", DETIC_PATH)

sys.path.insert(0, f"{DETIC_PATH}/third_party/CenterNet2/")
sys.path.insert(0, f"{DETIC_PATH}/")

# import some common detectron2 utilities
from detectron2.config import get_cfg

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.mypredictor import myVisualizationDemo

import clip
import asyncio

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


class Detector:
    def __init__(self):
        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(
            f"{DETIC_PATH}/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
        )
        cfg.MODEL.WEIGHTS = f"{DETIC_PATH}/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'  # load later
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = (
            False  # For better visualization purpose. Set to False for all classes.
        )
        cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = (
            f"{DETIC_PATH}/datasets/metadata/lvis_v1_train_cat_info.json"
        )
        cfg.freeze()
        self.demo = myVisualizationDemo(cfg)
        print("Detector init success")
        self.overlay = None
        self.max_confidence = None

        self.detections = None
        self.color_image = None
        self.detic_image = None

        self.query_dict = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def detect(self, color_image):
        self.query_dict = dict()
        self.color_image = color_image
        self.detections, visualized_output = self.demo.run_on_image2(self.color_image)
        self.detic_image = visualized_output.get_image()[:, :, ::-1]
        # cv2.imshow("result1", self.detic_image)
        # cv2.waitKey(0)
        if self.detections:
            self.overlay = np.zeros_like(self.detections[0]["mask"], dtype=np.uint8)
            self.max_confidence = np.zeros_like(self.overlay, dtype=np.float32)
            for i, detection in enumerate(self.detections, start=1):
                mask = detection["mask"].astype(np.uint8)

                # # 创建一个空白图像，大小与mask相同，初始化为0
                # mask_image = np.zeros_like(mask, dtype=np.uint8)
                # # 将mask中像素值为255的区域填充为白色
                # mask_image[mask == 1] = 255
                # # 在原始图像上绘制mask
                # result_image = cv2.bitwise_and(color_image, color_image, mask=mask_image)
                # # 显示结果图像
                # cv2.imshow("Masked Image", result_image)
                # cv2.waitKey(0)
                # print(detection["class_name"])

                # 更新overlay，将当前mask与之前的结果叠加
                self.overlay = np.where(mask, i, self.overlay)
                # 更新max_confidence，更新每个像素位置上的最大置信度
                self.max_confidence = np.where(mask, detection["confidence"], self.max_confidence)
                # print(self.overlay)
            return True
        else:
            return False
            # 转换为灰度图像以便应用ColorMap

    def query_mask(self, x, y):
        index = self.overlay[y, x]
        if index == 0:
            return np.zeros(512)
        else:
            if index in self.query_dict:
                return self.query_dict[index]
            else:
                try:
                    x1, y1, x2, y2 = self.detections[index - 1]['bbox']
                except:
                    print(index)
                    print(self.detections)
                roi_image = self.color_image[int(y1):int(y2), int(x1):int(x2)]
                # cv2.imshow('Result4', roi_image)
                # cv2.waitKey(0)
                roi_features = self.get_clip_features(roi_image)
                # print(self.detections[index-1]['class_name'])
                self.query_dict[index] = roi_features
                return roi_features

    def query_mask_vector(self, x_vector, y_vector):
        roi_features_vector = []
        for x, y in zip(x_vector, y_vector):
            roi_features = self.query_mask(x, y)
            roi_features_vector.append(roi_features)
        self.draw_detic(x_vector, y_vector)
        return np.array(roi_features_vector)

    def get_clip_features(self, roi_image):
        # print(roi_image)
        if roi_image.dtype != "uint8":
            roi_pil = IP.fromarray(roi_image.astype('uint8'))
        else:
            roi_pil = IP.fromarray(roi_image)
        roi = self.preprocess(roi_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            roi_features = self.clip_model.encode_image(roi).squeeze().cpu()
            # roi_features = self.clip_model.encode_image(roi)
        return np.array(roi_features)

    def draw_detic(self, x_vector, y_vector):

        overlay_gray = cv2.cvtColor(self.overlay, cv2.COLOR_GRAY2BGR)
        overlay_color = cv2.applyColorMap(overlay_gray * 10, cv2.COLORMAP_JET)
        cv2.imshow('Result2', overlay_color)
        cv2.waitKey(1)

        detic_image = self.detic_image.copy()
        # 设置文字参数
        font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型
        font_scale = 0.4  # 文字大小
        font_color = (255, 255, 255)  # 文字颜色
        font_thickness = 1  # 文字粗细
        for x, y in zip(x_vector, y_vector):
            if not isinstance(x, int) and not isinstance(y, int):
                x = x.cpu().item()
                y = y.cpu().item()
            cv2.circle(detic_image, (x, y), radius=2, color=(255, 0, 0), thickness=-1)  # 半径为5的实心圆
            # 绘制文字
            if self.overlay[y, x] != 0:
                text_size = cv2.getTextSize(str(self.overlay[y, x]), font, font_scale, font_thickness)[0]
                text_origin = (x - text_size[0] // 2, y - text_size[1] // 2)
                cv2.putText(detic_image, str(self.overlay[y, x]), text_origin, font, font_scale, font_color, font_thickness)
        cv2.imshow('Result3', detic_image)
        cv2.waitKey(1)
