# from typing import Dict, List, Literal
import os
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from PIL import Image

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.config.config import CfgNode
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.utils.colormap import random_color
from detectron2.utils.video_visualizer import VideoVisualizer, _create_text_labels
from detectron2.data import MetadataCatalog
from detectron2.structures.instances import Instances


from fire import Fire


class SortVideoVisualizer(VideoVisualizer):
    def __init__(self, metadata, instance_mode):
        super().__init__(metadata, instance_mode=instance_mode)
        self.color_mapping = {}

    def _assign_colors(self, ids):
        if len(ids) > 0 and not isinstance(ids[0], int):
            ids = [_id.item() for _id in ids]
            
        for _id in ids:
            if not _id in self.color_mapping:
                self.color_mapping[_id] = random_color(rgb=True, maximum=1)

        return [self.color_mapping[_id] for _id in ids]

    
    def draw_instance_predictions(self, frame, predictions):
        frame_visualizer = Visualizer(frame, self.metadata)
        num_instances = len(predictions)
        if num_instances == 0:
            return frame_visualizer.output

        boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None
        ids = predictions.ids if predictions.has("ids") else list(range(0, len(predictions)))

        print('Keypoints')
        print(keypoints)

        if predictions.has("pred_masks"):
            masks = predictions.pred_masks
        else:
            masks = None

        colors = self._assign_colors(ids)

        labels = [f"{self.metadata.get('thing_classes', None)[class_id]} ({instance_id})" for class_id, instance_id in zip(classes, ids)]

        if self._instance_mode == ColorMode.IMAGE_BW:
            # any() returns uint8 tensor
            frame_visualizer.output.img = frame_visualizer._create_grayscale_image(
                (masks.any(dim=0) > 0).numpy() if masks is not None else None
            )
            alpha = 0.3
        else:
            alpha = 0.5

        frame_visualizer.overlay_instances(
            boxes=None if masks is not None else boxes,  # boxes are a bit distracting
            masks=masks,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )

        return frame_visualizer.output

class Detection():
    def __init__(self, x1, y1, x2, y2, score, class_name, instance_id) -> None:
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.w = self.x2 - self.x1
        self.h = self.y2 - self.y1
        self.score = score 
        self.class_name = class_name
        self.instance_id = instance_id

    def get_center(self):
        x = int((self.x1 + self.x2) / 2)
        y = int((self.y1 + self.y2) / 2)
        return x, y

    def to_dict(self) -> dict:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2, 
            "y2": self.y2,
            "w": self.w,
            "h": self.h,
            "score": self.score, 
            "class_name": self.class_name,
            "instance_id": self.instance_id
        }
        
    def __repr__(self) -> str:
        return json.dumps(self.to_dict())

    def __str__(self):
        return json.dumps(self.to_dict())

class Detection9D():
    def __init__(self, x, y, z, l, w, h, alpha, beta, gamma, score, class_name, instance_id) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.l = l
        self.w = w
        self.h = h
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.score = score 
        self.class_name = class_name
        self.instance_id = instance_id

    def get_center(self):
        return [self.x, self.y, self.z]

    def get_size(self):
        return [self.l, self.w, self.h]

    def get_rotation_euler(self):
        return [self.alpha, self.beta, self.gamma]
    
    def get_rotation_matrix(self):
        return R.from_euler("xyz", np.array([self.alpha, self.beta, self.gamma])).as_matrix()
    
    def get_tensor(self):
        return [self.x, self.y, self.z, self.l, self.w, self.h, self.alpha, self.beta, self.gamma]

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "l": self.l,
            "w": self.w,
            "h": self.h,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "score": self.score, 
            "class_name": self.class_name,
            "instance_id": self.instance_id
        }
        
    def __repr__(self) -> str:
        return json.dumps(self.to_dict())

    def __str__(self):
        return json.dumps(self.to_dict())

class Detector():

    def __init__(self, mode): # : Literal["seg", "det"]
        self.shared_directory = "./share"
    
        self._set_cfg(mode)

        self.predictor = None
        self._set_predictor()
    
    def _set_cfg(self, mode) -> CfgNode: # : Literal["seg", "det"]
        cfg = get_cfg()

        if mode == "det":
            config_file, checkpoint_url, visualizer = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", SortVideoVisualizer
        elif mode == "seg":
            config_file, checkpoint_url, visualizer = "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml", "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml", VideoVisualizer
        elif mode == "keyp":
            config_file, checkpoint_url, visualizer = "COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml", "COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml", SortVideoVisualizer
        
        else:
            raise ValueError(f"Mode {mode} unkown")

        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        
        # cfg.MODEL.DEVICE = os.environ["GPU_ID"]
        # print("model running on", cfg.MODEL.DEVICE)

        if torch.cuda.device_count() == 0:
            cfg.MODEL.DEVICE = "cpu"
        else:
            cfg.MODEL.DEVICE = "cuda:0"

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
        
        self.cfg = cfg
        self.vis = visualizer(MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), ColorMode.IMAGE)

    def _set_predictor(self):
        os.environ["FVCORE_CACHE"] = str(Path(self.shared_directory, "artifacts"))
        self.predictor = DefaultPredictor(self.cfg)
        
    def get_prediction(self, image: np.ndarray) -> Instances:
        prediction = self.predictor(image)
        # {'instances': Instances(num_instances=0, image_height=216, image_width=384, fields=[pred_boxes: Boxes(tensor([], device='cuda:0', size=(0, 4))), scores: tensor([], device='cuda:0'), pred_classes: tensor([], device='cuda:0', dtype=torch.int64)])} 

        # https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        return prediction

    def visualize_prediction(self, image: np.ndarray, prediction: dict) -> np.ndarray:

        if "panoptic_seg" in prediction:
            panoptic_seg = prediction["panoptic_seg"]
            vis_output = self.vis.draw_panoptic_seg_predictions(image[:, :, ::-1], panoptic_seg[0].cpu(), panoptic_seg[1])
        else:
            print(prediction["instances"])
            vis_output = self.vis.draw_instance_predictions(image[:, :, ::-1], prediction["instances"].to("cpu"))

        return vis_output.get_image()[:, :, ::-1]


    def get_detection_classes_list(self) : # -> List["str"]
        """
        returns:
            a list, where the index is the class id and the item is the class name (str)
        """
        return MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes


def run(path_to_image: str, mode): # : Literal["seg", "det"]
    detector = Detector(mode=mode)
    image = np.array(Image.open(path_to_image))
    prediction = detector.get_prediction(image)
    visualization = detector.visualize_prediction(image, prediction)
    concatenated = np.concatenate([image, visualization], 1)
    plt.imshow(concatenated)
    plt.show()



if __name__ == "__main__":
    Fire(run)