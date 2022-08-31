# 2D bounding box detection & tracking
from typing import Dict, List, Union
from typing_extensions import Literal
from pathlib import Path

# imports
#from typing import Dict, List, Literal, Union
from pathlib import Path

import json
from detectron2.structures.instances import Instances
from detectron2.structures import Boxes

from scipy.spatial.transform import Rotation

import scipy.io as sio

from scipy.ndimage import median_filter as scp_mf

import numpy as np

from torch import tensor

import cv2

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.artist import Artist

from PIL import Image

import pandas as pd
from fire import Fire

# from pytransform3d import camera

# import open3d as o3d

from .object_detection import Detection, Detection9D, Detector
from .sort import Sort

from .DMPPE_ROOTNET_RELEASE.demo.inference import inference_root_net_single_frame

from .DMPPE_POSENET_RELEASE.demo.inference import inference_pose_net_single_frame

from .visualization import draw_3Dpose

from .DMPPE_POSENET_RELEASE.common.utils.vis import vis_keypoints, vis_3d_multiple_skeleton

from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
import torch

# root net
from .DMPPE_ROOTNET_RELEASE.main.config import cfg as root_cfg
from .DMPPE_ROOTNET_RELEASE.main.model import get_pose_net as  get_root_net

# pose net
from .DMPPE_POSENET_RELEASE.main.config import cfg as pose_cfg
from .DMPPE_POSENET_RELEASE.main.model import get_pose_net

import os.path as osp

from os import mkdir

import pathlib

import glob
from PIL import Image

from datetime import datetime

import open3d as o3d

import sys
sys.path.append(".")
sys.path.append("..")

from mmdet3d.apis.inference import direct_inference_mono_3d_detector, init_model, show_result_meshlab

from tqdm import tqdm

MAX_FRAMES = 10
det_id2str = {0: 'pedestrian', 1:'cyclist', 2:'car'}

from .ab3dmot.Xinshuo_PyToolbox.xinshuo_visualization.geometry_vis import random_colors
# TODO: mit mmdet_viz vereinheitlichen
from .ab3dmot.AB3DMOT_libs.mmdet.utils import draw_camera_bbox3d_on_img, points_cam2img

from .ab3dmot.AB3DMOT_libs.model import AB3DMOT #
from .ab3dmot.AB3DMOT_libs.mmdet.cam_box3d import CameraInstance3DBoxes

max_color = 30
colors = random_colors(max_color)       # Generate random colors
score_threshold = -10000

import pytransform3d.visualizer as pv

import open3d.visualization.rendering as rendering

from open3d import geometry

import itertools

# JSON Serialization
def _to_dict(obj):
    return obj.to_dict()

class CameraIntrinsics():
    def __init__(self, f_x=1492, f_y=1492, h=1920, w=1440, o_x=720, o_y=960) -> None: # iphone default intrinsics
        # The following for values must be in the same unit realtively – however, their absolute value doesn't matter and is not related to the data points
        self.f_x = f_x
        self.f_y = f_y
        self.h = h
        self.w = w
        self.o_x = o_x
        self.o_y = o_y

        # The following value is in units of the data points
        self.virtual_image_distance = 1

    @staticmethod
    def from_o3d_camera_json(path: str) -> "CameraIntrinsics":
        o3d_camera_dict = json.loads(Path(path).read_text())
        
        return CameraIntrinsics(
            o3d_camera_dict["intrinsic_matrix"][0],
            o3d_camera_dict["intrinsic_matrix"][4],
            o3d_camera_dict["height"],
            o3d_camera_dict["width"],
            o3d_camera_dict["intrinsic_matrix"][6],
            o3d_camera_dict["intrinsic_matrix"][7],
        )

    def to_o3d_intrinsics(self):
        return o3d.camera.PinholeCameraIntrinsic(
            self.w,
            self.h,
            self.f_x,
            self.f_y,
            self.o_x,
            self.o_y
        )

    def as_matrix_3x3(self) -> np.ndarray:
        return np.array([
                [self.f_x,  0,          self.o_x],
                [0,         self.f_y,   self.o_y],
                [0,         0,          1]
            ])
    
    def as_matrix_3x4(self) -> np.ndarray:
        return np.array([
                [self.f_x,  0,          self.o_x, 0],
                [0,         self.f_y,   self.o_y, 0],
                [0,         0,          1,        0]
            ])

    def to_dict(self):
        return {
            "f_x": self.f_x,
            "f_y": self.f_y,
            "h": self.h,
            "w": self.w,
            "o_x": self.o_x,
            "o_y": self.o_y
        }

    def __repr__(self) -> str:
        return json.dumps(self.to_dict())

class VideoFrames():
    def __init__(self, filename, downsampling=1, throttle=1, flip=False, landscape=False, pad=[], resize=[]) -> None:
        self.filename = filename
        self.capture = cv2.VideoCapture(filename) #apiPreference=cv2.CAP_MSMF
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.idx = 0
        self.downsampling = downsampling
        self.first_frame = None
        self.throttle = throttle
        self.flip = flip
        self.landscape = landscape
        self.pad = pad
        self.resize = resize


    def __iter__(self):
        return self

    def __next__(self):
        success, frame = self.capture.read()
        if not success:
            raise StopIteration

        self.idx += 1

        frame = cv2.resize(frame, self.get_output_size())

        if len(self.resize) == 2:
            frame = cv2.resize(frame, self.resize)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #  not needed, frames are processed as BGR by openCV
        
        if self.flip: 
            frame = cv2.flip(frame, 0)
            frame = cv2.flip(frame, 1)
        
        if self.landscape:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) # cv2.ROTATE_90_COUNTERCLOCKWISE
        
        if len(self.pad) == 4:
            top = self.pad[0]
            bottom = self.pad[1]
            left = self.pad[2]
            right = self.pad[3]
            frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        return frame

    def get_output_size(self):
        
        # 
        if len(self.resize) == 2:
            return int(self.resize[0] + self.pad[2] + self.pad[3]), int(self.resize[1] + self.pad[0] + self.pad[1]) 
        else:
            return int(self.width * self.downsampling), int(self.height * self.downsampling)
            

    def frame_at(self, idx):
        if idx == 0:
            if self.first_frame is not None: 
                return self.first_frame
            else:
                self.first_frame = next(self)
                return self.first_frame

        if idx < self.idx:
            raise ValueError(f"Can only skip, cannot jump back, or re-read!, current idx: {self.idx}, desired idx: {idx}")
        
        if idx == self.idx:
            return next(self)
        
        assert idx >= self.idx

        diff = idx - self.idx
        for _ in range(diff):
            next(self)

        return next(self)

    def show_interactively(self):
        frame = next(self)
        while frame is not None:
            frame = next(self)
            cv2.imshow("", frame)
            cv2.waitKey(int(1000 / self.fps))

    def __len__(self) -> int:
        return self.frame_count

class Perception():
    def _instances_to_detections(self, instances: Instances, class_ids): #  -> List[Detection]
        detections = list()
        for i in range(len(instances)):
            x1, y1, x2, y2 = instances.pred_boxes.tensor[i].tolist()
            # x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
            score, pred_class = instances.scores[i].item(), class_ids[instances.pred_classes[i]]
            
            instance_id = None
            if instances.has("ids"):
                instance_id = instances.ids[i].item()

            detection = Detection(x1, y1, x2, y2, score, pred_class, instance_id)
            detections.append(detection)
        return detections

    def _instances_to_detections9d(self, instances: dict) -> List[Detection9D]:
        detections = list()
        for i in range(len(instances['scores_3d'])):
            box = instances['boxes_3d'].tensor[i].tolist()
            if len(box) == 9: 
                x, y, z, l, w, h, alpha, beta, gamma = box
            elif len(box) == 7: 
                x, y, z, l, w, h, beta = box
                alpha = 0.
                gamma = 0.
            else:
                print("Send help!")
            # x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
            score, pred_class = instances['scores_3d'][i].item(), instances['labels_3d'][i].item()
            
            instance_id = None
            if 'ids' in instances.keys():
                instance_id = instances['ids'][i].item()

            detection = Detection9D(x, y, z, l, w, h, alpha, beta, gamma, score, det_id2str[pred_class], instance_id)
            detections.append(detection)
        return detections
    
    def set_intrinsics(self, focal, princpt):
        self.focal=focal
        self.princpt=princpt
         
    def track_human_sequence(self, video_frames: VideoFrames, visualize=False, score_threshold=.8, mode="det", file_name="None", fps_divisor=1): #-> Dict[int, List[Detection]]

        detector = Detector(mode)
        
        result_file = file_name.split("/")[-1].split(".")[0]

        date = datetime.now().strftime("%Y%m%d_%I%M%S")
        print(f"filename_{date}")

        root_dir = f"./output3/{date}-" + result_file + "-humans/"

        pathlib.Path(root_dir).mkdir(parents=True, exist_ok=True) 

        output_fps = int(video_frames.fps / fps_divisor)  #* video_frames.frame_count
        print("OutputFPS: ", output_fps)
        print("VideoOriginalFPS: ", video_frames.fps)


        if visualize: 
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")
            out = cv2.VideoWriter(root_dir + result_file + "_bbox_detections.mp4", fourcc, output_fps, video_frames.get_output_size()) # video_frames.fps
            #fourcc_gif = cv2.VideoWriter_fourcc('G','I','F')
            out_2d = cv2.VideoWriter(root_dir + result_file + "_keypoint_2d.mp4", fourcc, output_fps, video_frames.get_output_size())
            #out_3d = cv2.VideoWriter(root_dir + result_file + "_keypoint_3d.mp4", fourcc, video_frames.fps, video_frames.get_output_size())

        
        frame_id = 0

        sort_per_class_id = dict()

        # save intermediate results
        raw_detection_sequence = {}
        tracked_detection_sequence = {}
        pred_2d_save = {}
        pred_3d_save = {}

        # instantiate root model # snapshot load
        root_model_path = './perception_pipeline/DMPPE_ROOTNET_RELEASE/demo/rootnet_18.pth.tar' # './snapshot_%d.pth.tar' % int(args.test_epoch)
        assert osp.exists(root_model_path), 'Cannot find model at ' + root_model_path
        print('Load checkpoint from {}'.format(root_model_path))
        root_model = get_root_net(root_cfg, False)
        root_model = DataParallel(root_model).cuda()
        root_ckpt = torch.load(root_model_path)
        root_model.load_state_dict(root_ckpt['network'])
        root_model.eval()


        # instantiate pose model # snapshot load
        pose_model_path = './perception_pipeline/DMPPE_POSENET_RELEASE/demo/posenet_24.pth.tar'
        assert osp.exists(pose_model_path), 'Cannot find model at ' + pose_model_path
        print('Load checkpoint from {}'.format(pose_model_path))
        joint_num = 21
        pose_model = get_pose_net(pose_cfg, False, joint_num)
        pose_model = DataParallel(pose_model).cuda()
        pose_ckpt = torch.load(pose_model_path)
        pose_model.load_state_dict(pose_ckpt['network'])
        pose_model.eval()

        person_id = 0

        # TODO: load networks outside of loop

        image_list_2d = []

        image_list_3d = []
        
        for frame in tqdm(iter(video_frames)):

            if frame_id % fps_divisor == 0:

                print("Frame ID: ", frame_id)

                frame_original = frame.copy()

                # SIMPLE FRAME-BASED BOUNDING BOX PREDICTIONS
                prediction = detector.get_prediction(frame)
                instances = prediction["instances"].to("cpu")
                instances = instances[instances.scores > score_threshold]

                # filter for person
                instances = instances[instances.pred_classes == person_id]

                prediction["instances"] = instances
                raw_detections = self._instances_to_detections(instances, detector.get_detection_classes_list())
                # print(frame_id, detections)
                raw_detection_sequence[frame_id] = [raw_detection for raw_detection in raw_detections]                    

                # CLASS-WISE TRACKING
                tracked_detection_sequence[frame_id] = []

                class_ids = np.unique(instances.pred_classes.numpy())

                for class_id in class_ids:
                    if not class_id in sort_per_class_id:
                        sort_per_class_id[class_id] = Sort(max_age=20, min_hits=7, iou_threshold=.2)
                    
                    sort = sort_per_class_id[class_id]
                    
                    class_detections = instances[instances.pred_classes == class_id]
                    dets = np.concatenate([class_detections.pred_boxes.tensor.numpy(), class_detections.scores.unsqueeze(axis=1).numpy()], axis=1)
                    dets_with_id = sort.update(dets)
                    ids = dets_with_id[:,4].astype(np.int32)

                    tracked_instances = Instances(
                            instances.image_size, 
                            pred_boxes=Boxes(tensor(dets_with_id[:,0:4])), 
                            pred_classes=tensor([class_id]*dets_with_id.shape[0]),
                            ids=tensor(ids),
                            scores=tensor([0]*dets_with_id.shape[0])) # the tracking algorithm unfortunately eliminated this field

                    tracked_detections = self._instances_to_detections(tracked_instances, detector.get_detection_classes_list())

                    tracked_detection_sequence[frame_id].extend([tracked_detection for tracked_detection in tracked_detections])


                    
                    if visualize:
                        # Visualization
                        frame = detector.visualize_prediction(frame, {
                            "instances": tracked_instances
                        })

                # tracked_detections: single frame input for root and pose net

                # 2d detections
                bbox_list = [[det.x1, det.y1, det.w, det.h] for det in tracked_detection_sequence[frame_id]]

                # TODO: check if tracking id changes in root net and posenet - or if the order stays the same
                object_id_list = [det.instance_id for det in tracked_detection_sequence[frame_id]]

                # root_net inference:

                root_depth_list = inference_root_net_single_frame(frame_original.copy(), bbox_list, root_model, focal=self.focal, princpt=self.princpt)
                # TODO: save

                # pose_net inference
                output_pose_2d_list, output_pose_3d_list, vis_img = inference_pose_net_single_frame(frame_original.copy(), bbox_list, root_depth_list, pose_model, focal=self.focal, princpt=self.princpt)
                # TODO: save

                if visualize: 
                    cv2.imwrite(root_dir + result_file + '_pose_2d_{:04d}.jpg'.format(frame_id), vis_img)

                # visualize 3d human poses in surface plot

                # MuCo joint set
                joint_num = 21
                joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
                flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
                skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )

                pred_3d_kpt = np.array(output_pose_3d_list)
                pred_2d_kpt = np.array(output_pose_2d_list)
                pred_3d_save[frame_id] = output_pose_3d_list
                pred_2d_save[frame_id] = output_pose_2d_list

                pose_3d_plus_id = dict(zip(object_id_list, output_pose_3d_list))
                with open(root_dir + "hkp3d_{:04d}.json".format(frame_id), "wt") as file:
                    json.dump(pose_3d_plus_id, file)

                pose_2d_plus_id = dict(zip(object_id_list, output_pose_2d_list))
                with open(root_dir + "hkp2d_{:04d}.json".format(frame_id), "wt") as file:
                    json.dump(pose_2d_plus_id, file)
                
                # TODO: 
                if visualize and len(root_depth_list) > 0:
                    vis_img_3d = vis_3d_multiple_skeleton(pred_3d_kpt, np.ones_like(pred_3d_kpt), skeleton, root_dir + result_file, vis_img, frame_id, original=False)

                # draw_3Dpose(vis_img, np.array(output_pose_2d_list), np.array(output_pose_3d_list))

                if visualize: 
                    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    # for debugging:
                    
                    cv2.imwrite(root_dir + "{}_bboxes_frame_{:04d}.jpg".format(result_file, frame_id), frame)
                    out.write(frame)
                    out_2d.write(vis_img)
                    #if len(root_depth_list) > 0:
                        #out_3d.write(vis_img_3d)
                
                

            frame_id += 1
            print(".", end="")
            

            # print("root_depth_list", root_depth_list)
            # print("output_pose_3d_list", output_pose_3d_list)
        
        #sio.savemat(root_dir + "preds_3d_kpt_mupots.mat" , pred_3d_save)
        #sio.savemat(root_dir +" preds_2d_kpt_mupots.mat", pred_2d_save)

        print(pred_3d_save[0])

        with open(root_dir + "preds_3d_kpt_mupots.json", "wt") as file:
            json.dump(pred_3d_save, file)
        
        with open(root_dir + "preds_2d_kpt_mupots.json", "wt") as file:
            json.dump(pred_2d_save, file)

        with open(root_dir + "human_raw_detections.json", "wt") as file:
            json.dump(raw_detection_sequence, file, default=_to_dict)
        
        with open(root_dir + "human_tracked_detections.json", "wt") as file:
            json.dump(tracked_detection_sequence, file, default=_to_dict)

        if visualize:
            out.release() 
            out_2d.release() 
            #out_3d.release() 
        
            img_array = []
            
            # print(glob.glob(root_dir + '/' + result_file + '_pose_3d_*.jpg'))
            list_3d_images = glob.glob(root_dir + '/' + result_file + '_pose_3d_*.jpg')
            # for filename in list_3d_images:
            #     img = cv2.imread(filename)
            #     height, width, layers = img.shape
            #     size = (width,height)
            #     img_array.append(img)

            output_file_name = root_dir + result_file + "_keypoint_3d.mp4"
            output_file_name = root_dir + result_file + "_keypoint_3d.gif"

            frames = [Image.open(image) for image in sorted(list_3d_images)]

            print(len(frames))

            print(list_3d_images)
            frame_one = frames[0]


            sec = video_frames.frame_count / video_frames.fps
            test = len(frames) / sec
            print(test)
            print(1000/test)
            
            frame_one.save(output_file_name, format="GIF", append_images=frames,
                save_all=True, duration= 1000/test, loop=0) # int(len(frames) / output_fps) *

        # out_3d = cv2.VideoWriter(output_file_name, fourcc, output_fps, size)
 
        # for i in range(len(img_array)):
        #     out_3d.write(img_array[i])
        # out_3d.release()

        return tracked_detection_sequence

    def track_3d_bbbox_sequence(self, video_frames: VideoFrames, visualize=False, score_threshold=-0.1, intrinsics=None, file_name="", args_config=None, args_checkpoint=None, fps_divisor=1) -> Dict[int, List[Detection9D]]:

        if args_config is None:
            args_config = './input/smoke9D_objectron_generalize_full_ds_multigpu_two_classes_video.py'
            args_checkpoint = './output/generalize_chair_book_2022_05_04_multi_full/epoch_10.pth'

        args_device = 'cuda:0'

        result_file = file_name.split("/")[-1].split(".")[0]

        date = datetime.now().strftime("%Y%m%d_%I%M%S")
        print(f"filename_{date}")

        root_dir = f"./output3/{date}-" + result_file + "-cars/"

        pathlib.Path(root_dir).mkdir(parents=True, exist_ok=True) 

        output_fps = int(video_frames.fps / fps_divisor) # * video_frames.frame_count
        if output_fps ==0:
            output_fps = 1
        print("OutputFPS: ", output_fps)
        print("VideoOriginalFPS: ", video_frames.fps)
        
        detector = init_model(args_config, args_checkpoint, device=args_device)

        #print(intrinsics.as_matrix_3x3())

        #detector = Detector("det")
        
        if visualize: 
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")
            #out_raw = cv2.VideoWriter(root_dir + result_file + "_3dbbox_detections_raw.mp4", fourcc, video_frames.fps, video_frames.get_output_size())
            out = cv2.VideoWriter(root_dir + result_file + "_3dbbox_detections_tracked.mp4", fourcc, output_fps, video_frames.get_output_size())
        
        frame_id = 0

        sort_per_class_id = dict()

        raw_detection_sequence = {}
        tracked_detection_sequence = {}

        mot_tracker = AB3DMOT(max_age=20, min_hits=7) 

        print("Video Count: ", video_frames.frame_count)
        
        for frame in tqdm(iter(video_frames)):

            if frame_id % fps_divisor == 0:

                #cv2.imwrite(root_dir + result_file + '_raw_image_{:04d}.jpg'.format(frame_id), frame)

                # SIMPLE FRAME-BASED PREDICTIONS

                #prediction = detector.get_prediction(frame)

                # TODO: resize image and camera intrinsics
                result, data = direct_inference_mono_3d_detector(detector, frame, intrinsics.as_matrix_3x3())

                result = result[0]['img_bbox']
                # print(result)
                # print(result['boxes_3d'].tensor)
                # print(len(result['scores_3d']))
                # print(result['labels_3d'])
                
                # instances = Instances(image_size = frame.shape, fields={
                #     'pred_boxes': result['boxes_3d'],  #.tensor
                #     'scores': result['scores_3d'], 
                #     'pred_classes': result['labels_3d']})

                # print(instances)
                # print(instances.pred_boxes)

                output = []
                for ann_id in range(len(result['scores_3d'])):
                    class_type = result['labels_3d'][ann_id]
                    box_2d = [-1, -1, -1, -1]
                    score = result['scores_3d'][ann_id]
                    box_3d = result['boxes_3d'].tensor[ann_id].tolist()
                    if len(box_3d) == 7:
                        box_3d = box_3d[:6] + [0., box_3d[6], 0.]

                    output.append([frame_id] + [class_type.tolist()] + box_2d + [score.tolist()] + box_3d)
                
                #print(output)
                
                #result['boxes_3d'].tensor, result['scores_3d'], result['labels_3d']

                

                #{'instances': Instances(num_instances=0, image_height=216, image_width=384, fields=[pred_boxes: Boxes(tensor([], device='cuda:0', size=(0, 4))), scores: tensor([], device='cuda:0'), pred_classes: tensor([], device='cuda:0', dtype=torch.int64)])}

                #instances = instances[instances.scores > score_threshold]
                #TODO: prediction["instances"] = instances

                #results to instances

                # function results to detections
                # TODO: raw_detections = self._instances_to_detections(instances, detector.get_detection_classes_list())

                raw_detections =self._instances_to_detections9d(result)

                #print(raw_detections)

                # print(frame_id, detections)
                raw_detection_sequence[frame_id] = [raw_detection for raw_detection in raw_detections]

                with open(root_dir + "cbbox_{:04d}.json".format(frame_id), "wt") as file:
                    json.dump(raw_detection_sequence[frame_id], file, default=_to_dict)          

                # CLASS-WISE TRACKING
                # AB3DMOT

                tracked_detection_sequence[frame_id] = []

                dets = np.array(output)
                # print(dets.shape)
                # print("len", len(dets))
                # if len(dets) == 0: 
                #     print("No detections")
                #     dets = np.empty((1,16))
                #     dets[:] = np.NaN
                if len(dets.shape) == 1: dets = np.expand_dims(dets, axis=0) 

                # print("Dets: ", dets)
                # print("Dets shape: ", dets.shape)
                # print(dets.size)

                # frame, class,  x1, y1, w, h,  score
                # 0,     1,      2, 3, 4, 5,    6
                additional_info = dets[: , 1:7] # dets[:, 0] == frame		

                # x, y, z, h, w, l, alpha, beta, gamma in camera coordinate follwing mmdet convention
                # 7, 6, 8,  9, 10, 11,  12, 13, 14
                dets = dets[: , 7:16] # seq_dets[:,0] == frame
                dets_all = {'dets': dets, 'info': additional_info}

                # TODO: track in global coordinates, not camera coordinates
                trackers = mot_tracker.update(dets_all)

                tracked_detections = []
                # saving results, loop over each tracklet			
                for d in trackers:
                    #print(d)
                    bbox3d_tmp = d[0:9]       # h, w, l, x, y, z, theta in camera coordinate
                    #print(bbox3d_tmp)
                    #print(" ID : ", d[9])
                    id_tmp = d[9]
                    #print(id_tmp)
                    # ori_tmp = d[8:11]
                    #print('Class: ', d[10])
                    pred_class = d[10]
                    type_tmp = det_id2str[pred_class] # 11
                    bbox2d_tmp_trk = d[11:15] # 16
                    conf_tmp = d[15]

                    # save in detection format with track ID, can be used for dection evaluation and tracking visualization
                    str_to_srite = '%s -1 -1 %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d\n' % (type_tmp,
                        bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
                        bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[3], bbox3d_tmp[4], bbox3d_tmp[5], 
                        bbox3d_tmp[6], bbox3d_tmp[7], bbox3d_tmp[8], conf_tmp, id_tmp)
                    print(str_to_srite)

                    x, y, z, l, w, h, alpha, beta, gamma = bbox3d_tmp
                    detection = Detection9D(x, y, z, l, w, h, alpha, beta, gamma, conf_tmp, type_tmp, id_tmp)
                    tracked_detections.append(detection)


                # Visualization

                # load detections, N x 15
                #dets = pass

                # class_ids = np.unique(instances.pred_classes.numpy())
                # for class_id in class_ids: # probably not needed in new tracker
                #     if not class_id in sort_per_class_id:
                #         sort_per_class_id[class_id] = Sort(max_age=20, min_hits=7, iou_threshold=.2)
                    
                #     sort = sort_per_class_id[class_id]
                    
                #     class_detections = instances[instances.pred_classes == class_id]
                #     dets = np.concatenate([class_detections.pred_boxes.tensor.numpy(), class_detections.scores.unsqueeze(axis=1).numpy()], axis=1)
                #     dets_with_id = sort.update(dets)
                #     ids = dets_with_id[:,4].astype(np.int32)

                # tracked_instances = Instances(
                #         instances.image_size, 
                #         pred_boxes=Boxes(tensor(dets_with_id[:,0:4])), 
                #         pred_classes=tensor([class_id]*dets_with_id.shape[0]),
                #         ids=tensor(ids),
                #         scores=tensor([0]*dets_with_id.shape[0])) # the tracking algorithm unfortunately eliminated this field

                # tracked_detections = self._instances_to_detections(tracked_instances, detector.get_detection_classes_list())
                    

                tracked_detection_sequence[frame_id].extend([tracked_detection for tracked_detection in tracked_detections])
                    
                if visualize:
                    # Visualization
                    # frame = detector.visualize_prediction(frame, {
                    #     "instances": tracked_instances
                    # })

                    #print(tensor)

                    image_tmp = frame
                    for det in tracked_detections:
                        
                        tensor = det.get_tensor()
                        # print(tensor)
                        # tensor[2] = tensor[2] / 1.87
                        # print(tensor)
                        boxes = CameraInstance3DBoxes(np.array([tensor]))

                        if det.score < score_threshold:
                            continue

                        color_tmp = tuple([int(tmp * 255) for tmp in colors[int(det.instance_id) % max_color]])
                        

                        # TODO: add class label to plot
                        print(det.instance_id)
                        intm = intrinsics.as_matrix_3x3()
                        #intm[:2, :2] = intm[:2, :2] / 1.87
                        image_tmp = draw_camera_bbox3d_on_img(
                        boxes, image_tmp, intm, None, color=color_tmp, thickness=1, object_id=int(det.instance_id), class_type=det.class_name)

                if visualize: 
                    #image_tmp = cv2.cvtColor(image_tmp, cv2.COLOR_RGB2BGR)
                    out.write(image_tmp)

                if visualize:
                    
                    image_tmp = frame
                    for det in raw_detections:
                        tensor = det.get_tensor()
                        #tensor[2] = tensor[2] / 1.87
                        boxes = CameraInstance3DBoxes(np.array([tensor]))

                        if det.score < score_threshold or det.class_name != "car":
                            continue

                        # TODO: resolve later
                        det.instance_id = 0

                        color_tmp = tuple([int(tmp * 255) for tmp in colors[int(det.instance_id) % max_color]])
                        

                        # TODO: add class label to plot
                        print(det.instance_id)

                        intm = intrinsics.as_matrix_3x3()
                        #intm[:2, :2] = intm[:2, :2] / 1.87
                        print("Intrinscis: ", intm)

                        image_tmp = draw_camera_bbox3d_on_img(
                        boxes, image_tmp, intm, None, color=color_tmp, thickness=1, object_id=int(det.instance_id), class_type=det.class_name)

                        cv2.imwrite(root_dir + result_file + '_bbox_car_{:04d}.jpg'.format(frame_id), image_tmp)

                        # out_raw.write(image_tmp)

            frame_id += 1
            #print(".", end="")

            # if frame_id > 20:
            #     break

        with open(root_dir + "car_raw_detections.json", "wt") as file:
            json.dump(raw_detection_sequence, file, default=_to_dict)
        
        with open(root_dir + "car_tracked_detections.json", "wt") as file:
            json.dump(tracked_detection_sequence, file, default=_to_dict)

        if visualize:
            out.release() 
            #out_raw.release()

        return tracked_detection_sequence

class KeyFrameSelector():
    def __init__(self, ) -> None:
        self.filename = filename

    def get_distance_two_points(joint_one, joint_two):
        return np.linalg.norm(joint_one - joint_two)

    def get_minimum_distance_point_obb(box_tensor, point):
        # get the minimum distance of an arbitrary point in 3D space to the oriented bounding box

        rot_axis=2
        center = bbox3d[i, 0:3]
        dim = bbox3d[i, 3:6]
        yaw = np.zeros(3)
        yaw[rot_axis] = bbox3d[i, 6]
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)

        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)

        box3d.TriangleMesh

        line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set.paint_uniform_color(bbox_color)

        indices = box3d.get_point_indices_within_bounding_box(pcd.points)
        points_colors[indices] = in_box_color

        # draw bboxes on visualizer
        # vis.add_geometry(line_set)

        return None

class Event():
    def __init__(self, event_type, data, time_stamp) -> None:
        self.type = event_type
        self.time_stamp = time_stamp
        self.event_data = data

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "time_stamp": self.time_stamp,
            "event_data": self.event_data
        }

class CLI():
    @staticmethod
    def detect_old(mov_filename, mode, custom=True, landscape=False, fps_divisor=1):
        #camera_intrinsics = CameraIntrinsics()
        #trajectory = Trajectory(csv_filename)

        print(custom)
        print(landscape)
        print(fps_divisor)

        video_frames = VideoFrames(mov_filename, flip=custom, landscape=landscape)

        perception = Perception()

        if custom:
            perception.set_intrinsics([1447, 1447], [924, 720]) # portrait
        else:
            perception.set_intrinsics([1500, 1500], None) # standard

        if landscape: 
            perception.set_intrinsics([1447, 1447], [962, 682])# landscape

        tracked_detection_sequence = perception.track_human_sequence(video_frames, visualize=True, mode=mode, file_name=mov_filename,  fps_divisor=fps_divisor)
        
        #video_frames = VideoFrames(mov_filename)


        # output for a single input video:

        # dictionary: for every frame:
        # list of person keypoints in 3d 
        # list of object 3d bounding boxes



        # + SMOKE Output

    @staticmethod
    def detect_humans(mov_filename, mode, custom=True, landscape=False, fps_divisor=1, visualize=False):
        #camera_intrinsics = CameraIntrinsics()
        #trajectory = Trajectory(csv_filename)

        print(custom)
        print(landscape)
        print(fps_divisor)

        video_frames = VideoFrames(mov_filename, flip=custom, landscape=landscape)

        perception = Perception()

        if custom:
            perception.set_intrinsics([1447, 1447], [924, 720]) # portrait
        else:
            perception.set_intrinsics([1500, 1500], None) # standard

        if landscape: 
            # kitti intrinsics: 721, 721
            perception.set_intrinsics([1447, 1447], [962, 753])# landscape # other way round: 682

        tracked_detection_sequence = perception.track_human_sequence(video_frames, visualize=visualize, mode=mode, file_name=mov_filename,  fps_divisor=fps_divisor)
        
        #video_frames = VideoFrames(mov_filename)


        # output for a single input video:

        # dictionary: for every frame:
        # list of person keypoints in 3d 
        # list of object 3d bounding boxes

    @staticmethod
    def detect_cars(mov_filename, kitti=True, custom=True, landscape=False, fps_divisor=1, args_config=None, args_checkpoint=None, f_x=1445, f_y=1445, h=1440, w=1920, o_x=960, o_y=753):
        
        video_frames = VideoFrames(mov_filename, downsampling=1, throttle=1, flip=False, landscape=False, pad=[])

        # load camera intrinsics from json

        if kitti:
            kitti_w = 1280 
            kitti_h = 384

            resize_factor = h/kitti_h

            f_x = int(f_x / resize_factor)
            f_y = int(f_y / resize_factor)
            # w = 530
            # h = 384

            w = int(w / resize_factor)
            h = int(h / resize_factor)

            o_x = int(o_x / resize_factor)
            pad_x = int((kitti_w - w) / 2)
            o_x += pad_x
            o_y = int(o_y / resize_factor)

            # 1440/384 = 3.75
            # 1920/3.75 = 530

            # 1280-530 = 750
            # 750 / 2 = 375
            video_frames = VideoFrames(mov_filename, downsampling=1, throttle=1, flip=True, landscape=True, pad=[0,0,pad_x, pad_x], resize=(h, w))

            # f_x = 
            # f_y = 

            # h = 
            # w = 

            # o_x = 
            # o_y = 

            # TODO: scaling of images for pretrained smoke

        else:
            f_x = 385
            f_y = 385
            # w = 530
            # h = 384
            o_x = 257
            o_x += 375
            o_y = 182

            w = 1280
            h = 384

            # 1440/384 = 3.75
            # 1920/3.75 = 530

            # 1280-530 = 750
            # 750 / 2 = 375
            video_frames = VideoFrames(mov_filename, pad=[0,0,375, 375], resize=(530, 384))

        f_x = 721 # TODO: check why hardcoding the focal length like kitti images is better than taking the original one
        f_y = 721
        print("Intrinsics: ", [f_x, f_y, h, w, o_x, o_y])

        camera_intrinsics = CameraIntrinsics(f_x, f_y, h, w, o_x, o_y)
        #camera_intrinsics[:2] *= res_divisor
        #trajectory = Trajectory(csv_filename)
        

        # debug video_frames
        cnt = 0
        # for frame in video_frames:
        #     print(cnt)

        #     # load & extract frame
            
        #     # save frame without detections

        #     cnt += 1

        output_name = mov_filename.split("/")[-1].split(".")[0]

        args_config = './configs/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_video.py'
        args_checkpoint = './configs/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_original.pth'

        perception7d = Perception()
        # todo: output without tracking of ab3dmot
        tracked_detection_sequence = perception7d.track_3d_bbbox_sequence(video_frames, visualize=True, intrinsics=camera_intrinsics, file_name=mov_filename, args_config=args_config, args_checkpoint=args_checkpoint, fps_divisor=fps_divisor)

    @staticmethod
    def visualize_cars(input_folder='./output3/20220818_032614-car_multi_person_1-cars', frame_index=0, f_x=1445, f_y=1445, h=1440, w=1920, o_x=960, o_y=753):
        
        list_car = glob.glob(input_folder+ '/cbbox_*.json')
        list_car.sort()

        bboxes_list = []

        for file_name in list_car:
            with open(file_name, "rt") as file:
                car_frame = json.load(file)
            
            bboxes_list.append(car_frame)

        # for frame_id in range(len(list_car)):
        video_id = int(list_car[frame_index].split('_')[-1].split('.')[0])

        pred_bboxes = [Detection9D(**det) for det in bboxes_list[video_id]]

        video_name= input_folder.split('-')[-2]

        video_frames = VideoFrames('./input_data/' + video_name +'.mov', flip=True, landscape=True)

        frame = video_frames.frame_at(video_id)

        # kitti_w = 1280 
        # kitti_h = 384

        # resize_factor = h/kitti_h

        # f_x = int(f_x / resize_factor)
        # f_y = int(f_y / resize_factor)
        # # w = 530
        # # h = 384

        # w = int(w / resize_factor)
        # h = int(h / resize_factor)

        # o_x = int(o_x / resize_factor)
        # pad_x = int((kitti_w - w) / 2)
        # o_x += pad_x
        # o_y = int(o_y / resize_factor)

        intrinsics = CameraIntrinsics(f_x, f_y, h, w, o_x, o_y)

        count = 0
        image_tmp = frame
        for det in pred_bboxes:
            tensor = det.get_tensor()
            tensor[2] = tensor[2] / 1.87
            # tensor = tensor[]
            boxes = CameraInstance3DBoxes([np.take(tensor, [0,1,2,3,4,5,7])], box_dim=7)

            if det.score < score_threshold or det.class_name != "car":
                continue

            # TODO: resolve later
            det.instance_id = count

            color_tmp = tuple([int(tmp * 255) for tmp in colors[int(det.instance_id) % max_color]])
            

            # TODO: add class label to plot
            print(det.instance_id)

            intm = intrinsics.as_matrix_3x3()
            #intm[:2, :2] = intm[:2, :2] / 1.87
            print("Intrinscis: ", intm)

            image_tmp = draw_camera_bbox3d_on_img(
            boxes, image_tmp, intm, None, color=color_tmp, thickness=3, object_id=int(det.instance_id), class_type=det.class_name)

            count += 1

        print(input_folder + '/bbox_car_viz{}.jpg'.format(video_id))
        cv2.imwrite(input_folder + '/bbox_car_viz{}.jpg'.format(video_id), image_tmp)

    @staticmethod
    def visualize_humans():
        pass

    # TODO: 
    @staticmethod
    def visualize_3D(input_file='./output2/20220816_075218-car_multi_person_1-scene_frames.json', frame_index=0):
        # load humans
        with open(input_file, "rt") as file:
            scene_frames_dict = json.load(file)

        print(scene_frames_dict.keys())
        print(scene_frames_dict[str(frame_index)])

        human_cars_dict = scene_frames_dict['0']

        pred_humans = human_cars_dict["humans"]
        
        keypoint_ids, keypoint_list = zip(*pred_humans.items())

        print(keypoint_ids)

        pred_objects = human_cars_dict["cars"]

        pred_bboxes = [Detection9D(**det) for det in pred_objects]

        print(pred_bboxes[0].get_tensor())


        date = datetime.now().strftime("%Y%m%d_%I%M%S")
        folder_name = input_file.split('-')[-2]
        output_path = f"./output3/{date}-" + folder_name + "-3d_viz_{:04d}".format(frame_index)
        print(output_path)


        # MuCo joint set
        joint_num = 21
        joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
        flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
        skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )
        
        # matplotlib

        

        video_frames = VideoFrames('./input_data/' + folder_name +'.mov', flip=True, landscape=True)

        frame = video_frames.frame_at(frame_index)

        pred_3d_kpt = np.array(keypoint_list)

        vis_img_3d = vis_3d_multiple_skeleton(pred_3d_kpt, np.ones_like(pred_3d_kpt), skeleton, output_path, frame, frame_index, original=False)

        # keypoint_list
        # pred_bboxes

        # fig = plt.figure(figsize= [20,10])
        # ax_0 = fig.add_subplot(121)
        # ax_0.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.axis('off')

        # ax = fig.add_subplot(122, projection='3d')

        # # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
        # cmap = plt.get_cmap('rainbow')
        # colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
        # colors = np.array([[c[2], c[1], c[0]] for c in colors])
        # # colors = [np.array((c[2], c[1], c[0])) for c in colors]

        # for l in range(len(kps_lines)):
        #     i1 = kps_lines[l][0]
        #     i2 = kps_lines[l][1]

        #     person_num = kpt_3d.shape[0]
        #     for n in range(person_num):
        #         x = np.array([kpt_3d[n,i1,0], kpt_3d[n,i2,0]])
        #         y = np.array([kpt_3d[n,i1,1], kpt_3d[n,i2,1]])
        #         z = np.array([kpt_3d[n,i1,2], kpt_3d[n,i2,2]])

        #         # if x.max() > 12.000:
        #         #     continue

        #         # x z -y vs. -z x -y

        #         if original:
        #             if kpt_3d_vis[n,i1,0] > 0 and kpt_3d_vis[n,i2,0] > 0:
        #                 ax.plot(x, z, -y, c=colors[l], linewidth=2)
        #             if kpt_3d_vis[n,i1,0] > 0:
        #                 ax.scatter(kpt_3d[n,i1,0], kpt_3d[n,i1,2], -kpt_3d[n,i1,1], c=colors[l], marker='o')
        #             if kpt_3d_vis[n,i2,0] > 0:
        #                 ax.scatter(kpt_3d[n,i2,0], kpt_3d[n,i2,2], -kpt_3d[n,i2,1], c=colors[l], marker='o')

        #         else:
        #             if kpt_3d_vis[n,i1,0] > 0 and kpt_3d_vis[n,i2,0] > 0:
        #                 ax.plot(-z, x, -y, c=colors[l], linewidth=2)
        #             if kpt_3d_vis[n,i1,0] > 0:
        #                 ax.scatter(-1 * kpt_3d[n,i1,2], kpt_3d[n,i1,0], -kpt_3d[n,i1,1], c=colors[l], marker='o')
        #             if kpt_3d_vis[n,i2,0] > 0:
        #                 ax.scatter(-1 * kpt_3d[n,i2,2], kpt_3d[n,i2,0], -kpt_3d[n,i2,1], c=colors[l], marker='o')    

        #fig = pv.figure(width=2000, height=2000)
        #fig.set_line_width(2)



        render = rendering.OffscreenRenderer(2000, 2000)

        geometry_list = []
        for person_idx in range(len(keypoint_ids)):
            
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(pred_3d_kpt[person_idx]),
                lines=o3d.utility.Vector2iVector(skeleton),
                )
            line_set.colors = o3d.utility.Vector3dVector(colors)
            
            geometry_list.append(line_set)
            render.scene.add_geometry(f"human_{person_idx}", mesh , mtl)
            #fig.add_geometry(line_set)

            render.scene.add_geometry(line_set)

        
        #coord = np.eye(4)
        #coord[2,2] *=-1
        #fig.plot_transform(coord)
        # fig.add_geometry(geometry_list)
        #fig.save_image(output_path + "_o3d.jpg")

        


        # render.scene.set_background([0, 0, 0, 0])

        

        # o3d.visualization.draw_geometries(geometry_list)
        
        #render.scene.camera.look_at([0, 0, 0], [0, 10, 0], [0, 0, 1])

        render.scene.show_axes(True)

        img = render.render_to_image()
        o3d.io.write_image(output_path + "_o3d_test_1.jpg", img, 9)

        #render.scene.camera.look_at([0, 0, 0], [-10, 0, 0], [0, 0, 1])
        img = render.render_to_image()
        o3d.io.write_image(output_path + "_o3d_test_2.jpg", img, 9)

        # box = o3d.geometry.TriangleMesh.create_box(

        # Alternatively 
        #o3d.visualization.draw_geometries(

    @staticmethod
    def merge_detections(car_dir='./output2/20220816_072310-car_multi_person_1-cars', human_dir='./output2/20220816_074530-car_multi_person_1-humans'):
        
        list_human = glob.glob(human_dir + '/hkp3d_*.json')

        list_car = glob.glob(car_dir + '/cbbox_*.json')

        list_human.sort()
        list_car.sort()
        print(list_human[:5])
        print(list_car[:5])

        keypoint_list = []
        bboxes_list = []

        for file_name in list_human:
            with open(file_name, "rt") as file:
                human_frame = json.load(file)
            
            keypoint_list.append(human_frame)
        
        #print(keypoint_list)

        for file_name in list_car:
            with open(file_name, "rt") as file:
                car_frame = json.load(file)
            
            bboxes_list.append(car_frame)

        #print(bboxes_list)

        assert len(keypoint_list) == len(bboxes_list), 'Length of keypoint & bboxes frames does not match!'

        merged_dict = {}

        for frame_id in range(len(keypoint_list)):
            
            video_id = int(list_human[frame_id].split('_')[-1].split('.')[0])

            merged_dict[video_id] = {
                "humans": keypoint_list[frame_id],
                "cars": bboxes_list[frame_id]
            }

        date = datetime.now().strftime("%Y%m%d_%I%M%S")

        folder_name = human_dir.split('-')[-2]

        output_path = f"./output3/{date}-" + folder_name + "-scene_frames.json"

        with open(output_path, "wt") as file:
            json.dump(merged_dict, file, default=_to_dict) 

        print(output_path)
        # load car frames

        # load human frames

    def get_root_joint():
        pass

    @staticmethod
    def extract_keyframes(input_file='./output3/20220819_091024-car_multi_person_7-scene_frames.json', fps=60, bbox_frame_index=500):
        # MuCo joint set
        joint_num = 21
        joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
        joints_name_dict = {}
        for i, name in enumerate(joints_name):
            joints_name_dict[name] = i

        pelvs_idx = 14

        print(joints_name[14])
        print(joints_name_dict['Pelvis'])

        flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
        skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )
        
        with open(input_file, "rt") as file:
            scene_frames_dict = json.load(file)

        frame_index = 0
        indices = list(scene_frames_dict.keys())
        #print(scene_frames_dict.keys())
        #print(scene_frames_dict[str(bbox_frame_index)])


        # 3D bounding boxes assumed static
        human_cars_dict = scene_frames_dict[str(bbox_frame_index)]

        pred_objects = human_cars_dict["cars"]

        print("Number of oriented bboxes: ", len(pred_objects))

        # humans need a for loop: 
        pred_humans = human_cars_dict["humans"] # reference frame 

        keypoint_ids, keypoint_list = zip(*pred_humans.items()) # TODO: also include human ids, which were not present in the selected frame

        # get list of human keypoints over all frames

        keypoints_along_frames = []

        # downsampling_rate = 10 
        # [::downsampling_rate]

        for index in indices:
            keypoints_along_frames.append(scene_frames_dict[index]["humans"])

        keypoints_by_id = {}
        #keypoints_by_id_frames = {}

        #print(indices)

        for frame_nr, keypoints_dict in enumerate(keypoints_along_frames):
            person_ids = list(keypoints_dict.keys())

            for p_id in person_ids:
                #print(person_ids)
                #print( keypoints_by_id)
                if p_id not in keypoints_by_id:
                    keypoints_by_id[p_id] = []
                    #keypoints_by_id_frames[p_id] = []
                
                keypoints_by_id[p_id].append((keypoints_dict[p_id], frame_nr)) # TODO: what if detection fails in one or several frames?
                #keypoints_by_id_frames[p_id].append(frame_nr)

        # distance of human hand joint to bounding boxes
        # https://stackoverflow.com/questions/44824512/how-to-find-the-closest-point-on-a-right-rectangular-prism-3d-rectangle/44824522#44824522
        # https://math.stackexchange.com/questions/2133217/minimal-distance-to-a-cube-in-2d-and-3d-from-a-point-lying-outside

        # heuristics:
        # A: Handshake / HandClap, 
        # B: Human touches the object ()

        # C: Humans look at each other
        # D: Pointing of Human
        # [x] E: Human starts walking 
        # [x] F: Human stops walking
        # G: Human Inside Car: complete skeleton inside bounding box
        # H: 
        # I: 
        # J: 

        # K
        # L
        # M

        print("Analyzed Human Skeleton IDs: ", keypoints_by_id.keys())
        ######################### Velocity ############################
        
        fig = plt.figure(figsize= [10, 15])
        
        joints_for_plotting = ['Spine', 'Pelvis', 'Thorax']
        joint_tmp = ['Spine', 'Pelvis', 'Thorax'][1] # 'Pelvis'# 'Thorax'# "Pelvis"# 'Spine'# 'Pelvis'
        wl = 60

        down_sampling = 60

        interval_length = 1 / fps * down_sampling

        event_log = []
        event_log_dict = {
            "initial_human_pose": [],
            "initial_object_pose": [],
            "person_starts_moving": [],
            "person_stops_moving": [],
            "hand_clap": []
        }

        for person_id in ['1', '2']: # keypoint_ids:
            ax = fig.add_subplot(len(keypoint_ids),1, int(person_id))
             # person_id = '1'
            keypoints_tuple = keypoints_by_id[person_id]

            keypoints_tmp = [k[0] for k in keypoints_tuple]

            keypoints_frames = [k[1] for k in keypoints_tuple]

            print(len(keypoints_tmp))

            # for k in test_keypoints[:5]:
            #     print("Pelvis: ", np.array(k[14])/1000)

            pelvic_pos_3d = []

            print("####################################### Keypoint Shape: ", np.array(keypoints_tmp).shape)

            for pos in keypoints_tmp:
                pelvic_pos_3d.append(pos[joints_name_dict[joint_tmp]]) # Pelvis

            print("Single Joint Array: ", np.array(pelvic_pos_3d).shape)

            fps = 60 / (int(indices[1]) - int(indices[0]))
            print("FPS: ", fps)

            print(pelvic_pos_3d[:5])

            time_delta = 1/fps * down_sampling

            pelvic_pos_3d = np.array(pelvic_pos_3d) / 1000
            #signals = [moving_average(pelvic_pos_3d[:, 0], wl), moving_average(pelvic_pos_3d[:, 1], wl), moving_average(pelvic_pos_3d[:, 2], wl)]
            

            from scipy.signal import butter, lfilter, freqz

            def butter_lowpass(cutoff, fs, order=5):
                return butter(order, cutoff, fs=fs, btype='low', analog=False)

            def butter_lowpass_filter(data, cutoff, fs, order=5):
                b, a = butter_lowpass(cutoff, fs, order=order)
                y = lfilter(b, a, data)
                return y

            # Filter requirements.
            order = 6
            fs = fps      # sample rate, Hz
            cutoff = 20 #3.667  # desired cutoff frequency of the filter, Hz

            # Get the filter coefficients so we can check its frequency response.
            b, a = butter_lowpass(cutoff, fs, order)

            # y = butter_lowpass_filter(data, cutoff, fs, order)

            # https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
            x = pelvic_pos_3d[:, 0]
            y = pelvic_pos_3d[:, 1]
            z = pelvic_pos_3d[:, 2]

            pelvic_pos_filtered = np.zeros_like(pelvic_pos_3d)
            pelvic_pos_filtered[:, 0] = butter_lowpass_filter(x, cutoff, fs, order)
            pelvic_pos_filtered[:, 1] = butter_lowpass_filter(y, cutoff, fs, order)
            pelvic_pos_filtered[:, 2] = butter_lowpass_filter(z, cutoff, fs, order)

            # median
            # signals = [scp_mf(pelvic_pos_3d[:, 0], wl), scp_mf(pelvic_pos_3d[:, 1], wl), scp_mf(pelvic_pos_3d[:, 2], wl)]

            def moving_average(x, w):
                average = np.convolve(x, np.ones(w), 'same') / w
                average[:w] = average[w]
                average[-w:] = average[-w]
                return average

            pelvic_pos_average = np.zeros_like(pelvic_pos_3d)
            pelvic_pos_average[:, 0] = moving_average(x, wl)
            pelvic_pos_average[:, 1] = moving_average(y, wl)
            pelvic_pos_average[:, 2] = moving_average(z, wl)

            print(pelvic_pos_filtered.shape)

            # downsample
            pelvic_pos_3d = pelvic_pos_3d[::down_sampling]
            pelvic_pos_filtered = pelvic_pos_filtered[::down_sampling]
            pelvic_pos_average = pelvic_pos_average[::down_sampling]

            pelvic_pos_diff = np.diff(pelvic_pos_3d, axis=0)
            pelvic_pos_diff_filtered = np.diff(pelvic_pos_filtered, axis=0) # / 1000

            pelvic_pos_diff_average = np.diff(pelvic_pos_average, axis=0)
            print("pelvic_pos_diff.shape: ", pelvic_pos_diff.shape)
            print(pelvic_pos_diff)
            print("Averaged Pose: ", pelvic_pos_diff_average)

            pelvic_velo_filtered = np.linalg.norm(pelvic_pos_diff_filtered, axis=1) / time_delta
            pelvic_velo_average = np.linalg.norm(pelvic_pos_diff_average, axis=1) / time_delta
            pelvic_velo = np.linalg.norm(pelvic_pos_diff, axis=1) / time_delta

            print("Velocity Shape: ", pelvic_velo.shape)
            #pelvic_velo = np.concatenate([np.array([pelvic_velo[0],]), pelvic_velo])
            # pelvic_velo_filtered = np.concatenate([np.array([pelvic_velo_filtered[0],]), pelvic_velo_filtered])
            # pelvic_velo_average = np.concatenate([np.array([pelvic_velo_average[0],]), pelvic_velo_average])

            pelvic_velo_grad = np.gradient(pelvic_pos_average, time_delta, axis=1)
            print("Grad", pelvic_velo_grad.shape)


            threshold_velo = 0.25
            moving_standing = np.zeros_like(pelvic_velo_average)

            moving_standing[pelvic_velo_average>threshold_velo] = 1

            #pelvic_velo = moving_average(pelvic_velo, int(fps))

            # pelvic_velo
            #pelvic_pos_3d
            
            print(pelvic_pos_3d.shape)
            # wl = 20
            #signals = [moving_average(pelvic_pos_3d[:, 0], 3), moving_average(pelvic_pos_3d[:, 1], 3), moving_average(pelvic_pos_3d[:, 2], 3)]
            #signals = [moving_average(pelvic_pos_3d[:, 0], wl), moving_average(pelvic_pos_3d[:, 1], wl), moving_average(pelvic_pos_3d[:, 2], wl)]
            #signals = [pelvic_pos_3d[:, 0], pelvic_pos_3d[:, 1], pelvic_pos_3d[:, 2]]
            signals = [pelvic_pos_average[:, 0], pelvic_pos_average[:, 1], pelvic_pos_average[:, 2]]
            time_axis = np.array(keypoints_frames)[::down_sampling] / fps
            ax.plot(time_axis, pelvic_pos_3d[:, 0])
            ax.plot(time_axis, pelvic_pos_3d[:, 1])
            ax.plot(time_axis, pelvic_pos_3d[:, 2])

            ax.plot(time_axis, signals[0])
            ax.plot(time_axis, signals[1])
            ax.plot(time_axis, signals[2])

            ax.plot(time_axis[1:], pelvic_velo)
            ax.plot(time_axis[1:], pelvic_velo_average)

            #ax.plot(time_axis[1:], pelvic_pos_diff)
            #ax.plot(time_axis[::down_sampling], pelvic_velo_average)
            #ax.plot(time_axis, pelvic_velo_grad)
            # ax.plot(time_axis[::down_sampling], pelvic_velo_filtered)
            # ax.plot(time_axis, moving_average(pelvic_velo, wl)[wl:-wl])

            ax2 = ax.twinx()
            ax2.set_ylabel('Moving/Standing')
            ax2.plot(time_axis[1:], moving_standing, 'k')


            event_type = 'initial_human_pose'
            event_data = (person_id, keypoints_tmp[0])
            time_stamp = time_axis[0]
            event_log.append(Event(event_type, event_data, time_stamp))

            for i in range(1,len(moving_standing)):
                if moving_standing[i] == 1:
                    plt.axvspan(time_axis[i+1]-0.5*interval_length, time_axis[i+1]+0.5*interval_length, color='g', alpha=0.2, lw=0)

                # person starts moving
                if moving_standing[i-1] == 0 and moving_standing[i] == 1:
                    
                    event_type = 'person_starts_moving'
                    event_data = (person_id, keypoints_tmp[i-1])
                    time_stamp = time_axis[i-1]
                    event_log.append(Event(event_type, event_data, time_stamp))
                
                # person stops moving
                if moving_standing[i-1] == 1 and moving_standing[i] == 0:

                    event_type = 'person_stops_moving'
                    event_data = (person_id, keypoints_tmp[i])
                    time_stamp = time_axis[i]
                    event_log.append(Event(event_type, event_data, time_stamp))

            ax2.set_yticks([0,1])
            ax2.set_yticklabels(['Standing','Moving'])
        
            ax.set_xlabel("time [s]")
            ax.set_ylabel(f"[{joint_tmp}]joint velocity [m/s]")
            ax.title.set_text(f'{joint_tmp} Joint Velocity: Skeleton [{person_id}] | F:{wl}')
            ax.legend(['X', 'Y', 'Z','X_f', 'Y_f', 'Z_f', 'Velo', 'Velo_f'])




            

            #ax.set_ylim([min( pelvic_pos_3d[:, 0]), 10000])

        # Hand Joint Detection

        # For Plotting
        date = datetime.now().strftime("%Y%m%d_%I%M%S")

        folder_name = input_file.split('-')[-2]

        output_path = f"./extracted_keyframes/{date}-" + folder_name + "-key_frames" #

        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 

        plt.savefig(output_path + f"/{joint_tmp}_joint_velocity_subsampled.pdf")

        
        ############################## Hand Joint Distances ################################

        joints_name_dict['L_Hand']
        joints_name_dict['R_Hand']

        human_id_combinations = itertools.combinations(keypoint_ids, 2)

        print(list(human_id_combinations))

        for c in human_id_combinations:
            pass

        hand_distances = []

        for frame_nr, keypoints_dict in enumerate(keypoints_along_frames):
            right_hand_joint_1 = keypoints_dict['1'][joints_name_dict['R_Hand']]
            right_hand_joint_2 = keypoints_dict['2'][joints_name_dict['R_Hand']]
            # print("RHJ1: ", right_hand_joint_1)
            # print("RHJ2: ", right_hand_joint_2)

            h_distance = np.linalg.norm(np.array(right_hand_joint_1) - np.array(right_hand_joint_2))

            hand_distances.append(h_distance)

        fig = plt.figure(figsize= [10, 5])

        ax = fig.add_subplot(1,1,1)

        ax.set_xlabel("time [s]")
        joint_tmp = 'R_Hand'
        ax.set_ylabel(f"[{joint_tmp}] joint distance [mm]")
        ax.title.set_text(f'{joint_tmp} Joint Distance: Skeleton [1/2]')

        time_axis = np.arange(len(keypoints_along_frames))/fps

        hand_distances = np.array(hand_distances)
        hand_distances_smoothed = moving_average(hand_distances, 3)
        plt.plot(time_axis, hand_distances)
        plt.plot(time_axis, hand_distances_smoothed)

        handshake_threshold = 250
        interval_length = 1 / fps

        only_one = False
        for i in range(1,len(hand_distances)):
            if hand_distances_smoothed[i] < handshake_threshold:
                plt.axvspan(time_axis[i]-0.5*interval_length, time_axis[i]+0.5*interval_length, color='g', alpha=0.2, lw=0)


                if only_one is False:  # TODO: check for first event of longer period
                    event_type = 'hand_clap'
                    event_data = {
                        '1': list(keypoints_along_frames[i]['1']),
                        '2': list(keypoints_along_frames[i]['2'])
                    }
                    print("Event_data: ", event_data)
                    time_stamp = time_axis[i]
                    event_log.append(Event(event_type, event_data, time_stamp))

                    only_one = True
        print(type(keypoints_along_frames[i]['1']))

        print(min(hand_distances))

        plt.savefig(output_path + "/hand_joint_distance.pdf")


        # save events as json


        ############################## Human Object Distances ################################
        detect_human_objct_relation = False

        cars = [Detection9D(**obj) for obj in pred_objects if obj['class_name'] == 'car']

        for idx, car in enumerate(cars):
            event_type = 'initial_object_pose'
            
            tensor = car.get_tensor()
            tensor[2] = tensor[2] / 1.87
            # tensor = tensor[]
            boxes = CameraInstance3DBoxes([np.take(tensor, [0,1,2,3,4,5,7])], box_dim=7)
            boxes_corners = boxes.corners
            corners = boxes_corners[0].tolist()
            event_data = (0, tensor) # corners
            time_stamp = idx 
            event_log.append(Event(event_type, event_data, time_stamp))

        print(event_log)

        # save event log
        log_file_name = output_path + '/' + folder_name + '-event_log.json'
        with open(log_file_name, "wt") as file:
            json.dump(event_log, file, default=_to_dict) 

        if detect_human_objct_relation:
            # cars = [Detection9D(**obj) for obj in pred_objects if obj['class_name'] == 'car']

            print(cars)

            def closestPointToBox(q, origin, v100, v010, v001):
                # (Vector3 q, Vector3 origin, Vector3 v100, Vector3 v010, Vector3 v001)

                px = v100
                py = v010
                pz = v001

                vx = (px - origin)
                vy = (py - origin)
                vz = (pz - origin)

                tx = np.dot( q - origin, vx ) / np.dot(vx,vx)
                ty = np.dot( q - origin, vy ) / np.dot(vy,vy)
                tz = np.dot( q - origin, vz ) / np.dot(vz,vz)

                # tx = tx < 0 ? 0 : tx > 1 ? 1 : tx;
                if tx < 0:
                    tx = 0
                elif tx > 1:
                    tx = 1
                # else:
                    # tx = tx
                
                # ty = ty < 0 ? 0 : ty > 1 ? 1 : ty;
                if ty < 0:
                    ty = 0
                elif ty > 1:
                    ty = 1
                
                # tz = tz < 0 ? 0 : tz > 1 ? 1 : tz;
                if tz < 0:
                    tz = 0
                elif tz > 1:
                    tz = 1

                p = tx * vx + ty * vy + tz * vz + origin

                return p

            first_car = cars[0]

            tensor = first_car.get_tensor()
            tensor[2] = tensor[2] / 1.87
            # tensor = tensor[]
            boxes = CameraInstance3DBoxes([np.take(tensor, [0,1,2,3,4,5,7])], box_dim=7)
            boxes_corners = boxes.corners

            print(boxes_corners)

                #  (x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)
                # 
                #              front z
                #                   /
                #                  /
                #    (x0, y0, z1) + -----------  + (x1, y0, z1)
                #                /|            / |
                #               / |           /  |
                # (x0, y0, z0) + ----------- +   + (x1, y1, z1)
                #              |  /      .   |  /
                #              | / origin    | /
                # (x0, y1, z0) + ----------- + -------> x right
                #              |             (x1, y1, z0)
                #              |
                #              v
                #         down y
            
            q = np.array(keypoints_dict['1'][joints_name_dict['Pelvis']]) / 1000
            origin = boxes_corners[0][3].numpy()
            v100 = boxes_corners[0][7].numpy()
            v010 = boxes_corners[0][0].numpy()
            v001 = boxes_corners[0][2].numpy()

            p = closestPointToBox(q, origin, v100, v010, v001)

            print(p)

            distance_human_object = np.linalg.norm(q - p)

            print(distance_human_object)

            def get_distance_human_object(human_id = '1', joint_name = 'Pelvis', obj=pred_objects[0]):
                
                distances = []
                for frame_nr, keypoints_dict in enumerate(keypoints_along_frames):
                    queried_joint = np.array(keypoints_dict['1'][joints_name_dict[joint_name]])


                    tensor = obj.get_tensor()
                    tensor[2] = tensor[2] / 1.87
                    # tensor = tensor[]
                    boxes = CameraInstance3DBoxes([np.take(tensor, [0,1,2,3,4,5,7])], box_dim=7)
                    boxes_corners = boxes.corners

                    q = queried_joint / 1000
                    origin = boxes_corners[0][3].numpy()
                    v100 = boxes_corners[0][7].numpy()
                    v010 = boxes_corners[0][0].numpy()
                    v001 = boxes_corners[0][2].numpy()

                    p = closestPointToBox(q, origin, v100, v010, v001)
                    distance_human_object = np.linalg.norm(q - p)

                    distances.append(distance_human_object)

                return distances
                    # h_distance = np.linalg.norm(np.array(right_hand_joint_1) - np.array(right_hand_joint_2))

                    #hand_distances.append(h_distance)

            car_distances = get_distance_human_object(human_id = '1', joint_name = 'Pelvis', obj=cars[1]) # TODO: loop efficiently through all objects and humans

            fig = plt.figure(figsize= [10, 5])

            ax = fig.add_subplot(1,1,1)

            ax.set_xlabel("time [s]")
            joint_tmp = 'Pelvis'
            ax.set_ylabel(f"[{joint_tmp}] joint distance to car [mm]")
            #ax.title.set_text(f'{joint_tmp} Joint Distance: Skeleton [1/2]')

            time_axis = np.arange(len(keypoints_along_frames))/fps

            car_distances = np.array(car_distances)# .T
            print(car_distances)
            car_distances_smoothed = moving_average(car_distances, 3)
            plt.plot(time_axis, car_distances)
            plt.plot(time_axis, car_distances_smoothed)

            # handshake_threshold = 250
            # interval_length = 1 / fps
            # for i in range(1,len(hand_distances)):
            #     if hand_distances_smoothed[i] < handshake_threshold:
            #         plt.axvspan(time_axis[i]-0.5*interval_length, time_axis[i]+0.5*interval_length, color='g', alpha=0.2, lw=0)

            print(min(hand_distances))

            plt.savefig(output_path + "/human_object_distance.pdf")
       
    

if __name__ == "__main__":
    Fire(CLI)