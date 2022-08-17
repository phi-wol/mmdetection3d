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

    def predict_3d_pose(self, ):
        pass

    def predict_root_joint(self, ):
        pass
    
    def set_intrinsics(self, focal, princpt):
        self.focal=focal
        self.princpt=princpt
         
    def track_human_sequence(self, video_frames: VideoFrames, visualize=False, score_threshold=.8, mode="det", file_name="None", fps_divisor=1): #-> Dict[int, List[Detection]]

        detector = Detector(mode)
        
        result_file = file_name.split("/")[-1].split(".")[0]

        date = datetime.now().strftime("%Y%m%d_%I%M%S")
        print(f"filename_{date}")

        root_dir = f"./output2/{date}-" + result_file + "-humans/"

        pathlib.Path(root_dir).mkdir(parents=True, exist_ok=True) 

        output_fps = int(video_frames.fps * video_frames.frame_count / fps_divisor)
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
                
                # TODO: 
                if len(root_depth_list) > 0:
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

        root_dir = f"./output2/{date}-" + result_file + "-cars/"

        pathlib.Path(root_dir).mkdir(parents=True, exist_ok=True) 

        output_fps = int(video_frames.fps * video_frames.frame_count / fps_divisor)
        print("OutputFPS: ", output_fps)
        print("VideoOriginalFPS: ", video_frames.fps)
        
        detector = init_model(args_config, args_checkpoint, device=args_device)

        #print(intrinsics.as_matrix_3x3())

        #detector = Detector("det")
        
        if visualize: 
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")
            out_raw = cv2.VideoWriter(root_dir + result_file + "_3dbbox_detections_raw.mp4", fourcc, video_frames.fps, video_frames.get_output_size())
            out = cv2.VideoWriter(root_dir + result_file + "_3dbbox_detections_tracked.mp4", fourcc, video_frames.fps, video_frames.get_output_size())
        
        frame_id = 0

        sort_per_class_id = dict()

        raw_detection_sequence = {}
        tracked_detection_sequence = {}

        mot_tracker = AB3DMOT(max_age=20, min_hits=7) 

        print("Video Count: ", video_frames.frame_count)
        
        for frame in tqdm(iter(video_frames)):

            if frame_id % fps_divisor == 0:

                cv2.imwrite(root_dir + result_file + '_raw_image_{:04d}.jpg'.format(frame_id), frame)

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
                        boxes = CameraInstance3DBoxes(np.array([tensor]))

                        if det.score < score_threshold:
                            continue

                        color_tmp = tuple([int(tmp * 255) for tmp in colors[int(det.instance_id) % max_color]])
                        

                        # TODO: add class label to plot
                        print(det.instance_id)
                        image_tmp = draw_camera_bbox3d_on_img(
                        boxes, image_tmp, intrinsics.as_matrix_3x3(), None, color=color_tmp, thickness=1, object_id=int(det.instance_id), class_type=det.class_name)

                if visualize: 
                    #image_tmp = cv2.cvtColor(image_tmp, cv2.COLOR_RGB2BGR)
                    out.write(image_tmp)

                if visualize:
                    
                    image_tmp = frame
                    for det in raw_detections:
                        tensor = det.get_tensor()
                        boxes = CameraInstance3DBoxes(np.array([tensor]))

                        if det.score < score_threshold or det.class_name != "car":
                            continue

                        # TODO: resolve later
                        det.instance_id = 0

                        color_tmp = tuple([int(tmp * 255) for tmp in colors[int(det.instance_id) % max_color]])
                        

                        # TODO: add class label to plot
                        print(det.instance_id)
                        image_tmp = draw_camera_bbox3d_on_img(
                        boxes, image_tmp, intrinsics.as_matrix_3x3(), None, color=color_tmp, thickness=1, object_id=int(det.instance_id), class_type=det.class_name)

                        cv2.imwrite(root_dir + result_file + '_bbox_car_{:04d}.jpg'.format(frame_id), image_tmp)

                        out_raw.write(image_tmp)

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
            out_raw.release()

        return tracked_detection_sequence

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
    def detect_humans(mov_filename, mode, custom=True, landscape=False, fps_divisor=1):
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

    @staticmethod
    def detect_cars(mov_filename, kitti=True, custom=True, landscape=False, fps_divisor=1, args_config=None, args_checkpoint=None, f_x=1445, f_y=1445, h=1440, w=1920, o_x=960, o_y=753):
        
        video_frames = VideoFrames(mov_filename, downsampling=1, throttle=1, flip=False, landscape=False, pad=[])

        # load camera intrinsics from json

        if kitti:
            kitti_w = 1280 
            kitti_h = 384

            f_x = 385
            f_y = 385
            # w = 530
            # h = 384
            o_x = 257
            o_x += 375
            o_y = 201# 182 # 200

            w = 1280
            h = 384

            # 1440/384 = 3.75
            # 1920/3.75 = 530

            # 1280-530 = 750
            # 750 / 2 = 375
            video_frames = VideoFrames(mov_filename, downsampling=1, throttle=1, flip=True, landscape=True, pad=[0,0,375, 375], resize=(384, 530))

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

        camera_intrinsics = CameraIntrinsics(f_x*2, f_y*2, h, w, o_x, o_y)
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
    def visualize_cars():
        pass

    @staticmethod
    def visualize_humans():
        pass

    @staticmethod
    def visualize_3D(input_file='./output2/20220816_075218-car_multi_person_1-scene_frames.json', ):
        # load humans
        with open(input_file, "rt") as file:
            scene_frames_dict = json.load(file)

        print(scene_frames_dict.keys())
        print(scene_frames_dict['0'])


    @staticmethod
    def merge_detections(car_dir='./output2/20220816_072310-car_multi_person_1-cars', human_dir='./output2/20220816_074530-car_multi_person_1-humans'):
        
        list_human = glob.glob(human_dir + '/hkp3d_*.json')

        print(list_human)

        list_car = glob.glob(car_dir + '/cbbox_*.json')

        print(list_car)

        keypoint_list = []
        bboxes_list = []

        for file_name in list_human:
            with open(file_name, "rt") as file:
                human_frame = json.load(file)
            
            keypoint_list.append(human_frame)
        
        print(keypoint_list)

        for file_name in list_car:
            with open(file_name, "rt") as file:
                car_frame = json.load(file)
            
            bboxes_list.append(car_frame)

        print(bboxes_list)

        assert len(keypoint_list) == len(bboxes_list), 'Length of keypoint & bboxes frames does not match!'

        merged_dict = {}

        for frame_id in range(len(keypoint_list)):

            merged_dict[frame_id] = {
                "humans": keypoint_list[frame_id],
                "cars": bboxes_list[frame_id]
            }

        date = datetime.now().strftime("%Y%m%d_%I%M%S")
        print(f"filename_{date}")

        folder_name = human_dir.split('-')[-2]

        output_path = f"./output2/{date}-" + folder_name + "-scene_frames.json"

        with open(output_path, "wt") as file:
            json.dump(merged_dict, file, default=_to_dict) 


        # load car frames

        # load human frames
        
    
    @staticmethod
    def extract_keyframes():
        pass

    

if __name__ == "__main__":
    Fire(CLI)