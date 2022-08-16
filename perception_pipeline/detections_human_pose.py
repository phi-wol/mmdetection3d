
# 2D bounding box detection & tracking


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

from .object_detection import Detection, Detector
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

import sys
sys.path.append(".")
sys.path.append("..")

# JSON Serialization
def _to_dict(obj):
    return obj.to_dict()

class VideoFrames():
    def __init__(self, filename, downsampling=1, throttle=1, flip=False, landscape=False) -> None:
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


    def __iter__(self):
        return self

    def __next__(self):
        success, frame = self.capture.read()
        if not success:
            raise StopIteration

        self.idx += 1

        frame = cv2.resize(frame, self.get_output_size())
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #  not needed, frames are processed as BGR by openCV
        
        if self.flip: 
            frame = cv2.flip(frame, 0)
            frame = cv2.flip(frame, 1)
        
        if self.landscape:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) # cv2.ROTATE_90_COUNTERCLOCKWISE

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

    def predict_3d_pose(self, ):
        pass

    def predict_root_joint(self, ):
        pass
    
    def set_intrinsics(self, focal, princpt):
        self.focal=focal
        self.princpt=princpt
         
    def track_detection_sequence(self, video_frames: VideoFrames, visualize=False, score_threshold=.8, mode="det", file_name="None", fps_divisor=1): #-> Dict[int, List[Detection]]

        detector = Detector(mode)
        
        result_file = file_name.split("/")[-1].split(".")[0]

        root_dir = "./output/" + result_file + "/"

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
        
        for frame in iter(video_frames):

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


                pred_2d_kpt = np.array(output_pose_2d_list)
                pred_3d_save[str(frame_id)] = output_pose_3d_list
                pred_2d_save[str(frame_id)] = output_pose_2d_list
                
                

            frame_id += 1
            print(".", end="")
            

            # print("root_depth_list", root_depth_list)
            # print("output_pose_3d_list", output_pose_3d_list)
        
        sio.savemat(root_dir + "preds_3d_kpt_mupots.mat" , pred_3d_save)
        sio.savemat(root_dir +" preds_2d_kpt_mupots.mat", pred_2d_save)

        with open(root_dir + "raw_detections.json", "wt") as file:
            json.dump(raw_detection_sequence, file, default=_to_dict)
        
        with open(root_dir + "tracked_detections.json", "wt") as file:
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

    # (self, video_frames: VideoFrames)

class CLI():
    @staticmethod
    def detect(mov_filename, mode, custom=True, landscape=False, fps_divisor=1):
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

        tracked_detection_sequence = perception.track_detection_sequence(video_frames, visualize=True, mode=mode, file_name=mov_filename,  fps_divisor=fps_divisor)
        
        #video_frames = VideoFrames(mov_filename)


        # output for a single input video:

        # dictionary: for every frame:
        # list of person keypoints in 3d 
        # list of object 3d bounding boxes



        # + SMOKE Output

if __name__ == "__main__":
    Fire(CLI)