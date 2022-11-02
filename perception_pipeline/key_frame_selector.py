import numpy as np
import data
import pathlib

from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, lfilter, freqz

from datetime import datetime

from .object_detection import Detection, Detection9D

import cv2

from .ab3dmot.AB3DMOT_libs.mmdet.cam_box3d import CameraInstance3DBoxes
import matplotlib.pyplot as plt

import json

# JSON Serialization
def _to_dict(obj):
    return obj.to_dict()

class Event():
    def __init__(self, event_type, object_id, data, time_stamp, frame_nr) -> None:
        self.type = event_type
        self.object_id = object_id
        self.time_stamp = time_stamp
        self.event_data = data
        self.frame_nr = frame_nr

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "object_id": self.object_id,
            "time_stamp": self.time_stamp,
            "event_data": self.event_data,
            "frame_nr": self.frame_nr
        }

class TouchEvent():
    def __init__(self, feature_name, start_frame) -> None:
        self.feature_name = feature_name
        # self.data = data # TODO: add skeleton parameters of current event
        self.start_frame = start_frame
        # self.end_time = end_time
        # self.frame_nr = frame_nr

    def update_event_time(self, end_frame):
        self.end_frame = end_frame
        self.event_duration = int((self.end_frame - self.start_frame)/2)
        self.event_time = self.start_frame + self.event_duration
        print("New Event detected: ", self.feature_name, self.event_duration)

    def to_dict(self) -> dict:
        pass

class TouchEvent2():
    def __init__(self, joint_name, feature_name, start_frame, end_frame) -> None:
        self.joint_name = joint_name
        self.feature_name = feature_name
        # self.data = data # TODO: add skeleton parameters of current event
        self.start_frame = start_frame
        self.end_frame = end_frame
        # self.end_time = end_time
        # self.frame_nr = frame_nr

        self.event_duration = int((self.end_frame - self.start_frame)/2)
        self.event_time = self.start_frame + self.event_duration

    def to_dict(self) -> dict:
        pass

class MovingEvent():
    def __init__(self, joint_name, start_frame, end_frame) -> None:
        self.joint_name = joint_name
        # self.feature_name = feature_name
        self.start_frame = start_frame
        self.end_frame = end_frame

class SkeletonDetection():
    def __init__(self, idx, skeleton_param, time_step):
        self.idx = idx
        self.skeleton_param = skeleton_param
        self.time_step = time_step

class SemanticBox(CameraInstance3DBoxes):
    def __init__(self, tensor, box_dim=9):
        super().__init__(tensor, box_dim, with_yaw=True, origin=(0.5, 0.5, 0.5))

    def get_closest_feature_point(self, query_point, feature_points):
        
        distances = np.linalg.norm(feature_points - query_point, axis=1)
        print("distances", distances)

        argmin = np.argmin(distances)
        print("argmin: ", argmin)
        print("Closest Feature Point: ", self.semantic_dict[argmin])

        return feature_points[argmin], argmin

class ChairBox(SemanticBox):
    def __init__(self, tensor, box_dim=7):
        super().__init__(tensor, box_dim)

        self.semantic_dict = {
            0: "Seat",
            1: "ChairBack",
            2: "L_Handle",
            3: "R_Handle"
            }
    
    def get_semantic_feature_points(self):
        corner_points = self.corners[0].numpy()

        feature_list = []

        # Seat
        pos_tmp = 0.25 * (corner_points[0] + corner_points[1] + corner_points[5] + corner_points[4])
        pos_tmp += 0.5 * (corner_points[2] - corner_points[1]) #height
        feature_list.append(pos_tmp)

        # ChairBack
        pos_tmp = corner_points[3] + 0.5 * (corner_points[7] - corner_points[3])
        feature_list.append(pos_tmp)

        # L Handle
        pos_tmp = corner_points[5] + 1/3 * (corner_points[4] - corner_points[5] )
        pos_tmp += 0.5 * (corner_points[6] - corner_points[5]) #height
        feature_list.append(pos_tmp)

        # R Handle
        pos_tmp = corner_points[1] + 1/3 * (corner_points[0] - corner_points[1])
        pos_tmp += 0.5 * (corner_points[2] - corner_points[1]) #height
        feature_list.append(pos_tmp)

        self.feature_points = np.vstack(feature_list)

        return self.feature_points

class BikeBox(SemanticBox):
    # TODO: encode semantic feature points into the standard model of the box coordinates

    # Simulate Semantics of Bicycle Safety Check: "M-CHECK"
    # https://www.youtube.com/watch?v=B6CFPFdVz5E

    # https://www.centurycycles.com/how-to/how-to-do-a-pre-ride-safety-check-pg1343.htm


    def __init__(self, tensor, box_dim=9):
        super().__init__(tensor, box_dim)

        # encode semantic feature points:
        # when the model detects a touch interaction with the bounding box, 
        # the nearest feature point is calculated and returned as semantic representation for the event
        self.semantic_dict = {
            0: "Saddle",            # Character sits on bike
            1: "L_Handlebar",       # Test the Brake
            2: "R_Handlebar",       # Test the Brake / Ring the bell
            3: "F_Light",           # Check Front Light
            4: "B_Light",           # Check Back Light
            5: "L_Pedal",           # "Turn" Left Pedale - Chain
            6: "R_Pedal",           # "Turn" Right Pedale
            7: "F_Wheel",           # Check Front Tire Air Pressure for proper inflation and treads (excessive wear)
            8: "B_Wheel",           # Check Front Tire Air Pressure

            # wheel quick release levers
        }

        # check for rotating movements

    def get_semantic_feature_points(self):

        """self.corners: torch.Tensor: Coordinates of corners of all the boxes in
        shape (N, 8, 3).

        Convert the boxes to  in clockwise order, in the form of
        (x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)
            0       1       2       3       4       5       6       7

        .. code-block:: none

                         front z
                              /
                             /   vorne
               (x0, y0, z1) + -----------  + (x1, y0, z1)
                           /|            / |
                          / |           /  |
            (x0, y0, z0) + ----------- +   + (x1, y1, z1)
                         |  /      .   |  /
                         | / origin    | /
            (x0, y1, z0) + ----------- + -------> x right
                         |             (x1, y1, z0)
                         |
                         v  oben
                    down y

                                 front z
                              /
                             /   oben
               (    03    ) + -----------  + (    07    )
                           /|            / |
                          / |       (6) /  |
            (    02    ) + ----------- +   + (    04    )
                         |  /(0)   .   |  /
                         | / origin    | / 
            (    01    ) + ----------- + -------> x right
                         |             (    05    ) vorne
                         |
                         v
                    down y
        """
        
        corner_points = self.corners[0].numpy()

        feature_list = []

        # Saddle
        pos_tmp = corner_points[3] + 1/3 * (corner_points[2] - corner_points[3])
        pos_tmp += 0.15 * (corner_points[0] - corner_points[3]) #height
        pos_tmp += 0.5 * (corner_points[7] - corner_points[3])
        feature_list.append(pos_tmp)

        # L_Handlebar
        handle_l = corner_points[6] + 0.25 * (corner_points[7] - corner_points[6])
        feature_list.append(handle_l)

        # R_Handlebar
        handle_r = corner_points[2] + 0.25 * (corner_points[3] - corner_points[2])
        feature_list.append(handle_r)

        # F_Light
        pos_tmp = corner_points[1] + 0.5 * (corner_points[5] - corner_points[1])
        pos_tmp += 0.5 * (corner_points[2] - corner_points[1]) #height
        pos_tmp += 0.2 * (corner_points[0] - corner_points[1]) 
        feature_list.append(pos_tmp)

        # B_Light
        pos_tmp = corner_points[0] + 0.5 * (corner_points[4] - corner_points[0])
        pos_tmp += 0.5 * (corner_points[3] - corner_points[0]) #height
        pos_tmp += 0.2 * (corner_points[1] - corner_points[0]) 
        feature_list.append(pos_tmp)

        # L_Pedal
        pos_tmp = corner_points[5] + 0.6 * (corner_points[4] - corner_points[5])
        pos_tmp += 0.2 * (corner_points[2] - corner_points[1]) #height
        pos_tmp += 0.2 * (corner_points[1] - corner_points[5])
        feature_list.append(pos_tmp)

        # R_Pedal
        pos_tmp = corner_points[1] + 0.6 * (corner_points[0] - corner_points[1])
        pos_tmp += 0.2 * (corner_points[2] - corner_points[1]) #height
        pos_tmp += 0.2 * (corner_points[5] - corner_points[1])
        feature_list.append(pos_tmp)

        # F_Wheel
        pos_tmp = corner_points[1] + 0.5 * (corner_points[5] - corner_points[1])
        pos_tmp += 0.25 * (corner_points[2] - corner_points[1]) #height
        pos_tmp += 0.2 * (corner_points[0] - corner_points[1]) 
        feature_list.append(pos_tmp)

        # B_Wheel
        pos_tmp = corner_points[0] + 0.5 * (corner_points[4] - corner_points[0])
        pos_tmp += 0.25 * (corner_points[3] - corner_points[0]) #height
        pos_tmp += 0.2 * (corner_points[1] - corner_points[0]) 
        feature_list.append(pos_tmp)

        self.feature_points = np.vstack(feature_list)

        return self.feature_points

class CarBox(SemanticBox):

    def __init__(self, tensor, box_dim=9):
        super().__init__(tensor, box_dim)

        self.semantic_dict = {
            0: "Seat",
            1: "L_Handlebar",
            2: "R_Handlebar",
        }

        # Scheinwerfer
        # Tür
        # Reifen
        # Motorhaube

        # Tankdeckel / Ladebuchse

    def get_semantic_feature_points(self):
        pass

class KeyFrameVisualizer(KeyFrameSelector):
    def __init__(self, input_file, scene_frames_dict=None, fps=60) -> None:
        super().__init__(input_file, scene_frames_dict, fps)

    

class KeyFrameSelector():

    # snap point: semantic feature points of bounding box 
    # box model

    # TODO: Events are detected by distance of joint to closest snap point of bbox:
    # first, check if hand joint is close enough (below predefined threshold) to object's bbox
    # when below, localize closest snap point: start touch event -> take mid time stamp for event
    # 

    def __init__(self, input_file, scene_frames_dict=None, fps=60) -> None:
        self.input_file = input_file
        # MuCo joint set
        self.joint_num = 21
        self.joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
        self.joints_name_dict = {}
        for i, name in enumerate(self.joints_name):
            self.joints_name_dict[name] = i
        
        self.fps = fps

        self.flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
        self.skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )
        self.scene_frames_dict = scene_frames_dict

        date = datetime.now().strftime("%Y%m%d_%I%M%S")
        folder_name = self.input_file.split('-')[-2]
        self.output_path = f"./extracted_keyframes/{date}-" + folder_name + "-key_frames" #
        pathlib.Path(self.output_path).mkdir(parents=True, exist_ok=True) 

        self.touch_threshold = 0.1

        self.down_sampling = 1 # TODO: additional down_sampling needed?

    def extract_chair_events(self):
        event_log = {}

        # load first (and only) object pose - assumed static
        self.pred_objects = self.get_initial_objects(bbox_frame_index=0)
        object_poses = self.get_initial_object_events()
        event_log["object_poses"] = object_poses

        self.keypoints_by_id = self.sort_skeletons_to_ids()
       
        human_poses=[]

        # get initial human poses:
        for person_idx in self.keypoints_by_id.keys():
            
            human_det = self.keypoints_by_id[person_idx][0]
            skeleton_param = human_det.skeleton_param
            time_step = human_det.time_step

            print("time_step", time_step)
            print(type(time_step))

            # encode event
            event = Event(
                event_type='initial_human',
                object_id=person_idx, 
                data=skeleton_param, # TODO: list?
                time_stamp=time_step/self.fps,
                frame_nr=time_step
                )
        
            human_poses.append(event)

        person_idx = '1'
        joints = ["L_Hand", "R_Hand", "Pelvis"]
        for j in joints:
            touch_events = self.get_distance_person_feature_points(person_idx = person_idx, joint_name = j, visualize=True)
            # TODO: transform TouchEvents() into normal json Events()
            for tevent in touch_events: 
                frame_index = tevent.event_time 
                event_type = tevent.joint_name + '_' + tevent.feature_name
                
                human_det = self.keypoints_by_id[person_idx][frame_index]
                skeleton_param = human_det.skeleton_param
                time_step = human_det.time_step
                json_event = Event(
                    event_type=event_type,
                    object_id=person_idx, 
                    data=skeleton_param, # TODO: list?
                    time_stamp=time_step/self.fps,
                    frame_nr=time_step
                )
                human_poses.append(json_event)

        event_log["human_poses"] = human_poses

        save_path = self.output_path + '/' + self.output_path.split('/')[-1] + ".json"

        with open(save_path, "wt") as file:
            json.dump(event_log, file, default=_to_dict) 

    def run_selection(self):
        self.pred_objects = self.get_initial_objects(bbox_frame_index=0)
        self.pred_object_poses = self.get_initial_object_events()

        self.keypoints_by_id = self.sort_skeletons_to_ids()

        self.plot_selected_joints_position(selected_joints=['Pelvis', 'R_Hand', 'L_Hand'], person_ids=['1'])

        distances = self.get_distance_person_object(person_idx = '1', joint_name = 'L_Hand', obj=self.pred_objects[0], visualize=True)

        distances = self.get_distance_person_object(person_idx = '1', joint_name = 'R_Hand', obj=self.pred_objects[0], visualize=True)

        distances = self.get_distance_person_object(person_idx = '1', joint_name = 'Pelvis', obj=self.pred_objects[0], visualize=True)

        obj=self.pred_objects[0]
        self.semantic_box = self.get_semantic_box(obj_dict=obj)
        self.feature_points = self.semantic_box.get_semantic_feature_points()

        events = self.get_distance_person_feature_points(person_idx = '1', joint_name = 'L_Hand', visualize=True)

        events = self.get_distance_person_feature_points(person_idx = '1', joint_name = 'R_Hand', visualize=True)

        events = self.get_distance_person_feature_points(person_idx = '1', joint_name = 'Pelvis', visualize=True)

        velocities = self.get_joint_velocity(person_idx = '1', joint_name = 'Pelvis', visualize=True, threshold=0.25)

        velocities = self.get_joint_velocity(person_idx = '1', joint_name = 'R_Hand', visualize=True, threshold=0.25)

        velocities = self.get_joint_velocity(person_idx = '1', joint_name = 'L_Hand', visualize=True, threshold=0.25)

        return self.extract_chair_events()

    def get_joint_series(self, selected_joint='R_Hand', person_idx='1', smoothing=None):
        
        joint_idx = self.joints_name_dict[selected_joint]

        keypoints = [det.skeleton_param for det in self.keypoints_by_id[person_idx]]
        frame_ids = np.array([int(det.time_step) for det in self.keypoints_by_id[person_idx]])

        joint_3d_array = np.array([skeleton[joint_idx] for skeleton in keypoints])

        if smoothing == "average": # moving average
            kernel_size = 10
            joint_3d_array = self.smooth_3d(array_3d=joint_3d_array, window_len=kernel_size)

        return frame_ids, joint_3d_array

    def plot_selected_joints_position(self, selected_joints=['Spine', 'Pelvis', 'Thorax'], person_ids=['1', '2']):
        fig = plt.figure(figsize= [10, 15])

        wl = 60
        down_sampling = 60
        interval_length = 1 / self.fps * down_sampling

        joint_idxs = []
        for name in selected_joints:
            joint_idxs.append(self.joints_name_dict[name])

        for person_idx in person_ids:
            ax = fig.add_subplot(len(person_ids),1, int(person_idx))

            for joint in selected_joints:
                # get all 3d points of the joints over time
                frame_ids, joint_3d_array = self.get_joint_series(joint, person_idx, "average")
                time_axis = frame_ids / float(self.fps)
                ax.plot(frame_ids, joint_3d_array)

                print(len(frame_ids)) # number of frames, in which the person was detected

        # For Plotting
        ax.set_xlabel("time [s]")
        ax.set_ylabel("distance to camera in [mm]")
        ax.title.set_text('test postion plotting')
        signal_list = []
        for s in selected_joints:
            signal_list.append(f"{s}_X")
            signal_list.append(f"{s}_Y")
            signal_list.append(f"{s}_Z")
        ax.legend(signal_list)

        text = "debug"
        plt.savefig(self.output_path + f"/{text}_joint_3d_position_subsampled.pdf")

    def plot_distance_to_features(self, selected_joints=['Spine', 'Pelvis', 'Thorax'], person_idx='1'):
        pass

    def get_distance_two_points(self,joint_one, joint_two):
        return np.linalg.norm(joint_one - joint_two)

    # smoothing methods
    def moving_average(self, x, kernel_size):
        average = np.convolve(x, np.ones(kernel_size), 'same') / kernel_size
        # average[:w] = average[w] # TODO: check
        # average[-w:] = average[-w]
        return average

    def smooth(self, x,window_len=11,window='flat'):
        # https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
        """smooth the data using a window with requested size.
        
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal 
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.
        
        input:
            x: the input signal 
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.

        output:
            the smoothed signal
            
        example:

        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)
        
        see also: 
        
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter
    
        TODO: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """

        if x.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")

        if window_len<3:
            return x


        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')

        y=np.convolve(w/w.sum(),s,mode='valid')
        return y[int(window_len/2-1):-int(window_len/2)]

    def smooth_3d(self, array_3d, window_len=11, window='flat'):
        array_smoothed = array_3d.copy()
        for i in range(3):
            array_smoothed[:, i] = self.smooth(x=array_3d[:, i], window_len=window_len, window=window)

        return array_smoothed

    def butter_lowpass(self, cutoff, fs, order=5):
        return butter(order, cutoff, fs=fs, btype='low', analog=False)

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def closestPointToBox(self, q, origin, v100, v010, v001):
        # distance of human hand joint to bounding boxes
        # https://stackoverflow.com/questions/44824512/how-to-find-the-closest-point-on-a-right-rectangular-prism-3d-rectangle/44824522#44824522
        # https://math.stackexchange.com/questions/2133217/minimal-distance-to-a-cube-in-2d-and-3d-from-a-point-lying-outside

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

    def get_distance_person_object(self, person_idx = '1', joint_name = 'R_Hand', obj=None, visualize=False, feature=True):
        if obj is None:
            obj = self.pred_objects[0]
        print("obj", obj)

        det_obj = Detection9D(**obj)

        tensor = det_obj.get_tensor()
        boxes = CameraInstance3DBoxes([tensor], box_dim=9)
        # TODO: subsample box by hardcoded semantic areas
        boxes_corners = boxes.corners

        origin = boxes_corners[0][3].numpy()
        v100 = boxes_corners[0][7].numpy()
        v010 = boxes_corners[0][0].numpy()
        v001 = boxes_corners[0][2].numpy()

        frame_ids, joint_3d_array = self.get_joint_series(joint_name, person_idx)

        semantic_box = self.get_semantic_box(obj_dict=obj)
        feature_points = semantic_box.get_semantic_feature_points()

        distances = []
        for idx in range(len(frame_ids)):
            queried_joint = joint_3d_array[idx]
            q = queried_joint / 1000

            p = self.closestPointToBox(q, origin, v100, v010, v001)
            distance_human_object = np.linalg.norm(q - p)

            distances.append(distance_human_object)

        # TODO: smoothing?
        if visualize:
            fig = plt.figure(figsize= [20, 10])
            ax = fig.add_subplot(1,1,1)
            ax.plot(frame_ids, distances)
            interval_length = 1 # 1/float(self.fps)

            event_list = []
            event_counter = 0 
            for i in range(1,len(distances)):

                if distances[i] < self.touch_threshold:
                    if event_counter == 0: # a new event detected
                        feature_point, argmin = semantic_box.get_closest_feature_point(query_point= joint_3d_array[i]/1000, feature_points=feature_points)
                        feature_name = semantic_box.semantic_dict[argmin] # TODO: take majority vote on the selected feature
                        # start_time = 
                        event_tmp = TouchEvent(feature_name=feature_name, start_frame=frame_ids[i]) # add complete skeleton parameter 

                    event_counter +=1 # continue in existing event
                    plt.axvspan(frame_ids[i]-0.5*interval_length, frame_ids[i]+0.5*interval_length, color='g', alpha=0.2, lw=0) # for plotting
                else:
                    if event_counter > 0: # existing event has ended
                        event_tmp.update_event_time(end_frame=frame_ids[i])
                        event_list.append(event_tmp)
                        event_counter = 0
                    else: # no event detected
                        event_counter = 0

            if event_counter > 0: # check final event
                event_tmp.update_event_time(end_frame=frame_ids[len(distances)-1])
                event_list.append(event_tmp)

            for ev in event_list:
                if ev.event_duration > 10: #TODO: use events for coloring the spaces
                    ax.annotate(ev.feature_name, xy=(ev.event_time, 0.1))
                    # ax.annotate(ev.feature_name, xy=(ev.event_time, 0.0),  xycoords='data',
                    #     xytext=(0.2, 0.2), textcoords='axes fraction',
                    #     arrowprops=dict(facecolor='black', shrink=0.05), 
                    #     horizontalalignment='right', verticalalignment='top')

            ax.set_xlabel("n frames")
            ax.set_ylabel("distance hand - bounding box [m]")
            ax.title.set_text('test distance plotting')
            text = "debug"
            plt.savefig(self.output_path + f"/{text}_{joint_name}_joint_distance.pdf")

        return distances

    def get_intervals(self, arr, threshold):
        ranges = np.where(np.diff(arr < threshold, prepend=0, append=0))[0].reshape(-1, 2)
        ranges[:, 1] -= 1
        return ranges
        
    def get_joint_velocity(self, joint_name='Pelvis', person_idx='1', visualize=False, threshold=0.25):

        frame_ids, joint_3d_array = self.get_joint_series(joint_name, person_idx)

        positions_smoothed = self.smooth_3d(array_3d=joint_3d_array/1000,window_len=60,window='flat')

        pos_diffs = np.diff(positions_smoothed, n=1, axis=0)
        sampling_n = 30
        pos_diffs_sampled = np.diff(positions_smoothed[::sampling_n], n=1, axis=0)

        time_delta = 1/self.fps # * self.down_sampling

        velos = np.linalg.norm(pos_diffs, axis=1) / time_delta
        velos_sampled = np.linalg.norm(pos_diffs_sampled, axis=1) / (time_delta * sampling_n)

        # velos = self.smooth()

        intervals = self.get_intervals(-1*velos, -1*threshold)
        min_event_duration = 25

        event_list = []
        [event_list.append(MovingEvent(joint_name, interval[0], interval[1])) for interval in intervals if interval[1] - interval[0] > min_event_duration]

        if visualize:
            
            fig = plt.figure(figsize= [20, 10])
            ax = fig.add_subplot(1,1,1)
            legend = []
            
            ax.plot(frame_ids[1:], velos)
            ax.plot(frame_ids[::sampling_n][1:], velos_sampled)
            
            # legend.append(self.semantic_box.semantic_dict[feat])

            for event in event_list:
                plt.axvspan(event.start_frame, event.end_frame, color='g', alpha=0.2, lw=0) # for plotting
                event_time = (event.end_frame - event.start_frame) / 2 + event.start_frame
                ax.annotate("Moving", xy=(event_time, 0.25))
                
            ax.set_xlabel("n frames")
            ax.set_ylabel("joint veloctiy [m/s]")
            ax.title.set_text('test velocity plotting')
            text = "debug"
            ax.legend(legend)
            plt.savefig(self.output_path + f"/{text}_{joint_name}_joint_velocity.pdf")

    def get_distance_person_feature_points(self, person_idx = '1', joint_name = 'R_Hand', visualize=False, threshold=0.25):

        frame_ids, joint_3d_array = self.get_joint_series(joint_name, person_idx)
        
        # semantic_box = self.get_semantic_box(obj_dict=obj)
        # feature_points = semantic_box.get_semantic_feature_points()

        feature_distances = {}

        # check distance for every feature point: 

        print("joint_3d_array", joint_3d_array.shape)
        print("self.feature_points:", self.feature_points.shape)

        event_list = []
        n_feat = len(self.semantic_box.semantic_dict.keys())
        #print("N-feat: ", n_feat)
        for feat_idx in range(n_feat):

            f_name = self.semantic_box.semantic_dict[feat_idx]

            distances = []
           
            diff = joint_3d_array/1000 - self.feature_points[feat_idx]
            #print('diff.shape', diff.shape)
            distances = np.linalg.norm(diff, axis=1)
            #print('distances.shape', distances.shape)

            distances_smoothed = self.smooth(x=distances,window_len=60,window='flat')

            # TODO: Refactor Event Extraction: needed several times self.extract_events(...)
            intervals = self.get_intervals(distances_smoothed, threshold)
            print(f_name, "intervals", intervals)
            
            min_event_duration = 25
            
            [event_list.append(TouchEvent2(joint_name, f_name, interval[0], interval[1])) for interval in intervals if interval[1] - interval[0] > min_event_duration]

            feature_distances[feat_idx] = distances_smoothed

        # filter events for semantically not necessary events
        event_list = [event for event in event_list if not (("Hand" in event.joint_name) and event.feature_name == "Seat")]

        if visualize:

            fig = plt.figure(figsize= [20, 10])
            ax = fig.add_subplot(1,1,1)
            legend = []
            for feat in range(n_feat):
                ax.plot(frame_ids, feature_distances[feat])
                legend.append(self.semantic_box.semantic_dict[feat])

            for event in event_list:
                
                plt.axvspan(event.start_frame, event.end_frame, color='g', alpha=0.2, lw=0) # for plotting
                event_time = (event.end_frame - event.start_frame) / 2 + event.start_frame
                ax.annotate(event.feature_name, xy=(event_time, 0.25))
            

            ax.set_xlabel("n frames")
            ax.set_ylabel("distance hand - feature point [m]")
            ax.title.set_text('test distance plotting')
            text = "debug"
            ax.legend(legend)
            plt.savefig(self.output_path + f"/{text}_{joint_name}_joint_distance_2_feat.pdf")

        return event_list

    def get_initial_objects(self, bbox_frame_index=0):
        # 3D bounding boxes assumed static
        human_cars_dict = self.scene_frames_dict[str(bbox_frame_index)]
        if "objects" in human_cars_dict:
            pred_objects = human_cars_dict["objects"]
        else:
            pred_objects = human_cars_dict["cars"]
        print("Number of oriented bboxes: ", len(pred_objects))

        return pred_objects

    def get_initial_object_events(self): 
        
        event_list = []
        for idx, obj in enumerate(self.pred_objects):
            semantic_box = self.get_semantic_box(obj_dict=obj)
            
            event_tmp = Event(
                event_type= "initial_box",
                object_id=idx, 
                data = semantic_box.tensor[0].tolist(),
                time_stamp=0,
                frame_nr=0
                )

            event_list.append(event_tmp)

        return event_list

    def get_object_series(self, object_idx=0):
        # if not assumed static
        # for obj in 

        object_along_frames = []
        object_indices = []

        self.frame_indices = list(self.scene_frames_dict.keys()) # real video indices

        for index in self.frame_indices:
            object_list = self.scene_frames_dict[index]["objects"]
            if object_idx < len(object_list):
                object_along_frames.append(object_list[object_idx])
                object_indices.append(index)

        return np.array(object_indices), object_along_frames

    def sort_skeletons_to_ids(self):

        # get list of human keypoints over all frames
        keypoints_along_frames = []

        self.frame_indices = list(self.scene_frames_dict.keys()) # real video indices

        # get all human keypoints
        for index in self.frame_indices:
            keypoints_along_frames.append(self.scene_frames_dict[index]["humans"])
            # list of dictionaries, indexed by person_id over all frames

        keypoints_by_id = {}

        for index, keypoints_dict in enumerate(keypoints_along_frames):
            person_ids = list(keypoints_dict.keys())

            for p_id in person_ids:
                if p_id not in keypoints_by_id:
                    keypoints_by_id[p_id] = []
                
                sdet = SkeletonDetection(idx=p_id, skeleton_param=keypoints_dict[p_id], time_step=int(self.frame_indices[index]))
                keypoints_by_id[p_id].append(sdet) 
        
        return keypoints_by_id

    def get_semantic_box(self, obj_dict):

        det_obj = Detection9D(**obj_dict)
        tensor = det_obj.get_tensor()
        print(obj_dict)
        if obj_dict['class_name'] == "bike":
            semantic_box = BikeBox(tensor=[tensor], box_dim=9)
        elif obj_dict['class_name'] == "chair":
            semantic_box = ChairBox(tensor=[tensor], box_dim=9)

        elif obj_dict['class_name'] == "car":
            semantic_box = CarBox(tensor=[tensor], box_dim=7)
        else:
            raise "Semantic properties of class not implemented!"

        return semantic_box

    def test_semantics_series(self, focus_idx=0):

        # all objects in the respective frames
        self.pred_objects = self.get_initial_objects(bbox_frame_index=focus_idx)

        self.keypoints_by_id = self.sort_skeletons_to_ids()
        
        frame_ids_r, joint_3d_array = self.get_joint_series(selected_joint='R_Hand', person_idx='1')
        r_hand = np.array(joint_3d_array[frame_idx]) / 1000
        frame_ids_l, joint_3d_array = self.get_joint_series(selected_joint='L_Hand', person_idx='1')
        l_hand = np.array(joint_3d_array[frame_idx]) / 1000

    def test_semantics(self, frame_idx=0, focus_idx=0):
        
        # all objects in the respective frames
        self.pred_objects = self.get_initial_objects(bbox_frame_index=focus_idx)

        self.keypoints_by_id = self.sort_skeletons_to_ids()
        
        frame_ids_r, joint_3d_array = self.get_joint_series(selected_joint='R_Hand', person_idx='1')
        r_hand = np.array(joint_3d_array[frame_idx]) / 1000
        frame_ids_l, joint_3d_array = self.get_joint_series(selected_joint='L_Hand', person_idx='1')
        l_hand = np.array(joint_3d_array[frame_idx]) / 1000
        # check sitting:
        frame_ids_p, joint_3d_array = self.get_joint_series(selected_joint='Pelvis', person_idx='1')
        pelvis = np.array(joint_3d_array[frame_idx]) / 1000

        # get all instances of an specific object along the video
        frame_ids_obj, objects_along_frames = self.get_object_series(object_idx=0)

        # get reference object in the first frame
        pred_obj = self.pred_objects[0]

        semantic_box = self.get_semantic_box(obj_dict=pred_obj)

        feature_points = semantic_box.get_semantic_feature_points()
        print("feature_points", feature_points)
        print("r_hand", r_hand)
        feature_point, argmin = semantic_box.get_closest_feature_point(query_point=r_hand, feature_points=feature_points)
        print("l_hand", l_hand)
        feature_point, argmin = semantic_box.get_closest_feature_point(query_point=l_hand, feature_points=feature_points)
        print("pelvis", pelvis)
        feature_point, argmin = semantic_box.get_closest_feature_point(query_point=pelvis, feature_points=feature_points)
