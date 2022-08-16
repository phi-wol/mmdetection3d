
#%%
# Imports
from ast import Pass
import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from DMPPE_POSENET_RELEASE.common.utils.vis import vis_3d_multiple_skeleton_box
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

import scipy.io

from mmdet_viz.cam_box3d import CameraInstance3DBoxes
from mmdet_viz.utils import draw_camera_bbox3d_on_img

#%% 

# load bounding box
#%% Bounding Box

# 1
box_inference_1 = np.array([[-0.47982168197631836, 0.5114896297454834, 13.817742347717285, 3.949936866760254, 1.5974817276000977, 1.6347150802612305, 1.1412615776062012]])
box_inference_1 = np.array([[-0.4630279541015625, 0.4935872554779053, 13.817742347717285, 3.949936866760254, 1.5974817276000977, 1.6347150802612305, 1.1424756050109863]])
# 15
box_inference_2 = np.array([[-0.5321626663208008, 0.49402642250061035, 13.339641571044922, 3.934598207473755, 1.5928183794021606, 1.6316965818405151, 1.1810637712478638]])

# original landscape intrinsics:

scale_factor = 2.2
intrinsics = np.array([
    [1447*scale_factor, 0, 962], 
    [0, 1447*scale_factor, 682],
    [0, 0, 1]]
    )

intrinsics = np.array([
    [1447*scale_factor, 0, 682], 
    [0, 1447*scale_factor, 962],
    [0, 0, 1]]
    )

# load skeletons

print("Size:", box_inference_1.size)

mat = scipy.io.loadmat('./output/car_multi_person_1/preds_3d_kpt_mupots.mat')
pred_3d_kpt = np.array(mat['0'])
print(mat['0'])

# visualize skeletons

joint_num = 21
joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )

# TODO: 

# load image

vis_img = cv2.imread('./output/car_multi_person_1/car_multi_person_1_pose_2d_0000.jpg') # in RGB

#vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

# draw bounding box on image

# convert 

original_intrinsics = np.array([])

bboxes3d = CameraInstance3DBoxes(box_inference_1, box_dim=7)

bbox_corners =  bboxes3d.corners[0]


vis_img = draw_camera_bbox3d_on_img(bboxes3d, # CameraInstance3DBoxes`, shape=[M, 9] TODO: change to 9
                              vis_img,
                              intrinsics, # intrinsics
                              None,
                              color= (255, 170, 50), # (0, 255, 0), green
                              thickness=3,
                              color_2 = (50, 170, 255), # green rgb bgr
                              object_id=None)

print(bbox_corners)

print("BBox Corners: ", bbox_corners*1000)

vis_img_3d = vis_3d_multiple_skeleton_box(pred_3d_kpt, np.ones_like(pred_3d_kpt), skeleton, './output/car_multi_person_1/', vis_img, 0, original=False, bbox=bbox_corners*1000)


#vis_3d_multiple_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None, image=None, frame_id=0, original=True)
