import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from .DMPPE_POSENET_RELEASE.common.utils.vis import vis_keypoints, vis_3d_multiple_skeleton

# function img = draw_2Dskeleton(img_name, pred_2d_kpt, num_joint, skeleton, colorList_joint, colorList_skeleton)

# vis_keypoints(vis_img, vis_kps, skeleton)# visualize single skeleton

def draw_2Dskeleton(img, pred_2d_kpt, num_joint, skeleton, colorList_joint, colorList_skeleton):
 

    imgWidth, imgHeight = img.shape[0], img.shape[1]

    # fig = plt.figure(figsize=(figure_width figure_height))
    # canvas = FigureCanvas(fig)

    # line_width = 4
    
    # num_skeleton = size(skeleton,1);

    # num_pred = size(pred_2d_kpt,1);
    # for i = 1:num_pred
    #     for j =1:num_skeleton
    #         k1 = skeleton(j,1);
    #         k2 = skeleton(j,2);
    #         plot([pred_2d_kpt(i,k1,1),pred_2d_kpt(i,k2,1)],[pred_2d_kpt(i,k1,2),pred_2d_kpt(i,k2,2)],'Color',colorList_skeleton(j,:),'LineWidth',line_width);
    #     end
    #     for j=1:num_joint
    #         scatter(pred_2d_kpt(i,j,1),pred_2d_kpt(i,j,2),100,colorList_joint(j,:),'filled');
    #     end
    # end
    
    # set(gca,'Units','normalized','Position',[0 0 1 1]);  %# Modify axes size

    # frame = getframe(gcf);
    # img = frame.cdata;
    
    # hold off;
    # close(f); 

    # return img


def draw_3Dskeleton(img, pred_3d_kpt, num_joint, skeleton, colorList_joint, colorList_skeleton):
 
    x = pred_3d_kpt[:,:,0]
    y = pred_3d_kpt[:,:,1]
    z = pred_3d_kpt[:,:,2]
    pred_3d_kpt[:,:,0] = -z
    pred_3d_kpt[:,:,1] = x
    pred_3d_kpt[:,:,2] = -y

    # [imgHeight, imgWidth, dim] = size(img);
    imgWidth, imgHeight = img.shape[0], img.shape[1]

    figure_height = 450
    figure_width = figure_height / imgHeight * imgWidth

    fig = plt.figure(figsize=[figure_width, figure_height])
    ax = fig.add_subplot(111, projection='3d')
    
    # matplotlib?
    #f = figure('Position',[100 100 figure_width figure_height]);
    #set(f, 'visible', 'off');
    #hold on;
    #grid on;
    line_width = 4
    point_width = 50
 
    num_skeleton = skeleton.shape[0] # size(skeleton,1);

    num_pred = pred_3d_kpt.shape[0] # size(pred_3d_kpt,1);

    for i in range(num_pred):
        for j in range(num_skeleton):
            k1 = skeleton[j,0]
            k2 = skeleton[j,1]
            ax.plot(
                [pred_3d_kpt[i,k1,0],pred_3d_kpt[i,k2,0]],
                [pred_3d_kpt[i,k1,1],pred_3d_kpt[i,k2,1]],
                [pred_3d_kpt[i,k1,2],pred_3d_kpt[i,k2,2]],
                c = colorList_skeleton[j,:],
                lw = line_width)

        for j in range(num_joint):
            ax.scatter(
                pred_3d_kpt[i,j,0],
                pred_3d_kpt[i,j,1],
                pred_3d_kpt[i,j,2],
                s = point_width, 
                c = colorList_joint[j,:])

   
    # set(gca, 'color', [255/255 255/255 255/255]);
    # set(gca,'XTickLabel',[]);
    # set(gca,'YTickLabel',[]);
    # set(gca,'ZTickLabel',[]);
    
    x = pred_3d_kpt[:,:,0]
    xmin = x.min() - 120000
    xmax = x.max() + 6000
    
    y = pred_3d_kpt[:,:,1]
    ymin =y.min()
    ymax = y.max()

    z = pred_3d_kpt[:,:,2]
    zmin = z.min()
    zmax = z.max()
    
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])

    # ax.set_zlim(1,2)

    ax.plot_surface(
        np.array([[xmin],[xmin]]), 
        np.array([[ymin, ymax], [ymin, ymax]]), 
        np.array([[zmax, zmax], [zmin, zmin]]),
        rstride=5, cstride=5, facecolors=img/255)
        # 'CData', img,'FaceColor','texturemap')

        
    
    # h_img = surf([xmin;xmin],[ymin ymax;ymin ymax],[zmax zmax;zmin zmin],'CData',img,'FaceColor','texturemap');
    
    # view(62,27);
    ax.view_init(62,27)

    save_path = "output/test_visualization.pdf"
    plt.savefig(save_path)
    plt.savefig("output/test_visualization.png")

def draw_3Dpose(image, pred_2d_kpt, pred_3d_kpt):

    colorList_skeleton = np.array([
    [255/255, 128/255, 0/255],
    [255/255, 153/255, 51/255],
    [255/255, 178/255, 102/255],
    [230/255, 230/255, 0/255],

    [255/255, 153/255, 255/255],
    [153/255, 204/255, 255/255],

    [255/255, 102/255, 255/255],
    [255/255, 51/255, 255/255],

    [102/255, 178/255, 255/255],
    [51/255, 153/255, 255/255],

    [255/255, 153/255, 153/255],
    [255/255, 102/255, 102/255],
    [255/255, 51/255, 51/255],

    [153/255, 255/255, 153/255],
    [102/255, 255/255, 102/255],
    [51/255, 255/255, 51/255]])

    colorList_joint = np.array([
    [255/255, 128/255, 0/255],
    [255/255, 153/255, 51/255],
    [255/255, 153/255, 153/255],
    [255/255, 102/255, 102/255],
    [255/255, 51/255, 51/255],
    [153/255, 255/255, 153/255],
    [102/255, 255/255, 102/255],
    [51/255, 255/255, 51/255],
    [255/255, 153/255, 255/255],
    [255/255, 102/255, 255/255],
    [255/255, 51/255, 255/255],
    [153/255, 204/255, 255/255],
    [102/255, 178/255, 255/255],
    [51/255, 153/255, 255/255],
    [230/255, 230/255, 0/255],
    [230/255, 230/255, 0/255],
    [255/255, 178/255, 102/255]])

    joint_num = 21
    joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
    flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
    skeleton = np.array(( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) ))


    skeleton = skeleton.reshape([2,16]).transpose()
    #skeleton = transpose(reshape(skeleton,[2,16])) + 1;

    # fp_img_name = fopen('../coco_img_name.txt');
    # preds_2d_kpt = load('preds_2d_kpt_coco.mat');
    # preds_3d_kpt = load('preds_3d_kpt_coco.mat');

    # img_2d_proj = draw_2Dskeleton(img,pred_2d_kpt,num_joint,skeleton,colorList_joint,colorList_skeleton);

    # vis_kps = np.array(output_pose_3d_list)
    #vis_3d_multiple_skeleton(pred_3d_kpt, np.ones_like(pred_3d_kpt), skeleton, 'output_pose_3d (x,y,z: camera-centered. mm.)')

    image_viz = draw_3Dskeleton(image,pred_3d_kpt,num_joint,skeleton,colorList_joint,colorList_skeleton)

    return image_viz


    # TODO: save image / write to output result video


# def draw_3Dpose_coco()
 
#     root_path = '/mnt/hdd1/Data/Human_pose_estimation/COCO/2017/val2017/'
#     save_path = './output/vis'
#     num_joint =  17

#     colorList_skeleton = np.array([
#     [255/255 128/255 0/255],
#     [255/255 153/255 51/255],
#     [255/255 178/255 102/255],
#     [230/255 230/255 0/255],

#     [255/255 153/255 255/255],
#     [153/255 204/255 255/255],

#     [255/255 102/255 255/255],
#     [255/255 51/255 255/255],

#     [102/255 178/255 255/255],
#     [51/255 153/255 255/255],

#     [255/255 153/255 153/255],
#     [255/255 102/255 102/255],
#     [255/255 51/255 51/255],

#     [153/255 255/255 153/255],
#     [102/255 255/255 102/255],
#     [51/255 255/255 51/255]])

#     colorList_joint = np.array([
#     [255/255 128/255 0/255],
#     [255/255 153/255 51/255],
#     [255/255 153/255 153/255],
#     [255/255 102/255 102/255],
#     [255/255 51/255 51/255],
#     [153/255 255/255 153/255],
#     [102/255 255/255 102/255,]
#     [51/255 255/255 51/255],
#     [255/255 153/255 255/255],
#     [255/255 102/255 255/255],
#     [255/255 51/255 255/255],
#     [153/255 204/255 255/255],
#     [102/255 178/255 255/255],
#     [51/255 153/255 255/255],
#     [230/255 230/255 0/255],
#     [230/255 230/255 0/255],
#     [255/255 178/255 102/255]])

#     skeleton = np.array([ [0, 16], [1, 16], [1, 15], [15, 14], [14, 8], [14, 11], [8, 9], [9, 10], [11, 12], [12, 13], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7] ])
    
#     skeleton = skeleton.reshape([2,16]).transpose()
#     #skeleton = transpose(reshape(skeleton,[2,16])) + 1;

#     # fp_img_name = fopen('../coco_img_name.txt');
#     # preds_2d_kpt = load('preds_2d_kpt_coco.mat');
#     # preds_3d_kpt = load('preds_3d_kpt_coco.mat');

#     %img = draw_2Dskeleton(img_path,pred_2d_kpt,num_joint,skeleton,colorList_joint,colorList_skeleton);
#     f = draw_3Dskeleton(img,pred_3d_kpt,num_joint,skeleton,colorList_joint,colorList_skeleton);

#     img_name = fgetl(fp_img_name);

#     # for every image
#     while ischar(img_name)
        
#         if isfield(preds_2d_kpt,img_name)
#             pred_2d_kpt = getfield(preds_2d_kpt,img_name);
#             pred_3d_kpt = getfield(preds_3d_kpt,img_name);

#             pred_2d_kpt # for single image
#             pred_3d_kpt # for single image
            
#             img_name = strsplit(img_name,'_'); 
#             img_name = strcat(img_name{2},'.jpg');
#             img_path = strcat(root_path,img_name);
            
#             %img = draw_2Dskeleton(img_path,pred_2d_kpt,num_joint,skeleton,colorList_joint,colorList_skeleton);
#             img = imread(img_path);
#             f = draw_3Dskeleton(img,pred_3d_kpt,num_joint,skeleton,colorList_joint,colorList_skeleton);
            
#             set(gcf, 'InvertHardCopy', 'off');
#             set(gcf,'color','w');
#             mkdir(save_path);
#             saveas(f, strcat(save_path,img_name));
#             close(f);
#         end

#         img_name = fgetl(fp_img_name);
#     end