import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from ...main.config import cfg
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


def vis_keypoints(img, kps, kps_lines, kp_thresh=0.4, alpha=1):

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = np.array([(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors])

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = np.array([[c[2], c[1], c[0]] for c in colors])

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1,0], kpt_3d[i2,0]])
        y = np.array([kpt_3d[i1,1], kpt_3d[i2,1]])
        z = np.array([kpt_3d[i1,2], kpt_3d[i2,2]])

        if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1,0] > 0:
            ax.scatter(kpt_3d[i1,0], kpt_3d[i1,2], -kpt_3d[i1,1], c=colors[l], marker='o')
        if kpt_3d_vis[i2,0] > 0:
            ax.scatter(kpt_3d[i2,0], kpt_3d[i2,2], -kpt_3d[i2,1], c=colors[l], marker='o')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()
    
    
    plt.show()
    cv2.waitKey(0)

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    return ax

def vis_3d_multiple_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None, image=None, frame_id=0, original=True):

    # kps_lines = skeleton

    fig = plt.figure(figsize= [20,10])
    ax_0 = fig.add_subplot(121)
    ax_0.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    ax = fig.add_subplot(122, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = np.array([[c[2], c[1], c[0]] for c in colors])
    # colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]

        person_num = kpt_3d.shape[0]
        for n in range(person_num):
            x = np.array([kpt_3d[n,i1,0], kpt_3d[n,i2,0]])
            y = np.array([kpt_3d[n,i1,1], kpt_3d[n,i2,1]])
            z = np.array([kpt_3d[n,i1,2], kpt_3d[n,i2,2]])

            # if x.max() > 12.000:
            #     continue

            # x z -y vs. -z x -y

            if original:
                if kpt_3d_vis[n,i1,0] > 0 and kpt_3d_vis[n,i2,0] > 0:
                    ax.plot(x, z, -y, c=colors[l], linewidth=2)
                if kpt_3d_vis[n,i1,0] > 0:
                    ax.scatter(kpt_3d[n,i1,0], kpt_3d[n,i1,2], -kpt_3d[n,i1,1], c=colors[l], marker='o')
                if kpt_3d_vis[n,i2,0] > 0:
                    ax.scatter(kpt_3d[n,i2,0], kpt_3d[n,i2,2], -kpt_3d[n,i2,1], c=colors[l], marker='o')

            else:
                if kpt_3d_vis[n,i1,0] > 0 and kpt_3d_vis[n,i2,0] > 0:
                    ax.plot(-z, x, -y, c=colors[l], linewidth=2)
                if kpt_3d_vis[n,i1,0] > 0:
                    ax.scatter(-1 * kpt_3d[n,i1,2], kpt_3d[n,i1,0], -kpt_3d[n,i1,1], c=colors[l], marker='o')
                if kpt_3d_vis[n,i2,0] > 0:
                    ax.scatter(-1 * kpt_3d[n,i2,2], kpt_3d[n,i2,0], -kpt_3d[n,i2,1], c=colors[l], marker='o')

            # original:
            # if kpt_3d_vis[n,i1,0] > 0 and kpt_3d_vis[n,i2,0] > 0:
            #     ax.plot(x, z, -y, c=colors[l], linewidth=2)
            # if kpt_3d_vis[n,i1,0] > 0:
            #     ax.scatter(kpt_3d[n,i1,0], kpt_3d[n,i1,2], -kpt_3d[n,i1,1], c=colors[l], marker='o')
            # if kpt_3d_vis[n,i2,0] > 0:
            #     ax.scatter(kpt_3d[n,i2,0], kpt_3d[n,i2,2], -kpt_3d[n,i2,1], c=colors[l], marker='o')

    
    #ax.set_title('3D human keypoints [PoseNet]')
    
    if filename is None:
        ax.set_title('3D human keypoints [PoseNet]')
    else:
        ax.set_title(filename.split("/")[0])

    print(filename.split("/")[0])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')

    if not original:
        x = -kpt_3d[:,:,2] # -z
        xmin = x.min() #- 120000 # - 120000
        xmax = x.max() #+ 6000
        
        y = kpt_3d[:,:,0] # x
        ymin =y.min() 
        ymax = y.max()

        z = -kpt_3d[:,:,1] # -y
        zmin = z.min()
        zmax = z.max()

        ax.view_init(27,0)


    # visual fix

    # min_all = min(xmin, zmin)
    # max_all = max(xmax, zmax)

    # ax.set_xlim([min_all, max_all])
    # ax.set_ylim([ymin, ymax])
    # ax.set_zlim([min_all, max_all])

        
        
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_zlim([zmin, zmax])

        ax = set_axes_equal(ax)

        plot_surface = False
    
        if plot_surface: 

            ax.plot_surface(
            np.array([[xmin],[xmin]]), 
            np.array([[ymin, ymax], [ymin, ymax]]), 
            np.array([[zmax, zmax], [zmin, zmin]]),
            rstride=1, cstride=1, facecolors=image/255) # np.flipud(image.swapaxes(0,1)/255)

            print(image)

    #ax.set_box_aspect(aspect = [1,1,1])
    #ax.legend()


    canvas = FigureCanvas(fig)
    canvas.draw()        # draw the canvas, cache the renderer
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(canvas.get_width_height()[::-1] + (3,))

    plt.savefig(filename + "_pose_3d_{:04d}.jpg".format(frame_id), bbox_inches='tight')

    return image
    
    # plt.show()
    # cv2.waitKey(0)

def vis_3d_multiple_skeleton_box(kpt_3d, kpt_3d_vis, kps_lines, filename=None, image=None, frame_id=0, original=True, bbox=None):

    # kps_lines = skeleton

    fig = plt.figure(figsize= [20,10])
    ax_0 = fig.add_subplot(121)
    ax_0.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    ax = fig.add_subplot(122, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = np.array([[c[2], c[1], c[0]] for c in colors])
    # colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]

        person_num = kpt_3d.shape[0]
        for n in range(person_num):
            x = np.array([kpt_3d[n,i1,0], kpt_3d[n,i2,0]])
            y = np.array([kpt_3d[n,i1,1], kpt_3d[n,i2,1]])
            z = np.array([kpt_3d[n,i1,2], kpt_3d[n,i2,2]])

            # if x.max() > 12.000:
            #     continue

            # x z -y vs. -z x -y

            if original:
                if kpt_3d_vis[n,i1,0] > 0 and kpt_3d_vis[n,i2,0] > 0:
                    ax.plot(x, z, -y, c=colors[l], linewidth=2)
                if kpt_3d_vis[n,i1,0] > 0:
                    ax.scatter(kpt_3d[n,i1,0], kpt_3d[n,i1,2], -kpt_3d[n,i1,1], c=colors[l], marker='o')
                if kpt_3d_vis[n,i2,0] > 0:
                    ax.scatter(kpt_3d[n,i2,0], kpt_3d[n,i2,2], -kpt_3d[n,i2,1], c=colors[l], marker='o')

            else:
                if kpt_3d_vis[n,i1,0] > 0 and kpt_3d_vis[n,i2,0] > 0:
                    ax.plot(-z, x, -y, c=colors[l], linewidth=2)
                if kpt_3d_vis[n,i1,0] > 0:
                    ax.scatter(-1 * kpt_3d[n,i1,2], kpt_3d[n,i1,0], -kpt_3d[n,i1,1], c=colors[l], marker='o')
                if kpt_3d_vis[n,i2,0] > 0:
                    ax.scatter(-1 * kpt_3d[n,i2,2], kpt_3d[n,i2,0], -kpt_3d[n,i2,1], c=colors[l], marker='o')

            # original:
            # if kpt_3d_vis[n,i1,0] > 0 and kpt_3d_vis[n,i2,0] > 0:
            #     ax.plot(x, z, -y, c=colors[l], linewidth=2)
            # if kpt_3d_vis[n,i1,0] > 0:
            #     ax.scatter(kpt_3d[n,i1,0], kpt_3d[n,i1,2], -kpt_3d[n,i1,1], c=colors[l], marker='o')
            # if kpt_3d_vis[n,i2,0] > 0:
            #     ax.scatter(kpt_3d[n,i2,0], kpt_3d[n,i2,2], -kpt_3d[n,i2,1], c=colors[l], marker='o')

    if bbox is not None:
        
        color_tmp = colors[0]
        ax.scatter(bbox[:, 0], bbox[:,2], -bbox[:, 1], s=20, color = color_tmp)#c=[color_tmp]) # TODO: orderchange
        line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7), 
                    (0,5), (1, 4), (0, 7), (3, 4))
        # top_indices = ((0,5), (1, 4), (0, 7), (3, 4))

        for index in line_indices:
            start = index[0]
            end = index[1]
            ax.plot(
                xs= [bbox[:, 0][start], bbox[:, 0][end]], 
                ys= [bbox[:, 2][start], bbox[:, 2][end]],
                zs= [-bbox[:, 1][start], -bbox[:, 1][end]],# TODO: orderchange
                color = color_tmp)

    
    #ax.set_title('3D human keypoints [PoseNet]')
    
    if filename is None:
        ax.set_title('3D human keypoints [PoseNet]')
    else:
        ax.set_title(filename.split("/")[0])

    print(filename.split("/")[0])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')

    if not original:
        x = -kpt_3d[:,:,2] # -z
        xmin = x.min() #- 120000 # - 120000
        xmax = x.max() #+ 6000
        
        y = kpt_3d[:,:,0] # x
        ymin =y.min() 
        ymax = y.max()

        z = -kpt_3d[:,:,1] # -y
        zmin = z.min()
        zmax = z.max()

        ax.view_init(27,0)


    # visual fix

    # min_all = min(xmin, zmin)
    # max_all = max(xmax, zmax)

    # ax.set_xlim([min_all, max_all])
    # ax.set_ylim([ymin, ymax])
    # ax.set_zlim([min_all, max_all])

        
        
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_zlim([zmin, zmax])

        ax = set_axes_equal(ax)

        plot_surface = False
    
        if plot_surface: 

            ax.plot_surface(
            np.array([[xmin],[xmin]]), 
            np.array([[ymin, ymax], [ymin, ymax]]), 
            np.array([[zmax, zmax], [zmin, zmin]]),
            rstride=1, cstride=1, facecolors=image/255) # np.flipud(image.swapaxes(0,1)/255)

            print(image)

    #ax.set_box_aspect(aspect = [1,1,1])
    #ax.legend()


    canvas = FigureCanvas(fig)
    canvas.draw()        # draw the canvas, cache the renderer
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(canvas.get_width_height()[::-1] + (3,))

    plt.savefig(filename + "_pose_3d_box_{:04d}.jpg".format(frame_id), bbox_inches='tight')

    return image

def vis_3d_multiple_skeleton_original(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]

        person_num = kpt_3d.shape[0]
        for n in range(person_num):
            x = np.array([kpt_3d[n,i1,0], kpt_3d[n,i2,0]])
            y = np.array([kpt_3d[n,i1,1], kpt_3d[n,i2,1]])
            z = np.array([kpt_3d[n,i1,2], kpt_3d[n,i2,2]])

            if kpt_3d_vis[n,i1,0] > 0 and kpt_3d_vis[n,i2,0] > 0:
                ax.plot(x, z, -y, c=colors[l], linewidth=2)
            if kpt_3d_vis[n,i1,0] > 0:
                ax.scatter(kpt_3d[n,i1,0], kpt_3d[n,i1,2], -kpt_3d[n,i1,1], c=colors[l], marker='o')
            if kpt_3d_vis[n,i2,0] > 0:
                ax.scatter(kpt_3d[n,i2,0], kpt_3d[n,i2,2], -kpt_3d[n,i2,1], c=colors[l], marker='o')

    #if filename is None:
    ax.set_title('3D human keypoints [PoseNet]')
    # else:
    #     ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()