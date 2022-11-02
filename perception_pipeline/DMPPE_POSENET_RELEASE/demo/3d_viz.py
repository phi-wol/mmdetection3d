

function f = draw_3Dskeleton(img, pred_3d_kpt, num_joint, skeleton, colorList_joint, colorList_skeleton)
 
    x = pred_3d_kpt(:,:,1);
    y = pred_3d_kpt(:,:,2);
    z = pred_3d_kpt(:,:,3);
    pred_3d_kpt(:,:,1) = -z;
    pred_3d_kpt(:,:,2) = x;
    pred_3d_kpt(:,:,3) = -y;

    [imgHeight, imgWidth, dim] = size(img);
    
    figure_height = 450;
    figure_width = figure_height / imgHeight * imgWidth;
    f = figure('Position',[100 100 figure_width figure_height]);
    set(f, 'visible', 'off');
    hold on;
    grid on;
    line_width = 4;
    point_width = 50;
 
    num_skeleton = size(skeleton,1);

    num_pred = size(pred_3d_kpt,1);
    for i = 1:num_pred
        for j =1:num_skeleton
            k1 = skeleton(j,1);
            k2 = skeleton(j,2);

            plot3([pred_3d_kpt(i,k1,1),pred_3d_kpt(i,k2,1)],[pred_3d_kpt(i,k1,2),pred_3d_kpt(i,k2,2)],[pred_3d_kpt(i,k1,3),pred_3d_kpt(i,k2,3)],'Color',colorList_skeleton(j,:),'LineWidth',line_width);
        end
        for j=1:num_joint
            scatter3(pred_3d_kpt(i,j,1),pred_3d_kpt(i,j,2),pred_3d_kpt(i,j,3),point_width,colorList_joint(j,:),'filled');
        end
    end
   
    set(gca, 'color', [255/255 255/255 255/255]);
    set(gca,'XTickLabel',[]);
    set(gca,'YTickLabel',[]);
    set(gca,'ZTickLabel',[]);
    
    x = pred_3d_kpt(:,:,1);
    xmin = min(x(:)) - 120000;
    xmax = max(x(:)) + 6000;
    
    y = pred_3d_kpt(:,:,2);
    ymin = min(y(:));
    ymax = max(y(:));

    z = pred_3d_kpt(:,:,3);
    zmin = min(z(:));
    zmax = max(z(:));
    
    xlim([xmin xmax]);
    ylim([ymin ymax]);
    zlim([zmin zmax]);
    
    h_img = surf([xmin;xmin],[ymin ymax;ymin ymax],[zmax zmax;zmin zmin],'CData',img,'FaceColor','texturemap');
    set(h_img);
    
    view(62,27);
end


function img = draw_2Dskeleton(img_name, pred_2d_kpt, num_joint, skeleton, colorList_joint, colorList_skeleton)
 
    img = imread(img_name);
    [imgHeight, imgWidth, dim] = size(image);

    f = figure;
    set(f, 'visible', 'off');
    imshow(img);
    hold on;
    line_width = 4;
    
    num_skeleton = size(skeleton,1);

    num_pred = size(pred_2d_kpt,1);
    for i = 1:num_pred
        for j =1:num_skeleton
            k1 = skeleton(j,1);
            k2 = skeleton(j,2);
            plot([pred_2d_kpt(i,k1,1),pred_2d_kpt(i,k2,1)],[pred_2d_kpt(i,k1,2),pred_2d_kpt(i,k2,2)],'Color',colorList_skeleton(j,:),'LineWidth',line_width);
        end
        for j=1:num_joint
            scatter(pred_2d_kpt(i,j,1),pred_2d_kpt(i,j,2),100,colorList_joint(j,:),'filled');
        end
    end
    
    set(gca,'Units','normalized','Position',[0 0 1 1]);  %# Modify axes size

    frame = getframe(gcf);
    img = frame.cdata;
    
    hold off;
    close(f); 

end


function draw_3Dpose_mupots()
 
    root_path = '/mnt/hdd1/Data/Human_pose_estimation/MU/mupots-3d-eval/MultiPersonTestSet/';
    save_path = './vis/';
    num_joint =  17;

    colorList_skeleton = [
    255/255 128/255 0/255;
    255/255 153/255 51/255;
    255/255 178/255 102/255;
    230/255 230/255 0/255;

    255/255 153/255 255/255;
    153/255 204/255 255/255;

    255/255 102/255 255/255;
    255/255 51/255 255/255;

    102/255 178/255 255/255;
    51/255 153/255 255/255;

    255/255 153/255 153/255;
    255/255 102/255 102/255;
    255/255 51/255 51/255;

    153/255 255/255 153/255;
    102/255 255/255 102/255;
    51/255 255/255 51/255;
    ];
    colorList_joint = [
    255/255 128/255 0/255;
    255/255 153/255 51/255;
    255/255 153/255 153/255;
    255/255 102/255 102/255;
    255/255 51/255 51/255;
    153/255 255/255 153/255;
    102/255 255/255 102/255;
    51/255 255/255 51/255;
    255/255 153/255 255/255;
    255/255 102/255 255/255;
    255/255 51/255 255/255;
    153/255 204/255 255/255;
    102/255 178/255 255/255;
    51/255 153/255 255/255;
    230/255 230/255 0/255;
    230/255 230/255 0/255;
    255/255 178/255 102/255;

    ];
    skeleton = [ [0, 16], [1, 16], [1, 15], [15, 14], [14, 8], [14, 11], [8, 9], [9, 10], [11, 12], [12, 13], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7] ];
    skeleton = transpose(reshape(skeleton,[2,16])) + 1;

    fp_img_name = fopen('../mupots_img_name.txt');
    preds_2d_kpt = load('preds_2d_kpt_mupots.mat');
    preds_3d_kpt = load('preds_3d_kpt_mupots.mat');

    img_name = fgetl(fp_img_name);
    while ischar(img_name)
        img_name_split = strsplit(img_name);
        folder_id = str2double(img_name_split(1)); frame_id = str2double(img_name_split(2));
        img_name = sprintf('TS%d/img_%06d.jpg',folder_id, frame_id);
        img_path = strcat(root_path,img_name);

        pred_2d_kpt = getfield(preds_2d_kpt,sprintf('TS%d_img_%06d',folder_id, frame_id));
        pred_3d_kpt = getfield(preds_3d_kpt,sprintf('TS%d_img_%06d',folder_id, frame_id));

        %img = draw_2Dskeleton(img_path,pred_2d_kpt,num_joint,skeleton,colorList_joint,colorList_skeleton);
        img = imread(img_path);
        f = draw_3Dskeleton(img,pred_3d_kpt,num_joint,skeleton,colorList_joint,colorList_skeleton);

        set(gcf, 'InvertHardCopy', 'off');
        set(gcf,'color','w');
        mkdir(strcat(save_path,sprintf('TS%d',folder_id)));
        saveas(f, strcat(save_path,img_name));
        close(f);

        img_name = fgetl(fp_img_name);
    end
        
end