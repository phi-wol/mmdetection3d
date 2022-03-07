ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV, MMDetection and MMSegmentation
RUN pip install mmcv-full==1.3.18 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
RUN pip install mmdet==2.19.0
RUN pip install mmsegmentation==0.20.0

# Install MMDetection3D
RUN conda clean --all
COPY . /mmdetection3d
WORKDIR /mmdetection3d
RUN git checkout v1.0.0.dev0
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

#ENTRYPOINT ["python", "demo/mono_det_demo.py", "/demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg", "/demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525_mono3d.coco.json", "/configs/smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py", "/checkpoints/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_20210929_015553-d46d9bb0.pth", "--out-dir", "/kitti_output"]
