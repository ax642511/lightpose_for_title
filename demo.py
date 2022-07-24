import argparse

import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width


class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img

def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape    # 实际高和宽
    scale = net_input_height_size / height  # 将实际高缩放到期望高的缩放倍数

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)    # 缩放后的图像
    scaled_img = normalize(scaled_img, img_mean, img_scale)     # 归一化图像
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]     
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)    # 填充到高宽为stride整数倍的值

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()     # 由HWC转成CHW（BGR格式）
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)     # 得到网络输出

    stage2_heatmaps = stages_output[-2]
    
   
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))    #最后一个stage的热图作为最终的热图
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)    # 热图放大upsample_ratio倍

    stage2_pafs = stages_output[-1] # 最后一个stage的paf
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))     # 最后一个stage的paf作为最终的paf
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)    # paf放大upsample_ratio倍

    return heatmaps, pafs, scale, pad    # 返回热图，paf，输入模型图像相比原始图像缩放倍数，输入模型图像padding尺寸


def run_demo(net, image_provider, height_size, cpu, track, smooth):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1
    
    #print(image_provider)
    l=0
    for img in image_provider:
        orig_img = img.copy()
        
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)  # 热图，paf，输入模型图像相比原始图像缩放倍数，输入模型图像padding尺寸

        total_keypoints_num = 0
        all_keypoints_by_type = []  # all_keypoints_by_type为18个list，每个list包含Ni个当前点的x、y坐标，当前点热图值，当前点在所有特征点中的index

        for kpt_idx in range(num_keypoints):  # 19th for bg,第19个为背景，之考虑前18个关节点
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)
        # 得到所有分配的人（前18维为每个人各个关节点在所有关节点中的索引，后两维为每个人得分及每个人关节点数量），及所有关节点信息    
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)

        #print('pose_entries等于：')
        #print(pose_entries)
        
        # all_keypoints是检测到的所有点数量和置信度
        #print('all_keypoints等于')
        #print(all_keypoints)

        for kpt_id in range(all_keypoints.shape[0]):    # 依次将每个关节点信息缩放回原始图像上
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []

        for n in range(len(pose_entries)):  # 依次遍历找到的每个人
            if len(pose_entries[n]) == 0:
                continue

            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        
        # 只显示骨骼点，不显示背景图
        t= list(img.shape)
        tempImage = np.zeros((int(t[0]),int(t[1]),3)) # 其中必须传入tuple
        for pose in current_poses:
            pose.draw(tempImage) # 在新生成的自定义图像上绘制pose的线条
        
        # pose中主要保存人体的关键点信息以及连接方式
        #for pose in current_poses:
            #pose.draw(tempImage)

        # cv2.addWeighted()函数————融合权重加法函数
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        
        for pose in current_poses:
            #cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                        #(pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if track:
                #cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            #cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
                pass
        # 保存姿态估计后的图片
        cv2.imwrite("./mediaed/{}.jpg".format(l),tempImage)
        #cv2.imshow('Lightweight Human Pose Estimation Python Demo', tempImage)
        #cv2.imwrite('./test/{}.jpg'.format(i),tempImage)
        l +=1
        #for i in 
        #cv2.imwrite('./mediaed/test01.jpg',img)
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoint_iter_370000.pth',required=False, help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    frame_provider = ImageReader(args.images)
    if args.video != '':
        frame_provider = VideoReader(args.video)
    else:
        args.track = 0

    run_demo(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth)
