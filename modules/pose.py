import cv2
from cv2 import KeyPoint
import numpy as np
import math
from modules.keypoints import BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS
from modules.one_euro_filter import OneEuroFilter
from sanjiaohanshu import sin

class Pose:
    num_kpts = 18
    kpt_names = ['nose', 'neck',
                 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                 'r_eye', 'l_eye',
                 'r_ear', 'l_ear']
    sigmas = np.array([.26, .79, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35],
                      dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [0, 224, 255]

    def __init__(self, keypoints, confidence):
        super().__init__()
        self.keypoints = keypoints

        # self.keypints和keypoints即输出的人体关键点
        
        print('self.keypoints等于：')
        print(self.keypoints)
        print('self.keypints的len等于：')
        print(len(self.keypoints))
                

        self.confidence = confidence

        #print(confidence)
        # 将self.keypoints传入get_bbox()函数中
        self.bbox = Pose.get_bbox(self.keypoints)
        
        self.id = None
        self.filters = [[OneEuroFilter(), OneEuroFilter()] for _ in range(Pose.num_kpts)]

    @staticmethod
    # 画人体矩形框
    def get_bbox(keypoints):
        # 对一个人所有关节点的x点信息进行处理
        found_keypoints = np.zeros((np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32)
        print(found_keypoints)       
        found_kpt_id = 0
        # 18次
        for kpt_id in range(Pose.num_kpts):
            if keypoints[kpt_id, 0] == -1:
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id]
            found_kpt_id += 1
        # 画矩形框    
        bbox = cv2.boundingRect(found_keypoints)
        return bbox
        

    def update_id(self, id=None):
        self.id = id
        if self.id is None:
            self.id = Pose.last_id + 1
            Pose.last_id += 1

    def draw(self, img):
        assert self.keypoints.shape == (Pose.num_kpts, 2)
        # 输出BODY_PARTS_PAF_IDS的长度=19，为啥是19：待解决
        # print(len(BODY_PARTS_PAF_IDS))
        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):      # 将躯干某个连接的单位向量映射到paf对应的通道
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]           # 当前躯干起点的id
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]        # 当前关节点在所有关节点中的索引

            # 分配x_a,y_a
            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]     # 当前关节点在原图像上的坐标
                cv2.circle(img, (int(x_a), int(y_a)), 3, Pose.color, -1)     # 原图画圆

            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]   # 当前躯干终点的id
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]   # 当前关节点在所有关节点中的索引
            
            if global_kpt_b_id != -1:   # 分配了当前关节点
                x_b, y_b = self.keypoints[kpt_b_id]     # 当前关节点在原图像上的坐标
                cv2.circle(img, (int(x_b), int(y_b)), 3, Pose.color, -1)    # 原图画圆
           
            # 连接各关键点
            if global_kpt_a_id != -1 and global_kpt_b_id != -1:     #起点和终点均分配
                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), Pose.color, 3)    # 画连接起点和终点的直线
                """
                if (sin(self.keypoints[9],self.keypoints[10])> float(0) and sin(self.keypoints[9],self.keypoints[10]) < float(0.5)) and (sin(self.keypoints[8],self.keypoints[9])> float(0) and sin(self.keypoints[8],self.keypoints[9]) < float(0.5)):
                    print("此时为站立状态")
                    cv2.putText(img,text='standing',org=(0,200),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=3,color=(255,0,0),thickness=5)
                if  (sin(self.keypoints[8],self.keypoints[9])> float(0.5) and sin(self.keypoints[8],self.keypoints[9]) < float(1)):
                    print("此时为蹲下状态状态")
                    cv2.putText(img,text='squating',org=(0,200),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=3,color=(255,0,0),thickness=5)
                if ((self.keypoints[4][1]-self.keypoints[3][1]) < float(0) and (self.keypoints[3][1]-self.keypoints[2][1]<float(0))):
                    print("抬起了右手")
                    cv2.putText(img,text='lift right hand',org=(0,300),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=3,color=(255,0,0),thickness=5)
                if ((self.keypoints[7][1]-self.keypoints[6][1]) < float(0) and (self.keypoints[6][1]-self.keypoints[5][1]<float(0))):
                    print("抬起了左手")
                    cv2.putText(img,text='lift left hand',org=(0,400),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=3,color=(255,0,0),thickness=5)
                """
                # 输出各连接关节点坐标
                # print((int(x_a),int(y_a)),(int(x_b),int(y_b)))
    
            # print('第{}个人的关键点信息：'.format(part_id))
            # print((int(x_a),int(y_a)),(int(x_b),int(y_b)))
        
            
        
                

def get_similarity(a, b, threshold=0.5):
    num_similar_kpt = 0
    for kpt_id in range(Pose.num_kpts):
        if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1:
            distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)
            area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
            similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id]))
            if similarity > threshold:
                num_similar_kpt += 1
    return num_similar_kpt


def track_poses(previous_poses, current_poses, threshold=3, smooth=False):
    """Propagate poses ids from previous frame results. Id is propagated,
    if there are at least `threshold` similar keypoints between pose from previous frame and current.
    If correspondence between pose on previous and current frame was established, pose keypoints are smoothed.

    :param previous_poses: poses from previous frame with ids
    :param current_poses: poses from current frame to assign ids
    :param threshold: minimal number of similar keypoints between poses
    :param smooth: smooth pose keypoints between frames
    :return: None
    """
    current_poses = sorted(current_poses, key=lambda pose: pose.confidence, reverse=True)  # match confident poses first
    mask = np.ones(len(previous_poses), dtype=np.int32)
    for current_pose in current_poses:
        best_matched_id = None
        best_matched_pose_id = None
        best_matched_iou = 0
        for id, previous_pose in enumerate(previous_poses):
            if not mask[id]:
                continue
            iou = get_similarity(current_pose, previous_pose)
            if iou > best_matched_iou:
                best_matched_iou = iou
                best_matched_pose_id = previous_pose.id
                best_matched_id = id
        if best_matched_iou >= threshold:
            mask[best_matched_id] = 0
        else:  # pose not similar to any previous
            best_matched_pose_id = None
        current_pose.update_id(best_matched_pose_id)

        if smooth:
            for kpt_id in range(Pose.num_kpts):
                if current_pose.keypoints[kpt_id, 0] == -1:
                    continue
                # reuse filter if previous pose has valid filter
                if (best_matched_pose_id is not None
                        and previous_poses[best_matched_id].keypoints[kpt_id, 0] != -1):
                    current_pose.filters[kpt_id] = previous_poses[best_matched_id].filters[kpt_id]
                current_pose.keypoints[kpt_id, 0] = current_pose.filters[kpt_id][0](current_pose.keypoints[kpt_id, 0])
                current_pose.keypoints[kpt_id, 1] = current_pose.filters[kpt_id][1](current_pose.keypoints[kpt_id, 1])
            current_pose.bbox = Pose.get_bbox(current_pose.keypoints)
