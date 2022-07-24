import copy
import cv2
import numpy as np
import image_processing
 
class JointsDataset():
    def __init__(self, num_joints,
                 image_size=[192, 256],
                 heatmap_size=[48, 64],
                 transform=None,
                 target_type='gaussian'):
        """
        :param num_joints:
        :param image_size:
        :param heatmap_size:
        :param transform:
        :param target_type:
        """
        self.num_joints = num_joints
        self.image_size = np.asarray(image_size)
        self.heatmap_size = np.asarray(heatmap_size)
        self.target_type = target_type
        self.sigma = 0.5
        self.transform = transform
 
    def generate_target(self, joints, joints_vis):
        """
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target(num_joints,heatmap_size[0],heatmap_size[1]), is target_heatmap
                 target_weight(1: visible, 0: invisible)
        """
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]
        assert self.target_type == 'gaussian', 'Only support gaussian map now!'
        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)
            tmp_size = self.sigma * 3
            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue
                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))
 
                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])
 
                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
 
        else:
            raise Exception("Error:{}".format(self.target_type))
        return target, target_weight
 
 
def create_joints_3d(num_joints, joint):
    joints_3d = np.zeros((num_joints, 3), dtype=np.float)
    joints_3d_vis = np.zeros((num_joints, 3), dtype=np.float)
    for ipt in range(num_joints):
        joints_3d[ipt, 0] = joint[ipt, 0]
        joints_3d[ipt, 1] = joint[ipt, 1]
        joints_3d[ipt, 2] = 0
        if joint[ipt, 0] > 0 and joint[ipt, 1] > 0:
            t_vis = 1
        else:
            t_vis = 0
        joints_3d_vis[ipt, 0] = t_vis
        joints_3d_vis[ipt, 1] = t_vis
        joints_3d_vis[ipt, 2] = 0
    return joints_3d, joints_3d_vis
 
 
def vis_joint_in_images(image, joints_3d, joints_3d_vis, skeleton):
    joints = joints_3d[:, 0:2]
    # c = np.mean(joints,axis=0)
    image = image_processing.draw_key_point_in_image(image, [joints], skeleton)
    image_processing.cv_show_image("dst_image", image)
 
 
def vis_heatmap_in_image(heatmap, image, waitKey=0):
    h, w, d = image.shape
    resize_heatmap = []
    for map in heatmap:
        map = image_processing.resize_image(map, resize_height=h, resize_width=w)
        resize_heatmap.append(map)
 
    heatmap = np.asarray(resize_heatmap)
    heatmap = np.sum(heatmap, axis=0)
    heatmap = np.clip(heatmap, 0, 1)
    image_processing.cv_show_image("heatmap", heatmap, waitKey=30)
    image = np.asarray(image / 255.0, dtype=np.float32)
    image[:, :, 0] = image[:, :, 0] + heatmap
    image_processing.cv_show_image("heatmap_in_image", image, waitKey=waitKey)
 
 
if __name__ == "__main__":
    
 
    skeleton = [[0, 1], [1, 2], [3, 4], [4, 5], [2, 6], [6, 3], [12, 11], [7, 12],
                [11, 10], [13, 14], [14, 15], [8, 9], [8, 7], [6, 7], [7, 13]]
    
    joint = np.asarray([[357.6537, 728.02435],
                        [357.6537, 561.3204],
                        [363.4021, 388.86795],
                        [455.3767, 388.86795],
                        [461.12512, 567.0688],
                        [461.12512, 733.77277],
                        [409.3894, 388.86795],
                        [403.641, 170.42822],
                        [403.641, 118.69249],
                        [406.5152, 29.592072],
                        [248.4338, 348.62906],
                        [248.4338, 256.65442],
                        [323.1632, 170.42822],
                        [489.86722, 164.6798],
                        [553.09973, 262.40283],
                        [547.3513, 348.62906]])

    num_joints = len(joint)
    image_path = "000000005134.jpg"
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    image_size = np.asarray([400, 500])
    scale = image_size / [w, h]
    joint = joint * scale
    image = image_processing.resize_image(image, resize_width=image_size[0], resize_height=image_size[1])
    joints_3d, joints_3d_vis = create_joints_3d(num_joints, joint)
    joints_obj = JointsDataset(num_joints=num_joints, image_size=image_size, heatmap_size=[48, 64])
    target_heatmap, target_weight = joints_obj.generate_target(joints=joints_3d, joints_vis=joints_3d_vis)
    vis_heatmap_in_image(target_heatmap, image, waitKey=0)
    vis_joint_in_images(image, joints_3d, joints_3d_vis, skeleton)