import math
import numpy as np

from constants_vuer import grd_yup2grd_zup, hand2inspire
from motion_utils import mat_update, fast_mat_inv


class VuerPreprocessor:
    def __init__(self):
        # 初始化头部矩阵    
        self.vuer_head_mat = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 1.5],
                                  [0, 0, 1, -0.2],
                                  [0, 0, 0, 1]])
        # 初始化右手腕矩阵
        self.vuer_right_hand_mat = np.array([[1, 0, 0, 0.5],
                                         [0, 1, 0, 1],
                                         [0, 0, 1, -0.5],
                                         [0, 0, 0, 1]])
        # 初始化左手腕矩阵
        self.vuer_left_hand_mat = np.array([[1, 0, 0, -0.5],       
                                        [0, 1, 0, 1],
                                        [0, 0, 1, -0.5],
                                        [0, 0, 0, 1]])

    def process(self, tv):
        # 更新头部、右手腕和左手腕矩阵
        # 这里的mat_update函数，如果传入的矩阵是奇异矩阵，则返回先前的矩阵
        self.vuer_head_mat = mat_update(self.vuer_head_mat, tv.head_matrix.copy())
        self.vuer_right_hand_mat = mat_update(self.vuer_right_hand_mat, tv.right_hand.copy())
        self.vuer_left_hand_mat = mat_update(self.vuer_left_hand_mat, tv.left_hand.copy())

        # 坐标系转换
        # @ 符号在Python中用于矩阵乘法，相当于np.matmul()函数
        # 将头部矩阵从y轴向上坐标系转换到z轴向上坐标系
        head_mat = grd_yup2grd_zup @ self.vuer_head_mat @ fast_mat_inv(grd_yup2grd_zup)
        
        # 将右手腕矩阵从y轴向上坐标系转换到z轴向上坐标系
        right_hand_mat = grd_yup2grd_zup @ self.vuer_right_hand_mat @ fast_mat_inv(grd_yup2grd_zup)
        
        # 将左手腕矩阵从y轴向上坐标系转换到z轴向上坐标系
        left_hand_mat = grd_yup2grd_zup @ self.vuer_left_hand_mat @ fast_mat_inv(grd_yup2grd_zup)
        
        # grd_yup2grd_zup 是一个转换矩阵，用于将y轴向上的坐标系转换为z轴向上的坐标系
        # fast_mat_inv() 函数用于计算矩阵的逆，右乘fast_mat_inv(grd_yup2grd_zup):这一步看起来有点奇怪,但它的作用是保持矩阵中的旋转部分不变,只转换平移部分。
        # 这样做的原因是,头部的方向(旋转)在两个坐标系中应该保持一致,我们只需要转换它的位置(平移)。

        # 计算相对于头部的左手腕位置
        rel_left_wrist_mat = left_hand_mat @ hand2inspire
        rel_left_wrist_mat[0:3, 3] = rel_left_wrist_mat[0:3, 3] - head_mat[0:3, 3]

        # 计算相对于头部的右手腕位置
        rel_right_wrist_mat = right_hand_mat @ hand2inspire  # wTr = wTh @ hTr
        rel_right_wrist_mat[0:3, 3] = rel_right_wrist_mat[0:3, 3] - head_mat[0:3, 3]

        # 将手指坐标转换为齐次坐标
        #tv.left_landmarks =（N，3） 
        #left_fingers = (4,N)
        #因为后续变换都是用的齐次矩阵的形式，所以要先把N个关键点（xyz）变为齐次形式
        left_fingers = np.concatenate([tv.left_landmarks.copy().T, np.ones((1, tv.left_landmarks.shape[0]))])
        right_fingers = np.concatenate([tv.right_landmarks.copy().T, np.ones((1, tv.right_landmarks.shape[0]))])

        # 坐标系转换
        left_fingers = grd_yup2grd_zup @ left_fingers
        right_fingers = grd_yup2grd_zup @ right_fingers

        # 计算手指关键点在hand坐标系下的位置
        rel_left_fingers = fast_mat_inv(left_hand_mat) @ left_fingers
        rel_right_fingers = fast_mat_inv(right_hand_mat) @ right_fingers
        #hand2inspire 是手坐标系到inspire坐标系的转换矩阵，其转置为inspire坐标系到手坐标系的转换矩阵
        # 计算手指关键点在inspire坐标系下的位置，转置后最终得到的矩阵为（N，3）
        rel_left_fingers = (hand2inspire.T @ rel_left_fingers)[0:3, :].T
        rel_right_fingers = (hand2inspire.T @ rel_right_fingers)[0:3, :].T
        #head在world坐标系下的位置，wrist
        return head_mat, rel_left_wrist_mat, rel_right_wrist_mat, rel_left_fingers, rel_right_fingers

    def get_hand_gesture(self, tv):
        # 更新右手腕和左手腕矩阵
        self.vuer_right_hand_mat = mat_update(self.vuer_right_hand_mat, tv.right_hand.copy())
        self.vuer_left_hand_mat = mat_update(self.vuer_left_hand_mat, tv.left_hand.copy())

        # 坐标系转换
        right_wrist_mat = grd_yup2grd_zup @ self.vuer_right_hand_mat @ fast_mat_inv(grd_yup2grd_zup)
        left_wrist_mat = grd_yup2grd_zup @ self.vuer_left_hand_mat @ fast_mat_inv(grd_yup2grd_zup)

        # 将手指坐标转换为齐次坐标
        left_fingers = np.concatenate([tv.left_landmarks.copy().T, np.ones((1, tv.left_landmarks.shape[0]))])
        right_fingers = np.concatenate([tv.right_landmarks.copy().T, np.ones((1, tv.right_landmarks.shape[0]))])

        # 坐标系转换
        left_fingers = grd_yup2grd_zup @ left_fingers
        right_fingers = grd_yup2grd_zup @ right_fingers

        # 计算相对于手腕的手指位置
        rel_left_fingers = fast_mat_inv(left_wrist_mat) @ left_fingers
        rel_right_fingers = fast_mat_inv(right_wrist_mat) @ right_fingers
        rel_left_fingers = (hand2inspire.T @ rel_left_fingers)[0:3, :].T
        rel_right_fingers = (hand2inspire.T @ rel_right_fingers)[0:3, :].T
        all_fingers = np.concatenate([rel_left_fingers, rel_right_fingers], axis=0)

        return all_fingers

