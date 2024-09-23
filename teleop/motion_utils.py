import numpy as np
import logging

def mat_update(prev_mat, mat, threshold=1e-6):
    """
    更新矩阵函数。

    参数:
    prev_mat (numpy.ndarray): 先前的矩阵
    mat (numpy.ndarray): 新的矩阵
    threshold (float): 判断矩阵是否奇异的阈值

    返回:
    numpy.ndarray: 如果新矩阵有效则返回新矩阵，否则返回先前的矩阵

    异常:
    ValueError: 如果输入不是numpy数组或者形状不正确
    """
    # 检查输入参数的有效性
    if not isinstance(prev_mat, np.ndarray) or not isinstance(mat, np.ndarray):
        raise ValueError("输入必须是numpy数组")
    
    if prev_mat.shape != mat.shape:
        raise ValueError("两个输入矩阵的形状必须相同")
    
    if len(mat.shape) != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("输入必须是方阵")

    # 使用阈值来判断行列式是否接近于0
    if np.abs(np.linalg.det(mat)) < threshold:
        logging.warning("警告：新矩阵接近奇异，使用先前的矩阵。")
        return prev_mat
    else:
        logging.info("矩阵更新成功。")
        return mat


def fast_mat_inv(mat):
    """
    快速计算4x4矩阵的逆矩阵。
    
    这个函数假设输入的矩阵是一个4x4的变换矩阵，通常用于3D空间中的旋转和平移。
    它利用了变换矩阵的特殊结构来快速计算逆矩阵，比通用的矩阵求逆方法更高效。
    
    参数:
    mat (numpy.ndarray): 输入的4x4变换矩阵
    
    返回:
    numpy.ndarray: 输入矩阵的逆矩阵
    
    异常:
    ValueError: 如果输入矩阵不是4x4的或者不是有效的变换矩阵
    
    计算步骤:
    1. 检查输入矩阵的有效性。
    2. 创建一个4x4的单位矩阵作为结果矩阵的基础。
    3. 提取输入矩阵的3x3旋转部分，并将其转置作为结果矩阵的旋转部分。
    4. 计算平移向量的逆，方法是将旋转矩阵的转置与原平移向量的负值相乘。
    5. 返回计算得到的逆矩阵。
    """
    try:
        if not isinstance(mat, np.ndarray) or mat.shape != (4, 4):
            raise ValueError("输入必须是4x4的numpy数组")
        
        # 检查是否是有效的变换矩阵
        if not np.allclose(mat[3, :], [0, 0, 0, 1]):
            raise ValueError("输入不是有效的变换矩阵")
        
        # 检查旋转矩阵部分是否正交
        rotation = mat[:3, :3]
        if not np.allclose(np.dot(rotation, rotation.T), np.eye(3), atol=1e-6):
            raise ValueError("旋转矩阵部分不是正交矩阵")
        
        ret = np.eye(4)  # 创建4x4单位矩阵
        ret[:3, :3] = rotation.T  # 旋转矩阵的逆等于其转置
        ret[:3, 3] = -rotation.T @ mat[:3, 3]  # 计算平移向量的逆
        return ret
    except Exception as e:
        logging.error(f"矩阵求逆失败：{str(e)}")
        raise
