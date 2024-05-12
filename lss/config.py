# -----------------------------------------config start-----------------------------------------
version = 'v1.0-mini'
dataroot = '/home/zed/data/nuscenes'  # 数据集路径
nepochs = 10000  # 训练最大的epoch数
gpuid = 0  # gpu的序号

H = 900
W = 1600  # 图片大小
resize_lim = (0.193, 0.225)  # resize的范围
final_dim = (128, 352)  # 数据预处理之后最终的图片大小
bot_pct_lim = (0.0, 0.22)  # 裁剪图片时，图像底部裁剪掉部分所占比例范围
rot_lim = (-5.4, 5.4)  # 训练时旋转图片的角度范围
rand_flip = True  # # 是否随机翻转
ncams = 5  # 训练时选择的相机通道数
max_grad_norm = 5.0
pos_weight = 2.13  # 损失函数中给正样本项损失乘的权重系数
logdir = './runs'  # 日志的输出文件

xbound = [-50.0, 50.0, 0.5]  # 限制x方向的范围并划分网格
ybound = [-50.0, 50.0, 0.5]  # 限制y方向的范围并划分网格
zbound = [-10.0, 10.0, 20.0]  # 限制z方向的范围并划分网格
dbound = [4.0, 45.0, 1.0]  # 限制深度方向的范围并划分网格

bsz = 2  # batchsize
nworkers = 0  # 线程数
lr = 1e-3  # 学习率
weight_decay = 1e-7  # 权重衰减系数

grid_conf = {  # 网格配置
    'xbound': xbound,
    'ybound': ybound,
    'zbound': zbound,
    'dbound': dbound,
}
data_aug_conf = {  # 数据增强配置
    'resize_lim': resize_lim,
    'final_dim': final_dim,
    'rot_lim': rot_lim,
    'H': H, 'W': W,
    'rand_flip': rand_flip,
    'bot_pct_lim': bot_pct_lim,
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': ncams,
}

# -----------------------------------------config end-----------------------------------------