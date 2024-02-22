import torch
import torch.nn as nn
import numpy as np
from pcdet.ops.ingroup_inds.ingroup_inds_op import ingroup_inds

import logging, datetime
logging.basicConfig(filename='../output/unitr_backbone_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


get_inner_win_inds_cuda = ingroup_inds

# 为每个位置分配一个唯一的嵌入向量
class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, input_channel, num_pos_feats):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Linear(input_channel, num_pos_feats),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Linear(num_pos_feats, num_pos_feats))

    def forward(self, xyz):
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding

# 这个函数范围的是，每个体素的全局索引，和每个体素在每个window内的位置坐标
@torch.no_grad()
def get_window_coors(coors, sparse_shape, window_shape, do_shift, shift_list=None, return_win_coors=False):

    if len(window_shape) == 2:
        win_shape_x, win_shape_y = window_shape
        win_shape_z = sparse_shape[-1]
    else:
        win_shape_x, win_shape_y, win_shape_z = window_shape 
    # window_shape: [[30, 30, 1]]
        
    # sparse_shape_list = [[32, 88, 1]] / [[360, 360, 1]]
    sparse_shape_x, sparse_shape_y, sparse_shape_z = sparse_shape
    assert sparse_shape_z < sparse_shape_x, 'Usually holds... in case of wrong order'

    # 根据sparse shape / patch shape 来确定window的大小
    max_num_win_x = int(np.ceil((sparse_shape_x / win_shape_x)) + 1) # plus one here to meet the needs of shift.
    max_num_win_y = int(np.ceil((sparse_shape_y / win_shape_y)) + 1) # plus one here to meet the needs of shift.
    max_num_win_z = int(np.ceil((sparse_shape_z / win_shape_z)) + 1) # plus one here to meet the needs of shift.
    max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z
    logging.info(f"max_num_win_per_sample: {max_num_win_per_sample}, {max_num_win_x}, {max_num_win_y}, {max_num_win_z}")

    ''''
        patch window 划分后的结果是: window_shape: [[30, 30, 1]], sparse_shape: [32, 88, 1]
        window数目是: x = 3, y = 4, z = 2, 总共24个window

        voxel window划分后的结果是: window_shape: [[30, 30, 1]], sparse_shape: [360, 360, 1]
        window num: 338, 13, 13, 2, 总共338个window
    '''
    # shifts_list: [0, 0, 0], [15, 15, 0]
    if do_shift:    # i == 1
        if shift_list is not None:
            shift_x, shift_y, shift_z = shift_list[0], shift_list[1], shift_list[2]
        else:
            shift_x, shift_y, shift_z = win_shape_x // 2, win_shape_y // 2, win_shape_z // 2 
    else: # i == 0 
        if shift_list is not None:      # shifts_list: [0, 0, 0], [15, 15, 0]
            shift_x, shift_y, shift_z = shift_list[0], shift_list[1], shift_list[2]
        else: 
            shift_x, shift_y, shift_z = win_shape_x, win_shape_y, win_shape_z 
    
    logging.info(f"shift x,y,z:     {shift_x}, {shift_y}, {shift_z}")
    logging.info(f"win shape x,y,z:   {win_shape_x}, {win_shape_y}, {win_shape_z}") # 30, 30, 1

    # compatibility between 2D window and 3D window
    if sparse_shape_z == win_shape_z:
        shift_z = 0

    shifted_coors_x = coors[:, 3] + shift_x
    shifted_coors_y = coors[:, 2] + shift_y
    shifted_coors_z = coors[:, 1] + shift_z
    # logging.info("Get window coords")

    logging.info(f"coords: x:{coors[:31, 3] }\ty:{coors[:31, 2]}\tz:{coors[:31, 1]}\t")
    logging.info(f"shift_coord: x:{shifted_coors_x[:31]}\ty:{shifted_coors_y[:31]}\tz:{shifted_coors_z[:31]}\t")
    logging.info(f"shift_coord: x:{shifted_coors_x.shape}\ty:{shifted_coors_y.shape}\tz:{shifted_coors_z.shape}\t")
    # shift_coord: x:torch.Size([50688])	y:torch.Size([50688])	z:torch.Size([50688])	
    
    win_coors_x = shifted_coors_x // win_shape_x # 整除,向下取整，得到的是原先的coors 要落到多少win 里面去，同时也就是对应的window 
    win_coors_y = shifted_coors_y // win_shape_y
    win_coors_z = shifted_coors_z // win_shape_z
    '''相当于是说，以x为例，0-39这么多，前面计算出来的每个对应的win_coors_x就是0和1，前面0-29个对应的就是win_coors_x = 0的window里面去了，
    同理，后面的30-39就对应到win_coors_x = 1的window里面了
    因而，在每个方向上都计算出来window数量，最后就是总共的window数量了。
    '''
    logging.info(f"win_coors x,y,z: {win_coors_x[:31], {win_coors_y[:31]}, {win_coors_z[:31]}}")
    logging.info(f"win coord shape: {win_coors_x.shape}, {win_coors_y.shape}, {win_coors_z.shape}") 
    # x:torch.Size([50688])	y:torch.Size([50688])	z:torch.Size([50688])	

    
    if len(window_shape) == 2:
        assert (win_coors_z == 0).all()

    batch_win_inds = coors[:, 0] * max_num_win_per_sample + \
                        win_coors_x * max_num_win_y * max_num_win_z + \
                        win_coors_y * max_num_win_z + \
                        win_coors_z
    # win_coors_x * max_num_win_y * max_num_win_z +  计算x坐标的权重 =  win_coors_x * 4 * 2
    # win_coors_y * max_num_win_z +  计算y坐标的权重 = win_coors_y * 2
    # 可以想象成按照这种方式来进行重新分配索引,为的是将这么多的voxel 分配到window里面去的索引
    logging.info(f"batch_win_inds: {batch_win_inds.shape}, {batch_win_inds[:31]}") # torch.Size([50688])
    # max_value, max_index = torch.max(batch_win_inds, dim=0)
    # logging.info(f"最大值:{max_value.item()}" )
    # logging.info(f"最大值的索引: {max_index.item()}")

    ''''
    这里计算得到的batch_win_inds是一个一位的张量，其中每个元素是一个全局窗口索引，使得全局索引能够唯一确定一个窗口：
    将 3D 窗口索引（win_coors_x, win_coors_y, win_coors_z）和 batch 索引（coors[:, 0]）转化为了一个一维的全局窗口索引 batch_win_inds
    '''

    coors_in_win_x = shifted_coors_x % win_shape_x # 取余数 计算x轴上的体素相对于窗口的坐标，得到的是每个体素相对于所属的window的内部坐标啦
    logging.info(f"Window partition------coors_in_win_x shape: {coors_in_win_x.shape}, {coors_in_win_x[:31]}")
    coors_in_win_y = shifted_coors_y % win_shape_y
    logging.info(f"Window partition------coors_in_win_y shape: {coors_in_win_y.shape}, {coors_in_win_y[:31]}")
    coors_in_win_z = shifted_coors_z % win_shape_z
    logging.info(f"Window partition------coors_in_win_z shape: {coors_in_win_z.shape}, {coors_in_win_z[:31]}")
    '''
        29 - 39 % 30 = 0 - 9    ==> window 内的位置，所有的值0-29，不会超过30，整好是每个window里面的坐标。
        29 - 39 // 30 = 0 - 1  ==> window 对应的id
    '''
    coors_in_win = torch.stack([coors_in_win_z, coors_in_win_y, coors_in_win_x], dim=-1)

    # coors_in_win = torch.stack([coors_in_win_x, coors_in_win_y], dim=-1)
    if return_win_coors: # false    
        batch_win_coords = torch.stack([win_coors_z, win_coors_y, win_coors_x], dim=-1)
        return batch_win_inds, coors_in_win, batch_win_coords
    
    logging.info(f"return: --batch_win_inds: {batch_win_inds.shape}, \tcoors_in_win: {coors_in_win.shape}, coors_in_win:{coors_in_win[:31]}") # torch.Size([50688]), 	torch.Size([50688, 3])
    # 这个batch_win_inds 是 Windows indexs of each voxel with shape (N)
    # 这个coors_in_win 是计算体素在每个window内的位置坐标，即，window的坐标系，范围一定是 0-30，0-30， 0-1
    return batch_win_inds, coors_in_win


def get_pooling_index(coors, sparse_shape, window_shape):
    win_shape_x, win_shape_y, win_shape_z = window_shape
    sparse_shape_x, sparse_shape_y, sparse_shape_z = sparse_shape
    
    max_num_win_x = int(np.ceil((sparse_shape_x / win_shape_x)))
    max_num_win_y = int(np.ceil((sparse_shape_y / win_shape_y))) 
    max_num_win_z = int(np.ceil((sparse_shape_z / win_shape_z))) 
    max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z

    coors_x = coors[:, 3] 
    coors_y = coors[:, 2]
    coors_z = coors[:, 1]

    win_coors_x = coors_x // win_shape_x
    win_coors_y = coors_y // win_shape_y
    win_coors_z = coors_z // win_shape_z

    batch_win_inds = coors[:, 0] * max_num_win_per_sample + \
                        win_coors_x * max_num_win_y * max_num_win_z + \
                        win_coors_y * max_num_win_z + \
                        win_coors_z

    coors_in_win_x = coors_x % win_shape_x
    coors_in_win_y = coors_y % win_shape_y
    coors_in_win_z = coors_z % win_shape_z
    coors_in_win = torch.stack([coors_in_win_z, coors_in_win_y, coors_in_win_x], dim=-1)

    index_in_win = coors_in_win_x * win_shape_y * win_shape_z + \
                    coors_in_win_y * win_shape_z + \
                    coors_in_win_z

    batch_win_coords = torch.stack([coors[:, 0], win_coors_z, win_coors_y, win_coors_x], dim=-1)
    return batch_win_inds, coors_in_win, index_in_win, batch_win_coords


def get_continous_inds(setnum_per_win):
    '''
    Args:
        setnum_per_win (Tensor[int]): Number of sets assigned to each window with shape (win_num).
    Returns:
        set_win_inds (Tensor[int]): Window indexs of each set with shape (set_num).
        set_inds_in_win (Tensor[int]): Set indexs inner window with shape (set_num).

    Examples:
        setnum_per_win = torch.tensor([1, 2, 1, 3])
        set_inds_in_win = get_continous_inds(setnum_per_win)
        # we can get: set_inds_in_win = tensor([0, 0, 1, 0, 0, 1, 2])
    '''
    set_num = setnum_per_win.sum().item() # set_num = 7
    setnum_per_win_cumsum = torch.cumsum(setnum_per_win, dim=0)[:-1] # [1, 3, 4]
    set_win_inds = torch.full((set_num,), 0, device=setnum_per_win.device)
    set_win_inds[setnum_per_win_cumsum] = 1 # [0, 1, 0, 1, 1, 0, 0]
    set_win_inds = torch.cumsum(set_win_inds, dim=0) # [0, 1, 1, 2, 3, 3, 3]
    
    roll_set_win_inds_left = torch.roll(set_win_inds, -1)  # [1, 1, 2, 3, 3, 3, 0]
    diff = set_win_inds - roll_set_win_inds_left # [-1, 0, -1, -1, 0, 0, 3]
    end_pos_mask = diff != 0
    template = torch.ones_like(set_win_inds)
    template[end_pos_mask] = (setnum_per_win - 1) * -1  # [ 0, 1, -1, 0, 1, 1, -2]
    set_inds_in_win = torch.cumsum(template,dim=0) # [0, 1, 0, 0, 1, 2, 0]
    set_inds_in_win[end_pos_mask] = setnum_per_win # [1, 1, 2, 1, 1, 2, 3]
    set_inds_in_win = set_inds_in_win - 1 # [0, 0, 1, 0, 0, 1, 2]
    
    return set_win_inds, set_inds_in_win