import torch
import torch.nn as nn
from torch_scatter import scatter_mean

import logging
import datetime

logging.basicConfig(filename='../output/unitr_MAP2BEV_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.setLevel(logging.INFO)
logging.info('This is an info message.')
class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict

class PointPillarScatter3d(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        
        self.model_cfg = model_cfg
        self.nx, self.ny, self.nz = self.model_cfg.INPUT_SHAPE # INPUT_SHAPE: [360, 360, 1]
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES # NUM_BEV_FEATURES: 128
        self.num_bev_features_before_compression = self.model_cfg.NUM_BEV_FEATURES // self.nz # 128

    def forward(self, batch_dict, **kwargs):
        logging.info("----------------------- MAP to BEV ----------------------")
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        
        batch_spatial_features = [] # 用来存储每个batch的空间特征
        batch_size = coords[:, 0].max().int().item() + 1
        logging.info(f"batch_size: {batch_size}")
        for batch_idx in range(batch_size):
            
            spatial_feature = torch.zeros(
                self.num_bev_features_before_compression, 
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            # spatial_feature shape是， 128 x (360x360x1)

            batch_mask = coords[:, 0] == batch_idx # batch_mask 用于选择与当前批次索引相对应的坐标
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] * self.ny * self.nx \
                    + this_coords[:, 2] * self.nx \
                    + this_coords[:, 3]
            '''类似这种计算索引：
                batch_win_inds = coors[:, 0] * max_num_win_per_sample + \
                        win_coors_x * max_num_win_y * max_num_win_z + \
                        win_coors_y * max_num_win_z + \
                        win_coors_z
                # win_coors_x * max_num_win_y * max_num_win_z +  计算x坐标的权重
                # win_coors_y * max_num_win_z +  计算y坐标的权重
                # 可以想象成按照这种方式来进行重新分配索引,为的是将这么多的voxel 分配到window里面去的索引
            '''
            indices = indices.type(torch.long)

            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t() # 转置

            dense_feats = scatter_mean(pillars,indices,dim=1) # 计算每个特征索引对应的平均特征值
            dense_len = dense_feats.shape[1]
            spatial_feature[:,:dense_len] = dense_feats

            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features_before_compression * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict
