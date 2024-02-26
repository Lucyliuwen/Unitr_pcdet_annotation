import copy
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from pcdet.models.model_utils.swin_utils import PatchEmbed
from pcdet.models.model_utils.unitr_utils import MapImage2Lidar, MapLidar2Image
from pcdet.models.model_utils.dsvt_utils import PositionEmbeddingLearned
from pcdet.models.backbones_3d.dsvt import _get_activation_fn, DSVTInputLayer
from pcdet.ops.ingroup_inds.ingroup_inds_op import ingroup_inds
get_inner_win_inds_cuda = ingroup_inds

import logging
import datetime

logging.basicConfig(filename='../output/unitr_backbone_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.setLevel(logging.INFO)
logging.info('This is an info message.')
logging.info('This is an info message.')

class UniTR(nn.Module):
    '''
    UniTR: A Unified and Efficient Multi-Modal Transformer for Bird's-Eye-View Representation.
    Main args:
        set_info (list[list[int, int]]): A list of set config for each stage. Eelement i contains 
            [set_size, block_num], where set_size is the number of voxel in a set and block_num is the
            number of blocks for stage i. Length: stage_num.
        d_model (list[int]): the number of filters in first linear layer of each transformer encoder
        nhead (list[int]): Number of attention heads for each stage. Length: stage_num.
        dim_feedforward list([int]): the number of filters in first linear layer of each transformer encoder
        dropout (float): Drop rate of set attention. 
        activation (string): Name of activation layer in set attention.
        checkpoint_blocks: block IDs (0 to num_blocks - 1) to use checkpoint.
        Note: In PyTorch 1.8, checkpoint function seems not able to receive dict as parameters. Better to use PyTorch >= 1.9.
        accelerate (bool): whether accelerate forward by caching image pos embed, image2lidar coords and lidar2image coords.
    '''

    def __init__(self, model_cfg, use_map=False, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.set_info = set_info = self.model_cfg.set_info
        self.d_model = d_model = self.model_cfg.d_model
        self.nhead = nhead = self.model_cfg.nhead
        self.stage_num = stage_num = 1  # only support plain bakbone
        self.num_shifts = [2] * self.stage_num #       [2]
        self.checkpoint_blocks = self.model_cfg.checkpoint_blocks
        self.image_pos_num, self.lidar_pos_num = set_info[0][-1], set_info[0][-1]
        self.accelerate = self.model_cfg.get('ACCELERATE', False)
        self.use_map = use_map

        self.image_input_layer = UniTRInputLayer(
            self.model_cfg.IMAGE_INPUT_LAYER, self.accelerate)
        self.lidar_input_layer = UniTRInputLayer(
            self.model_cfg.LIDAR_INPUT_LAYER)

        # image patch embedding
        patch_embed_cfg = self.model_cfg.PATCH_EMBED
        self.patch_embed = PatchEmbed(
            in_channels=patch_embed_cfg.in_channels,
            embed_dims=patch_embed_cfg.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_embed_cfg.patch_size,
            stride=patch_embed_cfg.patch_size,
            norm_cfg=patch_embed_cfg.norm_cfg if patch_embed_cfg.patch_norm else None
        )
        patch_size = [patch_embed_cfg.image_size[0] // patch_embed_cfg.patch_size,
                      patch_embed_cfg.image_size[1] // patch_embed_cfg.patch_size]
        self.patch_size = patch_size # 用于后面将不同的patch恢复到image上来
        patch_x, patch_y = torch.meshgrid(torch.arange(
            patch_size[0]), torch.arange(patch_size[1]))
        # 生成两个矩阵，# 0 到 patch_size[0]-1，对应patch坐标x; 0 到 patch_size[1]-1,对应patch坐标y


        patch_z = torch.zeros((patch_size[0] * patch_size[1], 1))
        self.patch_zyx = torch.cat(
            [patch_z, patch_y.reshape(-1, 1), patch_x.reshape(-1, 1)], dim=-1).cuda()
        # patch coords with batch id
        self.patch_coords = None

        # image branch output norm
        self.out_indices = self.model_cfg.out_indices
        for i in self.out_indices:
            layer = nn.LayerNorm(d_model[-1])
            layer_name = f'out_norm{i}'
            self.add_module(layer_name, layer)

        # Sparse Regional Attention Blocks
        dim_feedforward = self.model_cfg.dim_feedforward
        dropout = self.model_cfg.dropout
        activation = self.model_cfg.activation
        layer_cfg = self.model_cfg.layer_cfg
        block_id = 0
        for stage_id in range(stage_num):
            num_blocks_this_stage = set_info[stage_id][-1]
            dmodel_this_stage = d_model[stage_id]
            dfeed_this_stage = dim_feedforward[stage_id]
            num_head_this_stage = nhead[stage_id]
            block_list, norm_list = [], []
            for i in range(num_blocks_this_stage):
                block_list.append(
                    UniTRBlock(dmodel_this_stage, num_head_this_stage, dfeed_this_stage,
                               dropout, activation, batch_first=True, block_id=block_id,
                               dout=dmodel_this_stage, layer_cfg=layer_cfg)
                )
                norm_list.append(nn.LayerNorm(dmodel_this_stage))
                block_id += 1
            self.__setattr__(f'stage_{stage_id}', nn.ModuleList(block_list))
            self.__setattr__(
                f'residual_norm_stage_{stage_id}', nn.ModuleList(norm_list)) # 用于image norm
            if layer_cfg.get('split_residual', False):
                # use different norm for lidar and image
                lidar_norm_list = [nn.LayerNorm(
                    dmodel_this_stage) for _ in range(num_blocks_this_stage)]
                self.__setattr__(
                    f'lidar_residual_norm_stage_{stage_id}', nn.ModuleList(lidar_norm_list)) # 用于lidar norm

        # Fuse Backbone
        fuse_cfg = self.model_cfg.get('FUSE_BACKBONE', None)
        self.fuse_on = fuse_cfg is not None
        if self.fuse_on:
            # image2lidar
            image2lidar_cfg = fuse_cfg.get('IMAGE2LIDAR', None)
            self.image2lidar_on = image2lidar_cfg is not None
            if self.image2lidar_on:
                # block range of image2lidar
                self.image2lidar_start = image2lidar_cfg.block_start
                self.image2lidar_end = image2lidar_cfg.block_end
                self.map_image2lidar_layer = MapImage2Lidar(
                    image2lidar_cfg, self.accelerate, self.use_map)
                self.image2lidar_input_layer = UniTRInputLayer(
                    image2lidar_cfg.image2lidar_layer)
                self.image2lidar_pos_num = image2lidar_cfg.image2lidar_layer.set_info[0][1]
                # encode the position of each patch from the closest point in image space
                self.neighbor_pos_embed = PositionEmbeddingLearned(
                    2, self.d_model[-1])

            # lidar2image
            lidar2image_cfg = fuse_cfg.get('LIDAR2IMAGE', None)
            self.lidar2image_on = lidar2image_cfg is not None
            if self.lidar2image_on:
                # block range of lidar2image
                self.lidar2image_start = lidar2image_cfg.block_start
                self.lidar2image_end = lidar2image_cfg.block_end
                self.map_lidar2image_layer = MapLidar2Image(
                    lidar2image_cfg, self.accelerate, self.use_map)
                self.lidar2image_input_layer = UniTRInputLayer(
                    lidar2image_cfg.lidar2image_layer)
                self.lidar2image_pos_num = lidar2image_cfg.lidar2image_layer.set_info[0][1]

        self._reset_parameters()

    def forward(self, batch_dict):
        '''
        Args:
            bacth_dict (dict): 
                The dict contains the following keys
                - voxel_features (Tensor[float]): Voxel features after VFE. Shape of (N, d_model[0]), 
                    where N is the number of input voxels.
                - voxel_coords (Tensor[int]): Shape of (N, 4), corresponding voxel coordinates of each voxels.
                    Each row is (batch_id, z, y, x). 
                - camera_imgs (Tensor[float]): multi view images, shape of (B, N, C, H, W),
                    where N is the number of image views.
                - ...
        
        Returns:
            bacth_dict (dict):
                The dict contains the following keys
                - pillar_features (Tensor[float]):
                - voxel_coords (Tensor[int]):
                - image_features (Tensor[float]):
        '''
        logging.info("-------------- mm_backbone -------------")
        # lidar(3d) and image(2d) preprocess
        logging.info('*'*30+"Entering in the lidar(3d) and image(2d) preprocess part"+'*'*30)
        multi_feat, voxel_info, patch_info, multi_set_voxel_inds_list, multi_set_voxel_masks_list, multi_pos_embed_list = self._input_preprocess(
            batch_dict)

        logging.info('*'*30 + "Entering in lidar(3d) and image(3d) preprocess part"+'*'*30)
        # lidar(3d) and image(3d) preprocess
        if self.image2lidar_on:
            image2lidar_inds_list, image2lidar_masks_list, multi_pos_embed_list = self._image2lidar_preprocess(
                batch_dict, multi_feat, multi_pos_embed_list)
        
        logging.info('*'*30+"Entering in the lidar(2d) and image(2d) preprocess part"+'*'*30)
        # lidar(2d) and image(2d) preprocess
        if self.lidar2image_on:
            lidar2image_inds_list, lidar2image_masks_list, multi_pos_embed_list = self._lidar2image_preprocess(
                batch_dict, multi_feat, multi_pos_embed_list)
        
        output = multi_feat
        block_id = 0
        voxel_num = batch_dict['voxel_num']
        logging.info(f"batch_dict: {batch_dict.keys()}")
        logging.info(f"voxel_num: {voxel_num}")

        batch_dict['image_features'] = [] # 又新增了一个
        logging.info(f"batch_dict BEFORE BACKBONE: {batch_dict.keys()}")
 
        # block forward
        for stage_id in range(self.stage_num): # 1
            logging.info(f'===================== Stage{stage_id} ==============')
            block_layers = self.__getattr__(f'stage_{stage_id}')
            residual_norm_layers = self.__getattr__(f'residual_norm_stage_{stage_id}')
            for i in range(len(block_layers)):
                block = block_layers[i]
                residual = output.clone()
                if self.image2lidar_on and i >= self.image2lidar_start and i < self.image2lidar_end:
                    logging.info("process image 2 lidar" + "2 " *20)
                    output = block(output, image2lidar_inds_list[stage_id], image2lidar_masks_list[stage_id], multi_pos_embed_list[stage_id][i],
                                   block_id=block_id, voxel_num=voxel_num, using_checkpoint=block_id in self.checkpoint_blocks)
                    logging.info('output shape of image2lidar block: %s', output.shape)
                elif self.lidar2image_on and i >= self.lidar2image_start and i < self.lidar2image_end:
                    logging.info("process lidar 2 image" + "1 " * 20)
                    output = block(output, lidar2image_inds_list[stage_id], lidar2image_masks_list[stage_id], multi_pos_embed_list[stage_id][i],
                                   block_id=block_id, voxel_num=voxel_num, using_checkpoint=block_id in self.checkpoint_blocks)
                    logging.info('output shape of lidar2image block: %s', output.shape)
                else:
                    logging.info("process lidar and image origin" + "0 " * 20)
                    output = block(output, multi_set_voxel_inds_list[stage_id], multi_set_voxel_masks_list[stage_id], multi_pos_embed_list[stage_id][i],
                                   block_id=block_id, voxel_num=voxel_num, using_checkpoint=block_id in self.checkpoint_blocks)
                    logging.info('output shape of origin: %s', output.shape) # [92773, 128]
                
                logging.info("process normalization " + "+"*10)
                # use different norm for lidar and image
                if self.model_cfg.layer_cfg.get('split_residual', False):
                    output = torch.cat([self.__getattr__(f'lidar_residual_norm_stage_{stage_id}')[i](output[:voxel_num] + residual[:voxel_num]),
                                       residual_norm_layers[i](output[voxel_num:] + residual[voxel_num:])], dim=0)
                else:
                    output = residual_norm_layers[i](output + residual)
                
                logging.info('output shape with processed: %s', output.shape) # [92773, 128])

                block_id += 1
                # recover image feature shape
                if i in self.out_indices: # false
                    batch_spatial_features = self._recover_image(pillar_features=output[voxel_num:],
                                                                 coords=patch_info[f'voxel_coors_stage{self.stage_num - 1}'], indices=i)
                    batch_dict['image_features'].append(batch_spatial_features)
                
            logging.info('*'*30+"The UniTR block end"+"*"*30)

        logging.info(f'===================== Stage END ==============')
        batch_dict['pillar_features'] = batch_dict['voxel_features'] = output[:voxel_num]
        batch_dict['voxel_coords'] = voxel_info[f'voxel_coors_stage{self.stage_num - 1}']

        logging.info(f"batch_dict shape of :\tpillar_features: {batch_dict['pillar_features'].shape}; \t voxel_coords: {batch_dict['voxel_coords'].shape}")
        logging.info(f"Result of mm_BACKBONE: {batch_dict.keys()}")
        for key in batch_dict:
            try:
                logging.info(f"key: {key} -- shape: {batch_dict[key].shape}")
            except:
                logging.info(f"key: {key} -- value: {batch_dict[key]}")
        return batch_dict

    def _input_preprocess(self, batch_dict):
        logging.info('===================== this is input_preprocess==============')
        # image branch =》 patch
        imgs = batch_dict['camera_imgs']
        B, N, C, H, W = imgs.shape  # 6, 6, 3, 256, 704
        imgs = imgs.view(B * N, C, H, W)

        imgs, hw_shape = self.patch_embed(imgs)  # 8x [36, 2816, C] [32, 88]
        batch_dict['hw_shape'] = hw_shape # [32, 88]

        # 36*2816, C
        batch_dict['patch_features'] = imgs.view(-1, imgs.shape[-1])
        logging.info(f"patch_feature shape: {batch_dict['patch_features'].shape}")

        if self.patch_coords is not None and ((self.patch_coords[:, 0].max().int().item() + 1) == B*N):
            batch_dict['patch_coords'] = self.patch_coords.clone()
        else:
            batch_idx = torch.arange(
                B*N, device=imgs.device).unsqueeze(1).repeat(1, hw_shape[0] * hw_shape[1]).view(-1, 1)
            batch_dict['patch_coords'] = torch.cat([batch_idx, self.patch_zyx.clone()[
                                                   None, ::].repeat(B*N, 1, 1).view(-1, 3)], dim=-1).long()
            self.patch_coords = batch_dict['patch_coords'].clone()
        logging.info(f"patch_coords shape: {batch_dict['patch_coords'].shape}")

        patch_info = self.image_input_layer(batch_dict)

        patch_feat = batch_dict['patch_features']
        patch_set_voxel_inds_list = [[patch_info[f'set_voxel_inds_stage{s}_shift{i}']
                                      for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        patch_set_voxel_masks_list = [[patch_info[f'set_voxel_mask_stage{s}_shift{i}']
                                       for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        patch_pos_embed_list = [[[patch_info[f'pos_embed_stage{s}_block{b}_shift{i}']
                                  for i in range(self.num_shifts[s])] for b in range(self.image_pos_num)] for s in range(len(self.set_info))]
        
        '''
        set_voxel_inds_list = []
        for s in range(len(self.set_info)): # 1层 set_info: [[90, 4]] => len(set_info) = 1
            voxel_inds_stage = []
            for i in range(self.num_shifts[s]): # 2 = [0, 1]
                voxel_ind = voxel_info[f'set_voxel_inds_stage{s}_shift{i}']
                voxel_inds_stage.append(voxel_ind)
            set_voxel_inds_list.append(voxel_inds_stage)

        也就是  set_voxel_inds_stage0_shift0 -- shape: torch.Size([2, 506, 90])
                set_voxel_inds_stage0_shift1 -- shape: torch.Size([2, 514, 90])
        '''

        # lidar branch
        voxel_info = self.lidar_input_layer(batch_dict)
        voxel_feat = batch_dict['voxel_features']
        set_voxel_inds_list = [[voxel_info[f'set_voxel_inds_stage{s}_shift{i}']
                                for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        set_voxel_masks_list = [[voxel_info[f'set_voxel_mask_stage{s}_shift{i}']
                                 for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        pos_embed_list = [[[voxel_info[f'pos_embed_stage{s}_block{b}_shift{i}']
                            for i in range(self.num_shifts[s])] for b in range(self.lidar_pos_num)] for s in range(len(self.set_info))]

        logging.info('===================== Fusion: image2d and lidar3d => multi feature ==============')
        # multi-modality parallel
        voxel_num = voxel_feat.shape[0]
        batch_dict['voxel_num'] = voxel_num
        multi_feat = torch.cat([voxel_feat, patch_feat], dim=0)
        logging.info(f'Shape of : voxel_feat: {voxel_feat.shape}; patch_feat: {patch_feat.shape}:\n\t multi-feat: {multi_feat.shape}')

        multi_set_voxel_inds_list = [[torch.cat([set_voxel_inds_list[s][i], patch_set_voxel_inds_list[s][i]+voxel_num], dim=1)
                                      for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        multi_set_voxel_masks_list = [[torch.cat([set_voxel_masks_list[s][i], patch_set_voxel_masks_list[s][i]], dim=1)
                                       for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
                
        # logging.info(f'Shape of : voxel_feat: {voxel_feat.shape}; patch_feat: {patch_feat.shape}:\n\t multi-feat: {multi_feat.shape}')
        # # 使用 len() 函数获取外层列表的长度
        # outer_length = len(multi_set_voxel_inds_list)
        # # 使用 len() 函数和列表推导式获取内层列表的长度，并存储在一个列表中
        # inner_lengths = [len(inner_list) for inner_list in multi_set_voxel_inds_list]
        # logging.info(f"out_lengths: {outer_length}")
        # logging.info(f"inner_lengths: {inner_lengths}")
        
        multi_pos_embed_list = []
        for s in range(len(self.set_info)):
            block_pos_embed_list = []
            for b in range(self.set_info[s][1]):
                shift_pos_embed_list = []
                for i in range(self.num_shifts[s]):
                    if b < self.lidar_pos_num and b < self.image_pos_num:
                        shift_pos_embed_list.append(
                            torch.cat([pos_embed_list[s][b][i], patch_pos_embed_list[s][b][i]], dim=0))
                    elif b < self.lidar_pos_num and b >= self.image_pos_num:
                        shift_pos_embed_list.append(pos_embed_list[s][b][i])
                    elif b >= self.lidar_pos_num and b < self.image_pos_num:
                        shift_pos_embed_list.append(
                            patch_pos_embed_list[s][b][i])
                    else:
                        raise NotImplementedError
                block_pos_embed_list.append(shift_pos_embed_list)
            multi_pos_embed_list.append(block_pos_embed_list)

        return multi_feat, voxel_info, patch_info, multi_set_voxel_inds_list, multi_set_voxel_masks_list, multi_pos_embed_list

    def _image2lidar_preprocess(self, batch_dict, multi_feat, multi_pos_embed_list):
        logging.info('===================== image2lidar_preprocess==============')
        N = batch_dict['camera_imgs'].shape[1]
        voxel_num = batch_dict['voxel_num']
        image2lidar_coords_zyx, nearest_dist = self.map_image2lidar_layer(
            batch_dict)
        image2lidar_coords_bzyx = torch.cat(
            [batch_dict['patch_coords'][:, :1].clone(), image2lidar_coords_zyx], dim=1)
        image2lidar_coords_bzyx[:, 0] = image2lidar_coords_bzyx[:, 0] // N
        image2lidar_batch_dict = {}
        image2lidar_batch_dict['voxel_features'] = multi_feat.clone()
        image2lidar_batch_dict['voxel_coords'] = torch.cat(
            [batch_dict['voxel_coords'], image2lidar_coords_bzyx], dim=0)
        
        image2lidar_info = self.image2lidar_input_layer(image2lidar_batch_dict)

        image2lidar_inds_list = [[image2lidar_info[f'set_voxel_inds_stage{s}_shift{i}']
                                  for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        image2lidar_masks_list = [[image2lidar_info[f'set_voxel_mask_stage{s}_shift{i}']
                                   for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        image2lidar_pos_embed_list = [[[image2lidar_info[f'pos_embed_stage{s}_block{b}_shift{i}']
                                        for i in range(self.num_shifts[s])] for b in range(self.image2lidar_pos_num)] for s in range(len(self.set_info))]
        image2lidar_neighbor_pos_embed = self.neighbor_pos_embed(nearest_dist)

        for b in range(self.image2lidar_start, self.image2lidar_end):
            for i in range(self.num_shifts[0]):
                image2lidar_pos_embed_list[0][b -
                                              self.image2lidar_start][i][voxel_num:] += image2lidar_neighbor_pos_embed
                multi_pos_embed_list[0][b][i] += image2lidar_pos_embed_list[0][b -
                                                                               self.image2lidar_start][i]
        return image2lidar_inds_list, image2lidar_masks_list, multi_pos_embed_list

    def _lidar2image_preprocess(self, batch_dict, multi_feat, multi_pos_embed_list):
        logging.info('===================== lidar2image_preprocess ==================')
        N = batch_dict['camera_imgs'].shape[1]
        hw_shape = batch_dict['hw_shape']
        lidar2image_coords_zyx = self.map_lidar2image_layer(batch_dict)
        lidar2image_coords_bzyx = torch.cat(
            [batch_dict['voxel_coords'][:, :1].clone(), lidar2image_coords_zyx], dim=1)
        multiview_coords = batch_dict['patch_coords'].clone()
        multiview_coords[:, 0] = batch_dict['patch_coords'][:, 0] // N
        multiview_coords[:, 1] = batch_dict['patch_coords'][:, 0] % N
        multiview_coords[:, 2] += hw_shape[1]
        multiview_coords[:, 3] += hw_shape[0]
        lidar2image_batch_dict = {}
        lidar2image_batch_dict['voxel_features'] = multi_feat.clone()
        lidar2image_batch_dict['voxel_coords'] = torch.cat(
            [lidar2image_coords_bzyx, multiview_coords], dim=0)
        
        lidar2image_info = self.lidar2image_input_layer(lidar2image_batch_dict)

        lidar2image_inds_list = [[lidar2image_info[f'set_voxel_inds_stage{s}_shift{i}']
                                  for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        lidar2image_masks_list = [[lidar2image_info[f'set_voxel_mask_stage{s}_shift{i}']
                                   for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        lidar2image_pos_embed_list = [[[lidar2image_info[f'pos_embed_stage{s}_block{b}_shift{i}']
                                        for i in range(self.num_shifts[s])] for b in range(self.lidar2image_pos_num)] for s in range(len(self.set_info))]

        for b in range(self.lidar2image_start, self.lidar2image_end):
            for i in range(self.num_shifts[0]):
                multi_pos_embed_list[0][b][i] += lidar2image_pos_embed_list[0][b -
                                                                               self.lidar2image_start][i]
        return lidar2image_inds_list, lidar2image_masks_list, multi_pos_embed_list

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name:
                nn.init.xavier_uniform_(p)

    def _recover_image(self, pillar_features, coords, indices):
        pillar_features = getattr(self, f'out_norm{indices}')(pillar_features)
        batch_size = coords[:, 0].max().int().item() + 1
        batch_spatial_features = pillar_features.view(
            batch_size, self.patch_size[0], self.patch_size[1], -1).permute(0, 3, 1, 2).contiguous()
        return batch_spatial_features

class UniTRBlock(nn.Module):
    ''' Consist of two encoder layer, shift and shift back.
    '''

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=True, block_id=-100, dout=None, layer_cfg=dict()):
        super().__init__()

        encoder_1 = UniTR_EncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                       activation, batch_first, layer_cfg=layer_cfg)
        encoder_2 = UniTR_EncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                       activation, batch_first, dout=dout, layer_cfg=layer_cfg)
        self.encoder_list = nn.ModuleList([encoder_1, encoder_2])

    def forward(
            self,
            src,
            set_voxel_inds_list,
            set_voxel_masks_list,
            pos_embed_list,
            block_id,
            voxel_num=0,
            using_checkpoint=False,
    ):
        logging.info('===================== UniTR Block ==============')
        num_shifts = len(set_voxel_inds_list)

        output = src
        for i in range(num_shifts):
            set_id = block_id % 2
            shift_id = i
            set_voxel_inds = set_voxel_inds_list[shift_id][set_id]
            set_voxel_masks = set_voxel_masks_list[shift_id][set_id]
            pos_embed = pos_embed_list[shift_id]
            layer = self.encoder_list[i]
            logging.info(f"num_shifts: i == {i}")
            logging.info(f"set_voxel_inds: {set_voxel_inds.shape}")
            logging.info(f"set_voxel_masks: {set_voxel_masks.shape}")

            if using_checkpoint and self.training:
                output = checkpoint(
                    layer, output, set_voxel_inds, set_voxel_masks, pos_embed, voxel_num)
            else:
                output = layer(output, set_voxel_inds,
                               set_voxel_masks, pos_embed, voxel_num=voxel_num)

        return output

class UniTR_EncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=True, mlp_dropout=0, dout=None, layer_cfg=dict()):
        super().__init__()
        self.win_attn = SetAttention(
            d_model, nhead, dropout, dim_feedforward, activation, batch_first, mlp_dropout, layer_cfg)
        if dout is None:
            dout = d_model
        self.norm = nn.LayerNorm(dout)
        self.d_model = d_model

    def forward(self, src, set_voxel_inds, set_voxel_masks, pos=None, voxel_num=0):
        logging.info('===================== unitr encoder layer ==============')
        identity = src
        
        src = self.win_attn(src, pos, set_voxel_masks, set_voxel_inds, voxel_num=voxel_num)
        src = src + identity
        src = self.norm(src)

        return src
class SetAttention(nn.Module):

    def __init__(self, d_model, nhead, dropout, dim_feedforward=2048, activation="relu", batch_first=True, mlp_dropout=0, layer_cfg=dict()):
        super().__init__()
        self.nhead = nhead

        if batch_first:     # = True
            self.self_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=batch_first)
        else:
            self.self_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(mlp_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.d_model = d_model
        self.layer_cfg = layer_cfg

        use_bn = layer_cfg.get('use_bn', False)
        if use_bn:
            assert use_bn is False
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        if layer_cfg.get('split_ffn', False):
            # Implementation of lidar Feedforward model
            self.lidar_linear1 = nn.Linear(d_model, dim_feedforward) # 128 => 256
            self.lidar_dropout = nn.Dropout(mlp_dropout)
            self.lidar_linear2 = nn.Linear(dim_feedforward, d_model) # 256 => 128

            use_bn = layer_cfg.get('use_bn', False)
            if use_bn:
                assert use_bn is False
            else:
                self.lidar_norm1 = nn.LayerNorm(d_model) # 每个样本的特征进行独立归一化，而不是对整个样本集进行归一化。这种归一化可以帮助模型更好地进行训练和优化。
                self.lidar_norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Identity() #这样在模型的前向传播中，self.dropout1 将不会对输入做任何修改，直接返回输入本身。
        self.dropout2 = nn.Identity() # 如果不需要进行随机失活或其他操作，可以将其设置为空操作，以节省计算资源和保持输入的原始形状。

        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos=None, key_padding_mask=None, voxel_inds=None, voxel_num=0):
        logging.info('===================== Set Attention ==============')

        logging.info(f'src: {src.shape} -- {src}; \
                     \npos: {pos.shape} -- {pos}; \
                     \nvoxel_inds: {voxel_inds.shape} --{voxel_inds}; \
                     \nvoxel_num: {voxel_num}')
        logging.info(f"set_voxel_masks == key_padding_mask: {key_padding_mask.shape} -- {key_padding_mask}") 
        # 对于二进制掩码，“True”值表示相应的“key”值将被忽略关注的目的。 对于字节掩码，非零值表示相应的“key”值将被忽略。

        set_features = src[voxel_inds]  # [win_num, 36, d_model]
        logging.info(f"set_feature: {set_features.shape}") 

        if pos is not None:
            set_pos = pos[voxel_inds]
        else:
            set_pos = None

        if pos is not None:
            query = set_features + set_pos
            key = set_features + set_pos
            value = set_features
        
        logging.info(f"query: {query.shape}; key: {key.shape}; value; {value.shape}") 
        # query: torch.Size([1037, 90, 128]); key: torch.Size([1037, 90, 128]); value; torch.Size([1037, 90, 128])

        if key_padding_mask is not None: # 执行这部分
            src2 = self.self_attn(query, key, value, key_padding_mask)[0]
        else:
            src2 = self.self_attn(query, key, value)[0]
        logging.info(f"src2: {src2.shape} -- {src2}")
        
        flatten_inds = voxel_inds.reshape(-1) # 得到原来张量的总数量 变成一维的
        logging.info(f"flatten_inds: {flatten_inds.shape} -- {flatten_inds}")

        unique_flatten_inds, inverse = torch.unique(flatten_inds, return_inverse=True) # 去重操作，
        # inverse包含原先元素在unique中的位置索引
        logging.info(f"unique_flatten_inds: {unique_flatten_inds.shape} -- {unique_flatten_inds}\
                    \ninverse: {inverse.shape} -- {inverse}")
        
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device) # perm 是 0 - inverse 的长度的整数
        logging.info(f"perm: {perm.shape} -- {perm}")

        inverse, perm = inverse.flip([0]), perm.flip([0]) # flip函数用于沿着指定维度翻转元素 eg: 4，3，2，1 = 》 1，2，3，4
        # 因为inverse 和perm 翻转一致，所以还是对应的顺序
        logging.info(f"inverse: {inverse.shape} -- {inverse}\
                    \n perm: {perm.shape} -- {perm}")

        perm = inverse.new_empty(unique_flatten_inds.size(0)).scatter_(0, inverse, perm)
        logging.info(f"perm: {perm.shape} -- {perm}")

        src2 = src2.reshape(-1, self.d_model)[perm] # self.d_model = 128 先变换形状，然后按照perm中索引的顺序进行重新排列
        logging.info(f"src2: {src2.shape} -- {src2}")

        if self.layer_cfg.get('split_ffn', False): # 当不存在时，返回fasle 这部分就是对lidar和image单独进行ffn
            src = src + self.dropout1(src2) # src += src2

            lidar_norm = self.lidar_norm1(src[:voxel_num]) # 前面部分时lidar的，
            image_norm = self.norm1(src[voxel_num:]) # 后面的是image 的
            logging.info(f"lidar_norm: {lidar_norm.shape}\
                         \nimage_norm: {image_norm.shape}")
            src = torch.cat([lidar_norm, image_norm], dim=0) # 归一化后又拼接到一起

            # 又对拼接后到src 分别进行了，linear(125-> 256), activation, dropout, linear(256->128)
            lidar_linear2 = self.lidar_linear2(self.lidar_dropout(
                self.activation(self.lidar_linear1(src[:voxel_num]))))
            image_linear2 = self.linear2(self.dropout(
                self.activation(self.linear1(src[voxel_num:]))))
            
            src2 = torch.cat([lidar_linear2, image_linear2], dim=0) # 又拼接起来

            src = src + self.dropout2(src2)
            lidar_norm2 = self.lidar_norm2(src[:voxel_num])
            image_norm2 = self.norm2(src[voxel_num:])
            src = torch.cat([lidar_norm2, image_norm2], dim=0)
        else:
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(
                self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src


class UniTRInputLayer(DSVTInputLayer):
    ''' 
    This class converts the output of vfe to unitr input.
    We do in this class:
    1. Window partition: partition voxels to non-overlapping windows.
    2. Set partition: generate non-overlapped and size-equivalent local sets within each window.
    3. Pre-compute the downsample infomation between two consecutive stages.
    4. Pre-compute the position embedding vectors.

    Args:
        sparse_shape (tuple[int, int, int]): Shape of input space (xdim, ydim, zdim).
        window_shape (list[list[int, int, int]]): Window shapes (winx, winy, winz) in different stages. Length: stage_num.
        downsample_stride (list[list[int, int, int]]): Downsample strides between two consecutive stages. 
            Element i is [ds_x, ds_y, ds_z], which is used between stage_i and stage_{i+1}. Length: stage_num - 1.
        d_model (list[int]): Number of input channels for each stage. Length: stage_num.
        set_info (list[list[int, int]]): A list of set config for each stage. Eelement i contains 
            [set_size, block_num], where set_size is the number of voxel in a set and block_num is the
            number of blocks for stage i. Length: stage_num.
        hybrid_factor (list[int, int, int]): Control the window shape in different blocks. 
            e.g. for block_{0} and block_{1} in stage_0, window shapes are [win_x, win_y, win_z] and 
            [win_x * h[0], win_y * h[1], win_z * h[2]] respectively.
        shift_list (list): Shift window. Length: stage_num.
        input_image (bool): whether input modal is image.
    '''

    def __init__(self, model_cfg, accelerate=False):
        # dummy config
        model_cfg.downsample_stride = model_cfg.get('downsample_stride',[])
        model_cfg.normalize_pos = model_cfg.get('normalize_pos',False)
        super().__init__(model_cfg)

        self.input_image = self.model_cfg.get('input_image', False)
        self.key_name = 'patch' if self.input_image else 'voxel'
        # only support image input accelerate
        self.accelerate = self.input_image and accelerate
        self.process_info = None

    def forward(self, batch_dict):
        '''
        Args:
            bacth_dict (dict): 
                The dict contains the following keys
                - voxel_features (Tensor[float]): Voxel features after VFE with shape (N, d_model[0]), 
                    where N is the number of input voxels.
                - voxel_coords (Tensor[int]): Shape of (N, 4), corresponding voxel coordinates of each voxels.
                    Each row is (batch_id, z, y, x). 
                - ...

        Returns:
            voxel_info (dict):
                The dict contains the following keys
                - voxel_coors_stage{i} (Tensor[int]): Shape of (N_i, 4). N is the number of voxels in stage_i.
                    Each row is (batch_id, z, y, x).
                - set_voxel_inds_stage{i}_shift{j} (Tensor[int]): Set partition index with shape (2, set_num, set_info[i][0]).
                    2 indicates x-axis partition and y-axis partition. 
                - set_voxel_mask_stage{i}_shift{i} (Tensor[bool]): Key mask used in set attention with shape (2, set_num, set_info[i][0]).
                - pos_embed_stage{i}_block{i}_shift{i} (Tensor[float]): Position embedding vectors with shape (N_i, d_model[i]). N_i is the 
                    number of remain voxels in stage_i;
                - ...
        '''

        logging.info('===================== Window and Set Partition ==============')
        # logging.info(f"self.input_image: {self.input_image}; self.process_info is not None? {self.process_info is not None}\n; batch_dict['patch_coords'][:, 0][-1] == self.process_info['voxel_coors_stage0'][:, 0][-1]: {batch_dict['patch_coords'][:, 0][-1] == self.process_info['voxel_coors_stage0'][:, 0][-1]}")
        if self.input_image and self.process_info is not None and (batch_dict['patch_coords'][:, 0][-1] == self.process_info['voxel_coors_stage0'][:, 0][-1]):
            logging.info(f'Process: Input_image; self.input_image: {self.input_image}; {self.process_info is not None}')
            patch_info = dict()
            for k in (self.process_info.keys()):
                if torch.is_tensor(self.process_info[k]):
                    patch_info[k] = self.process_info[k].clone()
                else:
                    patch_info[k] = copy.deepcopy(self.process_info[k])
            # accelerate by caching pos embed as patch coords are fixed
            if not self.accelerate: #  = True 
                for stage_id in range(len(self.downsample_stride)+1):
                    for block_id in range(self.set_info[stage_id][1]):
                        for shift_id in range(self.num_shifts[stage_id]):
                            patch_info[f'pos_embed_stage{stage_id}_block{block_id}_shift{shift_id}'] = \
                                self.get_pos_embed(
                                    patch_info[f'coors_in_win_stage{stage_id}_shift{shift_id}'], stage_id, block_id, shift_id)
            return patch_info

        key_name = self.key_name
        coors = batch_dict[f'{key_name}_coords'].long()
        logging.info(f"used coords: {coors.shape};--------batch{key_name}_coords")

        info = {}
        # original input voxel coors
        info[f'voxel_coors_stage0'] = coors.clone()

        for stage_id in range(len(self.downsample_stride)+1):
            # window partition of corrsponding stage-map
            info = self.window_partition(info, stage_id)
            logging.info(f"window partition----------------info.keys:  {info.keys()}")
            for key in info.keys():
                logging.info(f"Key: {key} Shape: {info[key].shape}")
            
            # generate set id of corrsponding stage-map
            info = self.get_set(info, stage_id)
            logging.info(f"Set partition----------------info.keys:  {info.keys()}") # torch.Size([2, 917, 90])
            for key in info.keys():
                logging.info(f"Key: {key} Shape: {info[key].shape}")
            
            # 这是每次划分后的位置嵌入   
            for block_id in range(self.set_info[stage_id][1]): # 4
                for shift_id in range(self.num_shifts[stage_id]):
                    info[f'pos_embed_stage{stage_id}_block{block_id}_shift{shift_id}'] = \
                        self.get_pos_embed(
                            info[f'coors_in_win_stage{stage_id}_shift{shift_id}'], stage_id, block_id, shift_id)
                    
                    logging.info(f"info[f'pos_embed_stage{stage_id}_block{block_id}_shift{shift_id}'] Shape: {info[f'pos_embed_stage{stage_id}_block{block_id}_shift{shift_id}'].shape}")
                    
        info['sparse_shape_list'] = self.sparse_shape_list

        # save process info for image input as patch coords are fixed
        if self.input_image:
            self.process_info = {}
            for k in (info.keys()):
                if k != 'patch_feats_stage0':
                    if torch.is_tensor(info[k]):
                        self.process_info[k] = info[k].clone()
                    else:
                        self.process_info[k] = copy.deepcopy(info[k])

        logging.info(f'Window and set patition result: {info.keys()}')
        for key in info:
            try:
                logging.info(f"key: {key} -- shape: {info[key].shape}")
            except:
                logging.info(f"key: {key} -- {info[key]}")
                continue

        logging.info('===================== Window and Set Partition END ==============\n\n')
        return info
    
    def get_set_single_shift(self, batch_win_inds, stage_id, shift_id=None, coors_in_win=None):
        '''
        voxel_order_list[list]: order respectively sort by x, y, z
        '''
        logging.info(f"entered in get_set_single_shift()")
        device = batch_win_inds.device
        
        # max number of voxel in a window
        voxel_num_set = self.set_info[stage_id][0] # 90 # set_info: [[90, 4]]
        max_voxel = self.window_shape[stage_id][shift_id][0] * \
            self.window_shape[stage_id][shift_id][1] * \
            self.window_shape[stage_id][shift_id][2]
        # window_shape: [[30, 30, 1]] = 30 * 30 * 1 = 900 计算出来的voxel 最大数量
        logging.info(f"max number of voxels in a window: {max_voxel} ") # 900

        if self.model_cfg.get('expand_max_voxels', None) is not None: # image2lidar: 10, lidar2image: 30
            max_voxel *= self.model_cfg.get('expand_max_voxels', None)
            logging.info(f"max number of expanded voxels in a window: {max_voxel} ")

        logging.info(f"batch_win_inds: {batch_win_inds.shape}, {batch_win_inds}") # 所以这batch_win_inds 是为了将这么多的voxel 分配到window里面去的索引，这样unique之后就可以得到每个voxel对应的window的索引了
        '''
            所以 这里的batch_win_inds的作用就是在这 用于统计得到连续的window索引即这里的 0-107 共108个连续索引的window
            虽然在前面一直是加权的全局window索引 长度和voxel长度一样 
        '''
        contiguous_win_inds = torch.unique(batch_win_inds, return_inverse=True)[1] 
        # torch.unique(batch_win_inds, return_inverse=True) 返回两个张量，一个是包含所有唯一元素的张量（这个长度可能是小于原先的batch长度的），另一个是新的索引的张量（长度=原先的batch长度,只是里面的索引不同）
        # 注意，这里取了[1]则是用来获取这个索引张量, 也就是[0, 1, 1, 2, 3, 3, 4, 4, 4... ， 107, 107]表示的是50688这么多的feature token 里面要对应的108个window id
        logging.info(f'contiguous_win_inds: {contiguous_win_inds.shape}, {contiguous_win_inds}') # 50688 

        voxelnum_per_win = torch.bincount(contiguous_win_inds) 
        # 计算contiguous_win_inds中每个值的频数，也就是每个窗口中的体素数（voxel number）结果保存在voxelnum_per_win中
        logging.info(f"voxelnum_per_win: {voxelnum_per_win.shape} -- {voxelnum_per_win}")

        win_num = voxelnum_per_win.shape[0]
        logging.info(f"window number: {win_num}")

        setnum_per_win_float = voxelnum_per_win / voxel_num_set
        # 将每个window中voxel数量 除以 set中指定的voxel数量就是每个window里面的set数量了

        setnum_per_win = torch.ceil(setnum_per_win_float).long()
        logging.info(f"setnum_per_win: {setnum_per_win.shape} -- {setnum_per_win}") # 每个window里面的set数
        
        set_num = setnum_per_win.sum().item() 
        logging.info(f"set number: {set_num}") # 得到的是，总共划分的set数量

        setnum_per_win_cumsum = torch.cumsum(setnum_per_win, dim=0)[:-1] # [:-1]则是对结果进行切片操作，取除了最后一个元素以外的所有元素的累积和（）
        logging.info(f"setnum_per_win_cumsum累积和: {setnum_per_win_cumsum.shape}, {setnum_per_win_cumsum}")
        # 用于下面为 所有set 分配 对应的window ID

        set_win_inds = torch.full((set_num,), 0, device=device) # full()返回一个填充了标量值的张量
        # 创建了set_num 长度的一维张量，所有值为0，并且放到了device上去 
        logging.info(f"set_win_inds: {set_win_inds.shape}, {set_win_inds}")

        set_win_inds[setnum_per_win_cumsum] = 1
        logging.info(f"set_win_inds: {set_win_inds.shape}, {set_win_inds}")

        set_win_inds = torch.cumsum(set_win_inds, dim=0)
        logging.info(f"set_win_inds: {set_win_inds.shape}, {set_win_inds}") 
        ''' 得到的是 这么多的set 对应的window_id，比如说前面10个set都是对应到window0的 那么他们的值就都是 0

        这步之后就得到了： 为set_num的数量set 分配到了win_num的对应关系win_id,
        前面的setnum_per_win_cumsum累积和是用来一次同时为属于一个win的set设置win_id ,而不需要一边遍来。
        例如, 第一个win里面有10个set, 第二个win里面有20个set, 这样的话，
        set_win_inds得到的结果是: 前面set_win_inds[0:9] = win_id_1, set_win_inds[10:19] = win_id_2, 就是这样子的
        '''

        # input [0,0,0, 1, 2,2]
        roll_set_win_inds_left = torch.roll(set_win_inds, -1)  # [0,0, 1, 2,2,0]
        # 沿着指定维度将输入张量中的元素滚动（循环移位）
        # 将set_win_inds中的元素向左滚动（循环移位）一个位置。也就是将所有元素依次向前移动一个位置。
        logging.info(f"roll_set_win_inds_left: {roll_set_win_inds_left}, {roll_set_win_inds_left.shape}")

        diff = set_win_inds - roll_set_win_inds_left  # [0, 0, -1, -1, 0, 2]
        logging.info(f"diff = set_win_inds - roll_set_win_inds_left: {diff}")

        end_pos_mask = diff != 0
        logging.info(f"end_pos_mask: {end_pos_mask.shape}, {end_pos_mask}") # [false, false, true, true, false, true]
        # end_pos_mask中只有108个true 整好对应的是 setnum_per_win的长度的元素
        template = torch.ones_like(set_win_inds)
        template[end_pos_mask] = (setnum_per_win - 1) * -1  # [1,1,-2, 0, 1,-1] # 将end_pos_mask = True的地方，设置成这个位置值 
        
        logging.info(f"template: {template.shape}, {template}")
        # -1*（10-1）= -9； -1*（0-1） = 0

        set_inds_in_win = torch.cumsum(template, dim=0)  # [1,2,0, 0, 1,0]
        logging.info(f"set_inds_in_win: {set_inds_in_win.shape}, {set_inds_in_win}")

        set_inds_in_win[end_pos_mask] = setnum_per_win  # [1,2,3, 1, 1,2]
        logging.info(f"set_inds_in_win: {set_inds_in_win.shape}, {set_inds_in_win}")

        set_inds_in_win = set_inds_in_win - 1  # [0,1,2, 0, 0,1]
        logging.info(f"set_inds_in_win: {set_inds_in_win.shape}, {set_inds_in_win}")

        '''得到的是 每个set在所属window的set集合的索引ID 
        
        得到的是: 每个window里面set的索引，比如说，一个window里面的set 有10个，这样的话，这个set的索引就是0-9。
        如果一个window里面只有一个set,这样的话，这个set的索引就是0。这也就是为什么有好多个0了,这代表了好多window里面只有一个set
        '''
        
        offset_idx = set_inds_in_win[:, None].repeat(1, voxel_num_set) * voxel_num_set
        # 得到(set_inds_in_win, voxel_num_set)的张量，并且每个元素都乘以voxel_num_set, N = set_inds_in_win的长度，
        logging.info(f"offset_idx: {offset_idx.shape}, {offset_idx[:10, :]}, {offset_idx[-35:-30, :]}")
        # torch.Size([set_num, 90]) 设的是，每个set90个voxel一组，这样加上base_idx之后就可以得到[  0,   1,   2,  ...,  87,  88,  89],[ 90,  91,  92,  ..., 177, 178, 179],[180, 181, 182,  ..., 267, 268, 269]

        base_idx = torch.arange(0, voxel_num_set, 1, device=device) # 包含从0到voxel_num_set-1的整数值 [0, 90]
        logging.info(f"base_idx: {base_idx.shape}, {base_idx}")

        base_select_idx = offset_idx + base_idx
        logging.info(f"base_select_ind: {base_select_idx.shape},{base_select_idx[:10, :]}, {base_select_idx[-35:-30, :]}")
        '''
            得到的是: 每个set里包含voxel数量的索引  shape = [set_num, 90]
            eg: 第一个window里面的set数是10个 且包含的最多的voxel数量是900,那么得到的set里面的值是 0-89 90-179 ...
        '''

        umi_tmp2 = voxelnum_per_win[set_win_inds][:, None]
        logging.info(f"voxelnum_per_win[set_win_inds][:, None]: {umi_tmp2.shape}, {umi_tmp2[:10, :]}")

        base_select_idx = base_select_idx * voxelnum_per_win[set_win_inds][:, None]
        # 将base_select_idx与voxelnum_per_win中对应位置的元素进行逐元素相乘，结果保存在base_select_idx中
        logging.info(f"base_select_ind: {base_select_idx.shape}, {base_select_idx}")

        base_select_idx = base_select_idx.double() / (setnum_per_win[set_win_inds] * voxel_num_set)[:, None].double()
        logging.info(f"base_select_ind: {base_select_idx.shape}, {base_select_idx}")

        base_select_idx = torch.floor(base_select_idx) # 向下取整
        logging.info(f"base_select_ind: {base_select_idx.shape}, {base_select_idx}")

        select_idx = base_select_idx
        select_idx = select_idx + set_win_inds.view(-1, 1) * max_voxel
        logging.info(f"select_idx: {select_idx.shape}, {select_idx}")
        '''
            得到的是 应该是：set里面的voxel索引了，
            不同于上面的是，上面得到的索引，实际会超出真正的voxel总数，但是这里没有超出
            例如 如果一个set只有60个voxel的话, 虽然长度还是90 但是里面会有重复的元素，

            select_idx -> [594, 90]
            set_win_inds.view(-1, 1) -> [594, 1] 里面每个值都乘以900
            max_voxel是一个window里面最多包含的voxel数量
                
        '''

        # sort by y
        inner_voxel_inds = get_inner_win_inds_cuda(contiguous_win_inds) # 通过给定contiguous_win_inds中获取内部窗口的体素索引
        logging.info(f"contiguous_win_inds: {contiguous_win_inds.shape}, {contiguous_win_inds}") # 为每个voxel 分配了一个window的id 索引，所以长度是50688个
        logging.info(f"inner_voxel_inds: {inner_voxel_inds.shape}, {inner_voxel_inds[:30]}, {inner_voxel_inds}")
        # inner_voxel_inds: torch.Size([50688]), tensor([284, 288, 292,  ...,  45,  46,  47], device='cuda:0')

        global_voxel_inds = contiguous_win_inds * max_voxel + inner_voxel_inds # max_voxel = 900, 这样计算后得到的就是全局的voxel索引了
        logging.info(f"global_voxel_inds: {global_voxel_inds.shape}, {global_voxel_inds}")

        _, order1 = torch.sort(global_voxel_inds) # 返回的是排序后的张量，和排序后的索引
        logging.info(f"order1: {order1.shape}, {order1[:30]}, {order1}") # 原先位置的voxel在有序voxel里面的索引
        # order1 是对全局voxel索引 按照排序后的 每个voxel的位置索引 =》 要的就是这个排序索引

        # window_shape: [[[30, 30, 1], [30, 30, 1]]]
        logging.info(f"window_shape: {self.window_shape}")
        global_voxel_inds_sorty = contiguous_win_inds * max_voxel + \
            coors_in_win[:, 1] * self.window_shape[stage_id][shift_id][0] * self.window_shape[stage_id][shift_id][2] + \
            coors_in_win[:, 2] * self.window_shape[stage_id][shift_id][2] + \
            coors_in_win[:, 0]
        '''
          y, y * 30 * 1
          x, x * 1
          z, z
          所以是按照y偏移？
        '''
        _, order2 = torch.sort(global_voxel_inds_sorty) # 将全局voxel索引按照y方向排序，
        logging.info(f"global_voxel_inds_sorty: {global_voxel_inds_sorty}") # 得到的全局体素索引
        logging.info(f"order2: {order2.shape}, {order2[:30], {order2}}")
        # 这里面的order2得到的结果就是         数值在50688范围内的排序了

        inner_voxel_inds_sorty = -torch.ones_like(inner_voxel_inds)
        inner_voxel_inds_sorty.scatter_(dim=0, index=order2, src=inner_voxel_inds[order1])
        # 将 inner_voxel_inds 张量根据指定的索引方式 order2，分散到替换的 inner_voxel_inds_sorty 张量的相应位置上。
        # inner_voxel_inds[order1]元素 按照order2放到固定的位置上
        # 注意：order2和inner_voxel_inds[order1]是等长的，并且，inner_voxel_inds[order1]的第一个元素的输出位置是order2的第一个元素对应索引
        logging.info(f"inner_voxel_inds_sorty: {inner_voxel_inds_sorty.shape}, {inner_voxel_inds_sorty[:30]}, {inner_voxel_inds_sorty}")

        inner_voxel_inds_sorty_reorder = inner_voxel_inds_sorty
        voxel_inds_in_batch_sorty = inner_voxel_inds_sorty_reorder + \
            max_voxel * contiguous_win_inds
        logging.info(f"voxel_inds_in_batch_sorty:{voxel_inds_in_batch_sorty.shape}, {voxel_inds_in_batch_sorty[:30]}, {voxel_inds_in_batch_sorty}") # 每window shape 大小的递增

        voxel_inds_padding_sorty = -1 * torch.ones((win_num * max_voxel), dtype=torch.long, device=device)
        logging.info(f"voxel_inds_padding_sorty: {voxel_inds_padding_sorty.shape},  {voxel_inds_padding_sorty}")

        # 替换掉一些内容
        voxel_inds_padding_sorty[voxel_inds_in_batch_sorty] = torch.arange(0, voxel_inds_in_batch_sorty.shape[0], dtype=torch.long, device=device)
        # arrange 的内容是： 0-voxel_inds_in_batch_sorty.shape[0], 递增序列
        # 替换掉，voxel_inds_padding_sorty[voxel_inds_in_batch_sorty] 这里50688个位置的，每个索引的位置，的值是arrange顺序排序的0-1-2-3---50688
        logging.info(f"voxel_inds_padding_sorty: {voxel_inds_padding_sorty.shape}, {voxel_inds_padding_sorty[:31]}, {voxel_inds_padding_sorty}")

        # sort by x # window_shape: [[[30, 30, 1], [30, 30, 1]]]
        global_voxel_inds_sorty = contiguous_win_inds * max_voxel + \
            coors_in_win[:, 2] * self.window_shape[stage_id][shift_id][1] * self.window_shape[stage_id][shift_id][2] + \
            coors_in_win[:, 1] * self.window_shape[stage_id][shift_id][2] + \
            coors_in_win[:, 0]
        '''
            x, x * 30 * 1
            y, y * 1
            z, z
            所以是按照x排序
        '''
        _, order2 = torch.sort(global_voxel_inds_sorty)

        inner_voxel_inds_sortx = -torch.ones_like(inner_voxel_inds)
        inner_voxel_inds_sortx.scatter_(dim=0, index=order2, src=inner_voxel_inds[order1])
        inner_voxel_inds_sortx_reorder = inner_voxel_inds_sortx
        voxel_inds_in_batch_sortx = inner_voxel_inds_sortx_reorder + \
            max_voxel * contiguous_win_inds
        voxel_inds_padding_sortx = -1 * \
            torch.ones((win_num * max_voxel), dtype=torch.long, device=device)
        voxel_inds_padding_sortx[voxel_inds_in_batch_sortx] = torch.arange(
            0, voxel_inds_in_batch_sortx.shape[0], dtype=torch.long, device=device)

        set_voxel_inds_sorty = voxel_inds_padding_sorty[select_idx.long()] # 对索引进行切片操作
        set_voxel_inds_sortx = voxel_inds_padding_sortx[select_idx.long()]
        logging.info(f"set_voxel_inds_sort---x: {set_voxel_inds_sortx.shape}, {set_voxel_inds_sortx}") 
        logging.info(f"set_voxel_inds_sort----y: {set_voxel_inds_sorty.shape}, {set_voxel_inds_sorty}")

        all_set_voxel_inds = torch.stack(
            (set_voxel_inds_sorty, set_voxel_inds_sortx), dim=0) # 最后叠加到一块，所以变成了【2， set_num, 90】

        return all_set_voxel_inds

    def get_pos_embed(self, coors_in_win, stage_id, block_id, shift_id):
        '''
        Args:
        coors_in_win: shape=[N, 3], order: z, y, x
        '''
        # [N,]
        window_shape = self.window_shape[stage_id][shift_id]  # window_shape: [[30, 30, 1]]
        embed_layer = self.posembed_layers[stage_id][block_id][shift_id]
        # logging.info(f"emed_layer: {embed_layer }")

        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        elif window_shape[-1] == 1: # window_shape: [[30, 30, 1]]
            if self.sparse_shape[-1] == 1: # sparse_shape: [32, 88, 1]
                ndim = 2
            else:
                ndim = 3
            win_x, win_y = window_shape[:2]
            win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3

        assert coors_in_win.size(1) == 3
        z, y, x = coors_in_win[:, 0] - win_z/2, coors_in_win[:, 1] - win_y/2, coors_in_win[:, 2] - win_x/2

        if self.normalize_pos: # false
            x = x / win_x * 2 * 3.1415 #[-pi, pi]
            y = y / win_y * 2 * 3.1415 #[-pi, pi]
            z = z / win_z * 2 * 3.1415 #[-pi, pi]
        
        if ndim==2:
            location = torch.stack((x, y), dim=-1)
        else:
            location = torch.stack((x, y, z), dim=-1)
        pos_embed = embed_layer(location)

        return pos_embed
