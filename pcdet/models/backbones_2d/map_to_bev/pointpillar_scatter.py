import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        device = pillar_features.device
        batch_size = coords[:, 0].max().int().item() + 1
        assert batch_size == 1, 'bev_index_map 路径仅支持 batch_size=1'

        # 若外部提供 bev_index_map（每帧由 Python 侧构造的 (G,) 索引表，pad 行 = M），
        # 用 Gather 替代 scatter。310P 上 ScatterND 是单核标量实现（~96ms），
        # Gather 多核向量化（~7ms），数值完全等价。
        if 'bev_index_map' in batch_dict:
            index_map = batch_dict['bev_index_map']
            M = pillar_features.shape[0]
            pillars_padded = torch.cat([
                pillar_features,
                torch.zeros(1, pillar_features.shape[-1], dtype=pillar_features.dtype, device=device),
            ], dim=0)
            bev = pillars_padded[index_map]  # (G, C) -> GatherV2
            spatial_feature = bev.transpose(0, 1).view(
                batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
            batch_dict['spatial_features'] = spatial_feature
            return batch_dict

        # NPU 310P 的 aicpu GatherElements 对布尔掩码索引 / 索引赋值不稳定（errorCode 0x2a），
        # 因此把不规则 scatter 放到 CPU 完成后再搬回设备。
        pf_cpu = pillar_features.detach().cpu()
        coords_cpu = coords.detach().cpu()

        batch_spatial_features = []
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pf_cpu.dtype)

            batch_mask = coords_cpu[:, 0] == batch_idx
            this_coords = coords_cpu[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pf_cpu[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features.to(device)
        return batch_dict


class PointPillarScatter3d(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        
        self.model_cfg = model_cfg
        self.nx, self.ny, self.nz = self.model_cfg.INPUT_SHAPE
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_bev_features_before_compression = self.model_cfg.NUM_BEV_FEATURES // self.nz

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features_before_compression,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] * self.ny * self.nx + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features_before_compression * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict