_base_ = '../pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py'

# use normal BN instead of SyncBN
model = dict(
    pts_voxel_encoder=dict(
        norm_cfg=dict(type='BN1d')),
    pts_backbone=dict(
        norm_cfg=dict(type='BN2d')),
    pts_neck=dict(
        norm_cfg=dict(type='BN2d'))
)
