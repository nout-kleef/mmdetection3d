_base_ = '../../_base_/models/hv_pointpillars_secfpn_inhouse.py'

voxel_size = [0.16, 0.16, 4]
_point_cloud_range = [0, -39.68, -2, 69.12, 39.68, 2]

model = dict(
    voxel_layer=dict(
        voxel_size=voxel_size,
        point_cloud_range=_point_cloud_range
    ),
    voxel_encoder=dict(
        in_channels=6,
        voxel_size=voxel_size,
        point_cloud_range=_point_cloud_range
    )
)