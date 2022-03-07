_base_ = '../../_base_/models/hv_pointpillars_secfpn_inhouse.py'

voxel_size = [0.16, 0.16, 4]

model = dict(
    voxel_layer=dict(
        voxel_size=voxel_size
    ),
    voxel_encoder=dict(
        in_channels=6,
        voxel_size=voxel_size
    )
)