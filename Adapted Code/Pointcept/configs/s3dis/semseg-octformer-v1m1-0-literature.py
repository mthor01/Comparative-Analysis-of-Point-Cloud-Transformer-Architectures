_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 16 # bs: total bs in all gpus
mix_prob = 0.8
enable_amp = False
empty_cache = False

# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="OctFormer-v1m1",
        in_channels=9,  
        num_classes=13,
        fpn_channels=168,
        channels=(96, 192, 384, 384), 
        num_blocks=(2, 2, 18, 2),  
        num_heads=(6, 12, 24, 24),  
        patch_size=26, 
        stem_down=2,
        head_up=2,
        dilation=4,
        drop_path=0.5,
        nempty=True,
        octree_depth=11, 
        octree_full_depth=2,
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

# scheduler settings
epoch = 600
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="MultiStepWithWarmupLR",
    milestones=[0.6, 0.9],
    gamma=0.1,
    warmup_rate=0.05,
    warmup_scale=1e-5,
)


# dataset settings
dataset_type = "S3DISDataset"
data_root = "data/s3dis"

data = dict(
    num_classes=13,
    ignore_index=-1,
    names=[
        "ceiling",
        "floor",
        "wall",
        "beam",
        "column",
        "window",
        "door",
        "table",
        "chair",
        "sofa",
        "bookcase",
        "board",
        "clutter",
    ],
    train=dict(
        type=dataset_type,
        split=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.1),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_min_coord=True,
            ),
            dict(type="SphereCrop", point_max=100000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "normal", "segment"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
    ),

    val=dict(
        type=dataset_type,
        split="Area_5",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample", 
                grid_size=0.01, 
                hash_type="fnv", 
                mode="train", 
                return_min_coord=True,
                return_inverse=True,
            ),
            #dict(type="SphereCrop", point_max=100000, mode="center"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "normal", "segment", "origin_segment", "inverse"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
    ),

    test=dict(
        type=dataset_type,
        split="Area_5",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="test",
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "normal", "index"),
                    feat_keys=("coord", "color", "normal"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [
                    dict(type="RandomScale", scale=[0.9, 0.9]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                    dict(type="RandomFlip", p=1),
                ],
                [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
                [
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.1, 1.1]),
                    dict(type="RandomFlip", p=1),
                ],
            ]
        )
    )
)
