_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 2  # bs: total bs in all gpus
enable_amp = True
empty_cache = True
mix_prob = 0

# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="OctFormer-v1m1",
        in_channels=3,  # color
        num_classes=13,
        channels=(64, 128, 256, 256),  # Reduced channel dimensions
        num_blocks=(2, 2, 10, 2),  # Reduced number of blocks in the 3rd stage
        num_heads=(4, 8, 16, 16),  # Reduced number of attention heads
        patch_size=16,  # Reduced patch size
        stem_down=2,
        head_up=2,
        dilation=4,
        drop_path=0.5,
        nempty=True,
        octree_scale_factor=10.24,
        octree_depth=10,  # Reduced octree depth
        octree_full_depth=2,
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

# scheduler settings
epoch = 600
optimizer = dict(type="AdamW", lr=0.003, weight_decay=0.05)
scheduler = dict(type="OneCycleLR",
                 max_lr=optimizer["lr"],
                 pct_start=0.05,
                 anneal_strategy="cos",
                 div_factor=10.0,
                 final_div_factor=1000.0)


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
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(type="NormalizeColor"),
            dict(type="GridSample", grid_size=0.04, hash_type="fnv", mode="train", return_min_coord=True),
            dict(type="SphereCrop", point_max=100000, mode="random"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "normal", "segment"),
                feat_keys=["color"],
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
            dict(type="NormalizeColor"),
            dict(type="GridSample", grid_size=0.04, hash_type="fnv", mode="train", return_min_coord=True),
            dict(type="SphereCrop", point_max=100000, mode="center"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "normal", "segment"),
                feat_keys=["color"],
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
            dict(type="GridSample", grid_size=0.04, hash_type="fnv", mode="train", return_min_coord=True),
        ],
        test_mode=True,
        test_cfg=dict(
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "normal", "index"),
                    feat_keys=["color"],
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