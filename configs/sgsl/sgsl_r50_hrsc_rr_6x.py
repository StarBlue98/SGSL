dataset_type = 'HRSCDataset'
data_root = '/data/HRSC/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(800, 512)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version='le90'),
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        rect_classes=[9, 11],
        version='le90'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 512),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='HRSCDataset',
        classwise=False,
        ann_file='/data/HRSC/ImageSets/trainval.txt',
        ann_subdir='/data/HRSC/FullDataSet/Annotations/',
        img_subdir='/data/HRSC/FullDataSet/AllImages/',
        img_prefix='/data/HRSC/FullDataSet/AllImages/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RResize', img_scale=(800, 512)),
            dict(
                type='RRandomFlip',
                flip_ratio=[0.25, 0.25, 0.25],
                direction=['horizontal', 'vertical', 'diagonal'],
                version='le90'),
            dict(
                type='PolyRandomRotate',
                rotate_ratio=0.5,
                angles_range=180,
                auto_bound=False,
                rect_classes=[9, 11],
                version='le90'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        version='le90'),
    val=dict(
        type='HRSCDataset',
        classwise=False,
        ann_file='/data/HRSC/ImageSets/test.txt',
        ann_subdir='/data/HRSC/FullDataSet/Annotations/',
        img_subdir='/data/HRSC/FullDataSet/AllImages/',
        img_prefix='/data/HRSC/FullDataSet/AllImages/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 512),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        version='le90'),
    test=dict(
        type='HRSCDataset',
        classwise=False,
        ann_file='/data/HRSC/ImageSets/test.txt',
        ann_subdir='/data/HRSC/FullDataSet/Annotations/',
        img_subdir='/data/HRSC/FullDataSet/AllImages/',
        img_prefix='/data/HRSC/FullDataSet/AllImages/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 512),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        version='le90'))
evaluation = dict(interval=72, metric='mAP')
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[48, 66])
runner = dict(type='EpochBasedRunner', max_epochs=72)
checkpoint_config = dict(interval=12)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
angle_version = 'le90'
model = dict(
    type='RotatedRetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RotatedATSSHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=1,
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range='le90',
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        reg_decoded_bbox=True,
        loss_bbox=dict(type='RotatedIoULossv1', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='Shape2ATSSObbAssigner',
            topk=9,
            angle_version='le90',
            iou_calculator=dict(type='RBboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))
work_dir = './work/sgsl_r50_hrsc_rr_6x'
auto_resume = False
gpu_ids = range(0, 1)
