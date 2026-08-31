from pathlib import Path

from detectron2.config import CfgNode as CN


def add_diffusiondet_config(cfg):
    """
    Add config for DiffusionDet
    """
    # ================= DiffusionDet =================
    cfg.MODEL.DiffusionDet = CN()
    cfg.MODEL.DiffusionDet.NUM_CLASSES = 80
    cfg.MODEL.DiffusionDet.NUM_PROPOSALS = 300

    # RCNN Head.
    cfg.MODEL.DiffusionDet.NHEADS = 8
    cfg.MODEL.DiffusionDet.DROPOUT = 0.0
    cfg.MODEL.DiffusionDet.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DiffusionDet.ACTIVATION = 'relu'
    cfg.MODEL.DiffusionDet.HIDDEN_DIM = 256
    cfg.MODEL.DiffusionDet.NUM_CLS = 1
    cfg.MODEL.DiffusionDet.NUM_REG = 3
    cfg.MODEL.DiffusionDet.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.DiffusionDet.NUM_DYNAMIC = 2
    cfg.MODEL.DiffusionDet.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.DiffusionDet.CLASS_WEIGHT = 2.0
    cfg.MODEL.DiffusionDet.GIOU_WEIGHT = 2.0
    cfg.MODEL.DiffusionDet.L1_WEIGHT = 5.0
    cfg.MODEL.DiffusionDet.DEEP_SUPERVISION = True
    cfg.MODEL.DiffusionDet.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.DiffusionDet.USE_FOCAL = True
    cfg.MODEL.DiffusionDet.USE_FED_LOSS = False
    cfg.MODEL.DiffusionDet.ALPHA = 0.25
    cfg.MODEL.DiffusionDet.GAMMA = 2.0
    cfg.MODEL.DiffusionDet.PRIOR_PROB = 0.01

    # Dynamic K
    cfg.MODEL.DiffusionDet.OTA_K = 5

    # Diffusion
    cfg.MODEL.DiffusionDet.SNR_SCALE = 2.0
    cfg.MODEL.DiffusionDet.SAMPLE_STEP = 1

    # Inference
    cfg.MODEL.DiffusionDet.USE_NMS = True

    # ================= ⭐⭐⭐ MAMBA（必须新增）⭐⭐⭐ =================
    cfg.MODEL.MAMBA = CN()
    cfg.MODEL.MAMBA.ENABLED = False

    cfg.MODEL.MAMBA.USE_RES4 = True
    cfg.MODEL.MAMBA.USE_RES5 = True

    cfg.MODEL.MAMBA.RES4_DEPTH = 6
    cfg.MODEL.MAMBA.RES4_NUM_HEADS = 8
    cfg.MODEL.MAMBA.RES4_WINDOW_SIZE = 14
    cfg.MODEL.MAMBA.RES4_TRANS_BLOCKS = [3, 4, 5]

    cfg.MODEL.MAMBA.RES5_DEPTH = 3
    cfg.MODEL.MAMBA.RES5_NUM_HEADS = 16
    cfg.MODEL.MAMBA.RES5_WINDOW_SIZE = 7
    cfg.MODEL.MAMBA.RES5_TRANS_BLOCKS = [2]

    # ================= Swin Backbones =================
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.SIZE = 'B'
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.OUT_FEATURES = (0, 1, 2, 3)

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # TTA.
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
    cfg.TEST.AUG.CVPODS_TTA = True
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = (
        [96, 10000], [96, 10000],
        [64, 10000], [64, 10000],
        [64, 10000], [0, 10000],
        [0, 10000], [0, 256],
        [0, 256], [0, 192],
        [0, 192], [0, 96],
        [0, 10000]
    )
        # ================= TEXT =================
    cfg.TEXT = CN()
    cfg.TEXT.VOCAB_SIZE = 30522          # 词表大小（你自己的 vocab 会覆盖）
    cfg.TEXT.MAX_LEN = 128               # 每条文本最大长度
    cfg.TEXT.HIDDEN_DIM = 256            # 对齐 DiffusionDet hidden_dim
    cfg.TEXT.NUM_LAYERS = 6
    cfg.TEXT.NUM_HEADS = 8
    cfg.TEXT.DROPOUT = 0.1

    # ⭐ 你自己生成的医学词表路径
    cfg.TEXT.VOCAB_PATH = str(Path(__file__).resolve().parents[1] / "medical_vocab.pkl")
