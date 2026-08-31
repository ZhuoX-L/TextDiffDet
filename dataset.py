import os
from detectron2.data.datasets import register_coco_instances

DATA_ROOT = os.environ.get("TEXTDIFFDET_DATA_ROOT", "datasets/limbfrac_ct")


register_coco_instances(
    "bone_train",
    {},
    os.path.join(DATA_ROOT, "train", "annotations.json"),
    os.path.join(DATA_ROOT, "train", "images")
)


register_coco_instances(
    "bone_val",
    {},
    os.path.join(DATA_ROOT, "val", "annotations.json"),
    os.path.join(DATA_ROOT, "val", "images")
)
