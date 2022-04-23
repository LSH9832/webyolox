from .data_augment import TrainTransform, ValTransform
from .data_prefetcher import DataPrefetcher
from .dataloading import DataLoader, get_yolox_datadir, worker_init_reset_seed
from .datasets import *
from .samplers import InfiniteSampler, YoloBatchSampler


def get_dataloader(
        batch_size,
        data_dir,
        json_file,
        train_dir,
        input_size=(416, 416),
        no_aug=False,
        cache_img=False,
        flip_prob=0.5,
        hsv_prob=1,
        degrees=10.0,
        translate=0.1,
        mosaic_scale=(0.1, 2),
        mixup_scale = (0.5, 1.5),
        shear=2.0,
        enable_mixup=True,
        mosaic_prob=1.0,
        mixup_prob=1.0,
        data_num_workers = 4,
        seed=None
):

    dataset = COCODataset(
        data_dir=data_dir,
        json_file=json_file,
        img_size=input_size,
        train_dir=train_dir,
        preproc=TrainTransform(
            max_labels=50,
            flip_prob=flip_prob,
            hsv_prob=hsv_prob
        ),
        cache=cache_img,
    )

    dataset = MosaicDetection(
        dataset,
        mosaic=not no_aug,
        img_size=input_size,
        preproc=TrainTransform(
            max_labels=120,
            flip_prob=flip_prob,
            hsv_prob=hsv_prob),
        degrees=degrees,
        translate=translate,
        mosaic_scale=mosaic_scale,
        mixup_scale=mixup_scale,
        shear=shear,
        enable_mixup=enable_mixup,
        mosaic_prob=mosaic_prob,
        mixup_prob=mixup_prob,
    )

    sampler = InfiniteSampler(len(dataset), seed=seed if seed else 0)

    batch_sampler = YoloBatchSampler(
        sampler=sampler,
        batch_size=batch_size,
        drop_last=False,
        mosaic=not no_aug,
    )

    dataloader_kwargs = {
        "num_workers": data_num_workers,
        "pin_memory": True,
        "batch_sampler": batch_sampler,
        "worker_init_fn": worker_init_reset_seed
    }

    train_loader = DataLoader(dataset, **dataloader_kwargs)

    return train_loader, dataset
