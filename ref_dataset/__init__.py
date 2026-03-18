from .transform import get_transform as get_transform_seg
from .custom_binary_dataset import CustomBinaryDataset
from .refdataset import Refdataset, collate_fn

def build_dataset(is_train, args, split='val'):

    data_path = args.data_path
    if is_train:
        split = 'train'
    else:
        split = split
    if args.data_set == 'custom_binary':
        dataset = CustomBinaryDataset(
            data_root=data_path,
            split=split,
            image_transforms=get_transform_seg((args.input_size, args.input_size), train=is_train),
            caption_index=getattr(args, "caption_index", 2),
        )
    elif args.data_set == 'refcoco':
        dataset = Refdataset(refer_data_root=data_path, dataset='refcoco', splitBy='unc', split=split, image_transforms=get_transform_seg((args.input_size, args.input_size), train=is_train))
    elif args.data_set == 'refcoco+':
        dataset = Refdataset(refer_data_root=data_path, dataset='refcoco+', splitBy='unc', split=split, image_transforms=get_transform_seg((args.input_size, args.input_size), train=is_train))
    elif args.data_set == 'refcocog':
        dataset = Refdataset(refer_data_root=data_path, dataset='refcocog', splitBy='umd', split=split, image_transforms=get_transform_seg((args.input_size, args.input_size), train=is_train))
    else:
        raise

    return dataset
