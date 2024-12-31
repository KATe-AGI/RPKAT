# --------------------------------------------------------
# RPKAT Main (validate)
# Adapted from Swin Transformer and EfficientViT
#   Swin: (https://github.com/microsoft/swin-transformer)
#   EfficientViT: (https://github.com/microsoft/Cream/tree/main/EfficientViT)
# --------------------------------------------------------
import argparse
import torch
import torch.backends.cudnn as cudnn
import json
import utils

from pathlib import Path
from timm.models import create_model
from data.datasets import build_dataset
from core.engine import evaluate

from model import build

def get_args_parser():
    parser = argparse.ArgumentParser(
        'RPKAT validation script', add_help=False)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=1, type=int)

    # Model parameters
    parser.add_argument('--model', default='RPKAT', type=str, metavar='MODEL',
                        help='Name of model to train')    
    parser.add_argument('--input-size', default=224,
                        type=int, help='images input size')

    # Finetuning params
    parser.add_argument('--finetune', default='path/to/checkpoint.pth', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default=r"path/to/dataset/HFUT-VL1",
                        help='dataset path')
    parser.add_argument('--output_dir', default='./results',
                        help='path where to save, empty for no saving')
    parser.add_argument('--data-set', default='HFUT-VL1', choices=[
                                                                    'HFUT-VL1', 'HFUT-VL2', 'Compcars_L',  # VLR
                                                                    'CompCars', 'Frontal-103', 'SBOD',  # VMMR
                                                                    'Compcars_C', 'UFPR-VCR', 'SBOD_C'  # VCR
                                                                    ], 
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order',
                                 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', default=True,
                        help='Perform evaluation only')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    return parser

def main(args):
    device = torch.device(args.device)

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Load dataset
    dataset_val, args.nb_classes = build_dataset(is_train=False, args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False
    )
       
    print(f"data_path: {args.data_path}")
    
    # Create the model
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        num_classes=args.nb_classes,
        pretrained=False,
    )

    # Load checkpoint if finetuning
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = utils.load_model(args.finetune, model)

        model.load_state_dict(checkpoint['model'], strict=False)

    model.to(device)

    # Save model and args if output directory is provided
    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "model.txt").open("a") as f:
            f.write(str(model))
    if args.output_dir and utils.is_main_process():
        with (output_dir / "args.txt").open("a") as f:
            f.write(json.dumps(args.__dict__, indent=2) + "\n")

    # Calculate parameters (M)
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f'number of params:: {n_parameters / 1e6:.1f} M')
    
    # Validate
    if args.eval:
        print(f"Validating model: {args.model}")
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'RPKAT validation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
