# --------------------------------------------------------
# RPKAT Main (train)
# Adapted from Swin Transformer and EfficientViT
#   Swin: (https://github.com/microsoft/swin-transformer)
#   EfficientViT: (https://github.com/microsoft/Cream/tree/main/EfficientViT)
# --------------------------------------------------------
import os
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import utils

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from data.datasets import build_dataset
from data.threeaugment import new_data_aug_generator
from core.engine import train_one_epoch, evaluate
from core.losses import Loss

from model import build

def get_args_parser():
    parser = argparse.ArgumentParser(
        'RPKAT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=200, type=int)

    # Model parameters
    parser.add_argument('--model', default='RPKAT', type=str, metavar='MODEL',
                        help='Name of model to train')    
    parser.add_argument('--input-size', default=224,
                        type=int, help='images input size')
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float,
                        default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu',
                        action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=0.02, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='agc',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.025,
                        help='weight decay (default: 0.025)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=25, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--ThreeAugment', action='store_true', default=True)
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug',
                        action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--set_bn_eval', action='store_true', default=False,
                        help='set BN layers to eval mode during finetuning.')

    # Dataset parameters
    parser.add_argument('--data-path', default=r"path/to/dataset/HFUT-VL1",
                        help='dataset path')
    parser.add_argument('--output_dir', default='./runs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--data-set', default='HFUT-VL1', choices=[
                                                                    'HFUT-VL1', 'HFUT-VL2', 'Compcars_L',  # VLR
                                                                    'CompCars', 'Frontal-103', 'SBOD',  # VMMR
                                                                    'Compcars_C', 'UFPR-VCR', 'SBOD_C'  # VCR
                                                                    ], 
                        type=str, help='Image Net dataset path')
    parser.add_argument('--scale-factor', default=1.0, type=float,
                        help='Factor to scale the training dataset (default: 1.0)')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order',
                                 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')  
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Perform evaluation only')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--save_freq', default=80, type=int,
                        help='frequency of model saving')
    return parser

def main(args):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    print(f"data_path: {args.data_path}")
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        num_classes=args.nb_classes,
        pretrained=False,
        # fuse=False,
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = utils.load_model(args.finetune, model)

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.l.weight', 'head.l.bias',
                  'head_dist.l.weight', 'head_dist.l.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but
        # before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model

    # Calculate parameters (M)
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f'number of params:: {n_parameters / 1e6:.1f} M')

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    criterion = Loss(criterion)

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if args.output_dir and utils.is_main_process():
        for file_name, content in [("model.txt", str(model)), ("args.txt", json.dumps(args.__dict__, indent=2) + "\n")]:
            with (output_dir / file_name).open("a") as f:
                f.write(content)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            print(f"Loading local checkpoint at {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')
        msg = model_without_ddp.load_state_dict(checkpoint['model'])
        print(msg)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(
                    model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    start_time = time.time()
    max_accuracy = 0.0
    min_loss = float('inf')
    best_epoch = -1
    best_epoch_loss = -1

    best_ckpt_path = None
    best_loss_ckpt_path = None

    # Initialize the log list
    all_epochs_log_stats = []

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, args.clip_mode, model_ema, mixup_fn,
            set_training_mode=True, set_bn_eval=args.set_bn_eval,
        )

        # Update the learning rate scheduler
        lr_scheduler.step(epoch)
        
        # evaluate after every epoch
        test_stats = evaluate(data_loader_val, model, device)
        
        # Update the best accuracy model
        if test_stats["acc1"] > max_accuracy:
            max_accuracy = test_stats["acc1"]
            best_epoch = epoch
            # Only save the core weight parameters of the best model
            checkpoint_state = {
                'model': model.state_dict(),
            }
            new_best_ckpt_path = output_dir / 'checkpoint_best.pth'
            torch.save(checkpoint_state, new_best_ckpt_path)
            
            if best_ckpt_path and best_ckpt_path.exists():
                os.remove(best_ckpt_path)
            max_accuracy_fmt = f"{max_accuracy:.2f}"
            best_ckpt_path = output_dir / f'checkpoint_best_acc_{max_accuracy_fmt}_{best_epoch}.pth'
            os.rename(new_best_ckpt_path, best_ckpt_path)
            print(f"Best accuracy checkpoint saved to {best_ckpt_path}")

        # Update the best loss model
        if test_stats["loss"] < min_loss:
            min_loss = test_stats["loss"]
            best_epoch_loss = epoch
            # Only save the core weight parameters of the best model
            checkpoint_state = {
                'model': model.state_dict(),
            }
            new_best_loss_ckpt_path = output_dir / 'checkpoint_best_loss.pth'
            torch.save(checkpoint_state, new_best_loss_ckpt_path)
            
            if best_loss_ckpt_path and best_loss_ckpt_path.exists():
                os.remove(best_loss_ckpt_path)
            min_loss_fmt = f"{min_loss:.4f}"
            best_loss_ckpt_path = output_dir / f'checkpoint_best_loss_{min_loss_fmt}_{best_epoch_loss}.pth'
            os.rename(new_best_loss_ckpt_path, best_loss_ckpt_path)
            print(f"Best loss checkpoint saved to {best_loss_ckpt_path}")

        # Save model checkpoints at specified frequency
        if epoch > 0 and (epoch % args.save_freq == 0):
            checkpoint_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'model_ema': get_state_dict(model_ema),
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            ckpt_path = output_dir / f'checkpoint_{epoch}.pth'
            torch.save(checkpoint_state, ckpt_path)

        # Output the current training and validation status
        print(f'Accuracy of the network on the {len(data_loader_val.dataset)} test images: {test_stats["acc1"]:.2f}%, [Max Accuracy: {max_accuracy:.2f}%, Best Epoch: {best_epoch}], [Min Loss: {min_loss:.4f}, Best Epoch: {best_epoch_loss}]')

        # Update the log
        epoch_log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters,
        }
        all_epochs_log_stats.append(epoch_log_stats)

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(epoch_log_stats) + "\n")

    # Automatically save the last checkpoint
    checkpoint_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': args.epochs,
        'model_ema': get_state_dict(model_ema),
        'scaler': loss_scaler.state_dict(),
        'args': args,
    }
    latest_ckpt_path = output_dir / f'checkpoint_latest_{args.epochs}.pth'
    torch.save(checkpoint_state, latest_ckpt_path)

    print(f"Latest checkpoint saved to {latest_ckpt_path}")

    # Record total training time
    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
            f.write(json.dumps({"Training_time": total_time_str}) + "\n")

    print(f'Training completed. Total time: {total_time_str}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'RPKAT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
