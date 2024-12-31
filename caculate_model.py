""" An example to use fvcore to calculate FLOPs, parameters and throughput of a model."""
'''
please install the following packages:
`pip install timm fvcore`

Example command:
python count.py --model RPKAT

code reference to: https://github.com/KATe-AGI/RPKAT/caculate_model.py
'''

import argparse
import torch
from timm.models import create_model
from fvcore.nn import FlopCountAnalysis, flop_count_table
from model import build

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Calculate FLOPs, parameters and throughput of a model")
    parser.add_argument(
        '--model',
        type=str,
        default='RPKAT',
        # choices=['RPKAT'],
        help='Model name')
    parser.add_argument(
        '--img_size',
        type=int,
        default=224,
        metavar='N',
        help='Input image size for the model (default: 224)'
    )
    return parser.parse_args()

def calculate_throughput(model, input_tensor):
    """Calculate the throughput of the model."""
    args = parse_args()
    _ = model(input_tensor[:2, ...])  # Dummy forward pass to initialize model

    batch_size = 100
    repetitions = 100
    _, c, h, w = input_tensor.shape
    input_tensor = torch.rand(batch_size, c, h, w).to("cuda")
    model = model.to("cuda")
    
    total_time = 0
    with torch.no_grad():
        for _ in range(repetitions):
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record()
            _ = model(input_tensor)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event) / 1000  # Seconds
            total_time += elapsed_time
    
    throughput = (repetitions * batch_size) / total_time
    print(f"Throughput of {args.model}: {throughput:.0f} images/s")

def main(args):
    """Main function to calculate FLOPs, parameters and throughput of a model."""
    # args = parse_args()

    model = create_model(args.model, num_classes=1000).to("cuda")
    model.eval()

    input_tensor = torch.rand(1, 3, args.img_size, args.img_size).to("cuda")

    # Calculate FLOPs (MACs) (G).
    flop = FlopCountAnalysis(model, input_tensor)
    print(flop_count_table(flop, max_depth=4))
    print(f"MACs (G) of {args.model}: {flop.total() / 1e9:.1f} G")

    # Calculate parameters (M)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters of {args.model}: {total_params / 1e6:.1f} M")

    # Calculate throughput (images/s)
    calculate_throughput(model, input_tensor)

if __name__ == '__main__':
    args = parse_args()
    main(args)