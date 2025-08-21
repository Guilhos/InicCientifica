import argparse
import torch
from torchdeq import get_deq, apply_norm, reset_norm
from torchdeq.utils import add_deq_args
from .layers import Injection, DEQFunc, Decoder
class DEQDemo(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.deq_func = DEQFunc(args)
        apply_norm(self.deq_func, args=args)
        self.deq = get_deq(args)
    def forward(self, x, z0):
        reset_norm(self.deq_func)
        f = lambda z: self.deq_func(z, x)
        return self.deq(f, z0)
def train(args, inj, deq, decoder, loader, loss, opt):
    for x, y in loader:
        z0 = torch.randn(args.z_shape)
        z_out, info = deq(inj(x), z0)
        l = loss(decoder(z_out[-1]), y)
        l.backward()
        opt.step()
        logger.info(f'Loss: {l.item()}, '
        +f'Rel: {info['rel_lowest'].mean().item()}'
        +f'Abs: {info['abs_lowest'].mean().item()}')
'''Add other arguments.'''
parser = argparse.ArgumentParser()
add_deq_args(parser)
args = parser.parse_args()
inj = Injection(args)
deq = DEQDemo(args)
decoder = Decoder(args)
''' Set up loader, logger, loss, opt, etc as in
standard PyTorch. '''
train(args, inj, deq, decoder, loader, loss, opt)