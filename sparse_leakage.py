"""
Minimal reproduction: positions vs values leakage in sparse collaborative inference.

A top-k sparse activation transmits two things -- the retained VALUES and the POSITIONS that locate
them. This script isolates each and trains a learned inverse (an MSE decoder) to reconstruct the input
from it, at a ResNet-18 split at layer2 on TinyImageNet.

Probes:
  dense               : the full activation z
  topk_values         : z on its top-k support (values + positions)
  positions_only      : the binary top-k mask (positions, no values)
  values_on_random    : the top-k values scattered onto a random support (values, no positions)

Expected: positions_only ~ topk_values >> values_on_random ~ dataset floor.

Usage:
  python sparse_leakage.py --data ./data/tiny-imagenet-200 --ckpt ./data/tiny_imagenet_resnet18.pth
"""
import argparse, math, os, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def normalize(x, dev):
    m = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1); s = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    return (x - m) / s

# ----------------------------- split model -----------------------------
class Split(nn.Module):
    """ResNet-18 split at the OUTPUT of layer2 (post-ReLU). Client = stem+layer1+layer2; server frozen."""
    def __init__(self, ckpt, num_classes=200, dev='cuda'):
        super().__init__()
        net = models.resnet18(weights=None); net.fc = nn.Linear(512, num_classes)
        sd = torch.load(ckpt, map_location='cpu'); sd = sd.get('state_dict', sd)
        net.load_state_dict(sd)
        self.client = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2)
        self.server = nn.Sequential(net.layer3, net.layer4, net.avgpool, nn.Flatten(1), net.fc)
        self.eval().to(dev); [p.requires_grad_(False) for p in self.parameters()]; self.dev = dev

# ----------------------------- probes -----------------------------
def topk_mask(z, rho):
    B = z.shape[0]; k = max(1, int(rho * z[0].numel()))
    flat = z.view(B, -1); _, idx = torch.topk(flat.abs(), k, dim=1)
    m = torch.zeros_like(flat); m.scatter_(1, idx, 1.0)
    return m.view_as(z)

def probe(name, z, rho, gen):
    if name == 'dense':
        return z, 16.0
    if name == 'topk_values':
        return z * topk_mask(z, rho), 16.0
    if name == 'positions_only':
        return topk_mask(z, rho), 0.0
    if name == 'values_on_random':                       # values, positions destroyed
        B = z.shape[0]; flat = z.view(B, -1); N = flat.shape[1]; k = max(1, int(rho * N))
        vals, idx = torch.topk(flat.abs(), k, dim=1); v = torch.gather(flat, 1, idx)
        o = torch.zeros_like(flat)
        for b in range(B):
            o[b, torch.randperm(N, generator=gen, device=z.device)[:k]] = v[b]
        return o.view_as(z), 16.0
    raise ValueError(name)

def analytical_rate(rho, b):                             # H2(rho) + rho*b bits/dim
    hb = 0.0 if rho in (0, 1) else -rho*math.log2(rho) - (1-rho)*math.log2(1-rho)
    return (0 if b == 0 else hb) + rho * b if b else hb

# ----------------------------- learned inverse (MSE decoder) -----------------------------
class Decoder(nn.Module):
    def __init__(self, cin=128, base=256, n_up=3):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(cin, base, 3, 1, 1), nn.BatchNorm2d(base), nn.ReLU(True))
        t = []
        for _ in range(n_up):
            t += [nn.Upsample(scale_factor=2, mode='nearest'),
                  nn.Conv2d(base, base, 3, 1, 1), nn.BatchNorm2d(base), nn.ReLU(True)]
        t += [nn.Conv2d(base, 3, 3, 1, 1), nn.Sigmoid()]
        self.trunk = nn.Sequential(*t)
    def forward(self, o): return self.trunk(self.head(o))

def recon_ssim(recon, x):
    r, t = recon.clamp(0, 1).cpu().numpy(), x.clamp(0, 1).cpu().numpy()
    return float(np.mean([ssim(np.transpose(t[i], (1, 2, 0)), np.transpose(r[i], (1, 2, 0)),
                               channel_axis=2, data_range=1.0) for i in range(len(r))]))

# ----------------------------- data -----------------------------
class TinyTrain(Dataset):
    def __init__(self, root, size=64):
        wn = sorted(os.listdir(os.path.join(root, 'train'))); self.w2i = {w: i for i, w in enumerate(wn)}
        self.samples = [(os.path.join(root, 'train', w, 'images', f), self.w2i[w])
                        for w in wn for f in os.listdir(os.path.join(root, 'train', w, 'images'))]
        self.tf = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor()])
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): p, y = self.samples[i]; return self.tf(Image.open(p).convert('RGB')), y

def tiny_val_images(root, n, size=64, seed=12345):
    tf = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor()])
    lines = [l.split('\t')[:2] for l in open(os.path.join(root, 'val', 'val_annotations.txt'))]
    paths = [os.path.join(root, 'val', 'images', fn) for fn, _ in lines]
    set_seed(seed); idx = torch.randperm(len(paths))[:n].tolist()
    return torch.stack([tf(Image.open(paths[i]).convert('RGB')) for i in idx])

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--rho', type=float, default=0.05); ap.add_argument('--n-eval', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); dev = a.device

    split = Split(a.ckpt, dev=dev)
    train = DataLoader(TinyTrain(a.data), batch_size=256, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    eval_x = tiny_val_images(a.data, a.n_eval)

    print(f"{'probe':<20}{'rate(bits/dim)':>16}{'SSIM':>10}")
    for name in ['dense', 'topk_values', 'positions_only', 'values_on_random']:
        set_seed(a.seed); gen = torch.Generator(device=dev).manual_seed(a.seed)
        dec = Decoder().to(dev); opt = torch.optim.Adam(dec.parameters(), lr=2e-3)
        dec.train()
        for _ in range(a.epochs):
            for x, _ in train:
                x = x.to(dev)
                with torch.no_grad():
                    o, _ = probe(name, split.client(normalize(x, dev)), a.rho, gen)
                loss = F.mse_loss(dec(o), x)
                opt.zero_grad(); loss.backward(); opt.step()
        dec.eval()
        recons, bits = [], 16.0
        with torch.no_grad():
            for i in range(0, len(eval_x), 256):
                o, bits = probe(name, split.client(normalize(eval_x[i:i+256].to(dev), dev)), a.rho, gen)
                recons.append(dec(o).cpu())
        s = recon_ssim(torch.cat(recons), eval_x)
        rho = 1.0 if name == 'dense' else a.rho
        print(f"{name:<20}{analytical_rate(rho, bits):>16.3f}{s:>10.3f}")

if __name__ == '__main__':
    main()
