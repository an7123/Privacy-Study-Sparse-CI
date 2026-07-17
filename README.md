# Sparse collaborative inference: positions vs values leakage (minimal)

A top-$k$ sparse activation transmits two things — the retained **values** and the **positions**
(the support) that locate them. This ~150-line script isolates each and trains a learned inverse to
reconstruct the input from it, at a ResNet-18 split at `layer2` on TinyImageNet.

**Finding:** the *positions* carry nearly all the leakage. Positions-only (a binary mask, all values
discarded) reconstructs almost as well as the full sparse code, while the values scattered onto random
positions collapse to the dataset floor.

Representative output (TinyImageNet, 95% sparsity, learned inverse):

```
probe                 rate(bits/dim)      SSIM
dense                         16.000     0.559
topk_values                    1.086     0.444
positions_only                 0.286     0.433
values_on_random               1.086     0.142
```

## Setup

```bash
pip install -r requirements.txt
```

You need TinyImageNet-200 in the standard layout and a ResNet-18 (200-class) checkpoint trained on it:

```
data/tiny-imagenet-200/train/<wnid>/images/*.JPEG
data/tiny-imagenet-200/val/{images/*.JPEG, val_annotations.txt}
data/tiny_imagenet_resnet18.pth      # standard torchvision resnet18 state_dict, fc -> 200 classes
```

## Run

```bash
python sparse_leakage.py --data ./data/tiny-imagenet-200 --ckpt ./data/tiny_imagenet_resnet18.pth
```

Runs in a few minutes per probe on one GPU. Knobs: `--rho` (keep fraction; 0.05 = 95% sparsity),
`--epochs`, `--n-eval`, `--device`.

## What it computes

- **split model** — ResNet-18 client = stem+layer1+layer2 (post-ReLU); server frozen (unused here).
- **probes** — `dense`, `topk_values`, `positions_only` (binary mask), `values_on_random`.
- **learned inverse** — an MSE convolutional decoder trained on the training split to map the
  transmitted object back to the image.
- **rate** — analytical $R = H_2(\rho) + \rho b$ bits per activation dimension ($b=16$ for FP16 values,
  $b=0$ for the binary mask).

This is a stripped-down core. The full study (white-box vs learned attackers, FaceScrub biometric
re-identification with an identity-disjoint control, matched-utility Pareto, spatial-vs-channel
decomposition, capacity-matched permutation-invariant payload attackers, structured sparsity, and a
defense sweep) lives in the companion repository.
