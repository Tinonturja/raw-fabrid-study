"""Protocol 5 detectors (PyTorch). See PROTOCOL_5_AMENDMENT_1.md, section A4.

Two feature families, three scoring pipelines:
  WRN-50 features (identical to strong_patchcore.Native, wrn50) -> 'patchcore' and 'gauss' scorers
  DINOv2 ViT-S/14 patch tokens -> 'dino' scorer
Every map is restored to the input's own height and width.
"""
import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

SEED = 13017
SIDE = 256
PATCH = 14
NORM = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


def gauss_kernel(sigma=4.0):
    r = int(4 * sigma)
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


KER = gauss_kernel()


def smooth(m):
    """(n, H, W) -> Gaussian sigma 4 with reflect padding, the same as Protocols 1 to 4."""
    k = KER.to(m.device)
    r = len(k) // 2
    m = m[:, None]
    m = F.conv2d(F.pad(m, (r, r, 0, 0), mode='reflect'), k.view(1, 1, 1, -1))
    m = F.conv2d(F.pad(m, (0, 0, r, r), mode='reflect'), k.view(1, 1, -1, 1))
    return m[:, 0]


def restore_cnn(scores, h, w):
    """(n, fh, fw) lattice scores -> (n, h, w): bilinear to the input size, then smoothing.
    For a 256 x 256 tile this is exactly strong_patchcore.smooth_up."""
    m = F.interpolate(scores[:, None], size=(h, w), mode='bilinear', align_corners=False)[:, 0]
    return smooth(m)


def restore_dino(scores, h, w):
    """(n, hp/14, wp/14) token scores -> (n, h, w). Amendment 1, A4: upsample to the PADDED size,
    then crop to the original height and width, then smooth. No boundary token is dropped."""
    hp, wp = scores.shape[-2] * PATCH, scores.shape[-1] * PATCH
    m = F.interpolate(scores[:, None], size=(hp, wp), mode='bilinear', align_corners=False)[:, 0]
    return smooth(m[:, :h, :w].contiguous())


def pad14(x):
    """Reflect-pad (n, 3, H, W) on the bottom and right to multiples of 14."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % PATCH, (-w) % PATCH
    return F.pad(x, (0, pw, 0, ph), mode='reflect') if (ph or pw) else x


class WRN:
    """WRN-50 layers 2+3, 3x3 average, layer-3 channel pairs averaged: 1024-d (strong_patchcore.Native)."""
    name = 'wrn50'

    def __init__(self, device):
        self.net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).eval().to(device)
        self.device = device

    def features(self, x):
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x.to(self.device)))))
        l2 = n.layer2(n.layer1(x))
        l3 = n.layer3(l2)
        l2 = F.avg_pool2d(l2, 3, 1, 1)
        l3 = F.avg_pool2d(l3, 3, 1, 1)
        l3 = l3.reshape(l3.shape[0], -1, 2, *l3.shape[-2:]).mean(2)
        l3 = F.interpolate(l3, size=l2.shape[-2:], mode='bilinear', align_corners=False)
        f = torch.cat([l2, l3], 1)
        return f.permute(0, 2, 3, 1)                       # (n, fh, fw, 1024)

    def valid(self, fh, fw, h, w):
        return torch.ones(fh, fw, dtype=torch.bool)

    def restore(self, s, h, w):
        return restore_cnn(s, h, w)


class DINO:
    """dinov2_vits14 last-layer patch tokens, L2-normalised."""
    name = 'dino'

    def __init__(self, device):
        self.net = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().to(device)
        self.device = device

    def features(self, x):
        x = pad14(x.to(self.device))
        hp, wp = x.shape[-2] // PATCH, x.shape[-1] // PATCH
        t = self.net.forward_features(x)['x_norm_patchtokens']
        t = F.normalize(t, dim=-1)
        return t.reshape(x.shape[0], hp, wp, -1)           # (n, hp, wp, 384)

    def valid(self, fh, fw, h, w):
        """Tokens whose 14 x 14 patch lies entirely inside the real image (bank only)."""
        rows = (torch.arange(fh) + 1) * PATCH <= h
        cols = (torch.arange(fw) + 1) * PATCH <= w
        return rows[:, None] & cols[None, :]

    def restore(self, s, h, w):
        return restore_dino(s, h, w)


def to_tensor(images):
    return torch.stack([NORM(im.convert('RGB')) for im in images])


@torch.inference_mode()
def sample_candidates(fam, tiles, total=128000, max_tiles=1000, batch=16, log=print):
    """Normal training tiles -> candidate vectors. Uses the same RNG procedure as strong_patchcore.Native.fit:
    one default_rng(SEED), per-tile choice without replacement, in tile order. If there are more than
    max_tiles tiles, max_tiles are drawn first with a separate RNG (SEED + 1)."""
    if len(tiles) > max_tiles:
        pick = np.sort(np.random.default_rng(SEED + 1).choice(len(tiles), max_tiles, replace=False))
        tiles = [tiles[i] for i in pick]
    per_tile = math.ceil(total / len(tiles))
    rng = np.random.default_rng(SEED)
    out = []
    for s in range(0, len(tiles), batch):
        imgs = [t() if callable(t) else t for t in tiles[s:s + batch]]
        f = fam.features(to_tensor(imgs))
        b, fh, fw, d = f.shape
        ok = torch.nonzero(fam.valid(fh, fw, SIDE, SIDE).ravel()).flatten().numpy()
        f = f.reshape(b, fh * fw, d)
        for i in range(b):
            idx = rng.choice(ok, min(per_tile, len(ok)), replace=False)
            out.append(f[i, torch.from_numpy(idx).to(f.device)].half().cpu())
        if s % 320 == 0:
            log(f'{fam.name} candidates {s}/{len(tiles)} tiles')
    X = torch.cat(out)
    return X, {'tiles': len(tiles), 'per_tile': per_tile, 'candidates': len(X), 'dim': int(X.shape[1])}


@torch.inference_mode()
def greedy_coreset(X, k, device, log=print):
    """Identical to strong_patchcore.Native.fit: 128-d Gaussian projection, farthest-point selection."""
    rng = np.random.default_rng(SEED)
    n, dim = len(X), int(X.shape[1])
    g = torch.Generator().manual_seed(SEED)
    proj = torch.randn(dim, 128, generator=g) / np.sqrt(128)
    P = torch.cat([(X[i:i + 32768].float() @ proj) for i in range(0, n, 32768)]).to(device)
    start = int(rng.integers(n))
    sel = torch.empty(k, dtype=torch.long, device=device)
    sel[0] = start
    mind = ((P - P[start]) ** 2).sum(1)
    for it in range(1, k):
        if it % 2000 == 0:
            log(f'coreset {it}/{k}')
        i = torch.argmax(mind)
        sel[it] = i
        mind = torch.minimum(mind, ((P - P[i]) ** 2).sum(1))
    sel = sel.cpu()
    assert len(set(sel.tolist())) == k
    return X[sel].float()


def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class PatchCore:
    name = 'patchcore'

    def __init__(self, bank, device):
        self.bank = bank.to(device)
        self.bank_sq = self.bank.square().sum(1)

    def __call__(self, q):
        out = []
        for j in range(0, len(q), 2048):
            qq = q[j:j + 2048].float()
            d = qq.square().sum(1, keepdim=True) + self.bank_sq - 2 * (qq @ self.bank.T)
            out.append(d.amin(1).clamp_min(0).sqrt())
        return torch.cat(out)


class SharedGauss:
    """Shared-Gaussian Mahalanobis baseline: one Gaussian over all sampled normal vectors."""
    name = 'gauss'

    def __init__(self, mu, prec, n, device, reg=0.01):
        self.mu = mu.float().to(device)
        self.prec = prec.float().to(device)
        self.info = {'reg': reg, 'n': n, 'mu_sha256': digest(self.mu), 'prec_sha256': digest(self.prec)}

    @staticmethod
    def fit(X, reg=0.01):
        """Mean and regularised precision over all sampled candidates (float64 on CPU)."""
        Xd = X.double()
        mu = Xd.mean(0)
        cov = torch.cov((Xd - mu).T) + reg * torch.eye(X.shape[1], dtype=torch.float64)
        return mu.float(), torch.linalg.inv(cov).float()

    def __call__(self, q):
        out = []
        for j in range(0, len(q), 4096):
            z = q[j:j + 4096].float() - self.mu
            out.append(((z @ self.prec) * z).sum(1).clamp_min(0).sqrt())
        return torch.cat(out)


class DinoKNN:
    name = 'dino'

    def __init__(self, bank, device):
        self.bank = F.normalize(bank.float(), dim=-1).to(device)

    def __call__(self, q):
        out = []
        for j in range(0, len(q), 2048):
            out.append(1 - (q[j:j + 2048].float() @ self.bank.T).amax(1))
        return torch.cat(out)


@torch.inference_mode()
def score_maps(fam, scorers, images):
    """Images of one shared size -> {scorer: (n, h, w) numpy maps at that size}."""
    x = to_tensor(images)
    h, w = x.shape[-2:]
    f = fam.features(x)
    n, fh, fw, d = f.shape
    q = f.reshape(-1, d)
    out = {}
    for sc in scorers:
        s = sc(q).reshape(n, fh, fw)
        out[sc.name] = fam.restore(s, h, w).cpu().numpy()
    return out
