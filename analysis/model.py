"""GaitPrint features and random-kernel comparison models."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import importlib.util
from pathlib import Path
import subprocess
import sys
from sklearn.preprocessing import Normalizer, StandardScaler
from sktime.transformations.panel.rocket import MiniRocketMultivariate, MultiRocketMultivariate


FS_HZ = 100


def vectorize_upper(mat):
    iu = np.triu_indices(mat.shape[0])
    return mat[iu]


def safe_corrcoef(x):
    corr = np.corrcoef(x)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def log_cov_features(x):
    cov = np.cov(x) + np.eye(x.shape[0], dtype=np.float64) * 1e-4
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-8, None)
    log_cov = (vecs * np.log(vals)) @ vecs.T
    return vectorize_upper(log_cov)


def spectral_features(x):
    feats = []
    freqs = np.fft.rfftfreq(x.shape[1], 1.0 / FS_HZ)
    spec = np.fft.rfft(x * np.hanning(x.shape[1]), axis=1)
    amp = np.abs(spec)
    power = amp ** 2
    harmonic_mask = (freqs >= 0.5) & (freqs <= 8.0)
    bands = [(0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 8.0), (8.0, 15.0), (15.0, 25.0)]
    for ch in range(x.shape[0]):
        p = power[ch] + 1e-12
        hp = p[harmonic_mask]
        hf = freqs[harmonic_mask]
        feats.extend(np.log1p(amp[ch, harmonic_mask]).tolist())
        feats.append(float((freqs * p).sum() / p.sum()))
        feats.append(float(-np.sum((p / p.sum()) * np.log(p / p.sum()))))
        if hp.size:
            top = np.argsort(hp)[-3:][::-1]
            for idx in top:
                feats.extend([float(hf[idx]), float(np.log1p(hp[idx]))])
        else:
            feats.extend([0.0] * 6)
        total = p.sum()
        for lo, hi in bands:
            mask = (freqs >= lo) & (freqs < hi)
            feats.append(float(np.log1p(p[mask].sum())))
            feats.append(float(p[mask].sum() / total))

    # Relative phase of each channel to pelvis acceleration magnitude's dominant gait bin.
    svm = np.sqrt((x[:3] ** 2).sum(axis=0))
    svm_spec = np.fft.rfft(svm * np.hanning(len(svm)))
    gait_mask = (freqs >= 0.5) & (freqs <= 3.0)
    if np.any(gait_mask):
        gait_bins = np.where(gait_mask)[0]
        dom_bin = gait_bins[np.argmax(np.abs(svm_spec[gait_bins]))]
        ref_phase = np.angle(svm_spec[dom_bin])
        for ch in range(x.shape[0]):
            delta = np.angle(spec[ch, dom_bin]) - ref_phase
            feats.extend([float(np.sin(delta)), float(np.cos(delta)), float(np.log1p(amp[ch, dom_bin]))])
    else:
        feats.extend([0.0] * (x.shape[0] * 3))
    return np.asarray(feats, dtype=np.float32)


def morphology_features(x):
    feats = []
    t = np.linspace(-1.0, 1.0, x.shape[1])
    for ch in range(x.shape[0]):
        sig = x[ch]
        feats.extend([
            float(sig.mean()),
            float(sig.std()),
            float(np.sqrt(np.mean(sig ** 2))),
            float(np.median(sig)),
            float(np.percentile(sig, 10)),
            float(np.percentile(sig, 25)),
            float(np.percentile(sig, 75)),
            float(np.percentile(sig, 90)),
            float(sig.max() - sig.min()),
            float(np.mean(np.abs(np.diff(sig)))),
            float(np.polyfit(t, sig, deg=1)[0]),
        ])
        centered = sig - sig.mean()
        sd = sig.std() + 1e-8
        feats.append(float(np.mean((centered / sd) ** 3)))
        feats.append(float(np.mean((centered / sd) ** 4)))
        for lag in (10, 20, 33, 50, 67):
            if lag < len(sig):
                a = centered[:-lag]
                b = centered[lag:]
                feats.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
            else:
                feats.append(0.0)
    svm = np.sqrt((x[:3] ** 2).sum(axis=0))
    for lag in (10, 20, 33, 50, 67):
        centered = svm - svm.mean()
        a = centered[:-lag]
        b = centered[lag:]
        feats.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    return np.asarray(feats, dtype=np.float32)


def phcf_features_one(win):
    raw = np.asarray(win, dtype=np.float32)
    d1 = np.diff(raw, axis=1, prepend=raw[:, :1])
    d2 = np.diff(d1, axis=1, prepend=d1[:, :1])
    aug = np.vstack([raw, d1, d2]).astype(np.float64)
    feats = [
        morphology_features(raw),
        spectral_features(raw),
        vectorize_upper(safe_corrcoef(raw)).astype(np.float32),
        vectorize_upper(safe_corrcoef(np.vstack([raw, d1]))).astype(np.float32),
        log_cov_features(aug).astype(np.float32),
    ]
    out = np.concatenate(feats)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def featurize(windows):
    return np.vstack([phcf_features_one(w) for w in windows]).astype(np.float32)


class PelvisKernelDuelFeatures:
    def __init__(self, groups=96, kernels_per_group=8, kernel_size=9, dilations=(1, 2, 4, 8), seed=42, batch_size=256):
        self.groups = groups
        self.kernels_per_group = kernels_per_group
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        self.seed = seed
        self.batch_size = batch_size
        self.weights = []

    def fit(self, x):
        rng = np.random.default_rng(self.seed)
        self.weights = []
        out_ch = self.groups * self.kernels_per_group
        for _ in self.dilations:
            w = rng.normal(0, 1, size=(out_ch, x.shape[1], self.kernel_size)).astype(np.float32)
            # Randomly drop some channels per kernel so kernels specialize.
            mask = rng.random(size=(out_ch, x.shape[1], 1)) < rng.uniform(0.35, 0.9, size=(out_ch, 1, 1))
            w *= mask.astype(np.float32)
            w -= w.mean(axis=(1, 2), keepdims=True)
            w /= np.linalg.norm(w.reshape(out_ch, -1), axis=1, keepdims=True).reshape(out_ch, 1, 1) + 1e-8
            self.weights.append(torch.from_numpy(w))
        return self

    def _features_batch(self, xb, device):
        feats = []
        xt = torch.from_numpy(xb).to(device)
        for dilation, weight_cpu in zip(self.dilations, self.weights):
            weight = weight_cpu.to(device)
            pad = dilation * (self.kernel_size // 2)
            resp = F.conv1d(xt, weight, padding=pad, dilation=dilation)
            resp = resp[:, :, : xt.shape[-1]]
            resp = resp.view(resp.shape[0], self.groups, self.kernels_per_group, resp.shape[-1])

            # Positive and negative competitions capture shape and inverse shape.
            for signed in (resp, -resp):
                winners = signed.argmax(dim=2)  # N, groups, time
                counts = []
                for k in range(self.kernels_per_group):
                    counts.append((winners == k).float().mean(dim=2))
                counts = torch.stack(counts, dim=2).reshape(resp.shape[0], -1)
                peaks = signed.amax(dim=3).reshape(resp.shape[0], -1)
                feats.extend([counts, peaks])
        return torch.cat(feats, dim=1).cpu().numpy().astype(np.float32)

    def transform(self, x):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        outs = []
        for start in range(0, len(x), self.batch_size):
            outs.append(self._features_batch(x[start : start + self.batch_size], device))
        return np.vstack(outs)


HYDRA_COMMIT = '144bb7aa3186654042f00f36b078370c620acb06'


def load_hydra_source(root: Path):
    head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],text=True).strip()
    if head != HYDRA_COMMIT or dirty:
        raise ValueError('HYDRA source must be clean at the recorded author commit')
    modules=[]
    for name in ['hydra','hydra_multivariate']:
        module_name='gaitprint_author_'+name
        if module_name not in sys.modules:
            spec=importlib.util.spec_from_file_location(module_name,root/'code'/f'{name}.py')
            module=importlib.util.module_from_spec(spec)
            sys.modules[module_name]=module
            spec.loader.exec_module(module)
        modules.append(sys.modules[module_name])
    return modules


class FeatureExtractor:
    def __init__(self, name: str, seed: int, threads: int, hydra_root: Path):
        self.name,self.seed,self.threads,self.hydra_root=name,seed,threads,hydra_root
        self.transforms=[]
        self.scalers=[]

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        if self.name in ['hydra','multirocket_hydra']:
            author,multivariate=load_hydra_source(self.hydra_root)
            self.hydra=multivariate.HydraMultivariate(x.shape[-1],x.shape[1],k=8,g=64)
            self.sparse_scaler=author.SparseScaler()
        if self.name=='gaitprint':
            for order,kernels in enumerate([3000,3000,2000]):
                transform=MiniRocketMultivariate(num_kernels=kernels,random_state=self.seed+order,n_jobs=self.threads)
                transform.fit(self._view(x,order),y)
                self.transforms.append(transform)
            self.multiscale=PelvisKernelDuelFeatures(groups=48,kernels_per_group=6,seed=self.seed,batch_size=256).fit(x)
        elif self.name=='minirocket':
            self.transforms=[MiniRocketMultivariate(num_kernels=10000,random_state=self.seed,n_jobs=self.threads).fit(self._view(x,0),y)]
        elif self.name in ['multirocket','multirocket_hydra']:
            self.transforms=[MultiRocketMultivariate(num_kernels=6250,random_state=self.seed,n_jobs=self.threads).fit(self._view(x,0),y)]
        elif self.name!='hydra':
            raise ValueError(f'Unknown model {self.name}')
        raw=self._raw(x)
        if self.name=='gaitprint':
            self.normalizer=Normalizer()
            raw=[self.normalizer.fit_transform(raw[0])]
        scaled=[]
        for i,block in enumerate(raw):
            if self.name in ['hydra','multirocket_hydra'] and i==0:
                scaled.append(self.sparse_scaler.fit_transform(torch.from_numpy(block)).numpy())
                self.scalers.append(None)
            else:
                scaler=StandardScaler(with_mean=self.name!='gaitprint')
                scaled.append(scaler.fit_transform(block))
                self.scalers.append(scaler)
        return np.concatenate(scaled,axis=1)

    @staticmethod
    def _view(x,order):
        return np.ascontiguousarray(np.diff(x,n=order,axis=2),dtype=np.float64)

    def _raw(self,x):
        blocks=[]
        if self.name in ['hydra','multirocket_hydra']:
            with torch.no_grad():
                blocks.append(self.hydra.batch(torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32)),batch_size=128).numpy())
        if self.name=='gaitprint':
            blocks.extend(np.asarray(t.transform(self._view(x,i)),dtype=np.float32) for i,t in enumerate(self.transforms))
            blocks.append(self.multiscale.transform(np.asarray(x,dtype=np.float32)))
            blocks.append(featurize(x))
            return [np.concatenate(blocks,axis=1)]
        blocks.extend(np.asarray(t.transform(self._view(x,0)),dtype=np.float32) for t in self.transforms)
        return blocks

    def transform(self,x):
        raw=self._raw(x)
        if self.name=='gaitprint':
            raw=[self.normalizer.transform(raw[0])]
        scaled=[]
        for block,scaler in zip(raw,self.scalers):
            if scaler is None:
                scaled.append(self.sparse_scaler.transform(torch.from_numpy(block)).numpy())
            else:
                scaled.append(scaler.transform(block))
        return np.concatenate(scaled,axis=1)
