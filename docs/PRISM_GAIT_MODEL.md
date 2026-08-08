# PRISM-Gait Model Architecture

## Overview

PRISM-Gait is a closed-set gait-identification model that converts a short
pelvis or lower-back inertial measurement unit (IMU) recording into a
participant-specific gait signature.

The selected model combines three complementary feature families:

1. a compressed multibranch MiniROCKET temporal representation;
2. competitive dilated-kernel features (CDKC);
3. a harmonic-covariance gait fingerprint (HCGF).

The feature blocks are concatenated into a 13,049-dimensional signature. A
cosine-normalized, scaled ridge classifier then produces one decision score for
each enrolled identity.

PRISM-Gait is a classical feature-fusion pipeline, not an end-to-end neural
network.

## Architecture Summary

```text
Six-channel pelvis/back IMU window
                |
                v
 Segmentation + movement gate + z-score
                |
                v
  +-------------+--------------+
  |             |              |
  v             v              v
MiniROCKET     CDKC           HCGF
 7,812        4,608            629
 features     features       features
  |             |              |
  +-------------+--------------+
                |
                v
      Feature concatenation
      13,049-d gait signature
                |
                v
 L2 normalization + feature scaling
                |
                v
   Cross-validated ridge classifier
                |
                v
 One decision score per enrolled identity
                |
                v
       Predicted participant ID
```

In mathematical form:

$$
\mathbf{X}\in\mathbb{R}^{6\times200}
\rightarrow \text{preprocessing}
\rightarrow
\begin{cases}
\text{Multibranch MiniROCKET} & 7{,}812\\
\text{CDKC} & 4{,}608\\
\text{HCGF} & 629
\end{cases}
\rightarrow
\mathbf{z}\in\mathbb{R}^{13{,}049}
\rightarrow
\text{Ridge CV}
\rightarrow
\mathbf{s}\in\mathbb{R}^{K}.
$$

Here, \(K\) is the number of identities enrolled in the current cohort.

## Model Input

### Signal channels

The primary treadmill model receives six synchronized pelvis or lower-back IMU
channels:

| Channel | Signal | Unit |
|---|---|---|
| 1 | Accelerometer x-axis, \(a_x\) | m/s² |
| 2 | Accelerometer y-axis, \(a_y\) | m/s² |
| 3 | Accelerometer z-axis, \(a_z\) | m/s² |
| 4 | Gyroscope x-axis, \(g_x\) | degrees/s |
| 5 | Gyroscope y-axis, \(g_y\) | degrees/s |
| 6 | Gyroscope z-axis, \(g_z\) | degrees/s |

### Input tensor

For the primary selected configuration:

| Property | Value |
|---|---:|
| Sampling frequency | 100 Hz |
| Window duration | 2 s |
| Samples per window | 200 |
| Channels | 6 |
| Input shape per window | \(6\times200\) |
| Window overlap | 50% |
| Window hop | 1 s / 100 samples |

For a batch of \(N\) windows, the implementation uses:

$$
\mathbf{X}_{\mathrm{batch}}\in\mathbb{R}^{N\times6\times200}.
$$

## Preprocessing

### 1. Window segmentation

Each continuous IMU recording is divided into 2-second windows with 50%
overlap.

### 2. Movement-quality gate

Acceleration vector magnitude is calculated as:

$$
\operatorname{SVM}(t)
=
\sqrt{a_x(t)^2+a_y(t)^2+a_z(t)^2}.
$$

A window is retained when:

$$
\operatorname{SD}(\operatorname{SVM})\geq0.4.
$$

This removes windows with insufficient walking-related acceleration variation.

### 3. Channel-wise normalization

Every retained window is independently z-score normalized along time:

$$
\widetilde{x}_c(t)
=
\frac{x_c(t)-\mu_c}{\sigma_c+\epsilon},
$$

where \(\mu_c\) and \(\sigma_c\) are the within-window mean and standard
deviation of channel \(c\).

## Feature Extraction

### 1. Compressed Multibranch MiniROCKET Core

The normalized window is represented at three temporal orders:

$$
\mathbf{X},\qquad
\Delta\mathbf{X}_t=\mathbf{X}_t-\mathbf{X}_{t-1},\qquad
\Delta^2\mathbf{X}_t=\Delta\mathbf{X}_t-\Delta\mathbf{X}_{t-1}.
$$

Each representation is passed through a separate MiniROCKET transformer.

| Branch | Requested kernels | Effective feature dimension | Main information represented |
|---|---:|---:|---|
| Raw signal, \(\mathbf{X}\) | 3,000 | 2,940 | Waveform morphology and recurring temporal motifs |
| First difference, \(\Delta\mathbf{X}\) | 3,000 | 2,940 | Slopes, transitions and local dynamics |
| Second difference, \(\Delta^2\mathbf{X}\) | 2,000 | 1,932 | Curvature, impacts and rapid transients |
| **Total** | **8,000** | **7,812** | Multiscale temporal representation |

MiniROCKET summarizes convolution responses primarily through
proportion-of-positive-values (PPV) features. The effective number of output
features is slightly smaller than the requested number of kernels because of
MiniROCKET's internal parameter grouping.

The complete temporal block is:

$$
\mathbf{z}_{\mathrm{temporal}}\in\mathbb{R}^{7{,}812}.
$$

### 2. Competitive Dilated-Kernel Component

The manuscript name for this component is **CDKC**. Its legacy implementation
name is **PKD**, or Pelvis Kernel Duel.

CDKC applies groups of random multichannel convolution kernels at four temporal
scales:

| Parameter | Value |
|---|---:|
| Kernel groups | 48 |
| Kernels per group | 6 |
| Kernel length | 9 samples |
| Dilations | 1, 2, 4 and 8 |
| Effective receptive fields | 9, 17, 33 and 65 samples |

At every time point, the six kernels within a group compete to produce the
largest response. The competition is repeated using both the positive and
sign-inverted convolution responses.

For every kernel, CDKC retains:

- winner frequency: the proportion of time points at which the kernel wins;
- peak response: the largest activation produced by that kernel.

The feature dimension is:

$$
48
\times6
\times4
\times2
\times2
=4{,}608,
$$

corresponding to groups, kernels per group, dilations, response signs and
summary statistics, respectively.

Therefore:

$$
\mathbf{z}_{\mathrm{CDKC}}\in\mathbb{R}^{4{,}608}.
$$

CDKC should be described as representing multiscale local gait-shape patterns.
It does not explicitly estimate gait phase.

### 3. Harmonic-Covariance Gait Fingerprint

The manuscript name for this component is **HCGF**. Its legacy implementation
name is **PHCF**, or the pelvis harmonic-covariance feature block.

HCGF contains deterministic signal descriptors:

| HCGF component | Description | Dimensions |
|---|---|---:|
| Morphology | Mean, SD, RMS, quantiles, range, slope, skewness, kurtosis, signal variation and autocorrelations | 113 |
| Spectral and harmonic | Gait-band spectrum, spectral centroid, entropy, dominant peaks, band energies and relative phase | 246 |
| Raw correlation | Upper triangle of the \(6\times6\) channel-correlation matrix | 21 |
| Raw + first-difference correlation | Upper triangle of a \(12\times12\) correlation matrix | 78 |
| Log covariance | Matrix-log representation of the \(18\times18\) raw/difference/second-difference covariance matrix | 171 |
| **Total** | Harmonic-covariance gait fingerprint | **629** |

Thus:

$$
\mathbf{z}_{\mathrm{HCGF}}\in\mathbb{R}^{629}.
$$

This block represents gait periodicity, inter-axis coordination, cross-channel
coupling and covariance geometry.

## Feature Fusion

The three feature blocks are concatenated directly:

$$
\mathbf{z}
=
\left[
\mathbf{z}_{\mathrm{temporal}},
\mathbf{z}_{\mathrm{CDKC}},
\mathbf{z}_{\mathrm{HCGF}}
\right].
$$

The resulting dimension is:

$$
7{,}812+4{,}608+629=13{,}049.
$$

The result should be described as a **13,049-dimensional feature signature**.
It is not a learned neural embedding.

## Classifier

The fused feature vector undergoes the following operations:

### 1. L2 normalization

$$
\mathbf{z}_{\mathrm{norm}}
=
\frac{\mathbf{z}}{\lVert\mathbf{z}\rVert_2}.
$$

This is called cosine normalization in the project because it makes vector
direction more important than the original feature magnitude.

### 2. Feature scaling

Features are scaled using `StandardScaler(with_mean=False)`.

### 3. Ridge classification

The final classifier is `RidgeClassifierCV`. Its regularization parameter is
selected from:

$$
\alpha\in\{0.01,\;0.1,\;1,\;10,\;100\}.
$$

The supervised ridge classifier is the final identity-discriminative learning
stage. The preceding CDKC and HCGF descriptors are not trained using gradient
descent.

## Model Outputs

### Window-level scores

For every input window, the ridge classifier produces:

$$
\mathbf{s}=
[s_1,s_2,\ldots,s_K],
$$

where \(s_k\) is the decision score for enrolled identity \(k\).

These values are uncalibrated decision scores, not probabilities.

### Window-level identity

The predicted identity is:

$$
\widehat{y}_{\mathrm{window}}
=
\underset{k}{\operatorname{argmax}}\;s_k.
$$

### Recording-level identity

Predictions from several overlapping windows can be aggregated to obtain a
single recording- or task-level identity.

- The primary treadmill evaluation uses majority voting across window-level
  predictions.
- Some FoG-STAR task analyses average the window decision-score vectors before
  selecting the highest-scoring identity.

Aggregation should therefore be shown as an optional downstream stage rather
than as part of the feature extractor.

## Training and Enrollment

For the primary treadmill protocol:

1. baseline walking windows form the enrollment/training data;
2. every training window is labeled with the corresponding participant ID;
3. MiniROCKET parameters and the ridge classifier are fitted using these
   labeled enrollment windows;
4. altered walking conditions are kept separate for evaluation.

The same feature architecture can be fitted in another cohort, but each cohort
requires its own identity enrollment. The current evidence does not represent
one globally frozen classifier containing identities from every dataset.

## Closed-Set Limitation

Core PRISM-Gait is a closed-set identification model. It selects among
identities seen during enrollment and does not inherently produce an
`unknown person` output.

Open-set operation requires an additional calibrated acceptance/rejection
threshold or external detector. Such a detector should not be drawn as part of
the core PRISM-Gait architecture unless the figure specifically describes the
open-set extension.

## Recommended Scientific Figure Structure

A clear left-to-right architecture figure can contain the following blocks.

### Block 1: Input

```text
Pelvis/lower-back IMU
ax, ay, az, gx, gy, gz
6 x 200 samples
2 s at 100 Hz
```

### Block 2: Preprocessing

```text
50% overlapping segmentation
Acceleration-SVM movement gate
Channel-wise z-score normalization
```

### Block 3: Parallel feature extraction

```text
Raw MiniROCKET                    2,940
First-difference MiniROCKET       2,940
Second-difference MiniROCKET      1,932
CDKC                              4,608
HCGF                                629
```

### Block 4: Fusion

```text
Feature concatenation
13,049-dimensional gait signature
```

### Block 5: Classifier

```text
L2 normalization
Feature scaling
RidgeClassifierCV
```

### Block 6: Output

```text
K identity decision scores
Argmax -> window-level participant ID
Optional recording-level aggregation
```

## Suggested Figure Caption

> **PRISM-Gait architecture.** Six-channel pelvis or lower-back accelerometer
> and gyroscope signals are segmented into overlapping 2-s windows, screened
> using an acceleration-magnitude movement gate, and normalized independently
> by channel. Three MiniROCKET branches encode the raw signal and its first and
> second temporal differences. These temporal features are fused with
> competitive dilated-kernel descriptors (CDKC) and a harmonic-covariance gait
> fingerprint (HCGF), producing a 13,049-dimensional feature signature.
> Following L2 normalization and feature scaling, a cross-validated multiclass
> ridge classifier produces one decision score per enrolled identity.
> Window-level scores or predictions are subsequently aggregated to obtain the
> recording-level identity.

## Interpretation and Claim Boundary

The model was designed to combine temporal, multiscale local-shape, harmonic
and cross-channel information. Experimental results support improved
identification robustness under controlled trunk-sway perturbations.

The current analyses do not prove that one particular harmonic, orientation,
phase or covariance mechanism causally produces the improvement. The model
should therefore be described as demonstrating improved robustness under trunk
sway, rather than as being invariant to every sensor-placement or gait
perturbation.

Synthetic orientation perturbations used elsewhere in the project should be
called **synthetic orientation perturbations**. They are not equivalent to a
real donning/doffing experiment.

The repository does not currently give a formal expansion of the acronym
`PRISM`. Its full form should not be invented unless it is explicitly defined
in the manuscript.

## Implementation References

- Input channels, windowing and normalization:
  `ablation/pelvis_harmonic_covariance_model.py`
- Multibranch MiniROCKET:
  `ablation/pelvis_mblite_fusion.py`
- CDKC/legacy PKD:
  `ablation/pelvis_kernel_duel_model.py`
- HCGF/legacy PHCF:
  `ablation/pelvis_harmonic_covariance_model.py`
- Feature fusion and classifier:
  `ablation/mblite_verification_suite.py`
- Existing architecture-figure source:
  `figs/mblite_architecture_figure.py`
- Existing vector architecture figure:
  `fig_prism_gait_architecture.pdf`
