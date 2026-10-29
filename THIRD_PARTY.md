# Author implementations

Full training fetches the following sources into `external/` and verifies their exact Git revisions and clean tracked files before use. The saved-evidence analysis does not download or execute them.

| Method | Source | Pinned revision | Upstream license file |
| --- | --- | --- | --- |
| HYDRA and MultiROCKET–HYDRA | [angus924/hydra](https://github.com/angus924/hydra) | `144bb7aa3186654042f00f36b078370c620acb06` | GPL-3.0, `LICENSE` |
| InceptionTime | [hfawaz/InceptionTime](https://github.com/hfawaz/InceptionTime) | `952d115e83fb3a66d75c5858c702f8dd5eeb18b7` | GPL-3.0, `LICENSE` |
| Multi-Model LSTM | [halam189/Gait_Recognition_LSTM](https://github.com/halam189/Gait_Recognition_LSTM) | `c837c85a2b545cfa5387bea5cb9e560504ff3ad5` | No separate license file at this revision; consult upstream terms |

HYDRA's multivariate transform and sparse scaler are loaded from the pinned source. InceptionTime's two original architecture methods are loaded from its source with the BatchNormalization namespace updated for Keras 3; the ensemble, baseline-only epoch selection and final refit are handled by this repository. The LSTM adapter uses the author's six-channel architecture with ten submodels, hidden size 40, two layers and the native 200-sample input. Author training scripts and data loaders are not executed.

MiniROCKET and MultiROCKET use the implementations distributed in `sktime==0.39.0` ([BSD-3-Clause license](https://github.com/sktime/sktime/blob/v0.39.0/LICENSE)). All downloaded code retains its upstream terms; this repository's license does not replace those terms. No upstream repository is vendored into this release.
