# Comparator architectures

These files adapt public model implementations to the same gait windows and
evaluation conditions used for PRISM-Gait.

| Runner name | Local adaptation | Public source |
|---|---|---|
| `rocket` | `rocket.py` (MiniROCKET and MultiROCKET through sktime) | [MiniROCKET](https://github.com/angus924/minirocket), [MultiROCKET](https://github.com/ChangWeiTan/MultiRocket) |
| `fcn-bilstm` | `fcn_bilstm.py` | [FCN-BiLSTM](https://github.com/utshabkg/gait_recognition) |
| `inceptiontime` | `inceptiontime.py` | [InceptionTime](https://github.com/hfawaz/InceptionTime) |
| `multi-model-lstm` | `multi_model_lstm.py` | [Multi-Model LSTM](https://github.com/halam189/Gait_Recognition_LSTM) |

Create the comparator environments:

```powershell
conda env create -f analysis/comparators/environment-deep.yml
conda env create -f analysis/comparators/environment-inceptiontime.yml
```

InceptionTime and Multi-Model LSTM load the authors' code directly. Place it at:

```powershell
git clone https://github.com/hfawaz/InceptionTime external/InceptionTime
git -C external/InceptionTime checkout 952d115e83fb3a66d75c5858c702f8dd5eeb18b7
git clone https://github.com/halam189/Gait_Recognition_LSTM external/Gait_Recognition_LSTM
```

Run one model or all models:

```powershell
python -m analysis.run_benchmark_models rocket
python -m analysis.run_benchmark_models all
```
