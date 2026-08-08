"""Five-fold subject-disjoint Multi-Model LSTM verification."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from ablation.multi_model_lstm_official_benchmark import (
    build_model, load_settings, make_loader, resample_time, set_seed, train_model,
)
from ablation.neural_verification_common import (
    CONDITIONS, OUTPUT, ROOT, create_logger, load_data, save_audit, save_scores,
    subject_folds, subset, trial_rows,
)


MODEL = "Multi-Model LSTM"
KEY = "multi_model_lstm"


def prepare(windows: list[np.ndarray]) -> np.ndarray:
    x = resample_time(np.stack(windows).astype(np.float32), 80)
    return x.reshape(len(x), -1)


def predict(model, windows: list[np.ndarray], batch: int, device) -> np.ndarray:
    loader = make_loader(prepare(windows), np.zeros(len(windows), dtype=np.int64), batch, False)
    output = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            scores, _ = model(inputs.to(device, non_blocking=True))
            output.append(scores.exp().cpu().numpy())
    return np.vstack(output)


def main() -> None:
    settings, data, logger = load_settings(), load_data(), create_logger(KEY)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    open_rows = []
    for fold, enrolled, impostors in subject_folds(data):
        split = subset(data, enrolled, impostors)
        set_seed(settings.seed + fold)
        x, y = prepare(split["train_x"]), split["train_y"]
        train, validation = train_test_split(
            np.arange(len(y)), test_size=settings.validation_fraction_treadmill,
            random_state=settings.seed, stratify=y,
        )
        model, history = train_model(
            build_model(len(enrolled), settings),
            make_loader(x[train], y[train], settings.train_batch_size, True),
            make_loader(x[validation], y[validation], settings.test_batch_size, False),
            settings, device,
        )
        fold_dir = OUTPUT / KEY / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), fold_dir / "model.pt")
        labels = np.arange(len(enrolled), dtype=np.int64)
        for condition in CONDITIONS:
            gx, gy, gm = split["genuine"][condition]
            open_rows += trial_rows(
                MODEL, fold, condition, predict(model, gx, settings.test_batch_size, device),
                gm, labels, np.asarray(gy),
            )
        for condition, (ix, im) in split["impostor"].items():
            open_rows += trial_rows(
                MODEL, fold, condition, predict(model, ix, settings.test_batch_size, device),
                im, labels, None,
            )
        logger.info("fold=%s enrolled=%s impostors=%s epochs=%s rows=%s",
                    fold, len(enrolled), len(impostors), len(history), len(open_rows))
    open_path = save_scores(KEY, "open_set", open_rows)

    set_seed(settings.seed)
    full = build_model(len(data["subjects"]), settings).to(device)
    full.load_state_dict(torch.load(
        ROOT / "baselines_out/treadmill_published_subject_sd/multi_model_lstm_official/treadmill_model.pt",
        map_location=device, weights_only=True,
    ))
    enrolled_rows, labels = [], np.arange(len(data["subjects"]), dtype=np.int64)
    for condition in CONDITIONS:
        enrolled_rows += trial_rows(
            MODEL, "", condition,
            predict(full, data["x_test_by_cond"][condition], settings.test_batch_size, device),
            data["meta_test_by_cond"][condition], labels,
            np.asarray(data["y_test_by_cond"][condition]), enrolled_protocol=True,
        )
    enrolled_path = save_scores(KEY, "enrolled", enrolled_rows)
    save_audit(KEY, {"model": MODEL, "folds": 5, "seed": 42, "device": str(device),
                    "open_scores": str(open_path), "enrolled_scores": str(enrolled_path)})


if __name__ == "__main__":
    main()
