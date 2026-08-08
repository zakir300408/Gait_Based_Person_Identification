"""Contract tests for the unmodified published Rocket pipelines."""

import unittest

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import Normalizer, StandardScaler

from ablation.run_strict_published_rocket_benchmarks import MODEL_SPECS, build_classifier


class StrictPublishedRocketTests(unittest.TestCase):
    def test_published_kernel_defaults(self):
        self.assertEqual(MODEL_SPECS["MiniRocket"]["num_kernels"], 10_000)
        self.assertEqual(MODEL_SPECS["MultiRocket"]["num_kernels"], 6_250)

    def test_pipeline_has_standard_scaler_and_full_ridge_grid_without_cosine_normalizer(self):
        for model_name in MODEL_SPECS:
            classifier = build_classifier(model_name, 42)
            pipeline = classifier.multivar_rocket_
            components = pipeline.get_params(deep=True).values()
            self.assertFalse(any(isinstance(step, Normalizer) for step in components))
            self.assertTrue(any(isinstance(step, StandardScaler) for step in components))
            ridge = pipeline.classifier
            self.assertIsInstance(ridge, RidgeClassifierCV)
            np.testing.assert_allclose(ridge.alphas, np.logspace(-3, 3, 10))


if __name__ == "__main__":
    unittest.main()
