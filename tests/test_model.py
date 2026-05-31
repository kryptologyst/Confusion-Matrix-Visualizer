import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import ConfusionMatrixTool


class TestConfusionMatrix:
    def test_compute(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 2, 2, 2])
        tool = ConfusionMatrixTool()
        results = tool.compute(y_true, y_pred)
        cm = np.array(results["confusion_matrix"])
        assert cm.shape == (3, 3)
        assert cm[0, 0] == 2

    def test_accuracy_perfect(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        tool = ConfusionMatrixTool()
        results = tool.compute(y, y)
        assert results["metrics"]["accuracy"] == 1.0
