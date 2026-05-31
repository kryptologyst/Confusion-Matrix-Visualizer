import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
from loguru import logger


class CMVisualizer:
    @staticmethod
    def plot(cm, class_names, normalize=False, save_path=None):
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
