import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from loguru import logger


class ConfusionMatrixTool:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.cm_: np.ndarray = None
        self.class_names_: list = []
        self.metrics_: dict = {}

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, class_names: list = None) -> dict:
        self.cm_ = confusion_matrix(y_true, y_pred)
        self.class_names_ = class_names or [f"class_{i}" for i in range(len(np.unique(y_true)))]
        self.metrics_ = classification_report(y_true, y_pred, target_names=self.class_names_, output_dict=True)
        logger.info(f"Confusion matrix: {self.cm_.shape}")
        return {
            "confusion_matrix": self.cm_.tolist(),
            "class_names": self.class_names_,
            "metrics": self.metrics_,
        }

    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray, feature_names=None, class_names=None) -> dict:
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=self.random_state, stratify=y,
        )
        model = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return self.compute(y_test, y_pred, class_names)

    def get_cm(self) -> np.ndarray:
        return self.cm_
