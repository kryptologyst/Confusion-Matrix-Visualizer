import typer
import sys
from loguru import logger

from .config import settings
from .data import load_data
from .model import ConfusionMatrixTool
from .visualizer import CMVisualizer

app = typer.Typer(help="Confusion Matrix Visualizer CLI")
logger.remove()
logger.add(sys.stderr, level=settings.log_level)


@app.command()
def visualize(
    dataset: str = typer.Option("wine", help="Dataset: iris, wine, breast_cancer"),
    normalize: bool = typer.Option(False, help="Normalize the matrix"),
):
    logger.info(f"Generating confusion matrix for {dataset}...")
    X, y, fn, cn = load_data(dataset)
    tool = ConfusionMatrixTool()
    results = tool.train_and_evaluate(X, y, class_names=cn)
    logger.info(f"Accuracy: {results['metrics']['accuracy']:.2%}")
    CMVisualizer.plot(
        np.array(results["confusion_matrix"]), cn, normalize=normalize,
        save_path=settings.plots_dir / "confusion_matrix.png",
    )
    logger.success("Done!")


if __name__ == "__main__":
    app()
