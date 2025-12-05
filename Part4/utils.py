from __future__ import annotations

import os
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.manifold import TSNE
from sklearn.metrics import make_scorer
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


# -------------------------
# Metrics
# -------------------------
def compute_rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


rmse_scorer = make_scorer(compute_rmse, greater_is_better=False)


# -------------------------
# Dataset
# -------------------------
class CustomImageDataset(Dataset):
    """Loads grayscale images and optional targets from a directory."""
    def __init__(
        self,
        images: Sequence[str],
        images_directory: str,
        target: Optional[Sequence[float]] = None,
        transform: Optional[transforms.Compose] = None,
    ):
        self.images = list(images)
        self.images_directory = images_directory
        self.target = None if target is None else np.asarray(target).reshape(-1)

        self.transform = transform or transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        path = os.path.join(self.images_directory, self.images[idx])
        img = Image.open(path)
        img = self.transform(img) if self.transform else img

        if self.target is None:
            return img
        return img, float(self.target[idx])


# -------------------------
# Simple CNN + Wrapper
# -------------------------
class SimpleCNN(nn.Module):
    """Small CNN that outputs both prediction and a compact feature vector."""
    def __init__(self, n_features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        # 48x48 -> 24x24 -> 12x12 with 8 channels -> 8*12*12
        self.fc1 = nn.Linear(8 * 12 * 12, n_features)
        self.fc2 = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 8 * 12 * 12)
        feats = self.fc1(x)
        out = self.fc2(feats)
        return out, feats


class MyCNN:
    """Thin training wrapper around SimpleCNN with feature extraction."""
    def __init__(
        self,
        n_features: int = 8,
        n_epochs: int = 20,
        batch_size: int = 50,
        learning_rate: float = 5e-4,
        device: Optional[str] = None,
    ):
        self.n_features = n_features
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[SimpleCNN] = None
        self.history: Optional[Dict[str, List[float]]] = None
        self.best_state_dict = None
        self.best_val_rmse = np.inf
        self.best_epoch = -1

    def fit(self, images: Sequence[str], y: Sequence[float], data_dir: str):
        """Train with a simple 75/25 split to track validation RMSE."""
        split = int(len(images) * 0.75)
        train_ds = CustomImageDataset(images[:split], data_dir, y[:split])
        val_ds = CustomImageDataset(images[split:], data_dir, y[split:])

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        self.model = SimpleCNN(self.n_features).to(self.device)
        opt = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        crit = nn.MSELoss()

        self.history = {"epoch": [], "train_rmse": [], "val_rmse": []}

        for epoch in range(1, self.n_epochs + 1):
            # train
            self.model.train()
            losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred, _ = self.model(xb)
                loss = crit(pred.squeeze(), yb.float())
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            train_rmse = float(np.sqrt(np.mean(losses))) if losses else np.nan

            # val
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred, _ = self.model(xb)
                    val_losses.append(crit(pred.squeeze(), yb.float()).item())
            val_rmse = float(np.sqrt(np.mean(val_losses))) if val_losses else np.nan

            # log
            self.history["epoch"].append(epoch)
            self.history["train_rmse"].append(train_rmse)
            self.history["val_rmse"].append(val_rmse)

            if val_rmse < self.best_val_rmse:
                self.best_val_rmse = val_rmse
                self.best_epoch = epoch
                self.best_state_dict = deepcopy(self.model.state_dict())

            if epoch % 5 == 0:
                print(f"Epoch {epoch:2d} | Train RMSE {train_rmse:.4f} | Val RMSE {val_rmse:.4f}")

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        return self.history

    def predict(self, images: Sequence[str], data_dir: str) -> np.ndarray:
        """Predict regression targets."""
        assert self.model is not None, "Call fit() first."
        ds = CustomImageDataset(images, data_dir)
        loader = DataLoader(ds, batch_size=len(ds), shuffle=False)

        self.model.eval()
        preds: List[np.ndarray] = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(self.device)
                out, _ = self.model(xb)
                preds.append(out.squeeze().cpu().numpy())
        return np.concatenate(preds, axis=0)

    def extract_features(self, images: Sequence[str], data_dir: str) -> np.ndarray:
        """Return the penultimate layer (embedding) for each image."""
        assert self.model is not None, "Call fit() first."
        ds = CustomImageDataset(images, data_dir)
        loader = DataLoader(ds, batch_size=len(ds), shuffle=False)

        self.model.eval()
        feats: List[np.ndarray] = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(self.device)
                _, z = self.model(xb)
                feats.append(z.cpu().numpy())
        return np.concatenate(feats, axis=0)


# -------------------------
# Visualization helpers
# -------------------------
def visualize_dataset_tSNE(
    dataset: Dataset,
    extract_features: bool = False,
    feature_extractor: Optional[MyCNN] = None,
    perplexity: float = 30,
    random_state: int = 42,
    zoom: float = 2.0,
):
    """t-SNE plot over raw pixels or CNN features with image hover."""
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    batch = next(iter(loader))

    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        images, labels = batch
        labels = labels.squeeze().numpy()
    else:
        images = batch
        labels = np.zeros(len(images))

    if extract_features:
        assert feature_extractor is not None, "feature_extractor required when extract_features=True"
        X = feature_extractor.extract_features(dataset.images, dataset.images_directory)
    else:
        X = images.squeeze().numpy().reshape(len(images), -1)

    init = "pca" if X.shape[1] >= 2 else "random"
    X2 = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init=init).fit_transform(X)

    colors = ["green", "yellow", "orange", "red", "black"]
    nodes = [0, 0.15, 0.3, 0.5, 1]
    mycmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(X2[:, 0], X2[:, 1], c=labels, cmap=mycmap, vmin=0, vmax=1, s=12)
    ax.axis("off")
    cb = plt.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("Risk", rotation=270, labelpad=10)

    im = OffsetImage(images[0].squeeze().numpy(), cmap="gray", zoom=zoom)
    ab = AnnotationBbox(im, (0, 0), xybox=(-40, 40), xycoords="data", boxcoords="offset points",
                        pad=0, arrowprops=dict(arrowstyle="->", color="gray", lw=1.0))
    ab.set_visible(False)
    ax.add_artist(ab)

    imgs = images.squeeze().numpy()

    def on_move(event):
        if event.inaxes != ax:
            if ab.get_visible():
                ab.set_visible(False)
                fig.canvas.draw_idle()
            return
        cont, info = sc.contains(event)
        if cont:
            i = info["ind"][0]
            ab.offsetbox = OffsetImage(imgs[i], zoom=zoom, cmap="gray")
            ab.xy = sc.get_offsets()[i]
            ab.set_visible(True)
            fig.canvas.draw_idle()
        elif ab.get_visible():
            ab.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.show()


def visualize_2Dconvolution(image: torch.Tensor, kernel_2d: np.ndarray):
    """Convolve a (1,H,W) image tensor with a 2D kernel and show inputs/outputs."""
    if image.ndim == 3 and image.shape[0] != 1:
        image = image[:1]  # keep single channel
    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    elif image.ndim == 3:
        image = image.unsqueeze(0)  # (1,1,H,W)

    # normalize to [0,1]
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    kernel = torch.as_tensor(kernel_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        transformed = nn.functional.conv2d(image, kernel)

    colors = ["magenta", "black", "green"]
    nodes = [0, 0.5, 1]
    mycmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

    plt.figure(figsize=(12, 4))
    # original
    plt.subplot(1, 3, 1)
    img = image.squeeze().cpu().numpy()
    plt.title(f"Original {img.shape}")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    # kernel
    plt.subplot(1, 3, 2)
    ker = kernel.squeeze().cpu().numpy()
    vmax = float(np.max(ker))
    vmin = float(min(np.min(ker), -vmax))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    plt.title(f"Kernel {ker.shape}")
    plt.imshow(ker, cmap=mycmap, norm=norm)
    plt.axis("off")

    # transformed
    plt.subplot(1, 3, 3)
    tr = transformed.squeeze().cpu().numpy()
    vmax = float(np.max(tr))
    vmin = float(min(np.min(tr), -vmax))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    plt.title(f"Transformed {tr.shape}")
    plt.imshow(tr, cmap=mycmap, norm=norm)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_regression_results(y_true: Sequence[float], y_pred: Sequence[float], title_prefix: str = ""):
    """Three quick plots: Pred vs True, Residuals, and Distributions."""
    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

    ax1 = plt.subplot(gs[0])
    ax1.set_title(f"{title_prefix}Predicted vs True")
    ax1.scatter(y_true, y_pred, s=10, alpha=0.7)
    ax1.plot([0, 1], [0, 1], "b--", alpha=0.3, label="Perfect")
    ax1.set_xlabel("y_true")
    ax1.set_ylabel("y_pred")
    ax1.legend()

    ax2 = plt.subplot(gs[1])
    ax2.set_title(f"{title_prefix}Residuals vs True")
    res = np.asarray(y_true) - np.asarray(y_pred)
    ax2.scatter(y_true, res, s=10, alpha=0.7, color="m")
    ax2.plot([0, 1], [0, 0], "m--", alpha=0.3)
    ax2.set_xlabel("y_true")
    ax2.set_ylabel("Residual")

    ax3 = plt.subplot(gs[2])
    ax3.set_title(f"{title_prefix}Distributions")
    bins = 20
    h_true, bins, _ = ax3.hist(y_true, bins=bins, alpha=0.3, color="green", edgecolor="green", label="y_true")
    ax3.hist(y_pred, bins=bins, alpha=0.3, color="orange", edgecolor="orange", label="y_pred")
    ax3.set_xlim(-0.05, 1.05)
    ax3.set_xlabel("Value")
    ax3.set_ylabel("Frequency")
    ax3.legend()

    plt.tight_layout()
    plt.show()


def plot_cnn_training_curves(history: Dict[str, List[float]], best_epoch: Optional[int] = None,
                             best_val_rmse: Optional[float] = None, title: str = "CNN Training Curve (RMSE)"):
    """Train/val RMSE per epoch with optional marker for best epoch."""
    assert all(k in history for k in ("epoch", "train_rmse", "val_rmse")), "history missing required keys"
    ep = history["epoch"]
    tr = history["train_rmse"]
    val = history["val_rmse"]

    plt.figure(figsize=(8, 5))
    plt.plot(ep, tr, label="Train RMSE", lw=2)
    plt.plot(ep, val, label="Val RMSE", lw=2)

    if best_epoch is not None and best_val_rmse is not None:
        plt.axvline(best_epoch, color="tab:green", ls="--", alpha=0.7, label=f"Best epoch = {best_epoch}")
        plt.scatter([best_epoch], [best_val_rmse], color="tab:green", zorder=3)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_mlp_loss_curve(mlp_or_pipeline, title: str = "MLP Loss Curve"):
    """Plot sklearn MLPRegressor loss curve. Accepts a Pipeline with step 'mlp'."""
    mlp = mlp_or_pipeline.named_steps.get("mlp", None) if hasattr(mlp_or_pipeline, "named_steps") else mlp_or_pipeline
    if mlp is None or not hasattr(mlp, "loss_curve_"):
        raise ValueError("Object has no 'loss_curve_' (is it a fitted MLPRegressor?)")
    plt.figure(figsize=(7, 4))
    plt.plot(mlp.loss_curve_, lw=2)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_permutation_importance(
    estimator, X: np.ndarray, y: np.ndarray, feature_names: Optional[Sequence[str]] = None,
    n_repeats: int = 10, random_state: int = 42, scoring=None, max_features: int = 20,
    title: str = "Permutation Importance"
):
    """Permutation importance barh plot for a fitted estimator or Pipeline."""
    result = permutation_importance(estimator, X, y, n_repeats=n_repeats, random_state=random_state, scoring=scoring)
    means = result.importances_mean
    stds = result.importances_std

    idx = np.argsort(means)[::-1]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    fnames = np.array(feature_names)[idx][:max_features]
    means = means[idx][:max_features]
    stds = stds[idx][:max_features]

    plt.figure(figsize=(8, 0.35 * len(fnames) + 1))
    y_pos = np.arange(len(fnames))
    plt.barh(y_pos, means, xerr=stds, align="center", color="tab:blue", ecolor="black", alpha=0.7)
    plt.yticks(y_pos, fnames)
    plt.gca().invert_yaxis()
    plt.xlabel("Mean importance (decrease in score)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_rmse_comparison(rmse_dict: Dict[str, float], title: str = "Model RMSE (lower is better)"):
    """Simple bar chart comparing RMSE across labels."""
    labels = list(rmse_dict.keys())
    values = [rmse_dict[k] for k in labels]
    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, values, color=["tab:gray" if i == 0 else "tab:blue" for i in range(len(labels))], alpha=0.85)
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.show()


# -------------------------
# Engineered image features
# -------------------------
# Small 3x3 bank
KERNEL_BANK: Dict[str, np.ndarray] = {
    "blur_soft": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float) / 16.0,
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
    "laplacian_4": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float),
    "laplacian_8": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float),
    "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float),
    "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float),
    "scharr_x": np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=float),
    "scharr_y": np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=float),
    "emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=float),
}


def unsharp(image_np: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Unsharp mask: original + alpha * (original - GaussianBlur(original))."""
    blur = ndi.gaussian_filter(image_np, sigma=1.0)
    return np.clip(image_np + alpha * (image_np - blur), 0, 1)


def dog(image_np: np.ndarray, sigma_small: float = 1.0, sigma_big: float = 2.5) -> np.ndarray:
    """Difference of Gaussians."""
    g1 = ndi.gaussian_filter(image_np, sigma_small)
    g2 = ndi.gaussian_filter(image_np, sigma_big)
    return g1 - g2


def response_stats(arr: np.ndarray) -> Dict[str, float]:
    """Basic statistics used as engineered features."""
    flat = arr.ravel()
    hist, _ = np.histogram(flat, bins=32, density=True)
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "energy": float(np.sum(np.abs(flat))),
        "variance": float(flat.var()),
        "entropy": float(1e-12 + (-np.sum(hist * np.log(hist + 1e-12)))),  # stable entropy
    }


def apply_conv_kernel(torch_tensor: torch.Tensor, kernel_2d: np.ndarray) -> np.ndarray:
    """Apply a 2D kernel to a (1,H,W) tensor; return numpy response."""
    k = torch.tensor(kernel_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    x = (torch_tensor - torch_tensor.min()) / (torch_tensor.max() - torch_tensor.min() + 1e-8)
    with torch.no_grad():
        y = nn.functional.conv2d(x.unsqueeze(0), k)  # (1,1,H,W)
    return y.squeeze().cpu().numpy()


def extract_engineered_kernel_features(
    image_filenames: Sequence[str],
    image_dir: str,
    kernels_dict: Dict[str, np.ndarray],
    include_unsharp: bool = True,
    include_dog: bool = True,
    dog_params: Tuple[float, float] = (1.0, 2.5),
    unsharp_alpha: float = 1.0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute engineered statistics per kernel (and optional unsharp/DoG) for each image.
    Returns (features, feature_names) where features has shape (n_samples, n_features).
    """
    stats_keys = ["mean", "std", "energy", "variance", "entropy"]

    def features_for_image(path: str) -> Tuple[List[float], List[str]]:
        img = Image.open(path).convert("L")
        arr = np.asarray(img, dtype=float)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        torch_img = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

        feats: List[float] = []
        names: List[str] = []

        for kname, kmat in kernels_dict.items():
            resp = apply_conv_kernel(torch_img, kmat)
            stats = response_stats(resp)
            for sk in stats_keys:
                feats.append(stats[sk])
                names.append(f"{kname}_{sk}")

        if include_unsharp:
            us = unsharp(arr, alpha=unsharp_alpha)
            stats = response_stats(us)
            for sk in stats_keys:
                feats.append(stats[sk])
                names.append(f"unsharp_{sk}")

        if include_dog:
            d = dog(arr, sigma_small=dog_params[0], sigma_big=dog_params[1])
            stats = response_stats(d)
            for sk in stats_keys:
                feats.append(stats[sk])
                names.append(f"dog_{sk}")

        return feats, names

    rows: List[List[float]] = []
    names_ref: Optional[List[str]] = None
    for fname in image_filenames:
        feats, names = features_for_image(os.path.join(image_dir, fname))
        rows.append(feats)
        if names_ref is None:
            names_ref = names

    assert names_ref is not None, "No images provided."
    return np.asarray(rows, dtype=float), names_ref


def eval_single_kernel(
    kernel_name: str,
    kernel_mat: np.ndarray,
    img_train: Sequence[str],
    img_test: Sequence[str],
    X_comb_train: np.ndarray,
    X_comb_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    base_mlp,  # pipeline or estimator
    image_dir_train: str,
    image_dir_test: Optional[str] = None,
) -> float:
    """
    Augment Tabular+CNN features with stats from a single kernel and return TEST RMSE.
    This is a convenience function; for model selection, prefer CV on TRAIN first.
    """
    image_dir_test = image_dir_test or image_dir_train
    kd = {kernel_name: kernel_mat}

    feats_tr, _ = extract_engineered_kernel_features(
        img_train, image_dir_train, kd, include_unsharp=False, include_dog=False
    )
    feats_te, _ = extract_engineered_kernel_features(
        img_test, image_dir_test, kd, include_unsharp=False, include_dog=False
    )

    X_tr = np.hstack([X_comb_train, feats_tr])
    X_te = np.hstack([X_comb_test, feats_te])

    model = clone(base_mlp)
    if hasattr(model, "set_params"):
        model.set_params(mlp__max_iter=2000)
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    return compute_rmse(y_test, y_pred)


# -------------------------
# Misc
# -------------------------
def verify_number_of_parameters(my_guess: int, model: nn.Module):
    """Quick check for total trainable parameter count."""
    true = sum(p.numel() for p in model.parameters())
    if int(my_guess) == int(true):
        print(f"You are correct! There are indeed {my_guess} parameters in this neural network.")
    elif my_guess < true:
        print(f"You are incorrect, there are more than {my_guess} parameters.\nTry again!")
    else:
        print(f"You are incorrect, there are fewer than {my_guess} parameters.\nTry again!")