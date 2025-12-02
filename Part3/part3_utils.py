# -*- coding: utf-8 -*-

import os
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from sklearn.manifold import TSNE
from sklearn.inspection import permutation_importance
import scipy.ndimage as ndi
from scipy.stats import entropy

import warnings
warnings.filterwarnings('ignore')


# -------------------------------
# Sample image grid visualization
# -------------------------------
def visualize_sample_images(dataset, gridsize=(3, 8), title="Sample images"):
    """
    Display a grid of sample images with their labels (risk).
    Expects a PyTorch-style dataset returning (image, label) pairs.
    """
    plt.close('all')
    data_loader = DataLoader(dataset, batch_size=max(1, gridsize[0] * gridsize[1]), shuffle=True)

    nrows, ncols = gridsize
    width, height = gridsize[::-1]
    fig, axes = plt.subplots(nrows, ncols, figsize=(width*1.5, height*1.5))
    axes = np.atleast_2d(axes)

    images, labels = next(iter(data_loader))
    total = min(nrows * ncols, len(images))

    for idx in range(total):
        i = idx // ncols
        j = idx % ncols
        ax = axes[i, j]
        image = images[idx].squeeze().numpy()
        label = labels[idx].item() if torch.is_tensor(labels[idx]) else labels[idx]

        ax.imshow(image, cmap='gray')
        ax.set_title(f"Risk: {label:.2f}")
        ax.axis('off')

    # Turn off any unused axes
    for idx in range(total, nrows * ncols):
        i = idx // ncols
        j = idx % ncols
        axes[i, j].axis('off')

    if title:
        fig.suptitle(title, y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()



# -------------------------------
# t-SNE visualization (with hover)
# -------------------------------
def visualize_dataset_tSNE(dataset, extract_features=False, feature_extractor=None, perplexity=30, random_state=42, zoom=2.0):
    """
    Visualize image data in 2D via t-SNE.
    - If extract_features=True, uses feature_extractor.extract_features(dataset.images, dataset.images_directory).
    - Otherwise, flattens pixel values (48x48) and uses them directly.
    Adds interactive hover to display the corresponding image.
    """
    # Create dataloader
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    batch = next(iter(data_loader))

    # Support datasets that return only images, or (images, labels)
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        images, labels = batch
        labels = labels.squeeze().numpy()
    else:
        images = batch
        labels = np.zeros(len(images))  # fallback if no labels provided

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)

    # If extract feature is True, first extract features then apply t-SNE
    if extract_features:
        assert feature_extractor is not None, "feature_extractor must be provided when extract_features=True"
        X = feature_extractor.extract_features(dataset.images, dataset.images_directory)
    else:
        X = images.squeeze().numpy().reshape(len(images), -1)

    X2 = tsne.fit_transform(X)

    # Define custom colormap for risk
    colors = ["green", "yellow", "orange", "red", "black"]
    nodes = [0, 0.15, 0.3, 0.5, 1]
    mycmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

    # Create plot
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(X2[:, 0], X2[:, 1], c=labels, cmap=mycmap, vmin=0, vmax=1, s=12)
    ax.axis('off')
    cb = plt.colorbar(scatter, ax=ax, pad=0.01)
    cb.set_label("Risk", rotation=270, labelpad=10)

    # Define annotation box and initialize with dummy values
    im = OffsetImage(images[0].squeeze().numpy(), cmap='gray', zoom=zoom)
    ab = AnnotationBbox(
        offsetbox=im,
        xy=(0, 0),
        xybox=(-40, 40),
        xycoords='data',
        boxcoords="offset points",
        pad=0,
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
    )
    ab.set_visible(False)
    ax.add_artist(ab)

    # Hover callback
    imgs = images.squeeze().numpy()
    def motion_hover(event):
        ab_visible = ab.get_visible()
        if event.inaxes == ax:
            is_contained, info = scatter.contains(event)
            if is_contained:
                idx = info['ind'][0]
                data_point_location = scatter.get_offsets()[idx]
                ab.offsetbox = OffsetImage(imgs[idx], zoom=zoom, cmap='gray')
                ab.xy = data_point_location
                ab.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if ab_visible:
                    ab.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', motion_hover)
    plt.show()


# -----------------------------------------
# Convolution visualization with custom kernel
# -----------------------------------------
def visualize_2Dconvolution(image, custom_kernel):
    """
    Convolve an image tensor (C,H,W) or (1,H,W) with a 2D custom kernel and visualize.
    """
    plt.close('all')

    if image.ndim == 3:
        image = image[:1]  # keep single channel
    image = image.unsqueeze(0) if image.ndim == 3 else image  # ensure shape (1,1,H,W) or (1,H,W)
    if image.ndim == 3:
        image = image.unsqueeze(0)

    # Normalize pixels between 0 and 1
    max_pixel_value = torch.max(image)
    min_pixel_value = torch.min(image)
    image_norm = (image - min_pixel_value) / (max_pixel_value - min_pixel_value + 1e-8)

    # Convert the custom kernel to a PyTorch tensor
    kernel = torch.as_tensor(custom_kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Apply the custom kernel using 2D convolution
    transformed_image = nn.functional.conv2d(image_norm, kernel)

    # Display the original and transformed images
    plt.figure(figsize=(12, 4))

    # Define colormap
    colors = ["magenta", "black", "green"]
    nodes = [0, 0.5, 1]
    mycmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

    # Original image
    plt.subplot(131)
    img = image_norm.squeeze().numpy()
    plt.title(f"Original Image {img.shape}")
    plt.axis('off')
    plt.imshow(img, cmap='gray')

    # Kernel
    plt.subplot(132)
    ker = kernel.squeeze().numpy()
    plt.title(f"Kernel {ker.shape}")
    plt.axis('off')
    vmax = np.max(ker)
    vmin = min(np.min(ker), -vmax)
    mynorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    plt.imshow(ker, cmap=mycmap, norm=mynorm)
    for (j, i), value in np.ndenumerate(ker):
        plt.text(i, j, f"{value:.3f}", ha='center', va='center', color="white", fontsize=8)

    # Transformed image
    plt.subplot(133)
    tr = transformed_image.squeeze().numpy()
    plt.title(f"Transformed Image {tr.shape}")
    plt.axis('off')
    vmax = np.max(tr)
    vmin = min(np.min(tr), -vmax)
    mynorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    plt.imshow(tr, cmap=mycmap, norm=mynorm)

    plt.tight_layout()
    plt.show()


# -----------------------------------------
# Verify number of model parameters (PyTorch)
# -----------------------------------------
def verify_number_of_parameters(my_guess, model):
    true_answer = sum(p.numel() for p in model.parameters())
    if int(my_guess) == int(true_answer):
        print(f"You are correct! There is indeed {my_guess} parameters in this neural network.")
    else:
        if my_guess < true_answer:
            print(f"You are incorrect, there is more than {my_guess} parameters in this neural network.\nTry again!")
        else:
            print(f"You are incorrect, there is less than {my_guess} parameters in this neural network.\nTry again!")


# -----------------------------------------
# Regression results visualization
# -----------------------------------------
def visualize_regression_results(y_true, y_pred, title_prefix=""):
    """
    Plots:
      - Predicted vs True scatter
      - Residuals vs True scatter
      - Distribution histograms
    """
    plt.close('all')

    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

    ax1 = plt.subplot(gs[0])
    ax1.set_title(f"{title_prefix}Predicted vs True")
    ax1.scatter(y_true, y_pred, marker='o', color='b', s=10, alpha=0.7)
    ax1.plot([0, 1], [0, 1], 'b--', label='Perfect predictions', alpha=0.3)
    ax1.set_xlabel('y_true')
    ax1.set_ylabel('y_pred')
    ax1.legend()

    ax2 = plt.subplot(gs[1])
    ax2.set_title(f"{title_prefix}Residuals vs True")
    residuals = (np.array(y_true) - np.array(y_pred))
    ax2.scatter(y_true, residuals, marker='o', color='m', s=10, alpha=0.7)
    ax2.plot([0, 1], [0, 0], 'm--', label='Zero error', alpha=0.3)
    ax2.set_xlabel('y_true')
    ax2.set_ylabel('Residual')
    ax2.legend(loc='best')

    ax3 = plt.subplot(gs[2])
    ax3.set_title(f"{title_prefix}Distribution of true and predicted")
    num_bins = 20
    hist_true, bins, _ = ax3.hist(y_true, bins=num_bins, alpha=0.3, color='green', edgecolor="green", label='y_true')
    hist_pred, _, _ = ax3.hist(y_pred, bins=bins, alpha=0.3, color='orange', edgecolor="orange", label='y_pred')
    ax3.set_xlim(-0.05, 1.05)
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Frequency')
    ax3.legend(loc='best')

    plt.tight_layout()
    plt.show()


# -----------------------------------------
# CNN training curves (history dict)
# -----------------------------------------
def plot_cnn_training_curves(history, best_epoch=None, best_val_rmse=None, title="CNN Training Curve (RMSE)"):
    """
    Plot training and validation RMSE per epoch from a history dict:
      history = {"epoch": [...], "train_rmse": [...], "val_rmse": [...]}
    """
    assert isinstance(history, dict) and all(k in history for k in ["epoch", "train_rmse", "val_rmse"]), \
        "history must be a dict with keys: epoch, train_rmse, val_rmse."

    epochs = history["epoch"]
    tr = history["train_rmse"]
    val = history["val_rmse"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, tr, label="Train RMSE", lw=2)
    plt.plot(epochs, val, label="Val RMSE", lw=2)

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


# -----------------------------------------
# MLP loss curve (from sklearn MLPRegressor)
# -----------------------------------------
def plot_mlp_loss_curve(mlp_or_pipeline, title="MLP Loss Curve"):
    """
    Plot the loss_curve_ of an sklearn MLPRegressor. Accepts either:
      - a fitted MLPRegressor
      - a fitted sklearn Pipeline with a 'mlp' step
    """
    # Extract MLPRegressor
    mlp = mlp_or_pipeline
    if hasattr(mlp_or_pipeline, "named_steps"):
        mlp = mlp_or_pipeline.named_steps.get("mlp", None)
    if mlp is None or not hasattr(mlp, "loss_curve_"):
        raise ValueError("Provided object does not have a 'loss_curve_' attribute (is it a fitted MLPRegressor?).")

    plt.figure(figsize=(7, 4))
    plt.plot(mlp.loss_curve_, lw=2)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# -----------------------------------------
# Permutation importance for (tabular) features
# -----------------------------------------
def plot_permutation_importance(estimator, X, y, feature_names=None, n_repeats=10, random_state=42, scoring=None, max_features=20, title="Permutation Importance"):
    """
    Compute and plot permutation importance for a fitted estimator on (X, y).
    Works with Pipeline or plain estimator. For Pipeline, importance is measured on pipeline output.
    """
    result = permutation_importance(estimator, X, y, n_repeats=n_repeats, random_state=random_state, scoring=scoring)
    importances_mean = result.importances_mean
    importances_std = result.importances_std

    idx = np.argsort(importances_mean)[::-1]
    if feature_names is None:
        feature_names = np.array([f"f{i}" for i in range(X.shape[1])])
    fnames = np.array(feature_names)[idx][:max_features]
    means = importances_mean[idx][:max_features]
    stds = importances_std[idx][:max_features]

    plt.figure(figsize=(8, 0.35 * len(fnames) + 1))
    y_pos = np.arange(len(fnames))
    plt.barh(y_pos, means, xerr=stds, align='center', color='tab:blue', ecolor='black', alpha=0.7)
    plt.yticks(y_pos, fnames)
    plt.gca().invert_yaxis()
    plt.xlabel("Mean importance (decrease in score)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# -----------------------------------------
# RMSE comparison bar chart
# -----------------------------------------
def plot_rmse_comparison(rmse_dict, title="Model RMSE Comparison (lower is better)"):
    """
    rmse_dict: dict of {label: rmse_value}
    """
    labels = list(rmse_dict.keys())
    values = [rmse_dict[k] for k in labels]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, values, color=['tab:gray' if i == 0 else 'tab:blue' for i in range(len(labels))], alpha=0.8)
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha='center', va='bottom')
    plt.ylabel("RMSE")
    plt.title(title)
    plt.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    plt.show()



#=============================================================
    #KERNEL
#============================================================


# Kernel bank (3x3 unless stated)
KERNEL_BANK = {
    "blur_soft": np.array([[1,2,1],
                           [2,4,2],
                           [1,2,1]], dtype=float) / 16.0,
    "sharpen": np.array([[0,-1,0],
                         [-1,5,-1],
                         [0,-1,0]], dtype=float),
    "laplacian_4": np.array([[0,1,0],
                             [1,-4,1],
                             [0,1,0]], dtype=float),
    "laplacian_8": np.array([[-1,-1,-1],
                             [-1, 8,-1],
                             [-1,-1,-1]], dtype=float),
    "sobel_x": np.array([[-1,0,1],
                         [-2,0,2],
                         [-1,0,1]], dtype=float),
    "sobel_y": np.array([[-1,-2,-1],
                         [ 0, 0, 0],
                         [ 1, 2, 1]], dtype=float),
    "scharr_x": np.array([[-3,0,3],
                          [-10,0,10],
                          [-3,0,3]], dtype=float),
    "scharr_y": np.array([[-3,-10,-3],
                          [ 0,  0,  0],
                          [ 3, 10,  3]], dtype=float),
    "emboss": np.array([[-2,-1,0],
                        [-1, 1,1],
                        [ 0, 1,2]], dtype=float),
}

def unsharp(image_np, alpha=1.0):
    """Unsharp mask: original + alpha*(original - GaussianBlur(original))."""
    blur = ndi.gaussian_filter(image_np, sigma=1.0)
    sharpened = image_np + alpha*(image_np - blur)
    return np.clip(sharpened, 0, 1)

def dog(image_np, sigma_small=1.0, sigma_big=2.5):
    """Difference of Gaussians."""
    g1 = ndi.gaussian_filter(image_np, sigma_small)
    g2 = ndi.gaussian_filter(image_np, sigma_big)
    return g1 - g2

def response_stats(arr):
    flat = arr.flatten()
    hist, _ = np.histogram(flat, bins=32, density=True)
    return {
        "mean": flat.mean(),
        "std": flat.std(),
        "energy": np.sum(np.abs(flat)),
        "variance": flat.var(),
        "entropy": entropy(hist + 1e-12),
    }

def apply_conv_kernel(torch_tensor, kernel_2d):
    """
    torch_tensor: shape (1, H, W)
    kernel_2d: np.array (kH, kW)
    """
    k = torch.tensor(kernel_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # Normalize input for stability (0-1)
    img_norm = (torch_tensor - torch_tensor.min()) / (torch_tensor.max() - torch_tensor.min() + 1e-8)
    with torch.no_grad():
        filtered = nn.functional.conv2d(img_norm.unsqueeze(0), k)  # (1,1,H,W)
    return filtered.squeeze().cpu().numpy()

def visualize_2Dconvolution(image, custom_kernel):
    
    plt.close('all')
    
    # Normalize pixels between 0 and 1
    max_pixel_value = torch.max(image)
    min_pixel_value = torch.min(image)
    image = (image-min_pixel_value)/(max_pixel_value-min_pixel_value)
   
    # Convert the custom kernel to a PyTorch tensor
    kernel = torch.FloatTensor(custom_kernel).unsqueeze(0).unsqueeze(0)

    # Apply the custom kernel using 2D convolution
    transformed_image = nn.functional.conv2d(image, kernel)

    # Display the original and transformed images
    plt.figure(figsize=(12, 4))
    
    # Define own colormap
    colors = ["magenta","black","green"]
    nodes = [0, 0.5, 1]
    mycmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))
    
    # Display original (normalized) image
    plt.subplot(131)
    image = image.squeeze().numpy()
    plt.title(f"Original Image {image.shape}")
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    
    # Display Kernel
    plt.subplot(132)
    kernel = kernel.squeeze().numpy()
    plt.title(f"Kernel {kernel.shape}")
    plt.axis('off')
    vmax = np.max(kernel)
    vmin = np.min([np.min(kernel),-vmax])
    mynorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    plt.imshow(kernel, cmap=mycmap, norm=mynorm)
    for (j,i),value in np.ndenumerate(kernel):
        plt.text(i,j,value,ha='center',va='center',color="white")
    
    # Display transformed image
    plt.subplot(133)
    transformed_image = transformed_image.squeeze().numpy()
    plt.title(f"Transformed Image {transformed_image.shape}")
    plt.axis('off')
    vmax = np.max(transformed_image)
    vmin = np.min([np.min(transformed_image),-vmax])         
    mynorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    plt.imshow(transformed_image, cmap=mycmap, norm=mynorm)
    
    plt.show()