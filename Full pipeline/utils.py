from sklearn.metrics import make_scorer
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from PIL import Image
import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

# RMSE scorer (negative, since sklearn maximizes score)
def compute_rmse(y_pred, y_true):
    y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)
    mse = np.mean((y_pred - y_true) ** 2)
    return np.sqrt(mse) 


rmse_scorer = make_scorer(compute_rmse, greater_is_better=False)

def run_rfecv(estimator, X, y, cv=5, min_features_to_select=1, step=1, n_jobs=-1, verbose=1):
    """
    Wrap RFECV for a given estimator and return selector + plotting data.
    For Pipeline estimators, pass the whole pipeline.
    """
    selector = RFECV(
        estimator=estimator,
        step=step,
        cv=cv,
        scoring=rmse_scorer,
        n_jobs=n_jobs,
        min_features_to_select=min_features_to_select,
        verbose=verbose
    )
    selector.fit(X, y)
    # RFECV stores cross-validation scores by feature counts in grid_scores_ (older) or cv_results_
    # Modern scikit-learn exposes cv_results_['mean_test_score'] mapped to n_features grid.
    mean_scores = selector.cv_results_['mean_test_score']  # negative RMSE values
    scores_rmse = -mean_scores                             # convert to RMSE
    n_features_range = range(1, X.shape[1] + 1)
    N_optimal = selector.n_features_
    support_mask = selector.support_
    ranking = selector.ranking_
    return selector, scores_rmse, n_features_range, N_optimal, support_mask, ranking

class CustomImageDataset(Dataset):
    def __init__(self, images, images_directory, target=None, transform=None):
        self.images = list(images)
        self.images_directory = images_directory
        self.target = None if target is None else np.asarray(target).reshape(-1)
        if transform is None:
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_directory, self.images[idx])
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        if self.target is not None:
            return image, float(self.target[idx])
        else:
            return image
        

class SimpleCNN(nn.Module):
    def __init__(self, n_features):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 48x48 -> 24x24 -> 12x12 with 8 channels -> 8*12*12
        self.fc1 = nn.Linear(8*12*12, n_features)
        self.fc2 = nn.Linear(n_features, 1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 8*12*12)
        features = self.fc1(x)
        out = self.fc2(features)
        return out, features
    


class MyCNN:
    def __init__(self, n_features=8, n_epochs=20, batch_size=50, learning_rate=5e-4, device=None):
        self.n_features = n_features
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.history = None
        self.best_state_dict = None
        self.best_val_rmse = np.inf
        self.best_epoch = -1

    def fit(self, images, y, data_dir):
        # Simple internal train/val split (75/25)
        split_ratio = 0.75
        split_index = int(len(images) * split_ratio)
        images_train = images[:split_index]
        y_train = y[:split_index]
        images_val = images[split_index:]
        y_val = y[split_index:]

        train_dataset = CustomImageDataset(images_train, data_dir, y_train)
        val_dataset = CustomImageDataset(images_val, data_dir, y_val)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        self.model = SimpleCNN(n_features=self.n_features).to(self.device)
        criterion = nn.MSELoss(reduction='mean')
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.history = {"epoch": [], "train_rmse": [], "val_rmse": []}

        for epoch in range(self.n_epochs):
            # Train
            self.model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs, _ = self.model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            train_rmse = np.sqrt(running_loss / max(1, (i+1)))

            # Validate
            self.model.eval()
            running_loss_val = 0.0
            with torch.no_grad():
                for j, (inputs, labels) in enumerate(val_loader):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs, _ = self.model(inputs)
                    loss = criterion(outputs.squeeze(), labels.float())
                    running_loss_val += loss.item()
            val_rmse = np.sqrt(running_loss_val / max(1, (j+1)))

            # Log
            self.history["epoch"].append(epoch + 1)
            self.history["train_rmse"].append(train_rmse)
            self.history["val_rmse"].append(val_rmse)

            # Track best
            if val_rmse < self.best_val_rmse:
                self.best_val_rmse = val_rmse
                self.best_epoch = epoch + 1
                self.best_state_dict = deepcopy(self.model.state_dict())

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:2d} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")

        # Optionally restore best weights
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        return self.history

    def predict(self, images, data_dir):
        dataset = CustomImageDataset(images, data_dir)
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out, _ = self.model(batch)
                preds.append(out.squeeze().cpu().numpy())
        return np.concatenate(preds, axis=0)

    def extract_features(self, images, data_dir):
        dataset = CustomImageDataset(images, data_dir)
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        self.model.eval()
        feats = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                _, features = self.model(batch)
                feats.append(features.cpu().numpy())
        return np.concatenate(feats, axis=0)