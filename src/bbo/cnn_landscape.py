#!/usr/bin/env python3
"""
CNN-Based Function Landscape Modeling for BBO

This module demonstrates using CNNs to model optimization landscapes as spatial images,
particularly useful for lower-dimensional functions (2D-3D).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from scipy.spatial.distance import cdist

class LandscapeGenerator:
    """Generate 2D landscape grids from sparse BBO evaluations"""
    
    def __init__(self, grid_size: int = 64, bounds: Tuple[float, float] = (0, 1)):
        self.grid_size = grid_size
        self.bounds = bounds
        
    def create_landscape_grid(self, X: np.ndarray, y: np.ndarray, 
                            interpolation: str = "rbf") -> np.ndarray:
        """
        Convert sparse evaluations to dense 2D grid for CNN processing
        
        Args:
            X: Input points [n_points, 2] (for 2D functions)
            y: Function values [n_points]
            interpolation: "rbf", "kriging", or "nearest"
        
        Returns:
            landscape: 2D grid [grid_size, grid_size]
        """
        if X.shape[1] != 2:
            raise ValueError("Landscape modeling currently supports 2D functions only")
        
        # Create regular grid
        x_grid = np.linspace(self.bounds[0], self.bounds[1], self.grid_size)
        y_grid = np.linspace(self.bounds[0], self.bounds[1], self.grid_size)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        
        if interpolation == "rbf":
            # RBF interpolation
            from scipy.interpolate import RBFInterpolator
            rbf = RBFInterpolator(X, y, kernel='gaussian', epsilon=0.1)
            z_grid = rbf(grid_points)
        elif interpolation == "nearest":
            # Nearest neighbor interpolation
            distances = cdist(grid_points, X)
            nearest_idx = np.argmin(distances, axis=1)
            z_grid = y[nearest_idx]
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation}")
        
        return z_grid.reshape(self.grid_size, self.grid_size)

class LandscapeCNN(nn.Module):
    """CNN for learning function landscapes and acquisition functions"""
    
    def __init__(self, grid_size: int = 64):
        super().__init__()
        self.grid_size = grid_size
        
        # Encoder: Function landscape -> Features
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
        )
        
        # Decoder: Features -> Acquisition landscape
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # 32x32
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),   # 64x64
            nn.ReLU(),
            
            nn.Conv2d(16, 1, kernel_size=3, padding=1),  # Final acquisition map
        )
        
        # Point value predictor
        self.value_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, landscape: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            landscape: [batch_size, 1, grid_size, grid_size]
        
        Returns:
            acquisition_map: [batch_size, 1, grid_size, grid_size]
            value_pred: [batch_size, 1]
        """
        # Encode landscape features
        features = self.encoder(landscape)
        
        # Generate acquisition map
        acquisition_map = self.decoder(features)
        
        # Predict overall function value
        value_pred = self.value_predictor(features)
        
        return acquisition_map, value_pred

class CNNAcquisitionOptimizer:
    """CNN-based acquisition function learning for BBO"""
    
    def __init__(self, grid_size: int = 64):
        self.grid_size = grid_size
        self.landscape_gen = LandscapeGenerator(grid_size)
        self.model = LandscapeCNN(grid_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.is_fitted = False
        
    def create_training_data(self, X_history: List[np.ndarray], 
                           y_history: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create training data from historical optimization runs
        
        Args:
            X_history: List of input arrays from different optimization runs
            y_history: List of output arrays from different optimization runs
        
        Returns:
            landscapes: [n_samples, 1, grid_size, grid_size]
            targets: [n_samples, 1, grid_size, grid_size] (target acquisition maps)
        """
        landscapes = []
        targets = []
        
        for X, y in zip(X_history, y_history):
            if X.shape[1] != 2:
                continue  # Skip non-2D functions
                
            # Create landscape from evaluations
            landscape = self.landscape_gen.create_landscape_grid(X, y)
            landscapes.append(landscape)
            
            # Create target acquisition map (simplified)
            # In practice, this would be the true optimal acquisition function
            y_best = np.max(y)
            target_acq = np.exp(-(landscape - y_best)**2 / (2 * 0.1**2))
            targets.append(target_acq)
        
        landscapes = torch.FloatTensor(np.array(landscapes)).unsqueeze(1)
        targets = torch.FloatTensor(np.array(targets)).unsqueeze(1)
        
        return landscapes, targets
    
    def fit(self, X_history: List[np.ndarray], y_history: List[np.ndarray], 
            epochs: int = 100) -> Dict[str, float]:
        """
        Train CNN to learn acquisition functions from landscape patterns
        """
        landscapes, targets = self.create_training_data(X_history, y_history)
        
        if len(landscapes) == 0:
            raise ValueError("No 2D functions found in training data")
        
        self.model.train()
        losses = []
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            pred_acq, pred_values = self.model(landscapes)
            
            # Loss: MSE on acquisition maps
            acq_loss = F.mse_loss(pred_acq, targets)
            
            # Additional regularization
            spatial_smooth_loss = self._spatial_smoothness_loss(pred_acq)
            
            total_loss = acq_loss + 0.1 * spatial_smooth_loss
            
            total_loss.backward()
            self.optimizer.step()
            
            losses.append(total_loss.item())
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")
        
        self.is_fitted = True
        return {"final_loss": losses[-1], "epochs": epochs}
    
    def _spatial_smoothness_loss(self, acquisition_maps: torch.Tensor) -> torch.Tensor:
        """Encourage spatial smoothness in acquisition maps"""
        # Gradient penalty for smoothness
        dx = torch.abs(acquisition_maps[:, :, :-1, :] - acquisition_maps[:, :, 1:, :])
        dy = torch.abs(acquisition_maps[:, :, :, :-1] - acquisition_maps[:, :, :, 1:])
        return torch.mean(dx) + torch.mean(dy)
    
    def predict_next_point(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Use CNN to predict optimal next evaluation point
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if X.shape[1] != 2:
            raise ValueError("CNN acquisition optimization supports 2D functions only")
        
        # Create landscape from current evaluations
        landscape = self.landscape_gen.create_landscape_grid(X, y)
        landscape_tensor = torch.FloatTensor(landscape).unsqueeze(0).unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            acq_map, value_pred = self.model(landscape_tensor)
        
        # Find peak of acquisition map
        acq_np = acq_map.squeeze().cpu().numpy()
        peak_idx = np.unravel_index(np.argmax(acq_np), acq_np.shape)
        
        # Convert grid coordinates back to continuous space
        x_next = np.array([
            peak_idx[1] / (self.grid_size - 1),  # x coordinate
            peak_idx[0] / (self.grid_size - 1)   # y coordinate
        ])
        
        return x_next, {
            "model_type": "CNN_Landscape",
            "acquisition_peak": float(acq_np[peak_idx]),
            "predicted_value": float(value_pred.item()),
            "landscape_shape": landscape.shape
        }
    
    def visualize_acquisition(self, X: np.ndarray, y: np.ndarray, 
                            save_path: Optional[str] = None):
        """Visualize function landscape and learned acquisition function"""
        if X.shape[1] != 2:
            print("Visualization only available for 2D functions")
            return
            
        landscape = self.landscape_gen.create_landscape_grid(X, y)
        landscape_tensor = torch.FloatTensor(landscape).unsqueeze(0).unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            acq_map, _ = self.model(landscape_tensor)
        
        acq_np = acq_map.squeeze().cpu().numpy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Function landscape
        im1 = ax1.imshow(landscape, cmap='viridis', origin='lower')
        ax1.scatter(X[:, 0] * (self.grid_size-1), X[:, 1] * (self.grid_size-1), 
                   c='red', s=50, alpha=0.7)
        ax1.set_title('Function Landscape')
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        plt.colorbar(im1, ax=ax1)
        
        # Acquisition function
        im2 = ax2.imshow(acq_np, cmap='hot', origin='lower')
        ax2.set_title('CNN Learned Acquisition Function')
        ax2.set_xlabel('X1')
        ax2.set_ylabel('X2')
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

def demo_cnn_landscape_optimization():
    """Demonstration of CNN-based landscape optimization"""
    print("🏔️ CNN Landscape-Based BBO Demo")
    print("=" * 40)
    
    # Generate synthetic 2D optimization history
    np.random.seed(42)
    n_runs = 5
    n_points_per_run = 20
    
    X_history = []
    y_history = []
    
    # Simulate multiple optimization runs on different 2D functions
    for run in range(n_runs):
        X_run = np.random.uniform(0, 1, (n_points_per_run, 2))
        # Different synthetic functions per run
        if run % 2 == 0:
            y_run = np.exp(-((X_run[:, 0] - 0.7)**2 + (X_run[:, 1] - 0.3)**2) / 0.1)
        else:
            y_run = np.sin(2 * np.pi * X_run[:, 0]) * np.cos(2 * np.pi * X_run[:, 1])
        
        y_run += 0.1 * np.random.randn(n_points_per_run)  # Add noise
        
        X_history.append(X_run)
        y_history.append(y_run)
    
    print(f"📊 Training on {n_runs} optimization runs with {n_points_per_run} points each")
    
    # Train CNN acquisition optimizer
    cnn_acq = CNNAcquisitionOptimizer(grid_size=32)
    fit_info = cnn_acq.fit(X_history, y_history, epochs=50)
    
    print(f"✅ CNN landscape model fitted successfully")
    print(f"📉 Final training loss: {fit_info['final_loss']:.6f}")
    
    # Test on new function
    X_test = np.random.uniform(0, 1, (10, 2))
    y_test = np.exp(-((X_test[:, 0] - 0.5)**2 + (X_test[:, 1] - 0.8)**2) / 0.2)
    
    x_next, report = cnn_acq.predict_next_point(X_test, y_test)
    
    print(f"🎯 CNN predicted next point: [{x_next[0]:.3f}, {x_next[1]:.3f}]")
    print(f"📈 Acquisition peak value: {report['acquisition_peak']:.6f}")
    print(f"🤖 Model type: {report['model_type']}")
    
    # Visualize (save instead of display for headless environments)
    cnn_acq.visualize_acquisition(X_test, y_test, 
                                 save_path="artifacts/cnn_acquisition_demo.png")
    
    print("🎉 CNN landscape optimization demo complete!")
    print("📊 Visualization saved to artifacts/cnn_acquisition_demo.png")

if __name__ == "__main__":
    demo_cnn_landscape_optimization()