#!/usr/bin/env python3
"""
CNN-Based Surrogate Models for BBO

This module demonstrates how to use CNNs as surrogate models for black-box optimization,
replacing or augmenting the current Gaussian Process approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass

@dataclass
class CNNConfig:
    """Configuration for CNN surrogate models"""
    input_dim: int
    hidden_channels: List[int] = None
    kernel_size: int = 3
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 100
    uncertainty_samples: int = 50  # For Monte Carlo dropout

class CNNSurrogate(nn.Module):
    """CNN-based surrogate model for function approximation"""
    
    def __init__(self, config: CNNConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        
        # Default hidden channels based on dimensionality
        if config.hidden_channels is None:
            if config.input_dim <= 2:
                config.hidden_channels = [32, 64, 32]
            elif config.input_dim <= 4:
                config.hidden_channels = [64, 128, 64]
            else:
                config.hidden_channels = [128, 256, 128]
        
        # Create spatial embedding for continuous inputs
        self.spatial_embed = nn.Linear(config.input_dim, 64)
        
        # CNN layers for spatial feature extraction
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        
        for out_channels in config.hidden_channels:
            self.conv_layers.append(nn.Conv1d(
                in_channels, out_channels, 
                kernel_size=config.kernel_size, 
                padding=config.kernel_size//2
            ))
            in_channels = out_channels
        
        # Adaptive pooling and final layers
        self.adaptive_pool = nn.AdaptiveAvgPool1d(16)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Output layers for mean and uncertainty
        final_size = config.hidden_channels[-1] * 16
        self.mean_head = nn.Linear(final_size, 1)
        self.log_var_head = nn.Linear(final_size, 1)
        
    def forward(self, x: torch.Tensor, return_var: bool = False) -> torch.Tensor:
        """
        Forward pass with optional uncertainty estimation
        
        Args:
            x: Input tensor [batch_size, input_dim]
            return_var: Whether to return uncertainty estimates
        """
        batch_size = x.shape[0]
        
        # Spatial embedding: [batch_size, input_dim] -> [batch_size, 64]
        embedded = F.relu(self.spatial_embed(x))
        
        # Reshape for conv1d: [batch_size, 1, 64]
        conv_input = embedded.unsqueeze(1)
        
        # CNN feature extraction
        features = conv_input
        for conv_layer in self.conv_layers:
            features = F.relu(conv_layer(features))
            features = self.dropout(features)
        
        # Adaptive pooling: [batch_size, channels, 16]
        pooled = self.adaptive_pool(features)
        flattened = pooled.view(batch_size, -1)
        
        # Output predictions
        mean = self.mean_head(flattened)
        
        if return_var:
            log_var = self.log_var_head(flattened)
            var = torch.exp(log_var)
            return mean, var
        
        return mean
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty using Monte Carlo dropout
        """
        self.train()  # Enable dropout for uncertainty estimation
        
        predictions = []
        for _ in range(self.config.uncertainty_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0).flatten()
        std = np.std(predictions, axis=0).flatten() 
        
        return mean, std

class CNNBayesianOptimizer:
    """CNN-based Bayesian Optimizer for BBO"""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.config = CNNConfig(input_dim=input_dim)
        self.model = CNNSurrogate(self.config)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate
        )
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Fit CNN surrogate model to data
        """
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1))
        
        self.model.train()
        losses = []
        
        for epoch in range(self.config.epochs):
            self.optimizer.zero_grad()
            
            # Forward pass with uncertainty
            mean_pred, var_pred = self.model(X_tensor, return_var=True)
            
            # Negative log-likelihood loss (assuming Gaussian)
            mse_loss = F.mse_loss(mean_pred, y_tensor)
            
            # Uncertainty regularization
            var_loss = torch.mean(var_pred)  # Encourage reasonable uncertainty
            
            total_loss = mse_loss + 0.1 * var_loss
            
            total_loss.backward()
            self.optimizer.step()
            
            losses.append(total_loss.item())
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")
        
        self.is_fitted = True
        return {"final_loss": losses[-1], "avg_loss": np.mean(losses[-10:])}
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and standard deviation
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_tensor = torch.FloatTensor(X)
        self.model.eval()
        
        with torch.no_grad():
            mean, var = self.model(X_tensor, return_var=True)
            mean = mean.cpu().numpy().flatten()
            std = torch.sqrt(var).cpu().numpy().flatten()
        
        return mean, std
    
    def predict_with_mc_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with Monte Carlo dropout uncertainty
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_tensor = torch.FloatTensor(X)
        return self.model.predict_with_uncertainty(X_tensor)

def cnn_expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    """
    Expected Improvement acquisition function for CNN predictions
    """
    from scipy.stats import norm
    
    sigma = np.maximum(sigma, 1e-9)
    imp = mu - y_best - xi
    z = imp / sigma
    ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
    return ei

# Example integration with existing BBO pipeline
def propose_next_point_cnn(X: np.ndarray, y: np.ndarray, *, acquisition: str = "ei", 
                          xi: float = 0.01, seed: int = 42, n_candidates: int = 10000) -> Tuple[np.ndarray, Dict]:
    """
    CNN-based next point proposal for BBO
    """
    dim = X.shape[1]
    
    # Fit CNN surrogate model
    cnn_optimizer = CNNBayesianOptimizer(input_dim=dim)
    fit_info = cnn_optimizer.fit(X, y)
    
    # Generate candidates
    np.random.seed(seed)
    X_cand = np.random.uniform(0.0, 1.0, size=(n_candidates, dim))
    
    # Predict with uncertainty
    mu, sigma = cnn_optimizer.predict_with_mc_uncertainty(X_cand)
    
    # Acquisition function
    y_best = float(np.max(y))
    if acquisition.lower() == "ei":
        scores = cnn_expected_improvement(mu, sigma, y_best, xi)
    else:
        raise ValueError(f"Acquisition '{acquisition}' not implemented for CNN")
    
    # Select best candidate
    best_idx = np.argmax(scores)
    x_next = X_cand[best_idx]
    
    return x_next, {
        "model_type": "CNN",
        "fit_info": fit_info,
        "mu_at_choice": float(mu[best_idx]),
        "sigma_at_choice": float(sigma[best_idx]),
        "ei_score": float(scores[best_idx]),
        "n_candidates": n_candidates
    }

if __name__ == "__main__":
    # Example usage
    print("🧠 CNN-Based BBO Surrogate Model Demo")
    print("=" * 40)
    
    # Generate synthetic data
    np.random.seed(42)
    dim = 3
    n_points = 50
    
    X = np.random.uniform(0, 1, (n_points, dim))
    y = np.sum(X**2, axis=1) + 0.1 * np.random.randn(n_points)  # Synthetic function
    
    print(f"📊 Training on {n_points} points in {dim}D space")
    
    # Test CNN surrogate
    cnn_opt = CNNBayesianOptimizer(input_dim=dim)
    fit_info = cnn_opt.fit(X, y)
    
    # Predict on test points
    X_test = np.random.uniform(0, 1, (10, dim))
    mu_pred, sigma_pred = cnn_opt.predict_with_mc_uncertainty(X_test)
    
    print(f"✅ CNN model fitted successfully")
    print(f"📈 Prediction mean range: [{mu_pred.min():.3f}, {mu_pred.max():.3f}]")
    print(f"🔧 Uncertainty range: [{sigma_pred.min():.3f}, {sigma_pred.max():.3f}]")
    
    # Test next point proposal
    x_next, report = propose_next_point_cnn(X, y, xi=0.01, n_candidates=1000)
    
    print(f"🎯 Next proposed point: {x_next}")
    print(f"📊 EI score: {report['ei_score']:.6f}")
    print(f"🤖 Model: {report['model_type']}")
    print("🎉 CNN-based BBO demonstration complete!")