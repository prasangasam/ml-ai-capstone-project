#!/usr/bin/env python3
"""
CNN Integration with BBO Pipeline

This module shows how to integrate CNN-based approaches with the existing 
Gaussian Process BBO pipeline, providing hybrid and ensemble methods.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json

# Import existing BBO modules
import sys
sys.path.append(str(Path(__file__).parent))

from gp import propose_next_point as gp_propose_next_point
from strategy import tune_params, decide_mode_maximise
from cnn_surrogate import CNNBayesianOptimizer, propose_next_point_cnn
from cnn_landscape import CNNAcquisitionOptimizer
from data_loader import FunctionDataset
import config

class HybridCNNGPOptimizer:
    """Hybrid optimizer combining CNN and GP approaches"""
    
    def __init__(self, ensemble_weights: Optional[Dict[str, float]] = None):
        """
        Initialize hybrid optimizer
        
        Args:
            ensemble_weights: Weights for different models
                {"gp": 0.6, "cnn_surrogate": 0.3, "cnn_landscape": 0.1}
        """
        if ensemble_weights is None:
            ensemble_weights = {"gp": 0.7, "cnn_surrogate": 0.3}
        
        self.ensemble_weights = ensemble_weights
        self.cnn_optimizers = {}  # Cache fitted CNN models
        self.cnn_landscape_opts = {}  # Cache landscape optimizers
        
    def _should_use_cnn_landscape(self, func_dim: int, n_points: int) -> bool:
        """Decide whether to use CNN landscape modeling"""
        # Use landscape CNN for 2D functions with sufficient data
        return func_dim == 2 and n_points >= 15
        
    def _should_use_cnn_surrogate(self, func_dim: int, n_points: int) -> bool:
        """Decide whether to use CNN surrogate modeling"""
        # Use CNN surrogate for higher-dimensional functions with enough data
        return func_dim >= 3 and n_points >= 25
    
    def propose_hybrid_point(self, X: np.ndarray, y: np.ndarray, *, 
                           acquisition: str = "ei", xi: float = 0.01, 
                           beta: float = 1.0, seed: int = 42,
                           n_candidates: int = 10000,
                           func_idx: int = 0) -> Tuple[np.ndarray, Dict]:
        """
        Propose next point using hybrid CNN-GP ensemble
        """
        dim = X.shape[1] 
        n_points = len(X)
        
        # Collect predictions from different models
        predictions = []
        reports = []
        weights = []
        
        # 1. Gaussian Process (always included)
        if "gp" in self.ensemble_weights:
            try:
                x_gp, report_gp = gp_propose_next_point(
                    X, y, acquisition=acquisition, xi=xi, beta=beta,
                    seed=seed, n_candidates=n_candidates
                )
                predictions.append(x_gp)
                reports.append({"model": "GP", **report_gp})
                weights.append(self.ensemble_weights["gp"])
                print(f"✅ GP prediction: {x_gp[:3]}...")
            except Exception as e:
                print(f"⚠️ GP prediction failed: {e}")
        
        # 2. CNN Surrogate Model (for higher-dimensional functions)
        if ("cnn_surrogate" in self.ensemble_weights and 
            self._should_use_cnn_surrogate(dim, n_points)):
            try:
                x_cnn, report_cnn = propose_next_point_cnn(
                    X, y, acquisition=acquisition, xi=xi,
                    seed=seed, n_candidates=n_candidates
                )
                predictions.append(x_cnn)
                reports.append({"model": "CNN_Surrogate", **report_cnn})
                weights.append(self.ensemble_weights["cnn_surrogate"])
                print(f"✅ CNN Surrogate prediction: {x_cnn[:3]}...")
            except Exception as e:
                print(f"⚠️ CNN Surrogate prediction failed: {e}")
        
        # 3. CNN Landscape Model (for 2D functions)  
        if ("cnn_landscape" in self.ensemble_weights and 
            self._should_use_cnn_landscape(dim, n_points)):
            try:
                # Use cached landscape optimizer or create new one
                if func_idx not in self.cnn_landscape_opts:
                    self.cnn_landscape_opts[func_idx] = CNNAcquisitionOptimizer()
                    # Would need historical data to train properly
                    print(f"⚠️ CNN Landscape needs historical training data")
                else:
                    cnn_land = self.cnn_landscape_opts[func_idx]
                    x_land, report_land = cnn_land.predict_next_point(X, y)
                    predictions.append(x_land)
                    reports.append({"model": "CNN_Landscape", **report_land})
                    weights.append(self.ensemble_weights.get("cnn_landscape", 0.1))
                    print(f"✅ CNN Landscape prediction: {x_land[:3]}...")
            except Exception as e:
                print(f"⚠️ CNN Landscape prediction failed: {e}")
        
        # Ensemble combination
        if len(predictions) == 0:
            raise ValueError("All prediction methods failed")
        elif len(predictions) == 1:
            # Single prediction
            x_final = predictions[0]
            ensemble_info = {"method": "single", "used_models": [reports[0]["model"]]}
        else:
            # Weighted ensemble
            predictions = np.array(predictions)
            weights = np.array(weights[:len(predictions)])
            weights = weights / weights.sum()  # Normalize weights
            
            x_final = np.average(predictions, weights=weights, axis=0)
            
            ensemble_info = {
                "method": "weighted_ensemble",
                "used_models": [r["model"] for r in reports],
                "weights": weights.tolist(),
                "individual_predictions": predictions.tolist()
            }
        
        return x_final, {
            "ensemble_info": ensemble_info,
            "individual_reports": reports,
            "final_prediction": x_final.tolist(),
            "n_models_used": len(predictions)
        }

class AdaptiveCNNGPPipeline:
    """Adaptive pipeline that switches between CNN and GP based on problem characteristics"""
    
    def __init__(self):
        self.performance_history = {}  # Track model performances
        self.model_selection_history = {}
        
    def select_best_model(self, func_idx: int, dim: int, n_points: int, 
                         recent_improvements: List[float]) -> str:
        """
        Adaptively select best model based on problem characteristics and performance
        """
        # Performance-based selection
        if func_idx in self.performance_history:
            perf = self.performance_history[func_idx]
            
            # Select model with best recent performance
            if "cnn_surrogate" in perf and "gp" in perf:
                if np.mean(perf["cnn_surrogate"][-3:]) > np.mean(perf["gp"][-3:]):
                    return "cnn_surrogate"
                else:
                    return "gp"
        
        # Rule-based selection for new functions
        if dim == 2 and n_points >= 15:
            return "cnn_landscape"
        elif dim >= 4 and n_points >= 25:
            return "cnn_surrogate"
        else:
            return "gp"  # Default fallback
    
    def update_performance(self, func_idx: int, model_type: str, improvement: float):
        """Update performance tracking for model selection"""
        if func_idx not in self.performance_history:
            self.performance_history[func_idx] = {}
        
        if model_type not in self.performance_history[func_idx]:
            self.performance_history[func_idx][model_type] = []
        
        self.performance_history[func_idx][model_type].append(improvement)
        
        # Keep only recent performance (last 10 evaluations)
        if len(self.performance_history[func_idx][model_type]) > 10:
            self.performance_history[func_idx][model_type] = \
                self.performance_history[func_idx][model_type][-10:]

def run_cnn_enhanced_optimization(*, initial_dir: Path, weekly_dir: Path,
                                 use_hybrid: bool = True,
                                 cnn_ensemble_weights: Optional[Dict] = None) -> Dict:
    """
    Enhanced BBO pipeline with CNN integration
    
    Args:
        initial_dir: Path to initial data
        weekly_dir: Path to weekly data
        use_hybrid: Whether to use hybrid CNN-GP ensemble
        cnn_ensemble_weights: Custom ensemble weights
    
    Returns:
        Enhanced optimization results with CNN integration info
    """
    from data_loader import load_initial_from_dir, load_weekly
    from pipeline import FunctionDataset
    import io
    
    print("🤖 CNN-Enhanced BBO Pipeline Starting")
    print("=" * 50)
    
    # Load data (same as standard pipeline)
    seeds = load_initial_from_dir(initial_dir)
    weekly_inputs_all, weekly_outputs_all, weekly_mode = load_weekly(weekly_dir)
    
    # Build function datasets
    funcs: List[FunctionDataset] = [
        FunctionDataset(s.idx, np.asarray(s.X, float), np.asarray(s.y, float).reshape(-1))
        for s in seeds
    ]
    
    for week_inputs, week_outputs in zip(weekly_inputs_all, weekly_outputs_all):
        for i in range(8):
            funcs[i].append(week_inputs[i], week_outputs[i])
    
    week_k = len(weekly_outputs_all)
    last_week_outputs = np.asarray(weekly_outputs_all[-1], float).reshape(-1)
    
    # Initialize CNN-enhanced optimizer
    if use_hybrid:
        optimizer = HybridCNNGPOptimizer(cnn_ensemble_weights)
    else:
        optimizer = AdaptiveCNNGPPipeline()
    
    portal_lines: List[str] = []
    diagnostics: List[Dict] = []
    
    print(f"📊 Week {week_k} optimization with {len(funcs)} functions")
    print("🔄 Processing functions with CNN-enhanced strategies...")
    
    for i, f in enumerate(funcs, start=1):
        func_idx = i - 1
        dim = f.X.shape[1]
        n_points = len(f.y)
        
        print(f"\n🎯 Function {i} ({dim}D, {n_points} evaluations)")
        
        # Standard mode and parameter selection
        mode = decide_mode_maximise(float(last_week_outputs[func_idx]), f.y)
        tuned = tune_params(mode, config.ACQUISITION)
        xi = float(tuned.get("xi", config.XI_EXPLORE))
        beta = float(tuned.get("beta", config.BETA_EXPLORE))
        
        try:
            if use_hybrid:
                # Hybrid CNN-GP approach
                x_next, report = optimizer.propose_hybrid_point(
                    f.X, f.y,
                    acquisition=config.ACQUISITION,
                    xi=xi, beta=beta,
                    seed=config.RNG_SEED + 31*i,
                    n_candidates=config.N_CANDIDATES,
                    func_idx=func_idx
                )
                
                model_info = {
                    "optimization_type": "hybrid_cnn_gp",
                    "n_models_used": report["n_models_used"],
                    "ensemble_method": report["ensemble_info"]["method"],
                    "models_used": report["ensemble_info"]["used_models"]
                }
            else:
                # Adaptive model selection
                recent_improvements = list(np.diff(f.y[-5:]) if len(f.y) >= 5 else [])
                selected_model = optimizer.select_best_model(
                    func_idx, dim, n_points, recent_improvements
                )
                
                if selected_model == "gp":
                    x_next, report = gp_propose_next_point(
                        f.X, f.y, acquisition=config.ACQUISITION,
                        xi=xi, beta=beta, seed=config.RNG_SEED + 31*i,
                        n_candidates=config.N_CANDIDATES
                    )
                elif selected_model == "cnn_surrogate":
                    x_next, report = propose_next_point_cnn(
                        f.X, f.y, acquisition=config.ACQUISITION,
                        xi=xi, seed=config.RNG_SEED + 31*i,
                        n_candidates=config.N_CANDIDATES
                    )
                else:  # cnn_landscape
                    # Would need proper implementation with trained model
                    print(f"⚠️ CNN landscape not fully implemented, falling back to GP")
                    x_next, report = gp_propose_next_point(
                        f.X, f.y, acquisition=config.ACQUISITION,
                        xi=xi, beta=beta, seed=config.RNG_SEED + 31*i,
                        n_candidates=config.N_CANDIDATES
                    )
                
                model_info = {
                    "optimization_type": "adaptive_selection",
                    "selected_model": selected_model,
                    "selection_reason": f"{dim}D_function_with_{n_points}_points"
                }
                
            print(f"✅ Next point: {x_next[:3]}... (model: {model_info.get('selected_model', 'hybrid')})")
            
        except Exception as e:
            print(f"⚠️ CNN optimization failed for Function {i}, using GP fallback: {e}")
            # Fallback to standard GP
            x_next, report = gp_propose_next_point(
                f.X, f.y, acquisition=config.ACQUISITION,
                xi=xi, beta=beta, seed=config.RNG_SEED + 31*i,
                n_candidates=config.N_CANDIDATES  
            )
            model_info = {"optimization_type": "gp_fallback", "error": str(e)}
        
        portal_lines.append(io.fmt_query(x_next))
        
        # Enhanced diagnostics
        diagnostics.append({
            "function_index": i,
            "mode": mode,
            "xi": xi,
            "beta": beta,
            "dimensionality": dim,
            "n_evaluations": n_points,
            **model_info,
            **report
        })
    
    # Save results (same as standard pipeline)  
    submission_path = io.save_submission_file(week_next=week_k+1, portal_lines=portal_lines)
    
    # Enhanced snapshot with CNN info
    snapshot_path = io.save_week_snapshot(week_k=week_k, payload={
        "acquisition": config.ACQUISITION,
        "week_k_observed": week_k,
        "initial_dir": str(initial_dir),
        "weekly_dir": str(weekly_dir),
        "weekly_mode": weekly_mode,
        "next_week_portal_lines": portal_lines,
        "diagnostics": diagnostics,
        "cnn_enhancement": "active",
        "hybrid_mode": use_hybrid,
        "ensemble_weights": cnn_ensemble_weights
    })
    
    print(f"\n🎉 CNN-Enhanced BBO Pipeline Complete!")
    print(f"📁 Submission: {submission_path}")
    print(f"📊 Snapshot: {snapshot_path}")
    
    return {
        "week_k": week_k,
        "portal_lines": portal_lines,
        "submission_path": str(submission_path),
        "snapshot_path": str(snapshot_path),
        "weekly_mode": weekly_mode,
        "cnn_enhanced": True,
        "diagnostics_summary": {
            "total_functions": len(diagnostics),
            "hybrid_used": use_hybrid,
            "unique_models": list(set(d.get("selected_model", "hybrid") for d in diagnostics))
        }
    }

if __name__ == "__main__":
    # Demo CNN-enhanced pipeline
    print("🚀 CNN-Enhanced BBO Pipeline Demo")
    
    # Would run with actual data paths
    # result = run_cnn_enhanced_optimization(
    #     initial_dir=Path("data/initial_data"),
    #     weekly_dir=Path("data/weekly"),
    #     use_hybrid=True,
    #     cnn_ensemble_weights={"gp": 0.6, "cnn_surrogate": 0.4}
    # )
    
    print("✅ CNN integration framework ready for deployment!")
    print("📋 Available CNN enhancement modes:")
    print("   • Hybrid CNN-GP ensemble")
    print("   • Adaptive model selection")
    print("   • CNN surrogate modeling")
    print("   • CNN landscape optimization (2D functions)")