# auto_repair/synthesis/pfs/probabilistic_model.py

"""
Probabilistic Model for Patch Ranking and Selection.

Implements:
- Prophet-style log-linear models [^75^]
- Neural patch correctness predictors
- Bayesian ensemble methods
- Online learning from feedback
"""

import json
import pickle
import logging
from pathlib import Path
from typing import (
    Dict, List, Optional, Tuple, Callable, Any, Union,
    Iterator
)
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from scipy import stats
from scipy.optimize import minimize

# Machine learning imports
try:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, using numpy implementations")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Internal imports
from auto_repair.config import get_config
from auto_repair.synthesis.pfs.templates import PatchInstance


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PatchFeatures:
    """Feature vector for a patch candidate."""
    # Modification features (indices 0-19)
    is_condition_change: float = 0.0
    is_null_guard: float = 0.0
    is_bounds_check: float = 0.0
    is_variable_replace: float = 0.0
    is_method_change: float = 0.0
    adds_loop: float = 0.0
    adds_exception_handler: float = 0.0
    modifies_arithmetic: float = 0.0
    changes_control_flow: float = 0.0
    adds_type_check: float = 0.0
    
    # Program value features (indices 20-39)
    var_used_in_condition: float = 0.0
    var_used_in_assignment: float = 0.0
    var_is_return_value: float = 0.0
    var_is_parameter: float = 0.0
    var_is_field: float = 0.0
    var_is_local: float = 0.0
    var_is_constant: float = 0.0
    var_has_similar_name: float = 0.0
    var_type_consistency: float = 0.0
    var_scope_match: float = 0.0
    
    # Context features (indices 40-49)
    cyclomatic_complexity: float = 0.0
    function_length: float = 0.0
    error_type_match: float = 0.0
    similar_to_historical_fix: float = 0.0
    template_confidence: float = 0.0
    location_rank: float = 0.0  # From GBFL
    ast_depth: float = 0.0
    num_variables_in_scope: float = 0.0
    is_in_loop: float = 0.0
    is_in_exception_handler: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector."""
        return np.array([
            # Modification features
            self.is_condition_change,
            self.is_null_guard,
            self.is_bounds_check,
            self.is_variable_replace,
            self.is_method_change,
            self.adds_loop,
            self.adds_exception_handler,
            self.modifies_arithmetic,
            self.changes_control_flow,
            self.adds_type_check,
            # Program value features
            self.var_used_in_condition,
            self.var_used_in_assignment,
            self.var_is_return_value,
            self.var_is_parameter,
            self.var_is_field,
            self.var_is_local,
            self.var_is_constant,
            self.var_has_similar_name,
            self.var_type_consistency,
            self.var_scope_match,
            # Context features
            self.cyclomatic_complexity,
            self.function_length,
            self.error_type_match,
            self.similar_to_historical_fix,
            self.template_confidence,
            self.location_rank,
            self.ast_depth,
            self.num_variables_in_scope,
            self.is_in_loop,
            self.is_in_exception_handler,
        ])
    
    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'PatchFeatures':
        """Create from numpy vector."""
        return cls(
            is_condition_change=vec[0],
            is_null_guard=vec[1],
            is_bounds_check=vec[2],
            is_variable_replace=vec[3],
            is_method_change=vec[4],
            adds_loop=vec[5],
            adds_exception_handler=vec[6],
            modifies_arithmetic=vec[7],
            changes_control_flow=vec[8],
            adds_type_check=vec[9],
            var_used_in_condition=vec[10],
            var_used_in_assignment=vec[11],
            var_is_return_value=vec[12],
            var_is_parameter=vec[13],
            var_is_field=vec[14],
            var_is_local=vec[15],
            var_is_constant=vec[16],
            var_has_similar_name=vec[17],
            var_type_consistency=vec[18],
            var_scope_match=vec[19],
            cyclomatic_complexity=vec[20],
            function_length=vec[21],
            error_type_match=vec[22],
            similar_to_historical_fix=vec[23],
            template_confidence=vec[24],
            location_rank=vec[25],
            ast_depth=vec[26],
            num_variables_in_scope=vec[27],
            is_in_loop=vec[28],
            is_in_exception_handler=vec[29],
        )


@dataclass
class TrainingExample:
    """Single training example for model."""
    features: PatchFeatures
    patch_code: str
    is_correct: bool  # Ground truth: did patch fix the bug?
    test_passed: bool  # Did tests pass?
    human_validated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbabilisticScore:
    """Score breakdown for a patch."""
    patch_id: str
    log_linear_score: float
    neural_score: Optional[float] = None
    bayesian_score: Optional[float] = None
    ensemble_score: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    feature_importance: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# PROPHET-STYLE LOG-LINEAR MODEL
# =============================================================================

class ProphetModel:
    """
    Prophet-style log-linear model for patch ranking [^75^].
    
    P(patch is correct) ∝ exp(φ · θ) * geometric_rank(location)
    
    where:
    - φ: feature vector
    - θ: learned weights
    - geometric_rank: position from fault localization
    """
    
    def __init__(
        self,
        feature_dim: int = 50,
        learning_rate: float = 0.01,
        regularization: float = 0.1,
        geometric_beta: float = 0.02
    ):
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.geometric_beta = geometric_beta  # β for geometric distribution
        
        # Model parameters
        self.weights: np.ndarray = np.zeros(feature_dim)
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
        # Training history
        self.training_losses: List[float] = []
        self.num_updates: int = 0
        
        self.logger = logging.getLogger(__name__)
    
    def score(
        self,
        features: PatchFeatures,
        location_rank: int = 1
    ) -> float:
        """
        Calculate patch correctness probability.
        
        Args:
            features: Patch feature vector
            location_rank: Rank from fault localization (1 = most suspicious)
        
        Returns:
            Probability score 0-1
        """
        phi = features.to_vector()
        
        # Normalize if scaler fitted
        if self.scaler and hasattr(self.scaler, 'mean_'):
            phi = self.scaler.transform(phi.reshape(1, -1))[0]
        
        # Log-linear score: exp(φ · θ)
        logit_score = np.dot(phi, self.weights)
        prob_score = np.exp(logit_score)
        
        # Geometric rank factor: (1-β)^(rank-1) * β
        geometric_factor = (1 - self.geometric_beta) ** (location_rank - 1) * self.geometric_beta
        
        # Combined score
        final_score = prob_score * geometric_factor
        
        return min(final_score, 1.0)  # Cap at 1.0
    
    def train(
        self,
        examples: List[TrainingExample],
        epochs: int = 100,
        verbose: bool = False
    ) -> float:
        """
        Train model using maximum likelihood estimation.
        
        Maximizes: Σ [correct * log(P) + (1-correct) * log(1-P)]
        """
        if not examples:
            self.logger.warning("No training examples provided")
            return 0.0
        
        # Prepare data
        X = np.array([ex.features.to_vector() for ex in examples])
        y = np.array([1.0 if ex.is_correct else 0.0 for ex in examples])
        
        # Fit scaler
        if self.scaler:
            X = self.scaler.fit_transform(X)
        
        # Use sklearn if available
        if SKLEARN_AVAILABLE:
            model = LogisticRegression(
                C=1.0/self.regularization,
                max_iter=epochs,
                verbose=1 if verbose else 0
            )
            model.fit(X, y)
            self.weights = model.coef_[0]
            
            # Calculate final loss
            probs = model.predict_proba(X)[:, 1]
            loss = -np.mean(y * np.log(probs + 1e-10) + (1-y) * np.log(1-probs + 1e-10))
        else:
            # Manual gradient descent
            loss = self._train_manual(X, y, epochs, verbose)
        
        self.training_losses.append(loss)
        self.num_updates += 1
        
        self.logger.info(f"Training complete. Final loss: {loss:.4f}")
        return loss
    
    def _train_manual(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        verbose: bool
    ) -> float:
        """Manual gradient descent training."""
        n = len(y)
        
        for epoch in range(epochs):
            # Forward pass
            logits = X @ self.weights
            probs = 1 / (1 + np.exp(-logits))  # Sigmoid
            
            # Loss: binary cross-entropy + L2 regularization
            loss = -np.mean(y * np.log(probs + 1e-10) + (1-y) * np.log(1-probs + 1e-10))
            loss += self.regularization * np.sum(self.weights ** 2)
            
            # Backward pass
            gradient = X.T @ (probs - y) / n + 2 * self.regularization * self.weights
            
            # Update
            self.weights -= self.learning_rate * gradient
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: loss = {loss:.4f}")
        
        return loss
    
    def update_online(
        self,
        example: TrainingExample
    ):
        """
        Online learning update from single example.
        
        For continuous learning from repair feedback.
        """
        phi = example.features.to_vector()
        
        if self.scaler and hasattr(self.scaler, 'mean_'):
            phi = self.scaler.transform(phi.reshape(1, -1))[0]
        
        # Current prediction
        logit = np.dot(phi, self.weights)
        prob = 1 / (1 + np.exp(-logit))
        
        # Target
        target = 1.0 if example.is_correct else 0.0
        
        # Gradient update
        gradient = phi * (prob - target) + 2 * self.regularization * self.weights
        self.weights -= self.learning_rate * gradient
        
        self.num_updates += 1
    
    def feature_importance(self) -> Dict[str, float]:
        """Get feature importance (weight magnitudes)."""
        feature_names = [
            "is_condition_change", "is_null_guard", "is_bounds_check",
            "is_variable_replace", "is_method_change", "adds_loop",
            "adds_exception_handler", "modifies_arithmetic",
            "changes_control_flow", "adds_type_check",
            "var_used_in_condition", "var_used_in_assignment",
            "var_is_return_value", "var_is_parameter",
            "var_is_field", "var_is_local", "var_is_constant",
            "var_has_similar_name", "var_type_consistency",
            "var_scope_match", "cyclomatic_complexity",
            "function_length", "error_type_match",
            "similar_to_historical_fix", "template_confidence",
            "location_rank", "ast_depth", "num_variables_in_scope",
            "is_in_loop", "is_in_exception_handler"
        ]
        
        return {
            name: abs(weight)
            for name, weight in zip(feature_names, self.weights)
        }
    
    def save(self, path: Path):
        """Save model to disk."""
        data = {
            'weights': self.weights.tolist(),
            'scaler_mean': self.scaler.mean_.tolist() if self.scaler and hasattr(self.scaler, 'mean_') else None,
            'scaler_scale': self.scaler.scale_.tolist() if self.scaler and hasattr(self.scaler, 'scale_') else None,
            'beta': self.geometric_beta,
            'num_updates': self.num_updates
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: Path):
        """Load model from disk."""
        with open(path) as f:
            data = json.load(f)
        
        self.weights = np.array(data['weights'])
        self.geometric_beta = data.get('beta', 0.02)
        self.num_updates = data.get('num_updates', 0)
        
        if self.scaler and data.get('scaler_mean'):
            self.scaler.mean_ = np.array(data['scaler_mean'])
            self.scaler.scale_ = np.array(data['scaler_scale'])


# =============================================================================
# NEURAL PATCH RANKER
# =============================================================================

class PatchRankingNN(nn.Module):
    """
    Neural network for patch correctness prediction.
    
    Deep architecture for capturing non-linear feature interactions.
    """
    
    def __init__(
        self,
        input_dim: int = 50,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class NeuralPatchRanker:
    """
    Neural ranker for patch correctness.
    
    Complements log-linear model with deep feature interactions.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "auto"
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for neural ranker")
        
        self.device = self._get_device(device)
        self.model = PatchRankingNN().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        if model_path and model_path.exists():
            self.load(model_path)
        
        self.logger = logging.getLogger(__name__)
    
    def _get_device(self, device: str) -> torch.Tensor:
        """Determine computation device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def score(self, features: PatchFeatures) -> float:
        """Predict correctness probability."""
        self.model.eval()
        
        x = torch.tensor(
            features.to_vector(),
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prob = self.model(x).item()
        
        return prob
    
    def train(
        self,
        examples: List[TrainingExample],
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> Dict[str, List[float]]:
        """
        Train neural ranker.
        """
        # Prepare data
        X = np.array([ex.features.to_vector() for ex in examples])
        y = np.array([1.0 if ex.is_correct else 0.0 for ex in examples])
        
        # Split train/val
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create datasets
        train_dataset = PatchDataset(X_train, y_train)
        val_dataset = PatchDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_losses = []
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                pred = self.model(batch_x).squeeze()
                loss = F.binary_cross_entropy(pred, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validate
            val_loss, val_acc = self._validate(val_loader)
            
            history['train_loss'].append(np.mean(train_losses))
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if epoch % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch}: train_loss={history['train_loss'][-1]:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )
        
        return history
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Run validation."""
        self.model.eval()
        losses = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                pred = self.model(batch_x).squeeze()
                loss = F.binary_cross_entropy(pred, batch_y)
                losses.append(loss.item())
                
                # Accuracy
                pred_binary = (pred > 0.5).float()
                correct += (pred_binary == batch_y).sum().item()
                total += len(batch_y)
        
        return np.mean(losses), correct / total if total > 0 else 0.0
    
    def save(self, path: Path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class PatchDataset(Dataset):
    """PyTorch dataset for patches."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# BAYESIAN ENSEMBLE
# =============================================================================

class BayesianEnsemble:
    """
    Bayesian ensemble of multiple ranking models.
    
    Combines predictions with uncertainty quantification.
    """
    
    def __init__(
        self,
        models: List[Callable[[PatchFeatures], float]],
        model_names: Optional[List[str]] = None
    ):
        self.models = models
        self.model_names = model_names or [f"model_{i}" for i in range(len(models))]
        self.model_weights = np.ones(len(models)) / len(models)  # Uniform prior
        
        # Historical performance for Bayesian update
        self.model_performance: Dict[str, List[bool]] = defaultdict(list)
    
    def score(self, features: PatchFeatures) -> Tuple[float, float]:
        """
        Ensemble prediction with uncertainty.
        
        Returns:
            (mean_prediction, uncertainty)
        """
        predictions = []
        
        for model in self.models:
            try:
                pred = model(features)
                predictions.append(pred)
            except Exception as e:
                logging.warning(f"Model failed: {e}")
                predictions.append(0.5)  # Neutral prediction
        
        predictions = np.array(predictions)
        
        # Weighted average
        weighted_mean = np.average(predictions, weights=self.model_weights)
        
        # Uncertainty: weighted standard deviation
        variance = np.average((predictions - weighted_mean) ** 2, weights=self.model_weights)
        uncertainty = np.sqrt(variance)
        
        return weighted_mean, uncertainty
    
    def update_weights(self, features: PatchFeatures, ground_truth: bool):
        """
        Bayesian update of model weights based on performance.
        
        Increases weight of models that predicted correctly.
        """
        for i, model in enumerate(self.models):
            try:
                pred = model(features) > 0.5  # Binary prediction
                correct = (pred == ground_truth)
                self.model_performance[self.model_names[i]].append(correct)
                
                # Update weight based on recent accuracy
                recent_correct = sum(self.model_performance[self.model_names[i]][-10:])
                self.model_weights[i] = 0.1 + recent_correct  # Prior + likelihood
            except Exception:
                self.model_weights[i] *= 0.9  # Decay weight for failing models
        
        # Normalize
        self.model_weights /= np.sum(self.model_weights)
    
    def get_model_confidences(self) -> Dict[str, float]:
        """Get current confidence in each model."""
        return {
            name: weight
            for name, weight in zip(self.model_names, self.model_weights)
        }


# =============================================================================
# MAIN PROBABILISTIC MODEL API
# =============================================================================

class ProbabilisticPatchModel:
    """
    Unified probabilistic model for patch ranking.
    
    Combines log-linear, neural, and Bayesian approaches.
    """
    
    def __init__(
        self,
        config=None,
        model_dir: Optional[Path] = None
    ):
        self.config = config or get_config().pfs
        self.model_dir = model_dir or Path("./models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sub-models
        self.prophet = ProphetModel(
            feature_dim=self.config.feature_dimensions,
            geometric_beta=0.02
        )
        
        self.neural: Optional[NeuralPatchRanker] = None
        if TORCH_AVAILABLE and self.config.enable_neural_ranker:
            self.neural = NeuralPatchRanker()
        
        self.ensemble: Optional[BayesianEnsemble] = None
        
        # Training data buffer for online learning
        self.training_buffer: List[TrainingExample] = []
        self.buffer_size = 1000
        
        self.logger = logging.getLogger(__name__)
    
    def score(
        self,
        patch: PatchInstance,
        location_rank: int = 1,
        return_breakdown: bool = False
    ) -> Union[float, ProbabilisticScore]:
        """
        Score patch using full probabilistic model.
        
        Args:
            patch: Patch candidate
            location_rank: Rank from fault localization
            return_breakdown: If True, return detailed score breakdown
        
        Returns:
            Probability score or full ProbabilisticScore
        """
        features = self._extract_features(patch)
        
        # Prophet score
        prophet_score = self.prophet.score(features, location_rank)
        
        # Neural score (if available)
        neural_score = None
        if self.neural:
            try:
                neural_score = self.neural.score(features)
            except Exception as e:
                self.logger.debug(f"Neural scoring failed: {e}")
        
        # Ensemble score
        if self.ensemble:
            ensemble_score, uncertainty = self.ensemble.score(features)
            confidence_interval = (
                max(0.0, ensemble_score - 2*uncertainty),
                min(1.0, ensemble_score + 2*uncertainty)
            )
        else:
            # Simple average if no ensemble
            scores = [s for s in [prophet_score, neural_score] if s is not None]
            ensemble_score = np.mean(scores) if scores else prophet_score
            confidence_interval = (ensemble_score * 0.8, min(1.0, ensemble_score * 1.2))
        
        if return_breakdown:
            return ProbabilisticScore(
                patch_id=patch.template_id,
                log_linear_score=prophet_score,
                neural_score=neural_score,
                bayesian_score=ensemble_score if self.ensemble else None,
                ensemble_score=ensemble_score,
                confidence_interval=confidence_interval,
                feature_importance=self.prophet.feature_importance()
            )
        
        return ensemble_score
    
    def rank_patches(
        self,
        patches: List[PatchInstance],
        location_ranks: Optional[Dict[str, int]] = None
    ) -> List[Tuple[PatchInstance, float]]:
        """
        Rank patches by correctness probability.
        
        Returns:
            List of (patch, score) sorted by score descending
        """
        scored = []
        
        for patch in patches:
            rank = location_ranks.get(patch.location, 1) if location_ranks else 1
            score = self.score(patch, rank)
            scored.append((patch, score))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored
    
    def train(
        self,
        training_data: List[TrainingExample],
        use_neural: bool = True,
        validation_split: float = 0.2
    ) -> Dict[str, any]:
        """
        Train all sub-models on historical data.
        
        Returns:
            Training metrics
        """
        results = {}
        
        # Train Prophet
        self.logger.info("Training Prophet model...")
        prophet_loss = self.prophet.train(training_data)
        results['prophet_loss'] = prophet_loss
        
        # Train neural (if enabled)
        if use_neural and self.neural:
            self.logger.info("Training neural ranker...")
            neural_history = self.neural.train(
                training_data,
                validation_split=validation_split
            )
            results['neural_history'] = neural_history
        
        # Initialize ensemble
        models = [lambda f, p=self.prophet: p.score(f)]
        if self.neural:
            models.append(lambda f, n=self.neural: n.score(f))
        
        self.ensemble = BayesianEnsemble(models, ["prophet", "neural"])
        
        # Save models
        self.save()
        
        return results
    
    def update_from_feedback(
        self,
        patch: PatchInstance,
        was_correct: bool,
        location_rank: int = 1
    ):
        """
        Online learning from repair feedback.
        
        Updates models based on whether patch actually fixed the bug.
        """
        features = self._extract_features(patch)
        example = TrainingExample(
            features=features,
            patch_code=patch.patched_code,
            is_correct=was_correct,
            test_passed=was_correct,  # Assume test passed if correct
            human_validated=True
        )
        
        # Update Prophet online
        self.prophet.update_online(example)
        
        # Buffer for batch neural update
        self.training_buffer.append(example)
        if len(self.training_buffer) > self.buffer_size:
            self.training_buffer.pop(0)
        
        # Update ensemble weights
        if self.ensemble:
            self.ensemble.update_weights(features, was_correct)
        
        self.logger.info(
            f"Updated from feedback: correct={was_correct}, "
            f"buffer_size={len(self.training_buffer)}"
        )
    
    def batch_update_neural(self):
        """Batch update neural model from buffer."""
        if not self.neural or len(self.training_buffer) < 10:
            return
        
        self.logger.info(f"Batch training neural on {len(self.training_buffer)} examples")
        self.neural.train(self.training_buffer, epochs=5)
    
    def _extract_features(self, patch: PatchInstance) -> PatchFeatures:
        """Extract features from patch instance."""
        # Use pre-computed features if available
        if patch.feature_vector:
            return PatchFeatures.from_vector(np.array(patch.feature_vector))
        
        # Otherwise extract from code
        return self._manual_feature_extraction(patch)
    
    def _manual_feature_extraction(self, patch: PatchInstance) -> PatchFeatures:
        """Manual feature extraction from patch code."""
        features = PatchFeatures()
        
        # Template-based features
        if "COND" in patch.template_id:
            features.is_condition_change = 1.0
        elif "NULL" in patch.template_id:
            features.is_null_guard = 1.0
        elif "BOUNDS" in patch.template_id:
            features.is_bounds_check = 1.0
        
        # Context features
        features.template_confidence = 0.7  # Default
        
        return features
    
    def save(self):
        """Save all models."""
        self.prophet.save(self.model_dir / "prophet_model.json")
        if self.neural:
            self.neural.save(self.model_dir / "neural_ranker.pt")
        
        # Save ensemble state
        if self.ensemble:
            with open(self.model_dir / "ensemble_weights.json", 'w') as f:
                json.dump(self.ensemble.get_model_confidences(), f)
        
        self.logger.info(f"Models saved to {self.model_dir}")
    
    def load(self):
        """Load all models."""
        prophet_path = self.model_dir / "prophet_model.json"
        if prophet_path.exists():
            self.prophet.load(prophet_path)
            self.logger.info("Loaded Prophet model")
        
        neural_path = self.model_dir / "neural_ranker.pt"
        if TORCH_AVAILABLE and neural_path.exists():
            self.neural = NeuralPatchRanker()
            self.neural.load(neural_path)
            self.logger.info("Loaded neural ranker")
        
        # Reconstruct ensemble
        models = [lambda f, p=self.prophet: p.score(f)]
        if self.neural:
            models.append(lambda f, n=self.neural: n.score(f))
        self.ensemble = BayesianEnsemble(models, ["prophet", "neural"])
        
        # Load ensemble weights
        ensemble_path = self.model_dir / "ensemble_weights.json"
        if ensemble_path.exists():
            with open(ensemble_path) as f:
                weights = json.load(f)
                for name, weight in weights.items():
                    if name in self.ensemble.model_names:
                        idx = self.ensemble.model_names.index(name)
                        self.ensemble.model_weights[idx] = weight
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get global feature importance."""
        return self.prophet.feature_importance()
    
    def explain_score(self, patch: PatchInstance) -> str:
        """Generate human-readable explanation of score."""
        score_breakdown = self.score(patch, return_breakdown=True)
        
        explanation = f"""
        Patch Score Analysis:
        - Overall Score: {score_breakdown.ensemble_score:.3f}
        - Confidence Interval: [{score_breakdown.confidence_interval[0]:.3f}, 
                                {score_breakdown.confidence_interval[1]:.3f}]
        
        Model Breakdown:
        - Log-linear (Prophet): {score_breakdown.log_linear_score:.3f}
        """
        
        if score_breakdown.neural_score:
            explanation += f"- Neural Network: {score_breakdown.neural_score:.3f}\n"
        
        explanation += "\nTop Contributing Features:\n"
        sorted_features = sorted(
            score_breakdown.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        for feature, importance in sorted_features:
            explanation += f"  - {feature}: {importance:.3f}\n"
        
        return explanation


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_training_example(
    patch: PatchInstance,
    source_code: str,
    test_result: bool,
    human_validated: bool = False
) -> TrainingExample:
    """
    Create training example from patch and test result.
    
    Args:
        patch: Patch candidate
        source_code: Original source code
        test_result: Whether tests passed with patch
        human_validated: Whether human confirmed correctness
    
    Returns:
        TrainingExample for model training
    """
    # Determine correctness
    is_correct = test_result and human_validated
    
    return TrainingExample(
        features=PatchFeatures.from_vector(np.array(patch.feature_vector)),
        patch_code=patch.patched_code,
        is_correct=is_correct,
        test_passed=test_result,
        human_validated=human_validated,
        metadata={
            "template_id": patch.template_id,
            "location": patch.location,
            "original_code": patch.original_code
        }
    )


def load_historical_patches(path: Path) -> List[TrainingExample]:
    """
    Load training data from historical patches directory.
    
    Expected format: JSON lines with patch data and outcomes.
    """
    examples = []
    
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            example = TrainingExample(
                features=PatchFeatures(**data['features']),
                patch_code=data['patch_code'],
                is_correct=data['is_correct'],
                test_passed=data['test_passed'],
                human_validated=data.get('human_validated', False),
                metadata=data.get('metadata', {})
            )
            examples.append(example)
    
    return examples


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_default_model: Optional[ProbabilisticPatchModel] = None

def get_probabilistic_model() -> ProbabilisticPatchModel:
    """Get or create default model instance."""
    global _default_model
    if _default_model is None:
        _default_model = ProbabilisticPatchModel()
        # Try to load existing models
        try:
            _default_model.load()
        except FileNotFoundError:
            pass  # No existing models
    return _default_model


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Create model
    model = ProbabilisticPatchModel()
    
    # Create dummy training data
    from auto_repair.synthesis.pfs.templates import PatchInstance
    
    training_data = []
    for i in range(100):
        features = PatchFeatures(
            is_null_guard=random.choice([0.0, 1.0]),
            template_confidence=random.random(),
            location_rank=random.randint(1, 10)
        )
        
        example = TrainingExample(
            features=features,
            patch_code=f"patch_{i}",
            is_correct=random.random() > 0.5,
            test_passed=random.random() > 0.3,
            human_validated=random.random() > 0.8
        )
        training_data.append(example)
    
    # Train
    print("Training model...")
    results = model.train(training_data, use_neural=False)
    print(f"Training complete: {results}")
    
    # Score a patch
    test_patch = PatchInstance(
        template_id="GUARD_NULL",
        original_code="x = obj.field",
        patched_code="if obj is not None:\n    x = obj.field",
        location="test.py::10",
        line_number=10,
        description="Add null guard",
        feature_vector=[0.0, 1.0, 0.0] + [0.0] * 47,  # is_null_guard = 1
        semantic_constraints=[],
        test_suggestions=[]
    )
    
    score = model.score(test_patch, location_rank=1)
    print(f"\nPatch score: {score:.3f}")
    
    # Explain score
    explanation = model.explain_score(test_patch)
    print(explanation)
    
    # Simulate feedback
    print("\nSimulating feedback...")
    model.update_from_feedback(test_patch, was_correct=True, location_rank=1)
    
    # Check updated score
    new_score = model.score(test_patch, location_rank=1)
    print(f"Updated score: {new_score:.3f}")
