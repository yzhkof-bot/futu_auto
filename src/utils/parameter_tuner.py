"""
Advanced parameter tuning utilities using various optimization algorithms.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from scipy.optimize import differential_evolution, minimize
from sklearn.model_selection import ParameterGrid
import random
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class OptimizationResult:
    """Result from parameter optimization."""
    best_params: Dict
    best_score: float
    all_results: List[Dict]
    optimization_history: List[float]
    method: str

class ParameterTuner:
    """
    Advanced parameter tuning using multiple optimization algorithms.
    """
    
    def __init__(self, 
                 objective_function: Callable,
                 parameter_bounds: Dict[str, Tuple[float, float]],
                 maximize: bool = True):
        """
        Initialize parameter tuner.
        
        Args:
            objective_function: Function to optimize (should return scalar score)
            parameter_bounds: Dictionary mapping parameter names to (min, max) bounds
            maximize: Whether to maximize or minimize the objective
        """
        self.objective_function = objective_function
        self.parameter_bounds = parameter_bounds
        self.maximize = maximize
        self.optimization_history = []
    
    def grid_search(self, 
                   parameter_grid: Dict[str, List],
                   n_jobs: int = 1) -> OptimizationResult:
        """
        Exhaustive grid search optimization.
        
        Args:
            parameter_grid: Dictionary mapping parameter names to lists of values
            n_jobs: Number of parallel jobs
            
        Returns:
            OptimizationResult object
        """
        
        print(f"Running grid search with {len(list(ParameterGrid(parameter_grid)))} combinations...")
        
        results = []
        best_score = -np.inf if self.maximize else np.inf
        best_params = None
        
        for i, params in enumerate(ParameterGrid(parameter_grid)):
            try:
                score = self.objective_function(params)
                
                results.append({
                    'parameters': params.copy(),
                    'score': score
                })
                
                self.optimization_history.append(score)
                
                # Update best
                if (self.maximize and score > best_score) or (not self.maximize and score < best_score):
                    best_score = score
                    best_params = params.copy()
                
                if i % 10 == 0:
                    print(f"Progress: {i+1}/{len(list(ParameterGrid(parameter_grid)))}, Best: {best_score:.4f}")
                    
            except Exception as e:
                print(f"Error with parameters {params}: {e}")
                continue
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=results,
            optimization_history=self.optimization_history.copy(),
            method="grid_search"
        )
    
    def random_search(self, 
                     n_iterations: int = 100,
                     seed: Optional[int] = None) -> OptimizationResult:
        """
        Random search optimization.
        
        Args:
            n_iterations: Number of random iterations
            seed: Random seed for reproducibility
            
        Returns:
            OptimizationResult object
        """
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        print(f"Running random search with {n_iterations} iterations...")
        
        results = []
        best_score = -np.inf if self.maximize else np.inf
        best_params = None
        
        for i in range(n_iterations):
            # Generate random parameters
            params = {}
            for param_name, (min_val, max_val) in self.parameter_bounds.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = random.randint(min_val, max_val)
                else:
                    params[param_name] = random.uniform(min_val, max_val)
            
            try:
                score = self.objective_function(params)
                
                results.append({
                    'parameters': params.copy(),
                    'score': score
                })
                
                self.optimization_history.append(score)
                
                # Update best
                if (self.maximize and score > best_score) or (not self.maximize and score < best_score):
                    best_score = score
                    best_params = params.copy()
                
                if i % 20 == 0:
                    print(f"Progress: {i+1}/{n_iterations}, Best: {best_score:.4f}")
                    
            except Exception as e:
                print(f"Error with parameters {params}: {e}")
                continue
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=results,
            optimization_history=self.optimization_history.copy(),
            method="random_search"
        )
    
    def differential_evolution_search(self, 
                                    popsize: int = 15,
                                    maxiter: int = 100,
                                    seed: Optional[int] = None) -> OptimizationResult:
        """
        Differential evolution optimization.
        
        Args:
            popsize: Population size multiplier
            maxiter: Maximum number of iterations
            seed: Random seed
            
        Returns:
            OptimizationResult object
        """
        
        print(f"Running differential evolution optimization...")
        
        # Prepare bounds for scipy
        bounds = list(self.parameter_bounds.values())
        param_names = list(self.parameter_bounds.keys())
        
        # Wrapper function for scipy
        def scipy_objective(x):
            params = dict(zip(param_names, x))
            
            # Convert to integers where needed
            for i, (param_name, (min_val, max_val)) in enumerate(self.parameter_bounds.items()):
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = int(round(x[i]))
            
            try:
                score = self.objective_function(params)
                self.optimization_history.append(score)
                
                # Differential evolution minimizes, so negate if we want to maximize
                return -score if self.maximize else score
                
            except Exception as e:
                print(f"Error in objective function: {e}")
                return np.inf
        
        # Run optimization
        result = differential_evolution(
            scipy_objective,
            bounds,
            popsize=popsize,
            maxiter=maxiter,
            seed=seed,
            disp=True
        )
        
        # Convert result back
        best_params = dict(zip(param_names, result.x))
        
        # Convert to integers where needed
        for param_name, (min_val, max_val) in self.parameter_bounds.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                best_params[param_name] = int(round(best_params[param_name]))
        
        best_score = -result.fun if self.maximize else result.fun
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=[],  # DE doesn't track all evaluations
            optimization_history=self.optimization_history.copy(),
            method="differential_evolution"
        )
    
    def bayesian_optimization(self, 
                            n_iterations: int = 50,
                            n_initial_points: int = 10) -> OptimizationResult:
        """
        Bayesian optimization using Gaussian Process.
        Note: This is a simplified implementation. For production use,
        consider using libraries like scikit-optimize or optuna.
        
        Args:
            n_iterations: Number of optimization iterations
            n_initial_points: Number of initial random points
            
        Returns:
            OptimizationResult object
        """
        
        print(f"Running Bayesian optimization with {n_iterations} iterations...")
        
        # Start with random search for initial points
        initial_result = self.random_search(n_initial_points)
        
        results = initial_result.all_results
        best_score = initial_result.best_score
        best_params = initial_result.best_params
        
        # Simple acquisition function (Upper Confidence Bound)
        for i in range(n_iterations - n_initial_points):
            # Generate candidate parameters (simplified - in practice use GP)
            params = {}
            for param_name, (min_val, max_val) in self.parameter_bounds.items():
                # Add some exploration around best parameters
                if best_params and param_name in best_params:
                    center = best_params[param_name]
                    range_size = (max_val - min_val) * 0.1  # 10% of range
                    
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        params[param_name] = max(min_val, min(max_val, 
                            int(center + random.uniform(-range_size, range_size))))
                    else:
                        params[param_name] = max(min_val, min(max_val,
                            center + random.uniform(-range_size, range_size)))
                else:
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        params[param_name] = random.randint(min_val, max_val)
                    else:
                        params[param_name] = random.uniform(min_val, max_val)
            
            try:
                score = self.objective_function(params)
                
                results.append({
                    'parameters': params.copy(),
                    'score': score
                })
                
                self.optimization_history.append(score)
                
                # Update best
                if (self.maximize and score > best_score) or (not self.maximize and score < best_score):
                    best_score = score
                    best_params = params.copy()
                
                if i % 10 == 0:
                    print(f"Progress: {i+n_initial_points+1}/{n_iterations}, Best: {best_score:.4f}")
                    
            except Exception as e:
                print(f"Error with parameters {params}: {e}")
                continue
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=results,
            optimization_history=self.optimization_history.copy(),
            method="bayesian_optimization"
        )
    
    def multi_objective_optimization(self, 
                                   objectives: List[Callable],
                                   weights: Optional[List[float]] = None,
                                   method: str = "random_search",
                                   **kwargs) -> OptimizationResult:
        """
        Multi-objective optimization using weighted sum approach.
        
        Args:
            objectives: List of objective functions
            weights: Weights for each objective (default: equal weights)
            method: Optimization method to use
            **kwargs: Additional arguments for the optimization method
            
        Returns:
            OptimizationResult object
        """
        
        if weights is None:
            weights = [1.0 / len(objectives)] * len(objectives)
        
        if len(weights) != len(objectives):
            raise ValueError("Number of weights must match number of objectives")
        
        print(f"Running multi-objective optimization with {len(objectives)} objectives...")
        
        # Create combined objective function
        original_objective = self.objective_function
        
        def combined_objective(params):
            scores = []
            for obj_func in objectives:
                try:
                    score = obj_func(params)
                    scores.append(score)
                except Exception as e:
                    print(f"Error in objective function: {e}")
                    scores.append(0)
            
            # Weighted sum
            combined_score = sum(w * s for w, s in zip(weights, scores))
            return combined_score
        
        # Temporarily replace objective function
        self.objective_function = combined_objective
        
        try:
            # Run optimization
            if method == "random_search":
                result = self.random_search(**kwargs)
            elif method == "grid_search":
                result = self.grid_search(**kwargs)
            elif method == "differential_evolution":
                result = self.differential_evolution_search(**kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            result.method = f"multi_objective_{method}"
            
        finally:
            # Restore original objective function
            self.objective_function = original_objective
        
        return result
    
    def walk_forward_optimization(self, 
                                data: pd.DataFrame,
                                train_window: int = 252,
                                test_window: int = 63,
                                step_size: int = 21,
                                method: str = "random_search",
                                **kwargs) -> List[OptimizationResult]:
        """
        Walk-forward optimization for time series data.
        
        Args:
            data: Time series data
            train_window: Training window size
            test_window: Test window size  
            step_size: Step size for walk-forward
            method: Optimization method
            **kwargs: Additional arguments for optimization method
            
        Returns:
            List of OptimizationResult objects for each period
        """
        
        print(f"Running walk-forward optimization...")
        
        results = []
        start_idx = train_window
        
        while start_idx + test_window <= len(data):
            print(f"\nOptimizing period: {data.index[start_idx-train_window]} to {data.index[start_idx-1]}")
            
            # Training data
            train_data = data.iloc[start_idx-train_window:start_idx]
            
            # Create objective function for this period
            original_objective = self.objective_function
            
            def period_objective(params):
                return original_objective(params, train_data)
            
            # Temporarily replace objective function
            self.objective_function = period_objective
            
            try:
                # Run optimization for this period
                if method == "random_search":
                    period_result = self.random_search(**kwargs)
                elif method == "grid_search":
                    period_result = self.grid_search(**kwargs)
                elif method == "differential_evolution":
                    period_result = self.differential_evolution_search(**kwargs)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # Add period information
                period_result.train_start = data.index[start_idx-train_window]
                period_result.train_end = data.index[start_idx-1]
                period_result.test_start = data.index[start_idx]
                period_result.test_end = data.index[min(start_idx+test_window-1, len(data)-1)]
                
                results.append(period_result)
                
            except Exception as e:
                print(f"Error in period optimization: {e}")
            
            finally:
                # Restore original objective function
                self.objective_function = original_objective
            
            start_idx += step_size
        
        return results
    
    def parameter_sensitivity_analysis(self, 
                                     base_params: Dict,
                                     sensitivity_range: float = 0.2,
                                     n_points: int = 10) -> Dict:
        """
        Analyze parameter sensitivity around best parameters.
        
        Args:
            base_params: Base parameter set
            sensitivity_range: Range to vary each parameter (as fraction)
            n_points: Number of points to test for each parameter
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        
        print("Running parameter sensitivity analysis...")
        
        sensitivity_results = {}
        
        for param_name, base_value in base_params.items():
            if param_name not in self.parameter_bounds:
                continue
            
            min_bound, max_bound = self.parameter_bounds[param_name]
            
            # Calculate range around base value
            if isinstance(base_value, int):
                param_range = max(1, int(base_value * sensitivity_range))
                test_values = range(
                    max(min_bound, base_value - param_range),
                    min(max_bound, base_value + param_range) + 1
                )
            else:
                param_range = base_value * sensitivity_range
                test_values = np.linspace(
                    max(min_bound, base_value - param_range),
                    min(max_bound, base_value + param_range),
                    n_points
                )
            
            param_scores = []
            
            for test_value in test_values:
                test_params = base_params.copy()
                test_params[param_name] = test_value
                
                try:
                    score = self.objective_function(test_params)
                    param_scores.append((test_value, score))
                except Exception as e:
                    print(f"Error testing {param_name}={test_value}: {e}")
                    continue
            
            sensitivity_results[param_name] = {
                'base_value': base_value,
                'test_values': [x[0] for x in param_scores],
                'scores': [x[1] for x in param_scores],
                'sensitivity': np.std([x[1] for x in param_scores]) if param_scores else 0
            }
        
        return sensitivity_results
    
    def get_optimization_summary(self, result: OptimizationResult) -> str:
        """
        Generate a summary of optimization results.
        
        Args:
            result: OptimizationResult object
            
        Returns:
            Formatted summary string
        """
        
        summary = f"""
PARAMETER OPTIMIZATION SUMMARY
{'='*50}

Method: {result.method}
Best Score: {result.best_score:.6f}
Total Evaluations: {len(result.optimization_history)}

Best Parameters:
{'-'*20}
"""
        
        for param, value in result.best_params.items():
            if isinstance(value, float):
                summary += f"{param}: {value:.4f}\n"
            else:
                summary += f"{param}: {value}\n"
        
        if result.optimization_history:
            summary += f"""
Optimization Progress:
{'-'*20}
Initial Score: {result.optimization_history[0]:.6f}
Final Score: {result.optimization_history[-1]:.6f}
Improvement: {result.optimization_history[-1] - result.optimization_history[0]:.6f}
"""
        
        return summary