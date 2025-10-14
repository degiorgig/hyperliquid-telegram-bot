"""
Bayesian Change Point Detection for Wyckoff Phase Transitions

This module implements Bayesian online change point detection to identify
regime changes in market structure. Particularly useful for detecting:
- Accumulation → Markup transitions
- Distribution → Markdown transitions
- Range → Trend transitions

Based on Adams & MacKay (2007) "Bayesian Online Changepoint Detection"
Optimized for crypto market microstructure and Wyckoff methodology.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from scipy import stats
from logging_utils import logger
from .wyckoff_types import WyckoffPhase, Timeframe


@dataclass
class ChangePoint:
    """Represents a detected change point in market structure"""
    index: int
    time: int  # milliseconds timestamp
    probability: float  # Confidence level (0-1)
    run_length: int  # How long since last change point
    from_regime: Optional[str] = None  # Previous market regime
    to_regime: Optional[str] = None  # New market regime
    metric_change: Optional[Dict[str, float]] = None  # What changed


class BayesianChangePointDetector:
    """
    Online Bayesian change point detection for market regime shifts.

    Uses a hazard function to model the probability of regime change
    and Student-t distribution to model price/volume dynamics robustly.
    """

    def __init__(
            self,
            hazard_rate: float = 250,  # Expected change every N periods
            alpha: float = 0.1,  # Student-t degrees of freedom parameter
            beta: float = 1.0,  # Student-t scale parameter
            sensitivity: float = 0.7  # Threshold for change point detection (0-1)
    ):
        """
        Args:
            hazard_rate: Expected periods between change points (lower = more sensitive)
            alpha: Prior belief strength (higher = more conservative)
            beta: Prior scale belief (higher = expects larger variations)
            sensitivity: Detection threshold (lower = more sensitive)
        """
        self.hazard_rate = hazard_rate
        self.alpha0 = alpha
        self.beta0 = beta
        self.sensitivity = sensitivity

        # Online statistics
        self.run_length_probs = np.array([1.0])  # P(run_length | data)
        self.run_lengths = np.array([0])

        # Sufficient statistics for Student-t (online update)
        self.alphas = np.array([alpha])
        self.betas = np.array([beta])
        self.means = np.array([0.0])
        self.kappas = np.array([1.0])

        self.change_points: List[ChangePoint] = []
        self.current_regime_start = 0

    def _hazard_function(self, r: int) -> float:
        """
        Hazard function: probability of change point at run length r.
        Uses exponential distribution (memoryless).
        """
        return 1.0 / self.hazard_rate

    def _update_statistics(self, x: float, r: int) -> Tuple[float, float, float, float]:
        """
        Update sufficient statistics for Student-t posterior.

        Args:
            x: New observation
            r: Run length

        Returns:
            Updated (alpha, beta, kappa, mean)
        """
        if r >= len(self.alphas):
            # Shouldn't happen, but safety check
            return self.alpha0, self.beta0, 1.0, 0.0

        alpha = self.alphas[r]
        beta = self.betas[r]
        kappa = self.kappas[r]
        mean = self.means[r]

        # Bayesian update for Student-t parameters
        kappa_new = kappa + 1
        mean_new = (kappa * mean + x) / kappa_new
        alpha_new = alpha + 0.5
        beta_new = beta + (kappa * (x - mean) ** 2) / (2 * kappa_new)

        return alpha_new, beta_new, kappa_new, mean_new

    def _predictive_probability(self, x: float, r: int) -> float:
        """
        Calculate predictive probability of observation x given run length r.
        Uses Student-t distribution.
        """
        if r >= len(self.alphas):
            return 1e-10  # Very small probability for safety

        alpha = self.alphas[r]
        beta = self.betas[r]
        kappa = self.kappas[r]
        mean = self.means[r]

        # Student-t parameters
        df = 2 * alpha
        loc = mean
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))

        # Avoid numerical issues
        scale = max(scale, 1e-6)

        try:
            prob = stats.t.pdf(x, df=df, loc=loc, scale=scale)
            return max(prob, 1e-10)  # Avoid zero probability
        except Exception as e:
            logger.warning(f"Error calculating predictive probability: {e}")
            return 1e-10

    def update(self, observation: float) -> Optional[ChangePoint]:
        """
        Update detector with new observation and detect change points.

        Args:
            observation: New price return, log-return, or z-score

        Returns:
            ChangePoint if detected, None otherwise
        """
        try:
            # Calculate predictive probabilities for all run lengths
            pred_probs = np.array([
                self._predictive_probability(observation, r)
                for r in range(len(self.run_length_probs))
            ])

            # Calculate growth probabilities (no change point)
            H = np.array([self._hazard_function(r) for r in range(len(self.run_length_probs))])
            growth_probs = self.run_length_probs * pred_probs * (1 - H)

            # Calculate change point probability (new run length = 0)
            cp_prob = (self.run_length_probs * pred_probs * H).sum()

            # Update run length distribution
            new_run_length_probs = np.zeros(len(self.run_length_probs) + 1)
            new_run_length_probs[0] = cp_prob
            new_run_length_probs[1:] = growth_probs

            # Normalize
            new_run_length_probs /= new_run_length_probs.sum()

            # Update sufficient statistics
            new_alphas = np.zeros(len(new_run_length_probs))
            new_betas = np.zeros(len(new_run_length_probs))
            new_kappas = np.zeros(len(new_run_length_probs))
            new_means = np.zeros(len(new_run_length_probs))

            # New regime (r=0)
            new_alphas[0] = self.alpha0
            new_betas[0] = self.beta0
            new_kappas[0] = 1.0
            new_means[0] = observation

            # Continuing regimes (r>0)
            for r in range(len(self.alphas)):
                alpha_new, beta_new, kappa_new, mean_new = self._update_statistics(observation, r)
                new_alphas[r + 1] = alpha_new
                new_betas[r + 1] = beta_new
                new_kappas[r + 1] = kappa_new
                new_means[r + 1] = mean_new

            # Store updated state
            self.run_length_probs = new_run_length_probs
            self.alphas = new_alphas
            self.betas = new_betas
            self.kappas = new_kappas
            self.means = new_means

            # Detect change point
            if cp_prob > self.sensitivity:
                # Find most likely run length before change
                most_likely_rl = np.argmax(self.run_length_probs[1:]) + 1

                cp = ChangePoint(
                    index=len(self.change_points),
                    time=0,  # Will be set by caller
                    probability=float(cp_prob),
                    run_length=int(most_likely_rl)
                )
                self.change_points.append(cp)
                self.current_regime_start = 0
                return cp

            self.current_regime_start += 1
            return None

        except Exception as e:
            logger.error(f"Error in Bayesian change point update: {e}")
            return None

    def get_current_regime_length(self) -> int:
        """Get length of current regime in periods"""
        return self.current_regime_start

    def reset(self):
        """Reset detector to initial state"""
        self.run_length_probs = np.array([1.0])
        self.alphas = np.array([self.alpha0])
        self.betas = np.array([self.beta0])
        self.means = np.array([0.0])
        self.kappas = np.array([1.0])
        self.change_points = []
        self.current_regime_start = 0


class WyckoffBayesianDetector:
    """
    Specialized Bayesian detector for Wyckoff phase transitions.

    Uses multiple metrics (price, volume, volatility) with ensemble detection
    to identify high-confidence regime changes.
    """

    def __init__(self, timeframe: Timeframe):
        """
        Initialize with timeframe-specific parameters.

        Args:
            timeframe: Trading timeframe for adaptive parameters
        """
        self.timeframe = timeframe

        # Adjust sensitivity based on timeframe
        # Shorter timeframes need more sensitivity
        sensitivity_map = {
            '15m': 0.65,
            '1h': 0.70,
            '4h': 0.75,
            '1d': 0.80
        }
        base_sensitivity = sensitivity_map.get(timeframe.value, 0.70)

        # Adjust hazard rate based on timeframe
        # Expected change points per timeframe
        hazard_map = {
            '15m': 100,  # ~25 hours
            '1h': 150,  # ~6.25 days
            '4h': 200,  # ~33 days
            '1d': 250  # ~8 months
        }
        base_hazard = hazard_map.get(timeframe.value, 150)

        # Create separate detectors for different metrics
        self.price_detector = BayesianChangePointDetector(
            hazard_rate=base_hazard,
            sensitivity=base_sensitivity,
            alpha=0.1,
            beta=1.0
        )

        self.volume_detector = BayesianChangePointDetector(
            hazard_rate=base_hazard * 0.8,  # Volume changes faster
            sensitivity=base_sensitivity * 1.1,  # Slightly more sensitive
            alpha=0.15,
            beta=1.5
        )

        self.volatility_detector = BayesianChangePointDetector(
            hazard_rate=base_hazard * 1.2,  # Volatility changes slower
            sensitivity=base_sensitivity * 0.9,
            alpha=0.1,
            beta=1.0
        )

        # Historical metrics for normalization
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.volatility_history: List[float] = []

        self.detected_transitions: List[Dict] = []

    def _normalize_metric(self, value: float, history: List[float], window: int = 20) -> float:
        """Normalize metric using z-score with rolling window"""
        if len(history) < 2:
            return 0.0

        recent = history[-window:] if len(history) >= window else history
        mean = np.mean(recent)
        std = np.std(recent)

        if std < 1e-6:
            return 0.0

        return (value - mean) / std

    def update(
            self,
            df: pd.DataFrame,
            current_phase: WyckoffPhase
    ) -> Optional[Dict]:
        """
        Update detectors with new candle data.

        Args:
            df: DataFrame with OHLCV data (must have at least 2 rows)
            current_phase: Current Wyckoff phase

        Returns:
            Dict with change point information if detected, None otherwise
        """
        if len(df) < 2:
            return None

        try:
            # Extract metrics from latest candle
            returns = df['c'].pct_change().iloc[-1]
            volume_change = df['v'].pct_change().iloc[-1]

            # Use ATR for volatility if available
            if 'ATR' in df.columns:
                volatility = df['ATR'].iloc[-1] / df['c'].iloc[-1]
            else:
                volatility = (df['h'].iloc[-1] - df['l'].iloc[-1]) / df['c'].iloc[-1]

            # Update histories
            self.price_history.append(returns)
            self.volume_history.append(volume_change)
            self.volatility_history.append(volatility)

            # Keep only recent history (memory efficiency)
            max_history = 500
            if len(self.price_history) > max_history:
                self.price_history = self.price_history[-max_history:]
                self.volume_history = self.volume_history[-max_history:]
                self.volatility_history = self.volatility_history[-max_history:]

            # Normalize metrics
            norm_price = self._normalize_metric(returns, self.price_history)
            norm_volume = self._normalize_metric(volume_change, self.volume_history)
            norm_volatility = self._normalize_metric(volatility, self.volatility_history)

            # Update each detector
            price_cp = self.price_detector.update(norm_price)
            volume_cp = self.volume_detector.update(norm_volume)
            vol_cp = self.volatility_detector.update(norm_volatility)

            # Ensemble detection: require at least 2/3 detectors to agree
            detections = [price_cp, volume_cp, vol_cp]
            num_detections = sum(1 for cp in detections if cp is not None)

            if num_detections >= 2:
                # High-confidence change point detected
                avg_probability = np.mean([
                    cp.probability for cp in detections if cp is not None
                ])

                transition = {
                    'time': int(df['time'].iloc[-1]),
                    'index': len(df) - 1,
                    'probability': float(avg_probability),
                    'current_phase': current_phase.value,
                    'price_changed': price_cp is not None,
                    'volume_changed': volume_cp is not None,
                    'volatility_changed': vol_cp is not None,
                    'price_regime_length': self.price_detector.get_current_regime_length(),
                    'volume_regime_length': self.volume_detector.get_current_regime_length(),
                    'metrics': {
                        'price_return': float(returns),
                        'volume_change': float(volume_change),
                        'volatility': float(volatility)
                    }
                }

                self.detected_transitions.append(transition)
                logger.info(
                    f"Bayesian change point detected: "
                    f"Phase={current_phase.value}, "
                    f"Probability={avg_probability:.2%}, "
                    f"Detectors={num_detections}/3"
                )

                return transition

            return None

        except Exception as e:
            logger.error(f"Error in Wyckoff Bayesian detector update: {e}")
            return None

    def get_regime_stability(self) -> float:
        """
        Get current regime stability score (0-1).
        Higher = more stable regime (less likely to change soon)
        """
        try:
            price_rl = self.price_detector.get_current_regime_length()
            volume_rl = self.volume_detector.get_current_regime_length()
            vol_rl = self.volatility_detector.get_current_regime_length()

            # Average regime length
            avg_rl = (price_rl + volume_rl + vol_rl) / 3.0

            # Normalize to 0-1 range (sigmoid)
            # Assumes regime is stable after ~50 periods
            stability = 1.0 / (1.0 + np.exp(-0.1 * (avg_rl - 50)))

            return float(stability)

        except Exception as e:
            logger.error(f"Error calculating regime stability: {e}")
            return 0.5

    def get_time_since_last_transition(self) -> int:
        """Get number of periods since last detected transition"""
        return self.price_detector.get_current_regime_length()

    def reset(self):
        """Reset all detectors"""
        self.price_detector.reset()
        self.volume_detector.reset()
        self.volatility_detector.reset()
        self.price_history = []
        self.volume_history = []
        self.volatility_history = []
        self.detected_transitions = []
