from typing import Dict, Final, Optional
import numpy as np
import pandas as pd
from logging_utils import logger
from .wyckoff_types import Timeframe

class AdaptiveThresholdManager:
    """Manages dynamic thresholds for Wyckoff analysis based on market conditions"""

    @staticmethod
    def get_wyckoff_sign_thresholds(df: pd.DataFrame, timeframe: Timeframe) -> Dict[str, float]:
        """
        Calculate adaptive thresholds for Wyckoff sign detection using percentiles.
        Uses rolling historical data to determine what constitutes 'significant' moves
        rather than relying on fixed multipliers.
        """
        if df.empty or len(df) < 30:
            # Fallback to conservative defaults
            return {
                "min_price_move": 0.008,
                "min_volume_surge": 2.0,
                "price_percentile_75": 0.015,
                "price_percentile_90": 0.025,
                "volume_percentile_75": 1.5,
                "volume_percentile_90": 2.5
            }

        try:
            # Use appropriate lookback based on timeframe
            lookback = min(len(df), timeframe.settings.wyckoff_trend_lookback * 5)
            recent_df = df.iloc[-lookback:]

            # Calculate price changes
            price_changes = recent_df['c'].pct_change().abs().dropna()
            volume_changes = (recent_df['v'] / recent_df['v'].rolling(
                timeframe.settings.volume_ma_window
            ).mean()).dropna()

            # Remove outliers using IQR method (keep 1.5*IQR range)
            def remove_outliers(series: pd.Series) -> pd.Series:
                q1, q3 = series.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                return series[(series >= lower_bound) & (series <= upper_bound)]

            clean_price_changes = remove_outliers(price_changes)
            clean_volume_changes = remove_outliers(volume_changes)

            # Calculate percentile-based thresholds
            price_p75 = clean_price_changes.quantile(0.75)
            price_p90 = clean_price_changes.quantile(0.90)
            volume_p75 = clean_volume_changes.quantile(0.75)
            volume_p90 = clean_volume_changes.quantile(0.90)

            # Apply timeframe-specific volatility factor
            volatility_factor = timeframe.settings.wyckoff_volatility_factor

            # Minimum thresholds: use median as baseline (more robust than mean)
            min_price_move = max(
                clean_price_changes.median() * 1.5,
                0.002  # Absolute minimum for crypto
            ) * volatility_factor

            min_volume_surge = max(
                clean_volume_changes.median() * 1.2,
                1.3  # Absolute minimum
            ) * volatility_factor

            return {
                "min_price_move": min_price_move,
                "min_volume_surge": min_volume_surge,
                "price_percentile_75": price_p75 * volatility_factor,
                "price_percentile_90": price_p90 * volatility_factor,
                "volume_percentile_75": volume_p75 * volatility_factor,
                "volume_percentile_90": volume_p90 * volatility_factor
            }

        except Exception as e:
            logger.warning(f"Error calculating Wyckoff sign thresholds: {e}")
            return {
                "min_price_move": 0.008,
                "min_volume_surge": 2.0,
                "price_percentile_75": 0.015,
                "price_percentile_90": 0.025,
                "volume_percentile_75": 1.5,
                "volume_percentile_90": 2.5
            }

    @staticmethod
    def get_spring_upthrust_thresholds(df: pd.DataFrame, timeframe: Timeframe) -> Dict[str, float]:
        """Calculate dynamic thresholds for spring/upthrust detection"""
        if df.empty or len(df) < 5:
            # Default fallback values
            return {"spring": 0.001, "upthrust": 0.001}
        
        try:
            # Calculate volatility using ATR
            atr = df['ATR'].iloc[-1]
            price = df['c'].iloc[-1]
            volatility = atr / price
                
            # Use timeframe_factor from settings
            timeframe_factor = timeframe.settings.spring_factor
            
            # Calculate adaptive thresholds
            base_threshold = 0.001
            volatility_multiplier = 10 * np.clip(volatility * 100, 0.25, 4.0)
            spring_threshold = base_threshold * (1 + volatility_multiplier) * timeframe_factor
            upthrust_threshold = spring_threshold * 1.05  # Slightly higher for upthrush
            
            # Calculate max wick threshold - higher for volatile markets
            max_wick_base = 0.2  # Base threshold for max wick size
            max_wick_threshold = max_wick_base * timeframe_factor * (1 + volatility * 10)
            
            return {
                "spring": spring_threshold,
                "upthrust": upthrust_threshold,
                "max_wick": max_wick_threshold
            }
        except Exception as e:
            logger.warning(f"Error calculating spring/upthrust thresholds: {e}")
            return {"spring": 0.001, "upthrust": 0.001}

    @staticmethod
    def get_liquidation_thresholds(df: pd.DataFrame, timeframe: Timeframe) -> Dict[str, float]:
        """Calculate dynamic thresholds for liquidation cascade detection"""
            
        try:
            # Check if dataframe has enough valid data
            if df.empty or len(df) < 5:
                return {
                    "vol_threshold": 2.5, 
                    "price_threshold": 0.04,
                    "velocity_threshold": 2.0,
                    "effort_threshold": 0.7
                }
            
            try:
                vol_pct = df['v'].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
                vol_std = vol_pct.std() if len(vol_pct) > 1 else 0.1
                
                price_pct = df['c'].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
                price_std = price_pct.std() if len(price_pct) > 1 else 0.01
            except Exception as calc_error:
                logger.warning(f"Error in liquidation thresholds std deviation calculations: {calc_error}")
                vol_std = 0.1
                price_std = 0.01
            
            # Use timeframe_factor from settings
            timeframe_factor = timeframe.settings.liquidation_factor
            
            vol_threshold = 2.5 * np.clip(1.0 / (vol_std * 10 + 0.5), 0.8, 1.4)
            price_threshold = max(0.02, min(0.06, price_std * 3.0)) * timeframe_factor
            velocity_threshold = 2.0 * timeframe_factor
            effort_threshold = 0.7 * timeframe_factor
            
            return {
                "vol_threshold": vol_threshold,
                "price_threshold": price_threshold,
                "velocity_threshold": velocity_threshold,
                "effort_threshold": effort_threshold
            }
        except Exception as e:
            logger.warning(f"Error calculating liquidation thresholds: {e}", exc_info=True)
            return {
                "vol_threshold": 2.5, 
                "price_threshold": 0.04,
                "velocity_threshold": 2.0,
                "effort_threshold": 0.7
            }
    
    @staticmethod
    def get_breakout_threshold(df: pd.DataFrame, timeframe: Timeframe) -> float:
        """Calculate dynamic threshold for breakout detection"""
        if df.empty or len(df) < 10:
            return 0.015  # Default fallback
        
        try:
            atr = df['ATR'].iloc[-1]
            price = df['c'].iloc[-1]
            volatility_factor = atr / price

            # Use timeframe_factor from settings
            timeframe_factor = timeframe.settings.breakout_factor
            
            # Calculate adaptive breakout threshold
            base_threshold = 0.015
            return max(0.01, base_threshold * (1 + volatility_factor * 5) * timeframe_factor)
        except Exception as e:
            logger.warning(f"Error calculating breakout threshold: {e}", exc_info=True)
            return 0.015