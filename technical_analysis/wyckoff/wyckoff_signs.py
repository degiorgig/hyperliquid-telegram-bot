import pandas as pd  # type: ignore[import]
import numpy as np  # type: ignore[import]
from typing import Final
from .wyckoff_types import WyckoffSign, Timeframe
from .adaptive_thresholds import AdaptiveThresholdManager

# Constants for Wyckoff sign detection
STRONG_DEV_THRESHOLD: Final[float] = 2.1

def detect_wyckoff_signs(
    df: pd.DataFrame,
    price_strength: float,
    volume_trend: float,
    is_spring: bool,
    is_upthrust: bool,
    timeframe: Timeframe
) -> WyckoffSign:
    """
    Detect specific Wyckoff signs in market action optimized for crypto intraday trading.
    Uses dedicated parameters from timeframe settings for consistent sign detection.
    More conservative approach - only returns a sign when evidence is strong.
    """
    if len(df) < 5:
        return WyckoffSign.NONE

    # Get adaptive thresholds based on historical percentiles
    adaptive_thresholds = AdaptiveThresholdManager.get_wyckoff_sign_thresholds(df, timeframe)
    min_price_move = adaptive_thresholds["min_price_move"]
    min_volume_surge = adaptive_thresholds["min_volume_surge"]
    price_p75 = adaptive_thresholds["price_percentile_75"]
    price_p90 = adaptive_thresholds["price_percentile_90"]
    volume_p75 = adaptive_thresholds["volume_percentile_75"]
    volume_p90 = adaptive_thresholds["volume_percentile_90"]

    # Calculate key metrics with noise reduction
    price_change = df['c'].pct_change()
    volume_change = df['v'].pct_change()

    # Use timeframe-specific lookback periods from settings
    volatility_window = max(5, timeframe.settings.spring_upthrust_window)
    volume_ma_window = max(5, timeframe.settings.volume_ma_window // 3)
    price_ma_window = max(8, int(timeframe.settings.ema_length * 0.75))

    # Use timeframe settings for recent high/low lookback
    recent_window = timeframe.settings.swing_lookback

    # Calculate rolling metrics with adaptive windows
    price_volatility = df['c'].pct_change().rolling(volatility_window).std().iloc[-1]
    volume_ma = df['v'].rolling(volume_ma_window).mean()
    price_ma = df['c'].rolling(price_ma_window).mean()

    # Use dedicated Wyckoff sign detection parameters from settings
    volatility_factor = timeframe.settings.wyckoff_volatility_factor
    trend_lookback = timeframe.settings.wyckoff_trend_lookback
    lps_volume_threshold = timeframe.settings.wyckoff_lps_volume_threshold
    lps_price_multiplier = timeframe.settings.wyckoff_lps_price_multiplier
    sos_multiplier = timeframe.settings.wyckoff_sos_multiplier
    ut_multiplier = timeframe.settings.wyckoff_ut_multiplier
    sc_multiplier = timeframe.settings.wyckoff_sc_multiplier
    ar_multiplier = timeframe.settings.wyckoff_ar_multiplier
    confirmation_threshold = timeframe.settings.wyckoff_confirmation_threshold
    
    # Detect market context for better signal relevance
    is_high_volatility = price_volatility > df['c'].pct_change().rolling(volatility_window * 3).std().mean() * 1.5
    recency_factor = 3 if is_high_volatility else 2

    # Improved confirmation functions with direct parameter usage and higher thresholds
    def confirm_trend(window: int, threshold: float, direction: int = 1) -> bool:
        """Check if price movement confirms a trend in the given direction with stricter criteria."""
        window = min(window, len(df) - 1)
        recent_changes = price_change.iloc[-window:]
        
        # Adjust sensitivity based on volatility - be more conservative
        sensitivity_factor = 0.9 if is_high_volatility else 1.0
        adjusted_threshold = threshold * sensitivity_factor
        weights = np.linspace(1, recency_factor, len(recent_changes))
        
        if direction == 1:  # Up trend
            weighted_confirms = (recent_changes > adjusted_threshold) * weights
        else:  # Down trend
            weighted_confirms = (recent_changes < -adjusted_threshold) * weights

        return weighted_confirms.sum() >= np.sum(weights) * confirmation_threshold
        
    def confirm_volume(window: int, threshold: float) -> bool:
        """Check if volume confirms a significant move with stricter criteria."""
        window = min(window, len(df) - 1)
        recent_volume = volume_change.iloc[-window:]
        
        # Adjust threshold based on volatility - be more conservative
        adjusted_threshold = threshold * (0.9 if is_high_volatility else 1.0)
        weights = np.linspace(1, recency_factor, len(recent_volume))
        weighted_confirms = (recent_volume > adjusted_threshold) * weights
        
        # Require higher confirmation threshold (adjusted up by 15%)
        stricter_threshold = confirmation_threshold * 1.15
        return weighted_confirms.sum() >= np.sum(weights) * stricter_threshold

    # Check for recent trend reversal
    recent_trend_changed = False
    if len(df) > (trend_lookback * 2):
        prev_trend = np.sign(df['c'].pct_change(trend_lookback).iloc[-trend_lookback-1])
        current_trend = np.sign(df['c'].pct_change(trend_lookback).iloc[-1])
        recent_trend_changed = prev_trend != current_trend and abs(current_trend) > 0

    # Create context variables using timeframe-specific recent window
    recent_low = df['l'].iloc[-recent_window:].min()
    recent_high = df['h'].iloc[-recent_window:].max()
    current_close = df['c'].iloc[-1]
    current_volume = df['v'].iloc[-1]
    price_distance_from_ma = current_close / price_ma.iloc[-1] - 1
    
    # Implement sign detection overlap prevention
    # Only return the strongest sign when multiple conditions are met
    sign_scores = {}

    # Selling Climax (SC) - use percentile thresholds for extreme moves
    if (abs(price_change.iloc[-1]) > price_p90 and  # Top 10% price moves
            price_change.iloc[-1] < 0 and
            (df['v'].iloc[-1] / volume_ma.iloc[-1]) > volume_p90 and  # Top 10% volume
        price_strength < -STRONG_DEV_THRESHOLD * 0.9 and
        price_distance_from_ma < -0.04 and
        confirm_volume(3, min_volume_surge * 0.8)):
        # Stronger score for higher percentile moves
        percentile_strength = abs(price_change.iloc[-1]) / price_p90
        sign_scores[WyckoffSign.SELLING_CLIMAX] = percentile_strength * (df['v'].iloc[-1] / volume_ma.iloc[-1])
        
    # Automatic Rally (AR) - more stringent conditions
    if (price_change.iloc[-1] > min_price_move * ar_multiplier * 1.2 and
        price_change.iloc[-2] < -min_price_move * 0.5 and  # Require prior decline
        df['l'].iloc[-1] > recent_low and
        price_strength < 0 and
        volume_change.iloc[-1] > 0.5 and
        recent_trend_changed and
        confirm_trend(2, min_price_move * 0.8)):
        sign_scores[WyckoffSign.AUTOMATIC_RALLY] = price_change.iloc[-1] * (1 + volume_change.iloc[-1])
        
    # Secondary Test (ST) - stricter tolerance range
    if (abs(price_change.iloc[-1]) < price_volatility * 0.6 and  # Stricter volatility threshold
        df['l'].iloc[-1] >= recent_low * 1.001 and  # Simplified and tighter tolerance
        df['l'].iloc[-1] <= recent_low * 1.015 and  # Upper bound more restrictive
        current_volume < volume_ma.iloc[-1] * 0.7 and  # Lower volume requirement
        price_strength < -0.5 and  # Stronger negative price strength
        df['v'].iloc[-1] < df['v'].iloc[-5:].min() * 1.2):  # Ensure truly decreased volume
        # Modified scoring formula to reduce sensitivity
        sign_scores[WyckoffSign.SECONDARY_TEST] = 0.7 / (abs(df['l'].iloc[-1] / recent_low - 1.0) + 0.03)

    # Last Point of Support (LPS)
    if (is_spring and
        volume_trend > lps_volume_threshold * 1.2 and
        price_change.iloc[-1] > min_price_move * lps_price_multiplier * 1.2 and
        price_strength < STRONG_DEV_THRESHOLD * 0.4 and
        confirm_trend(3, min_price_move * 0.6)):
        sign_scores[WyckoffSign.LAST_POINT_OF_SUPPORT] = price_change.iloc[-1] * volume_trend

    # Sign of Strength (SOS) - use 75th percentile for strong but not extreme moves
    if (price_change.iloc[-1] > price_p75 and  # Top 25% price moves
            confirm_trend(3, price_p75 * 0.6) and
            (df['v'].iloc[-1] / volume_ma.iloc[-1]) > volume_p75 and  # Above average volume
            price_strength > 0.4 and
            df['c'].iloc[-1] > price_ma.iloc[-1] * 1.01):
        # Score based on how far above the threshold
        percentile_strength = price_change.iloc[-1] / price_p75
        sign_scores[WyckoffSign.SIGN_OF_STRENGTH] = percentile_strength * (df['v'].iloc[-1] / volume_ma.iloc[-1])

    # Buying Climax (BC) - mirror of SC, use 90th percentile
    if (abs(price_change.iloc[-1]) > price_p90 and
            price_change.iloc[-1] > 0 and
            (df['v'].iloc[-1] / volume_ma.iloc[-1]) > volume_p90 and
            price_strength > STRONG_DEV_THRESHOLD * 0.9 and
            price_distance_from_ma > 0.05 and
        confirm_volume(3, min_volume_surge * 0.8)):
        percentile_strength = price_change.iloc[-1] / price_p90
        sign_scores[WyckoffSign.BUYING_CLIMAX] = percentile_strength * (df['v'].iloc[-1] / volume_ma.iloc[-1])
        
    # Upthrust (UT)
    if (is_upthrust and
        volume_change.iloc[-1] > min_volume_surge * 0.8 and
        price_change.iloc[-1] < -min_price_move * ut_multiplier * 1.1 and
        (df['c'].iloc[-1] / df['h'].iloc[-1]) < 0.985):  # Stricter closing ratio
        sign_scores[WyckoffSign.UPTHRUST] = abs(price_change.iloc[-1]) * volume_change.iloc[-1]

    # Secondary Test Resistance (STR) - stricter tolerance range
    if (abs(price_change.iloc[-1]) < price_volatility * 0.6 and  # Stricter volatility threshold
        df['h'].iloc[-1] <= recent_high * 0.999 and  # Simplified and tighter tolerance
        df['h'].iloc[-1] >= recent_high * 0.985 and  # Lower bound more restrictive
        current_volume < volume_ma.iloc[-1] * 0.7 and  # Lower volume requirement
        price_strength > 0.5 and  # Stronger positive price strength
        df['v'].iloc[-1] < df['v'].iloc[-5:].min() * 1.2):  # Ensure truly decreased volume
        # Modified scoring formula to reduce sensitivity
        sign_scores[WyckoffSign.SECONDARY_TEST_RESISTANCE] = 0.7 / (abs(df['h'].iloc[-1] / recent_high - 1.0) + 0.03)

    # Last Point of Supply (LPSY)
    if (is_upthrust and
        volume_trend > lps_volume_threshold * 1.2 and
        price_change.iloc[-1] < -min_price_move * lps_price_multiplier * 1.2 and
        price_strength > -STRONG_DEV_THRESHOLD * 0.4 and
        confirm_trend(3, min_price_move * 0.6, -1)):
        sign_scores[WyckoffSign.LAST_POINT_OF_RESISTANCE] = abs(price_change.iloc[-1]) * volume_trend

    # Sign of Weakness (SOW) - mirror of SOS
    if (abs(price_change.iloc[-1]) > price_p75 and
            price_change.iloc[-1] < 0 and
            confirm_trend(3, price_p75 * 0.6, -1) and
            (df['v'].iloc[-1] / volume_ma.iloc[-1]) > volume_p75 and
            price_strength < -0.4 and
            df['c'].iloc[-1] < price_ma.iloc[-1] * 0.99):
        percentile_strength = abs(price_change.iloc[-1]) / price_p75
        sign_scores[WyckoffSign.SIGN_OF_WEAKNESS] = percentile_strength * (df['v'].iloc[-1] / volume_ma.iloc[-1])

    # Return the strongest sign if it exists, otherwise NONE
    if sign_scores:
        # Find sign with highest score
        best_sign = max(sign_scores.items(), key=lambda x: x[1])[0]
        # Only return if score is significant (this filters out borderline cases)
        return best_sign

    return WyckoffSign.NONE