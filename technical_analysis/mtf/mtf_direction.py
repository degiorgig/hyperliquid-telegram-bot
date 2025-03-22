import pandas as pd  # type: ignore[import]
import numpy as np
import os
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Final
from dataclasses import dataclass
from technical_analysis.wyckoff_types import SignificantLevelsData
from logging_utils import logger


from .wyckoff_multi_timeframe_types import AllTimeframesAnalysis, MultiTimeframeDirection, TimeframeGroupAnalysis, MultiTimeframeContext

from technical_analysis.wyckoff_types import (
    WyckoffState, WyckoffPhase, MarketPattern, _TIMEFRAME_SETTINGS,
    is_bearish_action, is_bullish_action, is_bearish_phase, is_bullish_phase,
    CompositeAction, EffortResult, Timeframe, VolumeState, FundingState, VolatilityState, MarketLiquidity
)

from .wyckoff_multi_timeframe_types import (
    SHORT_TERM_TIMEFRAMES, INTERMEDIATE_TIMEFRAMES, LONG_TERM_TIMEFRAMES, CONTEXT_TIMEFRAMES,
    STRONG_MOMENTUM, MODERATE_MOMENTUM, WEAK_MOMENTUM,
    MIXED_MOMENTUM, LOW_MOMENTUM,
    SHORT_TERM_WEIGHT, INTERMEDIATE_WEIGHT, LONG_TERM_WEIGHT,
    DIRECTIONAL_WEIGHT, VOLUME_WEIGHT, PHASE_WEIGHT, MODERATE_VOLUME_THRESHOLD
)


def determine_overall_direction(analyses: List[TimeframeGroupAnalysis]) -> MultiTimeframeDirection:
    """Determine overall direction considering Wyckoff phase weights and uncertainty for each timeframe group."""
    if not analyses:
        return MultiTimeframeDirection.NEUTRAL

    # Group timeframes by their actual settings in _TIMEFRAME_SETTINGS
    timeframe_groups = {
        'short': [a for a in analyses if a.group_weight in {_TIMEFRAME_SETTINGS[tf].phase_weight 
                                                           for tf in SHORT_TERM_TIMEFRAMES}],
        'mid': [a for a in analyses if a.group_weight in {_TIMEFRAME_SETTINGS[tf].phase_weight 
                                                         for tf in INTERMEDIATE_TIMEFRAMES}],
        'long': [a for a in analyses if a.group_weight in {_TIMEFRAME_SETTINGS[tf].phase_weight 
                                                          for tf in LONG_TERM_TIMEFRAMES}],
        'context': [a for a in analyses if a.group_weight in {_TIMEFRAME_SETTINGS[tf].phase_weight
                                                            for tf in CONTEXT_TIMEFRAMES}]
    }

    # Count dominant phases with equal treatment for bullish/bearish
    # Only count confident phases (not uncertain)
    bullish_phases = sum(1 for a in analyses if is_bullish_phase(a.dominant_phase) and not a.uncertain_phase)
    bearish_phases = sum(1 for a in analyses if is_bearish_phase(a.dominant_phase) and not a.uncertain_phase)
    
    # Count strong volume signals by direction
    bullish_volume_signals = sum(1 for a in analyses if a.momentum_bias == MultiTimeframeDirection.BULLISH 
                                and a.volume_strength > MODERATE_VOLUME_THRESHOLD)
    bearish_volume_signals = sum(1 for a in analyses if a.momentum_bias == MultiTimeframeDirection.BEARISH 
                                and a.volume_strength > MODERATE_VOLUME_THRESHOLD)
    
    # Enhanced market structure consistency check - consider both phases and volume
    market_structure_bias = None
    phase_threshold = len(analyses) / 2.5  # Stricter threshold for phases
    
    # Calculate overall market structure considering both phases and volume
    if bearish_phases > bullish_phases and bearish_phases >= phase_threshold:
        market_structure_bias = MultiTimeframeDirection.BEARISH
    elif bullish_phases > bearish_phases and bullish_phases >= phase_threshold:
        market_structure_bias = MultiTimeframeDirection.BULLISH
        
    # Volume can override if extremely strong in one direction
    if bullish_volume_signals > bearish_volume_signals * 2 and bullish_volume_signals >= len(analyses) / 2:
        if market_structure_bias != MultiTimeframeDirection.BULLISH:
            # Consider as potential divergence, but only if strong enough
            market_structure_bias = MultiTimeframeDirection.BULLISH if bullish_volume_signals >= len(analyses) * 0.6 else market_structure_bias
    elif bearish_volume_signals > bullish_volume_signals * 2 and bearish_volume_signals >= len(analyses) / 2:
        if market_structure_bias != MultiTimeframeDirection.BEARISH:
            # Consider as potential divergence, but only if strong enough
            market_structure_bias = MultiTimeframeDirection.BEARISH if bearish_volume_signals >= len(analyses) * 0.6 else market_structure_bias
    
    def get_weighted_direction(group: List[TimeframeGroupAnalysis]) -> Tuple[MultiTimeframeDirection, float, float]:
        """
        Calculate the weighted directional bias for a group of timeframes.
        
        Args:
            group: List of timeframe group analyses
            
        Returns:
            Tuple containing:
            - The strongest directional bias (BULLISH, BEARISH, or NEUTRAL)
            - The weight/strength of that bias (0.0-1.0)
            - The average volume strength across all timeframes
        """
        if not group:
            return MultiTimeframeDirection.NEUTRAL, 0.0, 0.0

        group_total_weight = sum(a.group_weight for a in group)
        if group_total_weight == 0:
            return MultiTimeframeDirection.NEUTRAL, 0.0, 0.0
        
        def calculate_signal_weight(analysis: TimeframeGroupAnalysis, direction: MultiTimeframeDirection) -> float:
            """Calculate adjusted weight for a single timeframe analysis in the specified direction."""
            if analysis.momentum_bias != direction:
                return 0.0
                
            # Base weight normalized by total group weight
            base_weight = analysis.group_weight / group_total_weight
            
            # Apply multipliers for technical factors
            multipliers = [
                1 + analysis.volume_strength * 0.3,                       # Volume boost
                1.2 if analysis.volatility_state == VolatilityState.HIGH else 1.0,  # Volatility adjustment
                0.6 if analysis.uncertain_phase else 1.0,                  # Uncertainty penalty
                # Phase consistency factor - identical treatment for bullish/bearish
                0.7 if ((direction == MultiTimeframeDirection.BULLISH and is_bearish_phase(analysis.dominant_phase)) or
                      (direction == MultiTimeframeDirection.BEARISH and is_bullish_phase(analysis.dominant_phase))) 
                    else 1.0
            ]
            
            # Apply all multipliers to base weight
            return base_weight * np.prod(multipliers) # type: ignore
        
        # Calculate weighted signals for each direction
        weighted_signals = {
            direction: sum(calculate_signal_weight(a, direction) for a in group)
            for direction in [MultiTimeframeDirection.BULLISH, MultiTimeframeDirection.BEARISH]
        }
        
        # Calculate average volume
        avg_volume = sum(a.volume_strength for a in group) / len(group) if group else 0
        
        # Find strongest signal, defaulting to NEUTRAL if no clear direction
        if weighted_signals[MultiTimeframeDirection.BULLISH] == 0 and weighted_signals[MultiTimeframeDirection.BEARISH] == 0:
            return MultiTimeframeDirection.NEUTRAL, 0.0, avg_volume
        
        strongest_dir = max(weighted_signals.items(), key=lambda x: x[1])
        return strongest_dir[0], strongest_dir[1], avg_volume

    # Get weighted directions with volume context
    st_dir, st_weight, st_vol = get_weighted_direction(timeframe_groups['short'])
    mid_dir, mid_weight, mid_vol = get_weighted_direction(timeframe_groups['mid'])
    lt_dir, lt_weight, _ = get_weighted_direction(timeframe_groups['long'])
    ctx_dir, _, _ = get_weighted_direction(timeframe_groups['context'])

    # Calculate consistency scores to measure signal reliability
    dir_consistency = 0.0
    all_directions = [d for d in [st_dir, mid_dir, lt_dir, ctx_dir] if d != MultiTimeframeDirection.NEUTRAL]
    if all_directions:
        # Count occurrences of each non-neutral direction
        dir_counts = {} # type: Dict[MultiTimeframeDirection, int]
        for d in all_directions:
            dir_counts[d] = dir_counts.get(d, 0) + 1
        
        # Calculate direction consistency score (1.0 means all timeframes agree)
        max_count = max(dir_counts.values()) if dir_counts else 0
        dir_consistency = max_count / len(all_directions) if all_directions else 0.0
    
    # Detect high confidence directional moves
    high_confidence_direction = None
    if dir_consistency >= 0.75:  # At least 75% agreement among timeframes
        # Find the most common direction
        all_dirs = [d for d in [st_dir, mid_dir, lt_dir, ctx_dir] if d != MultiTimeframeDirection.NEUTRAL]
        if all_dirs:
            dir_counts = {}
            for d in all_dirs:
                dir_counts[d] = dir_counts.get(d, 0) + 1
            high_confidence_direction = max(dir_counts.items(), key=lambda x: x[1])[0]
            
            # Verify with volume confirmation
            direction_vols = {
                MultiTimeframeDirection.BULLISH: sum(v for d, _, v in [(st_dir, st_weight, st_vol), 
                                                                    (mid_dir, mid_weight, mid_vol)] 
                                             if d == MultiTimeframeDirection.BULLISH),
                MultiTimeframeDirection.BEARISH: sum(v for d, _, v in [(st_dir, st_weight, st_vol), 
                                                                     (mid_dir, mid_weight, mid_vol)]
                                              if d == MultiTimeframeDirection.BEARISH)
            }
            
            # Require adequate volume in the direction
            required_vol = MODERATE_VOLUME_THRESHOLD * 0.9  # Slightly lower threshold when we have consistency
            if direction_vols.get(high_confidence_direction, 0) < required_vol:
                high_confidence_direction = None  # Reset if volume doesn't confirm

    # Apply our detection logic, but first check for high confidence signals
    if high_confidence_direction:
        # If we have a high-confidence signal based on consistency across timeframes
        # and it doesn't contradict market structure, use it
        if not market_structure_bias or market_structure_bias == high_confidence_direction:
            return high_confidence_direction
        # If it contradicts market structure, we need stronger evidence
        elif (st_dir == high_confidence_direction and st_vol > MODERATE_VOLUME_THRESHOLD * 1.25 and
              mid_dir == high_confidence_direction):
            # Allow contradiction only with very strong evidence
            return high_confidence_direction

    # Check for high-conviction intraday moves first with enhanced criteria
    if st_dir != MultiTimeframeDirection.NEUTRAL:
        # Enhanced short-term signal requirements
        # 1. Strong weight
        # 2. Sufficient volume
        # 3. Agreement with at least one other timeframe OR very strong volume
        if (st_weight > 0.8 and st_vol > MODERATE_VOLUME_THRESHOLD and
            (mid_dir == st_dir or lt_dir == st_dir or 
             (st_vol > MODERATE_VOLUME_THRESHOLD * 1.5 and not any(a.uncertain_phase for a in timeframe_groups['short'])))):
            
            # Only reject if there's a direct conflict with intermediate timeframe
            if mid_dir != MultiTimeframeDirection.NEUTRAL and mid_dir != st_dir and mid_vol > st_vol:
                return MultiTimeframeDirection.NEUTRAL  # Conflict with stronger intermediate trend
            
            # Skip uncertain phases
            if any(a.uncertain_phase for a in timeframe_groups['short']):
                return MultiTimeframeDirection.NEUTRAL
                
            # Market structure consistency check with flexibility for strong signals
            if market_structure_bias and market_structure_bias != st_dir:
                # Allow contradiction only with very strong short-term signals
                if st_vol > MODERATE_VOLUME_THRESHOLD * 1.5 and st_weight > 0.9:
                    # This might be a reversal against the larger structure
                    return st_dir
                return MultiTimeframeDirection.NEUTRAL
                
            return st_dir

    # Check for strong intermediate trend - symmetric treatment
    if mid_dir != MultiTimeframeDirection.NEUTRAL and mid_weight > 0.7:
        # Skip uncertain phases
        if any(a.uncertain_phase for a in timeframe_groups['mid']):
            return MultiTimeframeDirection.NEUTRAL
            
        # Allow counter-trend short-term moves if volume is low
        if st_dir != MultiTimeframeDirection.NEUTRAL and st_dir != mid_dir:
            # Equal threshold for both bullish and bearish signals
            if st_vol < MODERATE_VOLUME_THRESHOLD:  # Using consistent threshold
                if market_structure_bias and market_structure_bias != mid_dir:
                    return MultiTimeframeDirection.NEUTRAL
                return mid_dir
            return MultiTimeframeDirection.NEUTRAL  # High volume conflict
        
        # Market structure consistency check
        if market_structure_bias and market_structure_bias != mid_dir:
            return MultiTimeframeDirection.NEUTRAL
            
        return mid_dir

    # Consider longer-term trend with confirmation - equal treatment
    if lt_dir != MultiTimeframeDirection.NEUTRAL and lt_weight > 0.6:
        # Skip uncertain phases
        if any(a.uncertain_phase for a in timeframe_groups['long']):
            return MultiTimeframeDirection.NEUTRAL
            
        if mid_dir == lt_dir or st_dir == lt_dir:
            if market_structure_bias and market_structure_bias != lt_dir:
                return MultiTimeframeDirection.NEUTRAL
            return lt_dir
            
        # Equal sensitivity for bearish and bullish signals
        if mid_vol < MODERATE_VOLUME_THRESHOLD and st_vol < MODERATE_VOLUME_THRESHOLD:  # Using consistent threshold
            if market_structure_bias and market_structure_bias != lt_dir:
                return MultiTimeframeDirection.NEUTRAL
            return lt_dir

    # Check for aligned moves even with lower weights - equal threshold
    if st_dir == mid_dir and st_dir != MultiTimeframeDirection.NEUTRAL:
        # Skip uncertain phases in both groups
        if any(a.uncertain_phase for a in timeframe_groups['short'] + timeframe_groups['mid']):
            return MultiTimeframeDirection.NEUTRAL
            
        # Same threshold for both bullish and bearish signals
        if (st_weight + mid_weight) / 2 > 0.45:
            if market_structure_bias and market_structure_bias != st_dir:
                return MultiTimeframeDirection.NEUTRAL
            return st_dir

    # When in doubt, respect market structure if phases are certain
    if market_structure_bias:
        # Only trust market structure if we have enough certain phases
        certain_phases = sum(1 for a in analyses if not a.uncertain_phase)
        if certain_phases >= len(analyses) / 2:
            return market_structure_bias

    return MultiTimeframeDirection.NEUTRAL
