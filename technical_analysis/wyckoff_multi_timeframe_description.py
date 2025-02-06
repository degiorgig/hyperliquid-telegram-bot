from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd  # type: ignore[import]

from .wyckoff_multi_timeframe_types import AllTimeframesAnalysis, MultiTimeframeDirection, TimeframeGroupAnalysis

from .wyckoff_types import (
    WyckoffState, WyckoffPhase, MarketPattern, 
    CompositeAction, EffortResult, Timeframe, VolumeState, FundingState, VolatilityState, MarketLiquidity, LiquidationRisk
)


def generate_all_timeframes_description(analysis: AllTimeframesAnalysis) -> str:
    """Generate comprehensive description including three timeframe groups."""
    alignment_pct = f"{analysis.alignment_score * 100:.0f}%"
    confidence_pct = f"{analysis.confidence_level * 100:.0f}%"

    # Get descriptions for all timeframe groups
    short_term_desc = _get_timeframe_trend_description(analysis.short_term)
    intermediate_desc = _get_timeframe_trend_description(analysis.intermediate)
    long_term_desc = _get_timeframe_trend_description(analysis.long_term)

    # Get market structure and context
    structure = _get_full_market_structure(analysis)
    context = _determine_market_context(analysis)
    
    # Get appropriate emoji
    emoji = _get_trend_emoji_all_timeframes(analysis)

    # Generate action plan
    insight = _generate_actionable_insight_all_timeframes(analysis)

    description = (
        f"{emoji} Market Structure Analysis:\n"
        f"Trend: {_determine_trend_strength(analysis)} {context}\n"
        f"Market Structure: {structure}\n\n"
        f"Long-Term View (8h-1d):\n{long_term_desc}\n"
        f"Mid-Term View (1h-4h):\n{intermediate_desc}\n"
        f"Near-Term View (15m-30m):\n{short_term_desc}\n\n"
        f"Signal Quality:\n"
        f"• Timeframe Alignment: {alignment_pct}\n"
        f"• Confidence Level: {confidence_pct}\n\n"
        f"{insight}"
    )

    return description

def _get_full_market_structure(analysis: AllTimeframesAnalysis) -> str:
    """Get comprehensive market structure description across three timeframes."""
    phases = [
        analysis.long_term.dominant_phase,
        analysis.intermediate.dominant_phase,
        analysis.short_term.dominant_phase
    ]
    dominant_phase = max(set(phases), key=phases.count)
    phase_alignment = phases.count(dominant_phase) / len(phases)

    biases = [
        analysis.long_term.momentum_bias,
        analysis.intermediate.momentum_bias,
        analysis.short_term.momentum_bias,
    ]
    dominant_bias = max(set(biases), key=biases.count)
    bias_alignment = biases.count(dominant_bias) / len(biases)

    # New logic for handling conflicting signals
    if phase_alignment > 0.75 and bias_alignment > 0.75:
        # Check for conflicting signals
        is_conflict = (
            (dominant_phase in [WyckoffPhase.MARKDOWN, WyckoffPhase.POSSIBLE_MARKDOWN] and 
             dominant_bias == MultiTimeframeDirection.BULLISH) or
            (dominant_phase in [WyckoffPhase.MARKUP, WyckoffPhase.POSSIBLE_MARKUP] and 
             dominant_bias == MultiTimeframeDirection.BEARISH)
        )
        
        if is_conflict:
            if dominant_phase in [WyckoffPhase.MARKDOWN, WyckoffPhase.POSSIBLE_MARKDOWN]:
                return f"Potential reversal, {dominant_phase.value} showing bullish momentum"
            else:
                return f"Potential reversal, {dominant_phase.value} showing bearish momentum"
        else:
            return f"Strong {dominant_phase.value} structure with {dominant_bias.value} momentum"
            
    elif bias_alignment > 0.75:
        return f"Mixed structure with dominant {dominant_bias.value} momentum"
    elif phase_alignment > 0.75:
        return f"Clear {dominant_phase.value} structure with mixed momentum"
    
    return "Complex structure with mixed signals across timeframes"

def _determine_market_context(analysis: AllTimeframesAnalysis) -> str:
    """
    Determine overall market context considering three timeframes.
    """
    # Weight by timeframe importance
    weights = [
        analysis.short_term.group_weight,
        analysis.intermediate.group_weight,
        analysis.long_term.group_weight
    ]
    total_weight = sum(weights)
    if total_weight == 0:
        return "undefined context"

    # Calculate weighted volume strength
    volume_strength = (
        analysis.short_term.volume_strength * weights[0] +
        analysis.intermediate.volume_strength * weights[1] +
        analysis.long_term.volume_strength * weights[2]
    ) / total_weight

    if analysis.overall_direction == MultiTimeframeDirection.NEUTRAL:
        if volume_strength > 0.7:
            return "high-volume ranging market"
        return "low-volume consolidation"

    context = analysis.overall_direction.value
    if volume_strength > 0.7 and analysis.confidence_level > 0.7:
        return f"high-conviction {context} trend"
    elif volume_strength > 0.5 and analysis.confidence_level > 0.6:
        return f"established {context} trend"
    
    return f"developing {context} bias"

def _determine_trend_strength(analysis: AllTimeframesAnalysis) -> str:
    """
    Determine overall trend strength considering three timeframes.
    """
    # Calculate weighted alignment
    alignments = [
        analysis.short_term.internal_alignment,
        analysis.intermediate.internal_alignment,
        analysis.long_term.internal_alignment
    ]
    weights = [
        analysis.short_term.group_weight,
        analysis.intermediate.group_weight,
        analysis.long_term.group_weight
    ]
    
    total_weight = sum(weights)
    if total_weight == 0:
        return "Undefined"
        
    weighted_alignment = sum(a * w for a, w in zip(alignments, weights)) / total_weight
    
    if weighted_alignment > 0.85:
        return "Extremely strong"
    elif weighted_alignment > 0.7:
        return "Very strong"
    elif weighted_alignment > 0.5:
        return "Strong"
    elif weighted_alignment > 0.3:
        return "Moderate"
    
    return "Weak"

def _get_trend_emoji_all_timeframes(analysis: AllTimeframesAnalysis) -> str:
    """
    Get appropriate trend emoji based on overall analysis state.
    """
    # First check if we have enough confidence
    if analysis.confidence_level < 0.4:
        return "📊"  # Low confidence
        
    # Get the overall trend strength
    trend_strength = analysis.alignment_score > 0.6 and analysis.confidence_level > 0.6
    
    match analysis.overall_direction:
        case MultiTimeframeDirection.BULLISH:
            if trend_strength:
                return "📈"  # Strong bullish
            return "↗️"  # Weak bullish
            
        case MultiTimeframeDirection.BEARISH:
            if trend_strength:
                return "📉"  # Strong bearish
            return "↘️"  # Weak bearish
            
        case MultiTimeframeDirection.NEUTRAL:
            # Check if we're in consolidation or in conflict
            if analysis.alignment_score > 0.6:
                return "↔️"  # Clear consolidation
            return "🔄"  # Mixed signals
            
    return "📊"  # Fallback for unknown states

def _generate_actionable_insight_all_timeframes(analysis: AllTimeframesAnalysis) -> str:
    """
    Generate comprehensive actionable insights considering all timeframes.
    """
    if analysis.confidence_level < 0.5:
        return "<b>Analysis:</b>\nLow confidence signals across timeframes.\n<b>Recommendation:</b>\nReduce exposure and wait for clearer setups."

    def get_full_context() -> tuple[str, str]:
        """Get base signal and action plan based on all timeframes."""
        if analysis.overall_direction == MultiTimeframeDirection.BULLISH:
            if analysis.confidence_level > 0.7:
                base_signal = (
                    f"Strong bullish alignment across multiple timeframes. "
                    f"Intermediate timeframe shows {analysis.intermediate.dominant_phase.value} phase, "
                    f"supported by {analysis.intermediate.dominant_action.value} and strong volume (strength: {analysis.intermediate.volume_strength:.2f}). "
                    f"High conviction uptrend."
                )
                action_plan = (
                    "Longs: Prioritize entries on dips with tight stop-losses below key support levels. "
                    "Consider adding to positions as the trend strengthens.\n"
                    "Shorts: Avoid, high risk of bull traps. If shorting, use extremely tight stop-losses."
                )
                return base_signal, action_plan

            base_signal = (
                f"Developing bullish structure with mixed timeframe signals. "
                f"Intermediate timeframe shows {analysis.intermediate.dominant_phase.value} phase, "
                f"indicating {analysis.intermediate.dominant_action.value}. "
                f"Volume strength is moderate ({analysis.intermediate.volume_strength:.2f}). Watch for confirmation signals."
            )
            action_plan = (
                "Longs: Scaled entries near support zones with careful risk management. "
                "Use smaller position sizes due to mixed signals.\n"
                "Shorts: Only consider at significant resistance with strong bearish signals. "
                "Confirm with price action before entering."
            )
            return base_signal, action_plan

        elif analysis.overall_direction == MultiTimeframeDirection.BEARISH:
            if analysis.confidence_level > 0.7:
                base_signal = (
                    f"Strong bearish alignment across multiple timeframes. "
                    f"Intermediate timeframe shows {analysis.intermediate.dominant_phase.value} phase, "
                    f"confirmed by {analysis.intermediate.dominant_action.value} and sustained selling pressure. "
                    f"High conviction downtrend with volume strength of {analysis.intermediate.volume_strength:.2f}."
                )
                action_plan = (
                    "Shorts: Focus on entries during rallies with tight stop-losses above key resistance levels. "
                    "Add to positions as the trend accelerates.\n"
                    "Longs: Avoid, high risk of bear traps. If longing, use extremely tight stop-losses."
                )
                return base_signal, action_plan

            base_signal = (
                f"Developing bearish structure with mixed timeframe signals. "
                f"Intermediate timeframe shows {analysis.intermediate.dominant_phase.value} phase, "
                f"indicating {analysis.intermediate.dominant_action.value}. "
                f"Awaiting further bearish confirmation. Volume strength is {analysis.intermediate.volume_strength:.2f}."
            )
            action_plan = (
                "Shorts: Scaled entries near resistance zones with strict risk control. "
                "Confirm bearish signals with price action and volume.\n"
                "Longs: Only attempt at major support with clear bullish reversal patterns. "
                "Be cautious of potential bear traps."
            )
            return base_signal, action_plan

        volume_context = "high-volume indecision" if analysis.intermediate.volume_strength > 0.7 else "low-volume consolidation"
        base_signal = f"Mixed signals across timeframes indicating a transitional or ranging market. {volume_context}."
        action_plan = (
            "Both Directions: Trade range extremes with confirmation. "
            "Use smaller position sizes and tighter stop-losses.\n"
            "Avoid large positions until a clear trend emerges. "
            "Focus on short-term trades."
        )
        return base_signal, action_plan

    base_signal, action_plan = get_full_context()

    # Add timeframe-specific insights
    timeframe_insights = []
    if analysis.short_term.momentum_bias != analysis.long_term.momentum_bias:
        timeframe_insights.append(
            f"Timeframe divergence: Long-term bias is {analysis.long_term.momentum_bias.value}, while short-term bias is {analysis.short_term.momentum_bias.value}. "
            f"Potential for trend reversal or continuation based on breakout direction. "
            f"Watch for a break of key levels to confirm the direction."
        )
    if analysis.short_term.dominant_phase != analysis.intermediate.dominant_phase:
        timeframe_insights.append(
            f"Phase mismatch: Short-term in {analysis.short_term.dominant_phase.value}, but mid-term in {analysis.intermediate.dominant_phase.value}. "
            f"Expect volatility as market seeks equilibrium. "
            f"Be prepared for rapid price swings and adjust stop-losses accordingly."
        )

    # Add risk warnings
    risk_warnings = []
    high_liq_risks = [tf.dominant_phase.value for tf in [analysis.short_term, analysis.intermediate, analysis.long_term] if tf.liquidation_risk == LiquidationRisk.HIGH]
    if high_liq_risks:
        risk_warnings.append(
            f"High liquidation risk on {', '.join(high_liq_risks)}. "
            f"Reduce leverage significantly to avoid forced liquidations. "
            f"Consider using isolated margin."
        )

    if analysis.short_term.volatility_state == VolatilityState.HIGH:
        risk_warnings.append(
            "High short-term volatility. "
            "Use smaller position sizes and wider stop-losses to account for rapid price swings. "
            "Avoid over-leveraging."
        )

    # Combine risk warnings if both high liquidation risk and high volatility are present
    if high_liq_risks and analysis.short_term.volatility_state == VolatilityState.HIGH:
        risk_warnings.append(
            "Combined high liquidation risk and high short-term volatility. "
            "Extreme caution is advised. Consider staying out of the market until conditions stabilize."
        )

    # Format the complete insight
    insights = [f"<b>Market Overview:</b>\n{base_signal}"]
    if timeframe_insights:
        insights.append("\n<b>Timeframe Analysis:</b>\n" + "\n".join(f"- {i}" for i in timeframe_insights))
    insights.append(f"\n<b>Trading Strategy:</b>\n{action_plan}")
    if risk_warnings:
        insights.append("\n<b>Risk Management:</b>\n" + "\n".join(f"- {w}" for w in risk_warnings))

    return "\n".join(insights)

 
def _get_timeframe_trend_description(analysis: TimeframeGroupAnalysis) -> str:
    """Generate trend description for a timeframe group."""
    return f"• {analysis.dominant_phase.value} phase {analysis.dominant_action.value} with " + (
        "strong volume" if analysis.volume_strength > 0.7 else
        "moderate volume" if analysis.volume_strength > 0.4 else
        "light volume"
    )
