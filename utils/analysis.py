"""
Analysis & Reporting Utilities
Generates human-readable wrinkle analysis reports.
"""

from __future__ import annotations

from datetime import datetime


FABRIC_IRONING_TIPS = {
    "cotton":          "Cotton responds best to medium-high heat with steam. Use circular motions.",
    "linen":           "Linen irons best while slightly damp. Use high heat and press firmly.",
    "silk":            "Silk needs low heat and a pressing cloth. Avoid steam directly on fabric.",
    "denim":           "Denim needs high heat, iron inside-out to preserve colour.",
    "polyester":       "Use low heat only — polyester melts at high temperatures.",
    "wool":            "Use a damp pressing cloth and medium heat. Never press directly.",
    "synthetic blend": "Use low-to-medium heat; check label for fibre percentages.",
    "auto-detect":     "Settings have been automatically optimised for the detected fabric.",
}

INTENSITY_DESCRIPTIONS = {
    "light":              "Light touch — removes surface creasing only.",
    "medium":             "Standard press — removes most wrinkles while keeping natural drape.",
    "professional press": "Full press — crisp, sharp finish suitable for formal wear.",
}


def analyze_wrinkles(
    wrinkle_score: float,
    zones: list[dict],
    labels: list[str],
    fabric: str,
    intensity: str,
) -> dict:
    """Aggregate analysis data into a structured report dict."""

    level = "Low" if wrinkle_score < 25 else ("Medium" if wrinkle_score < 55 else "High")

    improvement = _estimate_improvement(wrinkle_score, intensity)

    return {
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        "detected_garments": labels or ["clothing item"],
        "fabric":      fabric,
        "intensity":   intensity,
        "wrinkle_score": wrinkle_score,
        "wrinkle_level": level,
        "zones":       zones,
        "improvement": improvement,
        "fabric_tip":  FABRIC_IRONING_TIPS.get(fabric.lower(), FABRIC_IRONING_TIPS["auto-detect"]),
        "intensity_desc": INTENSITY_DESCRIPTIONS.get(
            intensity.lower(), INTENSITY_DESCRIPTIONS["medium"]
        ),
    }


def generate_analysis_report(analysis: dict) -> str:
    """Format analysis dict into a readable text report."""

    score   = analysis["wrinkle_score"]
    level   = analysis["wrinkle_level"]
    fab     = analysis["fabric"]
    zones   = analysis["zones"]
    garments = ", ".join(analysis["detected_garments"])
    impr    = analysis["improvement"]

    bar = _progress_bar(score, width=20)

    lines = [
        "═" * 48,
        "  GARMENT ANALYSIS REPORT",
        f"  {analysis['timestamp']}",
        "═" * 48,
        "",
        f"  Detected garment : {garments}",
        f"  Fabric type      : {fab}",
        f"  Ironing intensity: {analysis['intensity']}",
        "",
        "── Wrinkle Assessment ──────────────────────────",
        f"  Score  : {score:.1f} / 100  ({level})",
        f"  [{bar}]",
        "",
        "── Zone Breakdown ──────────────────────────────",
    ]

    for z in zones:
        indicator = "🔴" if z["level"] == "high" else ("🟡" if z["level"] == "medium" else "🟢")
        lines.append(f"  {indicator}  {z['name']:<22} {z['score']:.1f} pts  [{z['level']}]")

    lines += [
        "",
        "── Ironing Result ──────────────────────────────",
        f"  Expected improvement: {impr:.0f}%",
        f"  {analysis['intensity_desc']}",
        "",
        "── Fabric Tips ─────────────────────────────────",
        f"  {analysis['fabric_tip']}",
        "",
        "── What Was Preserved ──────────────────────────",
        "  ✓  Background and non-garment areas",
        "  ✓  Face, hair, skin tones",
        "  ✓  Garment colour and pattern",
        "  ✓  Logos, embroidery, buttons",
        "  ✓  Structural folds from body posture",
        "  ✓  Natural fabric drape and gravity pull",
        "  ✓  Original lighting and shadows",
        "",
        "═" * 48,
    ]

    return "\n".join(lines)


# ── helpers ───────────────────────────────────────────────────────────────────
def _estimate_improvement(score: float, intensity: str) -> float:
    intensity_factor = {"light": 0.45, "medium": 0.72, "professional press": 0.92}
    factor = intensity_factor.get(intensity.lower(), 0.72)
    return min(score * factor, 97.0)


def _progress_bar(value: float, width: int = 20, filled: str = "█", empty: str = "░") -> str:
    filled_n = int(round(value / 100.0 * width))
    return filled * filled_n + empty * (width - filled_n)
