"""Shared construction of the production seed-bank prompts."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import product

from .demographics import DEMOGRAPHICS


FACIAL_CUES = (
    "wide nose, large wide-set eyes, broad square chin",
    "narrow nose, small close-set eyes, pointed chin",
    "long straight nose, deep-set eyes, narrow jaw",
    "short rounded nose, hooded eyes, rounded chin",
    "slightly crooked nose, heavy brow, asymmetrical jaw",
    "broad flat nose, almond-shaped eyes, strong chin",
    "high cheekbones, downturned eyes, soft jaw",
    "full cheeks, close-set eyes, cleft chin",
)
LENS_FRAMINGS = (
    "26 mm phone close-up, face filling most of the frame",
    "35 mm environmental portrait, upper torso visible",
    "50 mm eye-level head-and-shoulders portrait",
    "85 mm tight headshot, compressed perspective",
)
PRESENTATION = (
    "wearing a plain crew-neck shirt, neutral expression, even soft light, "
    "uncluttered background, natural skin texture"
)


def character_prompt(
    demo_key: str,
    occupation: str,
    personal_detail: str,
    facial_cues: str,
    lens_framing: str,
) -> str:
    """Format one production prompt from an explicit variation tuple."""
    demo = DEMOGRAPHICS[demo_key]
    return (
        f"{demo.lock}, Facial features: {facial_cues}, "
        f"Camera: {lens_framing}, Background: {occupation}; {personal_detail}, "
        f"{PRESENTATION}"
    )


def all_character_prompts(demo_key: str) -> Iterator[str]:
    """Yield the complete production prompt space for one demographic."""
    demo = DEMOGRAPHICS[demo_key]
    for occupation, personal_detail, facial_cues, lens_framing in product(
        demo.occupations,
        demo.personal_details,
        FACIAL_CUES,
        LENS_FRAMINGS,
    ):
        yield character_prompt(
            demo_key,
            occupation,
            personal_detail,
            facial_cues,
            lens_framing,
        )
