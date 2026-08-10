"""
Self-rendered "image mode" GSM8K data (PLAN.md Section 3 Step 2, and the
Condition D/E images described in Section 1.7).

Renders a problem's TEXT onto a single image using plain, deterministic
typography (PIL/Pillow's ImageDraw) - NOT any generative/AI process. This
is a critical design choice, not an implementation detail: the rendering
pipeline has no access to, and is never run by, the models under study
(base or RL-trained) - it is pure preprocessing that happens once, before
any model ever sees the result. See the conversation/PLAN.md discussion of
why the real ZJU-REAL GSM8K-V dataset (a multi-scene illustrated
benchmark) is NOT used as the image stimulus source here - it doesn't
give the "same content, different representation" control this
experiment needs, unlike this literal text-rendering.

Rendering configuration is fixed and documented here (PLAN.md Section 11
requires "one consistent rendering configuration, documented precisely")
so results aren't an artifact of inconsistent rendering choices - prior
work found rendering-choice swings of up to ~47 accuracy points.

VERIFICATION STATUS: confirmed on live Modal infrastructure (2026-08-10,
scripts/validate_step2_on_modal.py) that FONT_PATH loads correctly (the
real DejaVu TTF, not a silent fallback) and that rendering produces a
fully legible, non-clipped image. This caught a real bug on the first
attempt: the original implementation used character-count-based wrapping
(textwrap.wrap), tuned against the narrower local-dev fallback font. Under
the real, wider production font, the same character count overflowed the
canvas and silently clipped the right edge of every line - invisible
locally, only caught by rendering against the actual production font.
Fixed by wrapping on measured pixel width (font.getlength) instead of
character count, which is correct regardless of which font is in use.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# --- Fixed rendering configuration (PLAN.md Section 11) ---------------------
# Production path: inside the pinned Modal image, with `fonts-dejavu-core`
# apt-installed (see configs/modal_app.py). Every image used in any
# reported result MUST be rendered with this exact font, not the local
# fallback below.
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_SIZE = 28
IMAGE_WIDTH = 800
MARGIN = 40
LINE_SPACING = 10
BACKGROUND_COLOR = "white"
TEXT_COLOR = "black"

_MAX_TEXT_WIDTH = IMAGE_WIDTH - 2 * MARGIN


def _load_font():
    try:
        return ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except OSError:
        # Local dev fallback (e.g. Windows machines without the Debian font
        # path) - fine for quick local testing of the rendering *logic*,
        # but NOT reproducible across machines. Any image that ends up in
        # an actual eval/training run must be rendered inside the pinned
        # Modal image where FONT_PATH resolves, not with this fallback.
        return ImageFont.load_default(size=FONT_SIZE)


def _wrap_to_pixel_width(text: str, font, max_width: int) -> list[str]:
    """
    Word-wrap by actual rendered pixel width (font.getlength), not
    character count. Character-count wrapping is NOT font-agnostic - it
    was tried first here and silently clipped text at the right edge of
    the image once run against the real production font (DejaVu), whose
    average glyph width differs from the local dev fallback font used
    during initial testing. Measuring real width is the only robust fix.
    """
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if font.getlength(candidate) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def render_problem_image(question_text: str) -> Image.Image:
    """
    Render a single GSM8K problem's text onto one image using the fixed
    configuration above. Deterministic: identical text + identical config
    always produces an identical image (byte-for-byte, given the same
    font file/rendering backend). Wrapping is measured in actual pixel
    width against the loaded font, so it stays correct regardless of
    which font is in use (see _wrap_to_pixel_width).
    """
    font = _load_font()
    wrapped_lines = _wrap_to_pixel_width(question_text, font, _MAX_TEXT_WIDTH)

    # Defensive check, not just a comment: every line must actually fit.
    # A single word longer than _MAX_TEXT_WIDTH would violate this (not
    # expected for GSM8K's plain-language problems, but fail loudly rather
    # than silently ship a clipped image if it ever happens).
    for line in wrapped_lines:
        measured = font.getlength(line)
        if measured > _MAX_TEXT_WIDTH:
            raise ValueError(
                f"Line exceeds max width even after wrapping "
                f"({measured:.0f}px > {_MAX_TEXT_WIDTH}px): {line!r}. "
                f"Likely an unsplittable long word - IMAGE_WIDTH or MARGIN "
                f"may need adjusting."
            )

    line_height = FONT_SIZE + LINE_SPACING
    text_height = len(wrapped_lines) * line_height
    image_height = text_height + 2 * MARGIN

    img = Image.new("RGB", (IMAGE_WIDTH, image_height), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    y = MARGIN
    for line in wrapped_lines:
        draw.text((MARGIN, y), line, font=font, fill=TEXT_COLOR)
        y += line_height

    return img


def render_and_save(question_text: str, out_path: str | Path) -> Path:
    """Render and save to disk (PNG), returning the path written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = render_problem_image(question_text)
    img.save(out_path)
    return out_path
