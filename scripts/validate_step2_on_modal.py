"""
Debug/validation script: confirms src/data/render.py actually resolves and
uses the pinned DejaVu font (FONT_PATH) when run inside the real Modal
container, not the local-dev fallback bitmap font. Local testing
(scripts/sanity_check_data.py) validated the wrapping/layout *logic* but
necessarily ran on the fallback font (no Debian font path on this dev
machine) - this closes that gap by running the same rendering code
against the real production font, where wrapping decisions could differ
due to font-metric differences.

Not part of the production pipeline - a one-off check, run once to
confirm configs/modal_app.py's fonts-dejavu-core apt_install actually
provides what render.py expects, then not needed again unless the image
or font config changes.

Usage: modal run scripts/validate_step2_on_modal.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.modal_app import app, image  # reuse the exact same pinned image;
# `image` already has src/ mounted to /root/src via add_local_dir - no
# separate Mount needed (modal.Mount doesn't exist in this SDK version,
# confirmed locally against the installed modal==1.5.3 before writing this).

SAMPLE_QUESTION = (
    "Every day, Wendi feeds each of her chickens three cups of mixed "
    "chicken feed, containing seeds, mealworms and vegetables to help "
    "keep them healthy.  She gives the chickens their feed in three "
    "separate meals. In the morning, she gives her flock of chickens 15 "
    "cups of feed.  In the afternoon, she gives her chickens another 25 "
    "cups of feed.  How many cups of feed does she need to give her "
    "chickens in the final meal of the day if the size of Wendi's flock "
    "is 20 chickens?"
)


@app.function(image=image, timeout=120)
def validate_render() -> dict:
    import sys as _sys

    _sys.path.insert(0, "/root")
    from src.data.render import FONT_PATH, render_problem_image
    from PIL import ImageFont

    result = {"font_path_configured": FONT_PATH}

    # Confirm the real TTF actually loads (not silently falling back).
    try:
        ImageFont.truetype(FONT_PATH, 28)
        result["font_load_status"] = "OK - real DejaVu TTF loaded"
    except OSError as e:
        result["font_load_status"] = f"FAILED - {e}"
        result["status"] = "FAIL"
        return result

    img = render_problem_image(SAMPLE_QUESTION)
    result["image_size"] = img.size
    result["status"] = "PASS" if img.size[0] > 0 and img.size[1] > 50 else "FAIL"

    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result["png_bytes"] = buf.getvalue()
    return result


@app.local_entrypoint()
def validate():  # unique name required - `app` is shared with modal_app.py,
    # which already registers a local_entrypoint called `main`; Modal
    # requires unique entrypoint names per App, not per file.
    result = validate_render.remote()
    print(f"font_path_configured: {result['font_path_configured']}")
    print(f"font_load_status: {result['font_load_status']}")
    if result["status"] == "PASS":
        print(f"image_size: {result['image_size']}")
        out_path = (
            Path(__file__).resolve().parent.parent
            / "scratch"
            / "modal_render_check.png"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(result["png_bytes"])
        print(f"saved to {out_path}")
    print(f"\nStatus: {result['status']}")
