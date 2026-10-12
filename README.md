# book_scanner

real-time page detection for a webcam. finds an open book or a sheet of paper in the live feed, tracks its four corners with temporal smoothing, and can split a two-page spread into separate left and right pages.

two approaches to the detection step: a **classical OpenCV pipeline** (edges and contours) and a **pretrained neural network** ([DocAligner](https://github.com/DocsaidLab/DocAligner)).


## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python page_identification_working.py
```

`docaligner-docsaid` is only needed for `page_identification_working.py`. The classical scripts run with just OpenCV and NumPy.

## Scripts

| Script | What it does |
| --- | --- |
| `page_identification_working.py` | Main script. Detects the page with DocAligner and tracks it. Single mode follows one quad; split mode tracks the left and right pages separately. |
| `page_identification_learning.py` | The classical detector: input channel → bilateral filter → Canny → morphological close → largest convex four-sided contour. Also has the filter views from the explorer. |
| `input_testing.py` | Interactive explorer for blur, Canny and edge-memory filters, used to choose the edge-detection parameters. |
| `common.py` | Shared helpers: camera setup, key handling, drawing, corner ordering. |
| `tracking.py` | Corner smoothing (`CornerTracker`) and the rough page box used in split mode. |

### Controls

`page_identification_working.py`

| Key | Action |
| --- | --- |
| `m` | Toggle single / split mode |
| `s` | Toggle smoothing |
| `0`, `q`, `Esc` | Quit |

`page_identification_learning.py` and `input_testing.py`

| Key | Action |
| --- | --- |
| `1`–`5` (`6` in the learning script) | Switch view: raw, grey, blur, Canny, edge memory (page highlight) |
| `w` / `s`, `d` / `a` | Blur kernel height / width |
| `i` / `k`, `j` / `l` | Canny high / low threshold |
| `c` | Cycle input channel: grayscale, HSV saturation, LAB lightness (learning script) |
| `0`, `Esc` | Quit |

## How it works

### Detection

| | Classical (`page_identification_learning.py`) | Learned (`page_identification_working.py`) |
| --- | --- | --- |
| Approach | Edge detection, then the largest convex quadrilateral contour | Pretrained DocAligner network predicts the four corners. The frame gets a CLAHE contrast boost and a black border first |
| Dependencies | OpenCV, NumPy | Adds `docaligner-docsaid` |
| Typical strengths | Fast, transparent, every stage can be inspected and tuned | Doesn't rely on clean edges, so it is typically more tolerant of clutter and low contrast |
| Typical weaknesses | Needs visible contrast between page and background; sensitive to Canny thresholds and lighting | Heavier per frame; a pretrained model trained by someone else |

