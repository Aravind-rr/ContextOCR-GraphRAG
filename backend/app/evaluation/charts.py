from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any


COLORS = ["#146c63", "#2f8f83", "#74b8ad", "#d89b2b", "#c95d4b", "#52616b"]


def _svg(title: str, body: str, width: int, height: int) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
<rect width="100%" height="100%" fill="#fff"/><style>text{{font-family:Arial,sans-serif;fill:#16302d}}.grid{{stroke:#dce6e3;stroke-width:1}}.axis{{stroke:#738783;stroke-width:1.2}}.label{{font-size:12px}}.title{{font-size:20px;font-weight:700}}</style>
<text x="24" y="30" class="title">{html.escape(title)}</text>{body}</svg>'''


def _write(path: Path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    return path.name


def bar_chart(path: Path, title: str, labels: list[str], values: list[float], *, y_label: str, width: int, height: int, maximum: float | None = None, x_label: str = "Category") -> str:
    left, top, right, bottom = 76, 55, 25, 85
    plot_w, plot_h = width - left - right, height - top - bottom
    top_value = maximum or max(values or [1])
    top_value = max(top_value, 1e-9)
    body = [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/><line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>']
    for tick in range(6):
        value = top_value * tick / 5
        y = top + plot_h - plot_h * tick / 5
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/><text x="{left-8}" y="{y+4:.1f}" text-anchor="end" class="label">{value:.2f}</text>')
    slot = plot_w / max(1, len(values)); bar_w = max(8, min(64, slot * .62))
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + slot * index + (slot - bar_w) / 2
        bar_h = plot_h * max(0, value) / top_value
        y = top + plot_h - bar_h
        short = label if len(label) <= 18 else label[:15] + "…"
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="3" fill="{COLORS[index % len(COLORS)]}"/><text x="{x+bar_w/2:.1f}" y="{y-7:.1f}" text-anchor="middle" class="label">{value:.3f}</text><text x="{x+bar_w/2:.1f}" y="{top+plot_h+19}" text-anchor="middle" class="label">{html.escape(short)}</text>')
    body.append(f'<text x="18" y="{top+plot_h/2}" transform="rotate(-90 18 {top+plot_h/2})" text-anchor="middle" class="label">{html.escape(y_label)}</text><text x="{left+plot_w/2}" y="{height-12}" text-anchor="middle" class="label">{html.escape(x_label)}</text>')
    return _write(path, _svg(title, "".join(body), width, height))


def histogram(path: Path, title: str, values: list[float], *, x_label: str, width: int, height: int, bins: int) -> str:
    low, high = min(values), max(values)
    if math.isclose(low, high):
        low, high = low - .5, high + .5
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int((value - low) / (high - low) * bins))
        counts[index] += 1
    labels = [f"{low + (high-low)*i/bins:.2f}" for i in range(bins)]
    return bar_chart(path, title, labels, [float(value) for value in counts], y_label="Documents", x_label=x_label, width=width, height=height, maximum=float(max(counts)))


def confusion_matrix_chart(path: Path, title: str, matrix: dict[str, dict[str, int]], labels: list[str], width: int, height: int) -> str:
    left, top = 190, 75
    size = min(width-left-50, height-top-80)
    cell = size / len(labels)
    maximum = max((value for row in matrix.values() for value in row.values()), default=1) or 1
    body = [f'<text x="{left+size/2}" y="{height-16}" text-anchor="middle" class="label">Predicted class</text><text x="22" y="{top+size/2}" transform="rotate(-90 22 {top+size/2})" text-anchor="middle" class="label">Actual class</text>']
    for row_index, actual in enumerate(labels):
        body.append(f'<text x="{left-12}" y="{top+row_index*cell+cell/2+4:.1f}" text-anchor="end" class="label">{html.escape(actual)}</text>')
        for column_index, predicted in enumerate(labels):
            value = matrix[actual][predicted]
            opacity = .16 + .84 * value / maximum
            x, y = left+column_index*cell, top+row_index*cell
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell:.1f}" height="{cell:.1f}" fill="#146c63" fill-opacity="{opacity:.3f}" stroke="#fff"/><text x="{x+cell/2:.1f}" y="{y+cell/2+5:.1f}" text-anchor="middle" class="label">{value}</text>')
    for index, label in enumerate(labels):
        body.append(f'<text x="{left+index*cell+cell/2:.1f}" y="{top+size+20:.1f}" text-anchor="middle" class="label">{html.escape(label)}</text>')
    return _write(path, _svg(title, "".join(body), width, height))


def scatter_chart(path: Path, title: str, points: list[tuple[float, float, str]], *, x_label: str, y_label: str, width: int, height: int) -> str:
    left, top, right, bottom = 80, 55, 30, 70
    plot_w, plot_h = width-left-right, height-top-bottom
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    if math.isclose(xmin, xmax): xmin, xmax = xmin-.05, xmax+.05
    if math.isclose(ymin, ymax): ymin, ymax = ymin-.05, ymax+.05
    body = [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/><line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>']
    for tick in range(6):
        x = left + plot_w*tick/5; y = top+plot_h-plot_h*tick/5
        body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" class="grid"/><line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/><text x="{x:.1f}" y="{top+plot_h+20}" text-anchor="middle" class="label">{xmin+(xmax-xmin)*tick/5:.2f}</text><text x="{left-8}" y="{y+4:.1f}" text-anchor="end" class="label">{ymin+(ymax-ymin)*tick/5:.2f}</text>')
    for index, (x_value, y_value, label) in enumerate(points):
        x = left + (x_value-xmin)/(xmax-xmin)*plot_w; y = top+plot_h-(y_value-ymin)/(ymax-ymin)*plot_h
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{COLORS[index % len(COLORS)]}"><title>{html.escape(label)}: {x_value:.4f}, {y_value:.4f}</title></circle>')
    body.append(f'<text x="{left+plot_w/2}" y="{height-18}" text-anchor="middle" class="label">{html.escape(x_label)}</text><text x="18" y="{top+plot_h/2}" transform="rotate(-90 18 {top+plot_h/2})" text-anchor="middle" class="label">{html.escape(y_label)}</text>')
    return _write(path, _svg(title, "".join(body), width, height))


def generate_charts(documents: list[dict[str, Any]], output: Path, config: dict[str, Any], evaluation_id: str, classification: dict[str, Any] | None = None) -> tuple[list[dict[str, str]], dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    width, height = int(config["charts"]["width"]), int(config["charts"]["height"])
    bins = int(config["charts"]["histogram_bins"])
    charts: list[dict[str, str]] = []
    sources: dict[str, Any] = {}
    context = f"Evaluation {evaluation_id[:8]}"

    def add(name: str, title: str, file_name: str, source: Any) -> None:
        charts.append({"name": name, "title": title, "file": file_name})
        sources[name] = source

    if len(documents) == 1:
        components = {key: value for key, value in documents[0]["quality_components"].items() if value is not None}
        labels, values = list(components), [100 * float(value) for value in components.values()]
        file_name = bar_chart(output/"quality_components.svg", f"Quality components — {context}", labels, values, y_label="Component score (0–100)", x_label="Metric component", width=width, height=height, maximum=100)
        add("quality_components", "Quality components", file_name, components)
    else:
        labels = [document["filename"] for document in documents]
        values = [float(document["final_quality_score"]) for document in documents]
        add("quality_by_document", "Quality score by document", bar_chart(output/"quality_by_document.svg", f"Quality score by document — {context}", labels, values, y_label="Quality score (0–100)", x_label="Document", width=width, height=height, maximum=100), dict(zip(labels, values)))
        add("quality_distribution", "Quality score distribution", histogram(output/"quality_distribution.svg", f"Quality score distribution — {context}", values, x_label="Quality score", width=width, height=height, bins=bins), values)
        class_names = ["High Quality", "Medium Quality", "Low Quality"]
        class_values = [float(sum(document["quality_class"] == name for document in documents)) for name in class_names]
        add("class_distribution", "Quality class distribution", bar_chart(output/"class_distribution.svg", f"Quality class distribution — {context}", class_names, class_values, y_label="Documents", x_label="Quality class", width=width, height=height, maximum=max(class_values or [1])), dict(zip(class_names, class_values)))

    for metric, title in (("cer", "CER distribution"), ("wer", "WER distribution"), ("noise_ratio", "Noise ratio distribution"), ("ocr_confidence", "OCR confidence distribution")):
        values = [float(document[metric]) for document in documents if document.get(metric) is not None]
        if len(values) >= 2:
            name = metric + "_distribution"
            add(name, title, histogram(output/f"{name}.svg", f"{title} — {context}", values, x_label=metric, width=width, height=height, bins=bins), values)

    for x_metric, y_metric, name, title in (
        ("noise_ratio", "final_quality_score", "noise_vs_quality", "Noise ratio vs quality score"),
        ("ocr_confidence", "final_quality_score", "confidence_vs_quality", "OCR confidence vs quality score"),
        ("valid_word_ratio", "final_quality_score", "valid_words_vs_quality", "Valid-word ratio vs quality score"),
        ("cer", "wer", "cer_vs_wer", "CER vs WER"),
    ):
        points = [(float(document[x_metric]), float(document[y_metric]), document["filename"]) for document in documents if document.get(x_metric) is not None and document.get(y_metric) is not None]
        if len(points) >= 2:
            add(name, title, scatter_chart(output/f"{name}.svg", f"{title} — {context}", points, x_label=x_metric, y_label=y_metric, width=width, height=height), points)

    gt_docs = [document for document in documents if document.get("cer") is not None and document.get("baseline_cer") is not None]
    if gt_docs:
        values = [sum(float(document[key]) for document in gt_docs)/len(gt_docs) for key in ("baseline_cer", "cer", "baseline_wer", "wer")]
        labels = ["OCR CER", "Corrected CER", "OCR WER", "Corrected WER"]
        add("correction_before_after", "Before vs after correction", bar_chart(output/"correction_before_after.svg", f"Before vs after correction — {context}", labels, values, y_label="Error rate", x_label="Pipeline output", width=width, height=height, maximum=max(1.0, max(values))), dict(zip(labels, values)))
    if classification:
        add("confusion_matrix", "Quality-class confusion matrix", confusion_matrix_chart(output/"confusion_matrix.svg", f"Quality-class confusion matrix — {context}", classification["confusion_matrix"], classification["labels"], width, height), classification["confusion_matrix"])
        macro = classification["macro"]
        labels = ["Precision", "Recall", "F1", "Specificity"]
        values = [float(macro[key]) for key in ("precision", "recall", "f1", "specificity")]
        add("classification_metrics", "Macro classification metrics", bar_chart(output/"classification_metrics.svg", f"Macro classification metrics — {context}", labels, values, y_label="Score", x_label="Metric", width=width, height=height, maximum=1), dict(zip(labels, values)))
    return charts, sources
