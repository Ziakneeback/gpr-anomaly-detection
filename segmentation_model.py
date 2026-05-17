"""Segmentation helpers for GPR anomaly detection.

The desktop application uses a robust statistical detector by default. This
module adds an optional U-Net segmentation layer for the dissertation/API part
of the project. If PyTorch is not installed or trained weights are unavailable,
the public functions fall back to the deterministic detector from
``gpr_anomaly_app.py`` so the application remains usable on a clean Python
installation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from gpr_anomaly_app import GPRProcessor


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    F = None


@dataclass
class SegmentationResult:
    mask: list[list[int]]
    model_name: str
    mode: str
    notes: str


if nn is not None:

    class DoubleConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)


    class UNetGPR(nn.Module):
        """Compact U-Net for binary segmentation of B-scan anomaly regions."""

        def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 16):
            super().__init__()
            self.enc1 = DoubleConv(in_channels, base)
            self.enc2 = DoubleConv(base, base * 2)
            self.enc3 = DoubleConv(base * 2, base * 4)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = DoubleConv(base * 4, base * 8)
            self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
            self.dec3 = DoubleConv(base * 8, base * 4)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
            self.dec2 = DoubleConv(base * 4, base * 2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
            self.dec1 = DoubleConv(base * 2, base)
            self.out = nn.Conv2d(base, out_channels, kernel_size=1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))

            d3 = self.up3(b)
            d3 = _match_and_concat(d3, e3)
            d3 = self.dec3(d3)

            d2 = self.up2(d3)
            d2 = _match_and_concat(d2, e2)
            d2 = self.dec2(d2)

            d1 = self.up1(d2)
            d1 = _match_and_concat(d1, e1)
            d1 = self.dec1(d1)
            return self.out(d1)


def _match_and_concat(decoder_tensor, encoder_tensor):
    if F is None:
        raise RuntimeError("PyTorch is required for U-Net inference")
    if decoder_tensor.shape[-2:] != encoder_tensor.shape[-2:]:
        decoder_tensor = F.interpolate(
            decoder_tensor,
            size=encoder_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return torch.cat([encoder_tensor, decoder_tensor], dim=1)


def _fallback_mask(data: list[list[float]], threshold: float) -> list[list[int]]:
    processed = GPRProcessor.preprocess(data)
    zones = GPRProcessor.detect_anomalies(processed, threshold=threshold)
    rows = len(processed)
    cols = len(processed[0])
    mask = [[0 for _ in range(cols)] for _ in range(rows)]
    for zone in zones:
        for trace in range(zone.trace_from, zone.trace_to + 1):
            for sample in range(zone.sample_from, zone.sample_to + 1):
                if 0 <= trace < rows and 0 <= sample < cols:
                    mask[trace][sample] = 1
    return mask


def predict_mask(
    data: list[list[float]],
    weights_path: str | None = None,
    threshold: float = 3.0,
    probability_threshold: float = 0.5,
) -> SegmentationResult:
    """Return a binary segmentation mask for anomalous B-scan regions.

    When PyTorch and trained weights are available, the function performs U-Net
    inference. Otherwise it returns a segmentation mask produced by the robust
    detector. This keeps the API and GUI operational without GPU dependencies.
    """

    if torch is None or weights_path is None:
        return SegmentationResult(
            mask=_fallback_mask(data, threshold),
            model_name="U-Net-compatible fallback",
            mode="robust-statistical-segmentation",
            notes="PyTorch weights are not loaded; mask is formed from connected anomaly zones.",
        )

    processed = GPRProcessor.preprocess(data)
    model = UNetGPR()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    tensor = torch.tensor(processed, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(tensor))[0, 0]
    mask = (probabilities >= probability_threshold).int().tolist()
    return SegmentationResult(
        mask=mask,
        model_name="U-Net",
        mode="torch-inference",
        notes="Mask is produced by U-Net binary segmentation.",
    )


def iou_score(predicted: Iterable[Iterable[int]], target: Iterable[Iterable[int]]) -> float:
    """Calculate Intersection over Union for two binary masks."""

    intersection = 0
    union = 0
    for pred_row, target_row in zip(predicted, target):
        for pred_value, target_value in zip(pred_row, target_row):
            pred = bool(pred_value)
            truth = bool(target_value)
            if pred and truth:
                intersection += 1
            if pred or truth:
                union += 1
    if union == 0:
        return 1.0
    return intersection / union

