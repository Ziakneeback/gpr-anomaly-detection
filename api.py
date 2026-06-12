"""REST API for GPR anomaly analysis.

Run after installing optional dependencies:

    uvicorn api:app --reload

The API reuses the same processing code as the desktop application. This keeps
the dissertation API module consistent with the working GUI implementation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from gpr_anomaly_app import GPRProcessor
from segmentation_model import iou_score, predict_mask


try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from pydantic import BaseModel
except Exception:  # pragma: no cover - optional dependency
    FastAPI = None
    File = None
    HTTPException = Exception
    UploadFile = None
    BaseModel = object


if FastAPI is not None:
    app = FastAPI(
        title="GPR Anomaly Detection API",
        description="REST API for detecting anomalous zones in GPR B-scan files.",
        version="1.0.0",
    )
else:
    app = None


if BaseModel is not object:

    class MaskPair(BaseModel):
        predicted: list[list[int]]
        target: list[list[int]]

else:

    class MaskPair:  # pragma: no cover - used only when pydantic is absent
        pass


def _zone_to_dict(zone):
    return {
        "trace": zone.trace_index,
        "sample": zone.sample_index,
        "trace_from": zone.trace_from,
        "trace_to": zone.trace_to,
        "sample_from": zone.sample_from,
        "sample_to": zone.sample_to,
        "score": round(zone.score, 4),
        "amplitude": round(zone.amplitude, 6),
        "confidence": round(zone.confidence, 4),
        "area": zone.area,
        "depth_m": round(zone.depth_m, 4),
        "latitude": zone.latitude,
        "longitude": zone.longitude,
        "coordinates": zone.coordinate_text,
        "label": zone.label,
        "reason": zone.reason,
    }


def analyze_path(path: str | Path, threshold: float = 3.0, use_unet: bool = False):
    imported = GPRProcessor.load_file(path)
    processed = GPRProcessor.preprocess(imported.data)
    zones = GPRProcessor.detect_anomalies(processed, threshold=threshold)
    sample_count = len(processed[0])
    gps_points = imported.gps_points or []
    for zone in zones:
        zone.depth_m = zone.sample_index / max(1, sample_count - 1) * 3.0
        if gps_points and len(processed) > 1:
            gps_index = round(zone.trace_index / (len(processed) - 1) * (len(gps_points) - 1))
            gps_index = max(0, min(len(gps_points) - 1, gps_index))
            gps = gps_points[gps_index]
            zone.latitude = gps["lat"]
            zone.longitude = gps["lon"]
            zone.coordinate_text = f"{gps['lat']:.6f}, {gps['lon']:.6f}"
        else:
            zone.coordinate_text = f"trace {zone.trace_index}, sample {zone.sample_index}"
    result = {
        "status": "ok",
        "source_type": imported.source_type,
        "details": imported.details,
        "traces": len(imported.data),
        "samples": len(imported.data[0]),
        "threshold": threshold,
        "zones_count": len(zones),
        "zones": [_zone_to_dict(zone) for zone in zones],
    }
    if use_unet:
        segmentation = predict_mask(imported.data, threshold=threshold)
        result["segmentation"] = {
            "model": segmentation.model_name,
            "mode": segmentation.mode,
            "notes": segmentation.notes,
            "mask_rows": len(segmentation.mask),
            "mask_cols": len(segmentation.mask[0]) if segmentation.mask else 0,
        }
    return result


if app is not None:

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "gpr-anomaly-detection"}


    @app.post("/analyze")
    async def analyze(
        file: UploadFile = File(...),
        threshold: float = 3.0,
        use_unet: bool = False,
    ):
        suffix = Path(file.filename or "input.gpr").suffix or ".gpr"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                tmp_path = Path(tmp.name)
            return analyze_path(tmp_path, threshold=threshold, use_unet=use_unet)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


    @app.post("/metrics/iou")
    def metrics_iou(payload: MaskPair):
        return {"iou": round(iou_score(payload.predicted, payload.target), 6)}


if __name__ == "__main__":
    if app is None:
        print("FastAPI is not installed. Install dependencies from requirements.txt first.")
    else:
        print("Run the API with: uvicorn api:app --reload")
