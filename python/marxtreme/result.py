"""Structured DetectResult container with strictly single-file persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from marxtreme.catalog import EventCatalog


@dataclass
class DetectResult:
    """Unified output container for ``marxtreme.detect()``."""

    events: EventCatalog | None
    clim: xr.DataArray | None = None
    thresh_high: xr.DataArray | None = None
    thresh_low: xr.DataArray | None = None
    intensity_ts: xr.DataArray | None = None
    daily_category_ts: xr.DataArray | None = None
    event_category_ts: xr.DataArray | None = None
    trend: xr.Dataset | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    out_path: Path | None = None
    _closers: list[Any] = field(default_factory=list, repr=False, compare=False)

    def close(self) -> None:
        """Release any open disk file handles held by lazily opened gridded arrays."""
        for obj in self._closers:
            try:
                obj.close()
            except Exception:
                pass
        self._closers.clear()

    def __enter__(self) -> "DetectResult":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        grids = [
            name
            for name in (
                "clim",
                "thresh_high",
                "thresh_low",
                "intensity_ts",
                "daily_category_ts",
                "event_category_ts",
                "trend",
            )
            if getattr(self, name) is not None
        ]
        storage = f"out_path={str(self.out_path)!r}" if self.out_path is not None else "in-memory"
        return f"DetectResult(\n  events={self.events!r},\n  grids={grids},\n  {storage}\n)"

    def to_dataset(self) -> xr.Dataset:
        """Bundle all present gridded DataArrays into a single ``xarray.Dataset``."""
        data_vars: dict[str, xr.DataArray] = {}
        for name in (
            "clim",
            "thresh_high",
            "thresh_low",
            "intensity_ts",
            "daily_category_ts",
            "event_category_ts",
        ):
            arr = getattr(self, name)
            if arr is not None:
                data_vars[name] = arr
        if self.trend is not None:
            for k, v in self.trend.data_vars.items():
                data_vars[f"trend_{k}"] = v
        clean_attrs = {
            k: (int(v) if isinstance(v, bool) else v)
            for k, v in self.metadata.items()
            if v is not None
        }
        return xr.Dataset(data_vars, attrs=clean_attrs)

    def save(
        self,
        path: str | Path,
        *,
        out_format: str | None = None,
        mat_layout: str = "v73",
    ) -> Path:
        """Save this result as ONE single self-contained file.

        Note: When overwriting ``self.out_path`` in place, lazy disk handles on ``self`` are
        closed prior to atomic replacement; reopen with ``marxtreme.open(path)`` to read
        the updated file.
        """
        from marxtreme.io_formats import (
            TABULAR_ONLY_FORMATS,
            OutputSpec,
            open_file_writer,
            resolve_out_format,
        )
        from marxtreme.streaming import estimate_max_tile_pixels, iter_spatial_tiles, parse_memory_limit

        out_file = Path(path)
        fmt = resolve_out_format(out_file, out_format=out_format, mat_layout=mat_layout)

        if fmt in TABULAR_ONLY_FORMATS:
            if self.events is None:
                raise ValueError(f"Cannot save format '{fmt}' because DetectResult.events is None.")
            return self.events.save(out_file, fmt=fmt, mat_layout=mat_layout)

        spec, grid_sources = OutputSpec.from_result(self, fmt)
        same_target = self.out_path is not None and out_file.resolve() == self.out_path.resolve()
        grid_offsets = self.events.grid_offsets.copy() if self.events is not None else None

        writer = open_file_writer(out_file, fmt, spec)
        max_tile = (
            estimate_max_tile_pixels(
                parse_memory_limit("512MB"),
                0,
                int(spec.det_time.size),
                return_intensity_ts=spec.write_intensity_ts,
                return_daily_category_ts=spec.write_daily_category_ts,
                return_event_category_ts=spec.write_event_category_ts,
                moving_n_years=int(spec.moving_years.size) if spec.moving_years is not None else 0,
                year_len=spec.year_len,
                intensity_itemsize=spec.intensity_dtype.itemsize,
            )
            if spec.spatial_dims
            else 1
        )
        try:
            for tile in iter_spatial_tiles(spec.spatial_dims, spec.spatial_shape, max_tile):
                for gname, gda, target_dims in grid_sources:
                    sub = gda.isel(tile.isel_dict) if spec.spatial_dims else gda
                    if target_dims and tuple(sub.dims) != target_dims:
                        sub = sub.transpose(*target_dims)
                    gextra = len(target_dims) - len(spec.spatial_dims)
                    writer.write_grid_tile(gname, tile, np.asarray(sub.values), gextra)

            ev_schema = None
            if self.events is not None:
                for part in self.events.iter_chunks():
                    if ev_schema is None:
                        ev_schema = part.schema
                    writer.write_events_table(part)

            if same_target:
                self.close()
            writer.finalize(ev_schema, grid_offsets)
        finally:
            writer.close()
        return out_file

    def to_mat(self, path: str | Path, layout: str = "v73") -> Path:
        """Export a single standalone MATLAB ``.mat`` file (``'v73'`` default or ``'detectc'`` v5)."""
        return self.save(path, out_format="mat", mat_layout=layout)

    @classmethod
    def open(cls, path: str | Path, events_in_memory: bool = False) -> "DetectResult":
        """Reopen any self-contained ``marxtreme`` output."""
        from marxtreme.io_formats import TREND_VAR_KEYS, open_single_file

        p = Path(path)
        ds, events, metadata, closers = open_single_file(p, events_in_memory=events_in_memory)

        def _get(name: str) -> xr.DataArray | None:
            return ds[name] if (ds is not None and name in ds) else None

        trend_vars = (
            {short: ds[full] for short, full in TREND_VAR_KEYS if full in ds}
            if ds is not None
            else {}
        )
        return cls(
            events=events,
            clim=_get("clim"),
            thresh_high=_get("thresh_high"),
            thresh_low=_get("thresh_low"),
            intensity_ts=_get("intensity_ts"),
            daily_category_ts=_get("daily_category_ts"),
            event_category_ts=_get("event_category_ts"),
            trend=xr.Dataset(trend_vars) if trend_vars else None,
            metadata=metadata,
            out_path=p,
            _closers=closers,
        )
