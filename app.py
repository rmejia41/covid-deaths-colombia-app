"""
COVID-19 Fatalities Analytics Dashboard — Colombia (2020–present)

Open data reference (source page):
https://www.datos.gov.co/Salud-y-Protecci-n-Social/Fallecidos-COVID-en-Colombia/jp5m-e7yr/about_data

GitHub raw dataset (provided by user):
https://github.com/rmejia41/open_datasets/raw/main/Fallecidos_COVID_en_Colombia_20251229_divipola_normalized.xlsx
"""

from __future__ import annotations

import os
import re
import unicodedata
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update
import plotly.express as px
import plotly.graph_objects as go

import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

import dash_leaflet as dl
from dash_extensions.javascript import assign


# ============================================================
# CONFIG — LOCAL FIRST + GITHUB RAW FALLBACK (NO API)
# ============================================================
OPEN_DATA_ABOUT_URL = (
    "https://www.datos.gov.co/Salud-y-Protecci-n-Social/"
    "Fallecidos-COVID-en-Colombia/jp5m-e7yr/about_data"
)

# GitHub raw dataset (fallback if local/repo file not found)
GITHUB_RAW_DATA_URL = (
    "https://github.com/rmejia41/open_datasets/raw/main/"
    "Fallecidos_COVID_en_Colombia_20251229_divipola_normalized.xlsx"
)

# Optional: override from env var (local path inside Render or your machine)
# Example (Windows):
#   set COVID_DATA_PATH=C:\path\to\file.xlsx
ENV_DATA_PATH = os.getenv("COVID_DATA_PATH", "").strip()

# Your local dev path (optional; used only if it exists)
LOCAL_DATA_PATH = Path(
    r"C:\Users\rmeji\OneDrive\Documents\AI projects\data\Fallecidos_COVID_en_Colombia_20251229_divipola_normalized.xlsx"
)

# Repo fallback paths (commit your dataset to one of these)
REPO_FALLBACK_PATHS = [
    Path("data") / "Fallecidos_COVID_en_Colombia_20251229_divipola_normalized.xlsx",
    Path("data") / "Fallecidos_COVID_en_Colombia_20251229_divipola_normalized.csv",
    Path("Fallecidos_COVID_en_Colombia_20251229_divipola_normalized.xlsx"),
    Path("Fallecidos_COVID_en_Colombia_20251229_divipola_normalized.csv"),
]

# Sandbox fallbacks (useful for local testing in some environments)
SANDBOX_FALLBACKS = [
    Path("/mnt/data/Fallecidos_COVID_en_Colombia_20251229_divipola_normalized.xlsx"),
    Path("/mnt/data/Fallecidos_COVID_en_Colombia_20251229_divipola_normalized.csv"),
]

# Where we cache the GitHub download (Render-friendly)
CACHE_DIR = Path("data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
GITHUB_CACHE_PATH = CACHE_DIR / "Fallecidos_COVID_en_Colombia_20251229_divipola_normalized.xlsx"

DEFAULT_BIN_DAYS = 7

THEME = dbc.themes.FLATLY
FIG_TEMPLATE_NAME = "flatly"
load_figure_template(FIG_TEMPLATE_NAME)
px.defaults.template = FIG_TEMPLATE_NAME


# ============================================================
# Leaflet JS bindings: tooltips + popups
# ============================================================
POINT_TO_LAYER = assign(
    """
function(feature, latlng){
  return L.circleMarker(latlng, {
    radius: 6,
    weight: 1,
    opacity: 1,
    fillOpacity: 0.7
  });
}
"""
)

ON_EACH_FEATURE = assign(
    """
function(feature, layer){
  if(!feature || !feature.properties){ return; }
  const p = feature.properties;

  // Cluster feature (supercluster).
  if(!!p.cluster){
    const n = p.point_count || 0;
    const html = `<b>Cluster</b><br>Cases: ${n.toLocaleString()}<br><span style="color:#6c757d">Zoom in to see individual cases.</span>`;
    layer.bindTooltip(html, {sticky:true, direction:"top"});
    layer.bindPopup(html);
    return;
  }

  // Individual case point.
  const dept = p.departamento ? `Departamento: ${p.departamento}<br>` : "";
  const mun  = p.municipio ? `Municipio: ${p.municipio}<br>` : "";
  const sex  = p.sex ? `Sex: ${p.sex}<br>` : "";
  const ag   = p.age_group ? `Age group: ${p.age_group}<br>` : "";
  const age  = (p.age_years !== undefined && p.age_years !== null && p.age_years !== "") ? `Age: ${p.age_years}<br>` : "";
  const date = p.fecha_de_muerte ? `Death date: ${p.fecha_de_muerte}<br>` : "";

  const html = `<b>Case location</b><br>` + dept + mun + sex + ag + age + date;

  layer.bindTooltip(html, {sticky:true, direction:"top"});
  layer.bindPopup(html);
}
"""
)

# ============================================================
# Helpers: normalization and parsing
# ============================================================
def safe_empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16),
    )
    fig.update_layout(template=FIG_TEMPLATE_NAME, margin=dict(l=30, r=20, t=60, b=30))
    return fig


def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = []
    for c in df.columns:
        s = str(c).strip()
        s = _strip_accents(s)
        s = s.lower()
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", "_", s).strip("_")
        cols.append(s)
    df.columns = cols
    return df


def _first_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _coerce_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _clean_sex(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.upper()
    mapping = {"M": "Male", "F": "Female", "MALE": "Male", "FEMALE": "Female"}
    x = x.map(lambda v: mapping.get(v, v.title()))
    x = x.replace({"": "Unknown", "Nan": "Unknown", "None": "Unknown"})
    return x.fillna("Unknown")


def _ensure_age_years(df: pd.DataFrame) -> pd.Series:
    c = _first_existing_col(df, ["edad", "age", "age_years"])
    if c:
        return pd.to_numeric(df[c], errors="coerce")
    return pd.Series([np.nan] * len(df), index=df.index)


def _make_age_group(age_years: pd.Series) -> pd.Series:
    bins = list(range(0, 81, 5)) + [np.inf]
    labels = [f"{bins[i]}-{bins[i + 1] - 1}" for i in range(len(bins) - 2)] + ["80+"]
    grp = pd.cut(age_years, bins=[-np.inf] + bins, labels=["Unknown"] + labels, right=True)
    return grp.astype(str).replace({"nan": "Unknown"})


def age_group_sort_key(label: str) -> tuple[int, int]:
    if label is None:
        return (10**9, 10**9)
    s = str(label).strip().lower().replace("–", "-").replace("—", "-")
    if s in {"unknown", "nan", "none", ""}:
        return (10**9, 10**9)
    m = re.match(r"^(\d+)-(\d+)$", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.match(r"^(\d+)\+$", s)
    if m:
        return (int(m.group(1)), 10**8)
    return (10**9 - 1, 10**9 - 1)


def sort_age_groups(labels: list[str]) -> list[str]:
    uniq = list(dict.fromkeys([str(x) for x in labels if str(x).strip() != ""]))
    return sorted(uniq, key=age_group_sort_key)


# ============================================================
# Local / GitHub source resolution (NO API)
# ============================================================
def download_github_dataset(url: str, target_path: Path) -> Path:
    """
    Download dataset from GitHub raw URL to a local cache path.
    Only downloads if the file does not already exist.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        return target_path

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    with open(target_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return target_path


def resolve_local_source() -> Path:
    """
    Resolution order (LOCAL FIRST, GITHUB RAW FALLBACK):
      1) COVID_DATA_PATH env var
      2) Hardcoded LOCAL_DATA_PATH (if exists)
      3) Repo fallback paths (./data or project root)
      4) GitHub raw dataset (download once, cached to ./data)
      5) Sandbox fallbacks
    """
    # 1) Explicit env var
    if ENV_DATA_PATH:
        p = Path(ENV_DATA_PATH)
        if p.exists():
            return p
        raise FileNotFoundError(f"COVID_DATA_PATH was set but file does not exist: {p}")

    # 2) Your local dev path (optional)
    if LOCAL_DATA_PATH.exists():
        return LOCAL_DATA_PATH

    # 3) Repo-committed files
    for p in REPO_FALLBACK_PATHS:
        if p.exists():
            return p

    # 4) GitHub raw fallback (download once)
    try:
        return download_github_dataset(GITHUB_RAW_DATA_URL, GITHUB_CACHE_PATH)
    except Exception as e:
        print(f"[WARN] Failed to download GitHub dataset: {e}")

    # 5) Sandbox fallbacks
    for p in SANDBOX_FALLBACKS:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Dataset not found.\n"
        "Tried:\n"
        "  - COVID_DATA_PATH\n"
        "  - LOCAL_DATA_PATH\n"
        "  - Repo /data directory or project root\n"
        "  - GitHub raw URL (download)\n"
        "  - Sandbox fallbacks\n"
    )


# ============================================================
# Color strategy (fix repeated colors when many age groups)
# ============================================================
def make_distinct_hsl_colors(n: int, s: int = 65, l: int = 45) -> list[str]:
    """Generate n visually distinct colors using evenly spaced hues."""
    n = max(int(n), 1)
    return [f"hsl({int(i * 360 / n)}, {s}%, {l}%)" for i in range(n)]


def stable_color_from_name(name: str) -> str:
    """Stable color derived from name hashing (fallback)."""
    h = int(hashlib.md5(str(name).encode("utf-8")).hexdigest(), 16)
    hue = h % 360
    return f"hsl({hue}, 65%, 45%)"


# ============================================================
# Load data (local file OR cached GitHub download)
# ============================================================
@lru_cache(maxsize=1)
def load_data_local() -> pd.DataFrame:
    src = resolve_local_source()

    if src.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(src)  # requires openpyxl
    elif src.suffix.lower() == ".csv":
        df = pd.read_csv(src, low_memory=False)
    else:
        raise ValueError(f"Unsupported file type: {src.suffix} (use CSV or XLSX)")

    df = _normalize_columns(df)

    death_col = _first_existing_col(
        df, ["fecha_de_muerte", "fecha_muerte", "fecha_de_fallecimiento", "fecha_fallecimiento"]
    )
    if not death_col:
        raise ValueError("Could not find a death date column (e.g., 'fecha_de_muerte') in the dataset.")

    df["fecha_de_muerte"] = _coerce_datetime(df[death_col])

    sym_col = _first_existing_col(df, ["fecha_de_inicio_de_sintomas", "fecha_inicio_sintomas", "fecha_sintomas"])
    dx_col = _first_existing_col(df, ["fecha_de_diagnostico", "fecha_diagnostico", "fecha_diagnosis"])
    df["fecha_de_inicio_de_sintomas"] = _coerce_datetime(df[sym_col]) if sym_col else pd.NaT
    df["fecha_de_diagnostico"] = _coerce_datetime(df[dx_col]) if dx_col else pd.NaT

    dep_col = _first_existing_col(df, ["departamento", "departamento_divipola", "nombre_departamento"])
    mun_col = _first_existing_col(df, ["municipio", "municipio_divipola", "nombre_municipio"])
    df["departamento"] = (
        df[dep_col].astype(str).replace({"nan": "Unknown", "": "Unknown"}).fillna("Unknown") if dep_col else "Unknown"
    )
    df["municipio"] = (
        df[mun_col].astype(str).replace({"nan": "Unknown", "": "Unknown"}).fillna("Unknown") if mun_col else "Unknown"
    )

    sex_col = _first_existing_col(df, ["sexo", "sex", "genero", "gender"])
    df["sex"] = _clean_sex(df[sex_col]) if sex_col else "Unknown"

    df["age_years"] = _ensure_age_years(df)
    df["age_group"] = _make_age_group(df["age_years"])

    lat_col = _first_existing_col(df, ["latitud", "latitude", "lat"])
    lon_col = _first_existing_col(df, ["longitud", "longitude", "lon", "lng"])
    if lat_col and lon_col:
        df["latitud"] = pd.to_numeric(df[lat_col], errors="coerce")
        df["longitud"] = pd.to_numeric(df[lon_col], errors="coerce")

    df = df.dropna(subset=["fecha_de_muerte"]).copy()
    df["active_date"] = df["fecha_de_muerte"].dt.normalize()
    df["year"] = df["active_date"].dt.year.astype("Int64")

    df["symptom_to_dx_days"] = np.nan
    m = df["fecha_de_inicio_de_sintomas"].notna() & df["fecha_de_diagnostico"].notna()
    df.loc[m, "symptom_to_dx_days"] = (
        df.loc[m, "fecha_de_diagnostico"] - df.loc[m, "fecha_de_inicio_de_sintomas"]
    ).dt.days

    df["symptom_to_death_days"] = np.nan
    m = df["fecha_de_inicio_de_sintomas"].notna() & df["fecha_de_muerte"].notna()
    df.loc[m, "symptom_to_death_days"] = (df.loc[m, "fecha_de_muerte"] - df.loc[m, "fecha_de_inicio_de_sintomas"]).dt.days

    return df


df_full = load_data_local()

global_min = df_full["active_date"].min().date().isoformat()
global_max = df_full["active_date"].max().date().isoformat()

years_all = sorted([int(y) for y in df_full["year"].dropna().unique().tolist() if pd.notna(y)])
departamentos_all = sorted(df_full["departamento"].dropna().unique().tolist())
municipios_all = sorted(df_full["municipio"].dropna().unique().tolist())
sexes_all = sorted(df_full["sex"].dropna().unique().tolist())
age_groups_all = sort_age_groups(df_full["age_group"].dropna().unique().tolist())

AGE_GROUP_COLORS = make_distinct_hsl_colors(len(age_groups_all))
AGE_COLOR_MAP = dict(zip(age_groups_all, AGE_GROUP_COLORS))
SEX_COLOR_MAP = {"Male": "hsl(210, 65%, 45%)", "Female": "hsl(340, 65%, 45%)", "Unknown": "hsl(0, 0%, 45%)"}


# ============================================================
# Filtering + Epi aggregation
# ============================================================
def filter_records(
    df: pd.DataFrame,
    year_value: str,
    date_start: str | None,
    date_end: str | None,
    departamentos: list[str],
    municipios: list[str],
    sexes: list[str],
    age_groups: list[str],
) -> pd.DataFrame:
    out = df.copy()

    if year_value and year_value != "ALL":
        out = out[out["year"] == int(year_value)]

    if date_start:
        out = out[out["active_date"] >= pd.to_datetime(date_start)]
    if date_end:
        out = out[out["active_date"] <= pd.to_datetime(date_end)]

    if departamentos:
        out = out[out["departamento"].isin(departamentos)]
    if municipios:
        out = out[out["municipio"].isin(municipios)]
    if sexes:
        out = out[out["sex"].isin(sexes)]
    if age_groups:
        out = out[out["age_group"].isin(age_groups)]

    return out


def aggregate_epi(df: pd.DataFrame, bin_days: int, stack_by: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["bin_mid", "bin_end", "group", "count", "bar_width_ms"])

    bin_days = max(int(bin_days), 1)

    group_col_map = {"none": None, "sex": "sex", "age_group": "age_group"}
    gcol = group_col_map.get(stack_by, None)
    if gcol and gcol not in df.columns:
        gcol = None

    tmp = df[["active_date"] + ([gcol] if gcol else [])].copy()
    tmp["active_date"] = pd.to_datetime(tmp["active_date"], errors="coerce")
    tmp = tmp.dropna(subset=["active_date"]).copy()
    if tmp.empty:
        return pd.DataFrame(columns=["bin_mid", "bin_end", "group", "count", "bar_width_ms"])

    tmp["group"] = tmp[gcol].astype(str) if gcol else "All"
    origin = tmp["active_date"].min()
    if pd.isna(origin):
        return pd.DataFrame(columns=["bin_mid", "bin_end", "group", "count", "bar_width_ms"])

    binned = (
        tmp.set_index(pd.DatetimeIndex(tmp["active_date"]))
        .groupby([pd.Grouper(freq=f"{bin_days}D", origin=origin, label="right", closed="right"), "group"])
        .size()
        .rename("count")
        .reset_index()
    )
    time_col = binned.columns[0]
    binned = binned.rename(columns={time_col: "bin_end"})
    binned["bin_mid"] = binned["bin_end"] - pd.to_timedelta(bin_days / 2.0, unit="D")
    binned["bar_width_ms"] = bin_days * 24 * 60 * 60 * 1000
    return binned[["bin_mid", "bin_end", "group", "count", "bar_width_ms"]]


def _ordered_groups(groups: list[str], stack_by: str) -> list[str]:
    if stack_by == "age_group":
        order = [g for g in age_groups_all if g in groups]
        extras = [g for g in groups if g not in set(order)]
        return order + sorted(extras)
    if stack_by == "sex":
        pref = ["Male", "Female", "Unknown"]
        order = [g for g in pref if g in groups]
        extras = [g for g in groups if g not in set(order)]
        return order + sorted(extras)
    return sorted(groups)


def make_epi_figure(binned: pd.DataFrame, bin_days: int, stack_by: str) -> go.Figure:
    if binned.empty:
        return safe_empty_figure("No records match the current filters.")

    fig = go.Figure()
    plot_df = binned.copy()
    plot_df["group"] = plot_df["group"].astype(str)

    groups = _ordered_groups(plot_df["group"].unique().tolist(), stack_by=stack_by)

    for g in groups:
        sub = plot_df[plot_df["group"] == g]

        color = None
        if stack_by == "age_group":
            color = AGE_COLOR_MAP.get(g, stable_color_from_name(g))
        elif stack_by == "sex":
            color = SEX_COLOR_MAP.get(g, stable_color_from_name(g))

        fig.add_trace(
            go.Bar(
                x=sub["bin_mid"],
                y=sub["count"],
                width=sub["bar_width_ms"],
                name=g,
                marker=dict(color=color) if color else None,
                hovertemplate=(
                    "<b>" + str(g) + "</b><br>"
                    "Bin center: %{x|%Y-%m-%d}<br>"
                    f"Bin width: {int(bin_days)} days<br>"
                    "Deaths: %{y}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template=FIG_TEMPLATE_NAME,
        title=f"Epidemic curve (fatalities) • Bin = {int(bin_days)} day(s) • Stacked",
        xaxis_title="Date (binned)",
        yaxis_title="Deaths",
        barmode="stack",
        legend_title="Group",
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
        legend=dict(itemsizing="constant"),
    )
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def compute_kpis(df: pd.DataFrame) -> dict[str, str]:
    if df.empty:
        return {
            "Total deaths": "0",
            "Peak day": "—",
            "Peak deaths": "—",
            "Median age": "—",
            "Median symptom→dx (days)": "—",
            "Median symptom→death (days)": "—",
        }

    total = len(df)
    daily = df.groupby("active_date").size().sort_index()
    peak_day = daily.idxmax().date().isoformat()
    peak_n = int(daily.max())

    med_age = np.nanmedian(df["age_years"].dropna()) if df["age_years"].notna().any() else np.nan
    med_s2dx = np.nanmedian(df["symptom_to_dx_days"].dropna()) if df["symptom_to_dx_days"].notna().any() else np.nan
    med_s2d = np.nanmedian(df["symptom_to_death_days"].dropna()) if df["symptom_to_death_days"].notna().any() else np.nan

    def fmt(x):
        if pd.isna(x):
            return "—"
        return f"{x:.1f}"

    return {
        "Total deaths": f"{total:,}",
        "Peak day": peak_day,
        "Peak deaths": f"{peak_n:,}",
        "Median age": fmt(med_age),
        "Median symptom→dx (days)": fmt(med_s2dx),
        "Median symptom→death (days)": fmt(med_s2d),
    }


def kpi_cards(kpis: dict[str, str]):
    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([html.Div(k, className="text-muted small"), html.Div(v, className="h4 mb-0")]),
                    className="shadow-sm",
                ),
                xs=12,
                sm=6,
                lg=4,
                xl=2,
            )
            for k, v in kpis.items()
        ],
        className="g-2",
    )


# ============================================================
# Charts (interactive)
# ============================================================
def make_delay_hist(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    if df.empty or col not in df.columns or not df[col].notna().any():
        return safe_empty_figure(f"{title}: requires the relevant date fields.")
    x = df[df[col].notna()].copy()
    x = x[(x[col] >= 0) & (x[col] <= 120)]
    if x.empty:
        return safe_empty_figure(f"{title}: no non-negative values after cleaning.")
    fig = px.histogram(x, x=col, nbins=60, title=title)
    fig.update_xaxes(title="Days")
    fig.update_yaxes(title="Deaths")
    fig.update_layout(template=FIG_TEMPLATE_NAME, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def make_rolling_deaths(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return safe_empty_figure("No data.")
    daily = df.groupby("active_date").size().rename("deaths").reset_index()
    daily["roll7"] = daily["deaths"].rolling(7, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["active_date"], y=daily["deaths"], mode="lines", name="Daily deaths"))
    fig.add_trace(go.Scatter(x=daily["active_date"], y=daily["roll7"], mode="lines", name="7-day average"))
    fig.update_layout(
        template=FIG_TEMPLATE_NAME,
        title="Daily deaths and 7-day rolling average",
        xaxis_title="Date",
        yaxis_title="Deaths",
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
    )
    return fig


def make_age_time_heatmap(df: pd.DataFrame) -> go.Figure:
    if df.empty or "age_group" not in df.columns:
        return safe_empty_figure("No data.")
    tmp = df.copy()
    tmp["month"] = tmp["active_date"].dt.to_period("M").dt.to_timestamp()
    pivot = tmp.groupby(["month", "age_group"]).size().rename("deaths").reset_index()
    order = sort_age_groups(pivot["age_group"].astype(str).tolist())
    pivot["age_group"] = pd.Categorical(pivot["age_group"].astype(str), categories=order, ordered=True)

    fig = px.density_heatmap(
        pivot,
        x="month",
        y="age_group",
        z="deaths",
        histfunc="sum",
        title="Deaths by month and age group",
    )
    fig.update_yaxes(title="", categoryorder="array", categoryarray=order)
    fig.update_xaxes(title="Month")
    fig.update_layout(template=FIG_TEMPLATE_NAME, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def make_deaths_by_department(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    if df.empty:
        return safe_empty_figure("No data.")
    grp = df.groupby("departamento").size().rename("deaths").reset_index()
    grp = grp.sort_values("deaths", ascending=False).head(top_n)
    fig = px.bar(
        grp.sort_values("deaths"),
        x="deaths",
        y="departamento",
        orientation="h",
        title=f"Top {top_n} departamentos by deaths",
    )
    fig.update_xaxes(title="Deaths")
    fig.update_yaxes(title="")
    fig.update_layout(template=FIG_TEMPLATE_NAME, margin=dict(l=40, r=20, t=60, b=40))
    return fig


# ============================================================
# GeoJSON clustered map — INDIVIDUAL CASES ONLY
# ============================================================
def build_case_geojson(df_cases: pd.DataFrame, max_points: int = 12000) -> dict:
    m = df_cases.dropna(subset=["latitud", "longitud"]).copy()
    if m.empty:
        return {"type": "FeatureCollection", "features": []}

    if len(m) > max_points:
        m = m.sample(max_points, random_state=7)

    m["latitud"] = pd.to_numeric(m["latitud"], errors="coerce")
    m["longitud"] = pd.to_numeric(m["longitud"], errors="coerce")
    m = m.dropna(subset=["latitud", "longitud"]).copy()
    if m.empty:
        return {"type": "FeatureCollection", "features": []}

    death_date_str = pd.to_datetime(m["fecha_de_muerte"], errors="coerce").dt.date.astype(str).replace("NaT", "")

    features = []
    for i, r in m.iterrows():
        lat = r["latitud"]
        lon = r["longitud"]
        props = {
            "departamento": str(r.get("departamento", "")),
            "municipio": str(r.get("municipio", "")),
            "sex": str(r.get("sex", "")),
            "age_group": str(r.get("age_group", "")),
            "fecha_de_muerte": str(death_date_str.loc[i]) if i in death_date_str.index else "",
        }
        age = r.get("age_years", np.nan)
        if not pd.isna(age):
            try:
                props["age_years"] = float(age)
            except Exception:
                pass

        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "properties": props,
            }
        )

    return {"type": "FeatureCollection", "features": features}


def compute_bounds_from_cases(df_cases: pd.DataFrame):
    m = df_cases.dropna(subset=["latitud", "longitud"]).copy()
    if m.empty:
        return None
    lat = pd.to_numeric(m["latitud"], errors="coerce").dropna()
    lon = pd.to_numeric(m["longitud"], errors="coerce").dropna()
    if lat.empty or lon.empty:
        return None
    return [[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]]


# ============================================================
# App + Layout
# ============================================================
app = Dash(__name__, external_stylesheets=[THEME], suppress_callback_exceptions=True)
server = app.server
app.title = "COVID-19 Fatalities — Colombia"

controls = dbc.Card(
    dbc.CardBody(
        [
            html.Div("Time window", className="fw-semibold"),
            dcc.DatePickerRange(
                id="date-range",
                min_date_allowed=global_min,
                max_date_allowed=global_max,
                start_date=global_min,
                end_date=global_max,
                display_format="YYYY-MM-DD",
            ),
            html.Hr(),
            html.Div("Filters", className="fw-semibold"),
            html.Label("Year"),
            dcc.Dropdown(
                id="year-filter",
                options=[{"label": "All years", "value": "ALL"}] + [{"label": str(y), "value": str(y)} for y in years_all],
                value="ALL",
                clearable=False,
            ),
            html.Label("Departamento"),
            dcc.Dropdown(
                id="depto-filter",
                options=[{"label": d.title(), "value": d} for d in departamentos_all],
                value=[],
                multi=True,
                placeholder="All",
            ),
            html.Label("Municipio"),
            dcc.Dropdown(
                id="mun-filter",
                options=[{"label": m.title(), "value": m} for m in municipios_all],
                value=[],
                multi=True,
                placeholder="All (use Departamento first to narrow)",
            ),
            html.Label("Sex"),
            dcc.Dropdown(
                id="sex-filter",
                options=[{"label": s, "value": s} for s in sexes_all],
                value=[],
                multi=True,
                placeholder="All",
            ),
            html.Label("Age group"),
            dcc.Dropdown(
                id="age-filter",
                options=[{"label": a, "value": a} for a in age_groups_all],
                value=[],
                multi=True,
                placeholder="All",
            ),
            html.Hr(),
            html.Div("Epicurve (stacked)", className="fw-semibold"),
            html.Label("Bin width (days)"),
            dcc.Slider(
                id="bin-days",
                min=1,
                max=30,
                step=1,
                value=DEFAULT_BIN_DAYS,
                marks={1: "1", 3: "3", 5: "5", 7: "7", 14: "14", 21: "21", 30: "30"},
            ),
            html.Br(),
            html.Label("Stack bars by"),
            dcc.Dropdown(
                id="stack-by",
                options=[
                    {"label": "None (total only)", "value": "none"},
                    {"label": "Sex", "value": "sex"},
                    {"label": "Age group", "value": "age_group"},
                ],
                value="sex",
                clearable=False,
            ),
            html.Hr(),
            dbc.Button("Download filtered deaths (CSV)", id="btn-download", n_clicks=0, color="primary", className="w-100"),
            dcc.Download(id="download-csv"),
        ]
    ),
    className="shadow-sm",
)

app.layout = dbc.Container(
    fluid=True,
    className="dbc",
    children=[
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.H2("COVID-19 Fatalities Analytics Dashboard — Colombia", className="mt-3 mb-1"),
                        html.Div(
                            [
                                "Open data reference: ",
                                html.A(
                                    "Ministerio de Salud y Protección Social (Datos Abiertos Colombia)",
                                    href=OPEN_DATA_ABOUT_URL,
                                    target="_blank",
                                ),
                            ],
                            className="text-muted",
                        ),
                    ]
                )
            )
        ),
        dbc.Row(
            [
                dbc.Col(controls, xs=12, md=4, lg=3),
                dbc.Col(
                    [
                        html.Div(id="kpi-area", className="mb-2"),
                        dcc.Tabs(
                            id="tabs",
                            value="tab-epi",
                            children=[
                                dcc.Tab(label="Epi curve", value="tab-epi"),
                                dcc.Tab(label="Charts", value="tab-charts"),
                                dcc.Tab(label="Map (clustered cases)", value="tab-map"),
                            ],
                        ),
                        html.Div(id="tab-content", className="pt-2"),
                    ],
                    xs=12,
                    md=8,
                    lg=9,
                ),
            ],
            className="g-2",
        ),
    ],
)


def epi_tab_layout():
    return html.Div(
        [
            dcc.Graph(id="epi-graph"),
            html.Div(id="table-note", className="text-muted small mt-2"),
            dash_table.DataTable(
                id="records-table",
                page_size=12,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"fontFamily": "Arial", "fontSize": 12, "padding": "6px"},
                style_header={"fontWeight": "bold"},
            ),
        ]
    )


def charts_tab_layout():
    return html.Div(
        [
            dbc.Row([dbc.Col(dcc.Graph(id="rolling-fig"), md=12)], className="g-2"),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="delay-s2dx-fig"), md=6),
                    dbc.Col(dcc.Graph(id="delay-s2d-fig"), md=6),
                ],
                className="g-2 mt-1",
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="heatmap-fig"), md=7),
                    dbc.Col(dcc.Graph(id="top-dept-fig"), md=5),
                ],
                className="g-2 mt-1",
            ),
        ]
    )


def map_tab_layout():
    return html.Div(
        [
            dbc.Alert(
                "Map shows individual cases (sampled for performance if needed). Hover for tooltip; click for popup. Zoom to split clusters.",
                color="info",
                className="py-2",
            ),
            dl.Map(
                id="leaflet-map",
                center=[4.57, -74.30],
                zoom=5,
                preferCanvas=True,
                style={"width": "100%", "height": "650px", "borderRadius": "8px"},
                children=[
                    dl.LayersControl(
                        [
                            dl.BaseLayer(
                                dl.TileLayer(url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"),
                                name="OpenStreetMap",
                                checked=True,
                            ),
                            dl.BaseLayer(
                                dl.TileLayer(url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"),
                                name="Carto Light",
                            ),
                            dl.BaseLayer(
                                dl.TileLayer(url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"),
                                name="Carto Dark",
                            ),
                        ]
                    ),
                    dl.ScaleControl(position="bottomleft"),
                    dl.GeoJSON(
                        id="map-geojson",
                        data={"type": "FeatureCollection", "features": []},
                        cluster=True,
                        zoomToBoundsOnClick=True,
                        superClusterOptions={"radius": 60},
                        options=dict(pointToLayer=POINT_TO_LAYER, onEachFeature=ON_EACH_FEATURE),
                    ),
                ],
            ),
        ]
    )


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab: str):
    if tab == "tab-epi":
        return epi_tab_layout()
    if tab == "tab-charts":
        return charts_tab_layout()
    if tab == "tab-map":
        return map_tab_layout()
    return html.Div("Unknown tab.")


@app.callback(
    Output("mun-filter", "options"),
    Output("mun-filter", "value"),
    Input("depto-filter", "value"),
    Input("year-filter", "value"),
    State("mun-filter", "value"),
)
def sync_municipio_options(selected_deptos, year_value, current_muns):
    df = df_full
    if year_value and year_value != "ALL":
        df = df[df["year"] == int(year_value)]

    if selected_deptos:
        df = df[df["departamento"].isin(selected_deptos)]
        muns = sorted(df["municipio"].dropna().unique().tolist())
    else:
        muns = municipios_all

    opts = [{"label": m.title(), "value": m} for m in muns]
    current_muns = current_muns or []
    valid_set = set(muns)
    new_value = [m for m in current_muns if m in valid_set]
    return opts, new_value


@app.callback(
    Output("kpi-area", "children"),
    Input("year-filter", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("depto-filter", "value"),
    Input("mun-filter", "value"),
    Input("sex-filter", "value"),
    Input("age-filter", "value"),
)
def update_kpis(year_value, start_date, end_date, deptos, muns, sexes, age_groups):
    f = filter_records(
        df_full,
        year_value=year_value or "ALL",
        date_start=start_date,
        date_end=end_date,
        departamentos=deptos or [],
        municipios=muns or [],
        sexes=sexes or [],
        age_groups=age_groups or [],
    )
    return kpi_cards(compute_kpis(f))


@app.callback(
    Output("epi-graph", "figure"),
    Output("records-table", "data"),
    Output("records-table", "columns"),
    Output("table-note", "children"),
    Input("year-filter", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("depto-filter", "value"),
    Input("mun-filter", "value"),
    Input("sex-filter", "value"),
    Input("age-filter", "value"),
    Input("bin-days", "value"),
    Input("stack-by", "value"),
)
def update_epi(year_value, start_date, end_date, deptos, muns, sexes, age_groups, bin_days, stack_by):
    f = filter_records(
        df_full,
        year_value=year_value or "ALL",
        date_start=start_date,
        date_end=end_date,
        departamentos=deptos or [],
        municipios=muns or [],
        sexes=sexes or [],
        age_groups=age_groups or [],
    )

    stack_by = str(stack_by or "sex")
    binned = aggregate_epi(f, bin_days=int(bin_days or 1), stack_by=stack_by)
    fig = make_epi_figure(binned, int(bin_days or 1), stack_by=stack_by)

    preferred_cols = [
        "active_date",
        "fecha_de_muerte",
        "fecha_de_inicio_de_sintomas",
        "fecha_de_diagnostico",
        "departamento",
        "municipio",
        "sex",
        "age_years",
        "age_group",
        "symptom_to_dx_days",
        "symptom_to_death_days",
        "latitud",
        "longitud",
    ]
    cols = [c for c in preferred_cols if c in f.columns]
    table_df = f[cols].copy()

    for c in ["fecha_de_muerte", "fecha_de_inicio_de_sintomas", "fecha_de_diagnostico", "active_date"]:
        if c in table_df.columns:
            table_df[c] = pd.to_datetime(table_df[c], errors="coerce").dt.date.astype(str).replace("NaT", "")

    max_preview = 2000
    note = ""
    if len(table_df) > max_preview:
        table_df = table_df.head(max_preview)
        note = f"Showing first {max_preview:,} rows (filtered total: {len(f):,})."

    data = table_df.to_dict("records")
    columns = [{"name": c.replace("_", " ").title(), "id": c} for c in table_df.columns]
    return fig, data, columns, note


@app.callback(
    Output("rolling-fig", "figure"),
    Output("delay-s2dx-fig", "figure"),
    Output("delay-s2d-fig", "figure"),
    Output("heatmap-fig", "figure"),
    Output("top-dept-fig", "figure"),
    Input("year-filter", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("depto-filter", "value"),
    Input("mun-filter", "value"),
    Input("sex-filter", "value"),
    Input("age-filter", "value"),
)
def update_charts(year_value, start_date, end_date, deptos, muns, sexes, age_groups):
    f = filter_records(
        df_full,
        year_value=year_value or "ALL",
        date_start=start_date,
        date_end=end_date,
        departamentos=deptos or [],
        municipios=muns or [],
        sexes=sexes or [],
        age_groups=age_groups or [],
    )
    return (
        make_rolling_deaths(f),
        make_delay_hist(f, "symptom_to_dx_days", "Delay: symptom onset → diagnosis (days)"),
        make_delay_hist(f, "symptom_to_death_days", "Delay: symptom onset → death (days)"),
        make_age_time_heatmap(f),
        make_deaths_by_department(f, top_n=15),
    )


@app.callback(
    Output("map-geojson", "data"),
    Output("leaflet-map", "bounds"),
    Input("year-filter", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("depto-filter", "value"),
    Input("mun-filter", "value"),
    Input("sex-filter", "value"),
    Input("age-filter", "value"),
)
def update_map(year_value, start_date, end_date, deptos, muns, sexes, age_groups):
    f = filter_records(
        df_full,
        year_value=year_value or "ALL",
        date_start=start_date,
        date_end=end_date,
        departamentos=deptos or [],
        municipios=muns or [],
        sexes=sexes or [],
        age_groups=age_groups or [],
    )

    if not {"latitud", "longitud"}.issubset(f.columns):
        return {"type": "FeatureCollection", "features": []}, no_update

    geojson = build_case_geojson(f, max_points=12000)
    bounds = compute_bounds_from_cases(f)
    return geojson, bounds or no_update


@app.callback(
    Output("download-csv", "data"),
    Input("btn-download", "n_clicks"),
    State("year-filter", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("depto-filter", "value"),
    State("mun-filter", "value"),
    State("sex-filter", "value"),
    State("age-filter", "value"),
    prevent_initial_call=True,
)
def download_filtered(n_clicks, year_value, start_date, end_date, deptos, muns, sexes, age_groups):
    f = filter_records(
        df_full,
        year_value=year_value or "ALL",
        date_start=start_date,
        date_end=end_date,
        departamentos=deptos or [],
        municipios=muns or [],
        sexes=sexes or [],
        age_groups=age_groups or [],
    )
    return dcc.send_data_frame(f.to_csv, "covid_colombia_filtered_deaths.csv", index=False)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", "8050")))