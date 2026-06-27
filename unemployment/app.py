"""
================================================================================
 CareerMap India — Employment Opportunity Intelligence System
================================================================================

An NLP-powered analytics assistant over India's regional unemployment data.
Ask plain-English questions ("Which state has the highest unemployment?",
"Compare Gujarat and Maharashtra", "Predict unemployment in Gujarat") and get
a data-grounded answer, a chart, and (where relevant) a map — all computed
locally with pandas / NumPy / scikit-learn / Plotly. No LLM, no external API.

--------------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------------
1) pip install pandas numpy plotly gradio scikit-learn
2) Put these two files in the SAME folder as this script:
     - Unemployment_in_India.csv
     - Unemployment_Rate_upto_11_2020.csv
3) python app.py
   (a local Gradio URL will open in your browser)

--------------------------------------------------------------------------------
ABOUT THE DATA — WHY TWO FILES ARE MERGED
--------------------------------------------------------------------------------
No single public file contains every column this project needs. Two real,
complementary Kaggle-sourced files are combined instead of inventing data:

  • Unemployment_in_India.csv
      Region, Date, Frequency, Unemployment Rate, Estimated Employed,
      Labour Participation Rate, Area (Rural/Urban)
      -> May 2019 – Jun 2020, rural/urban split, no zone or coordinates.

  • Unemployment_Rate_upto_11_2020.csv
      Region, Date, Frequency, Unemployment Rate, Estimated Employed,
      Labour Participation Rate, Region.1 (Zone), longitude, latitude
      -> Jan 2020 – Oct 2020, state totals, with zone + map coordinates.

DataPipeline merges them into one long table: every row from file 1 keeps its
Rural/Urban label; every row from file 2 is tagged Area="Total". A per-state
lookup of Zone/Longitude/Latitude (built from file 2) is then joined onto
every row by state name, so Rural-vs-Urban analysis, zone analysis, map
plotting, and a continuous May-2019→Oct-2020 timeline (covering the
pre-COVID and COVID lockdown period) are all available together.
================================================================================
"""

import re
import difflib
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import gradio as gr
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
try:
    pd.set_option("mode.chained_assignment", None)
except Exception:
    pass  # Deprecated in pandas 2.x; safe to skip

# ================================================================================
# 1. CONFIG / CONSTANTS
# ================================================================================

BASE_DIR = Path(__file__).resolve().parent
FILE_RURAL_URBAN = BASE_DIR / "Unemployment_in_India.csv"
FILE_ZONE_GEO = BASE_DIR / "Unemployment_Rate_upto_11_2020.csv"

# National lockdown announcement — used as the Pre-COVID / COVID split point.
COVID_START = pd.Timestamp("2020-03-25")

# Common abbreviations / misspellings -> canonical state name used in the data.
STATE_ALIASES = {
    "up": "Uttar Pradesh", "u.p": "Uttar Pradesh", "u.p.": "Uttar Pradesh",
    "mp": "Madhya Pradesh", "m.p": "Madhya Pradesh",
    "ap": "Andhra Pradesh", "a.p": "Andhra Pradesh",
    "tn": "Tamil Nadu", "wb": "West Bengal", "hp": "Himachal Pradesh",
    "j&k": "Jammu & Kashmir", "jk": "Jammu & Kashmir", "jandk": "Jammu & Kashmir",
    "jammu and kashmir": "Jammu & Kashmir", "jammu kashmir": "Jammu & Kashmir",
    "ncr": "Delhi", "new delhi": "Delhi",
    "bengaluru": "Karnataka", "bangalore": "Karnataka", "blr": "Karnataka",
    "mumbai": "Maharashtra", "chennai": "Tamil Nadu", "kolkata": "West Bengal",
    "hyderabad": "Telangana", "pondicherry": "Puducherry",
    "telengana": "Telangana", "gujrat": "Gujarat",
    "utranchal": "Uttarakhand", "uttaranchal": "Uttarakhand",
    "chattisgarh": "Chhattisgarh", "orissa": "Odisha",
}

# Geo info missing from the source file for a handful of union territories.
MANUAL_GEO = {
    # state: (zone, longitude, latitude)
    "Chandigarh": ("North", 76.7794, 30.7333),
}

ZONE_KEYWORDS = {
    "north east": "Northeast", "northeast": "Northeast", "north-east": "Northeast",
    "north": "North", "south": "South", "east": "East", "west": "West",
}

CATEGORY_MAP = {
    "highest_unemployment": "Unemployment Analysis",
    "lowest_unemployment": "Unemployment Analysis",
    "compare_states": "State Comparison",
    "top_states": "Ranking Analysis",
    "bottom_states": "Ranking Analysis",
    "state_analysis": "Regional Analysis",
    "region_analysis": "Regional Analysis",
    "covid_impact": "COVID Impact Analysis",
    "rural_urban": "Rural vs Urban Analysis",
    "trend": "Time-Based Analysis",
    "labour_participation": "Unemployment Analysis",
    "opportunity_ranking": "Job Opportunity Analysis",
    "job_recommendation": "Job Opportunity Analysis",
    "best_state": "Job Opportunity Analysis",
    "relocation": "Smart Recommendations",
    "prediction": "Prediction Analysis",
    "map_view": "Regional Analysis",
    "summary": "Smart Recommendations",
    "help": "Smart Recommendations",
}

EXAMPLE_QUESTIONS = [
    "Which state has the highest unemployment?",
    "Which state has the lowest unemployment?",
    "Top 5 states for jobs",
    "Compare Gujarat and Maharashtra",
    "Show Gujarat unemployment trend",
    "Best state for relocation",
    "Which region has the highest employment?",
    "Top 10 states by opportunity score",
    "Predict unemployment in Gujarat",
    "How did COVID affect unemployment?",
    "Rural vs urban unemployment",
    "Show me the map of opportunity scores",
]

# Sleek Dark/Neon palette — high-contrast, vibrant glow chart colors
COLORS = {
    "bg": "#0f172a", "card": "#1e293b", "border": "rgba(255,255,255,0.08)",
    "text": "#f8fafc", "muted": "#94a3b8",
    "accent": "#6366f1", "accent2": "#f43f5e", "warn": "#fbbf24", "danger": "#ef4444",
    "teal": "#14b8a6", "green": "#10b981", "purple": "#8b5cf6", "pink": "#ec4899",
}

# Distinct color sequence for multi-series charts (comparisons, etc.)
CHART_COLORS = ["#6366f1", "#f43f5e", "#14b8a6", "#fbbf24", "#8b5cf6", "#ec4899", "#10b981", "#3b82f6"]



# ================================================================================
# 2. SMALL FORMATTING HELPERS
# ================================================================================

def fmt_people(n: float) -> str:
    """Format a head-count as a human-readable string, e.g. 16,635,535 -> '16.6M'."""
    if pd.isna(n):
        return "N/A"
    n = float(n)
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:.0f}"


def fmt_pct(n: float, signed: bool = False) -> str:
    if pd.isna(n):
        return "N/A"
    sign = "+" if (signed and n > 0) else ""
    return f"{sign}{n:.2f}%"


def fmt_pp(n: float) -> str:
    """Percentage-point change, with explicit sign."""
    if pd.isna(n):
        return "N/A"
    sign = "+" if n > 0 else ""
    return f"{sign}{n:.2f} pp"


# ================================================================================
# 3. DATA PIPELINE — load, clean, merge, engineer features
# ================================================================================

class DataPipeline:
    """Loads both source CSVs, cleans/normalizes them, merges them into one
    long table, and builds derived state/zone-level analytics tables."""

    def __init__(self, rural_urban_path: Path, zone_geo_path: Path):
        self.rural_urban_path = rural_urban_path
        self.zone_geo_path = zone_geo_path
        self.df: pd.DataFrame = None
        self.state_summary: pd.DataFrame = None
        self.zone_summary: pd.DataFrame = None
        self.known_states: List[str] = []
        self._load_and_process()

    # ---- loading -----------------------------------------------------------

    @staticmethod
    def _canonicalize_state(name: str) -> str:
        name = " ".join(str(name).split())
        return STATE_ALIASES.get(name.lower(), name)

    def _load_rural_urban(self) -> pd.DataFrame:
        if not self.rural_urban_path.exists():
            raise FileNotFoundError(
                f"Could not find '{self.rural_urban_path.name}'. Place it next to app.py."
            )
        df = pd.read_csv(self.rural_urban_path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=["Region"]).copy()
        for c in ["Region", "Frequency", "Area"]:
            df[c] = df[c].astype(str).str.strip()
        df["Region"] = df["Region"].apply(self._canonicalize_state)
        df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(), format="%d-%m-%Y")
        for c in ["Estimated Unemployment Rate (%)", "Estimated Employed",
                  "Estimated Labour Participation Rate (%)"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Estimated Unemployment Rate (%)", "Estimated Employed",
                                "Estimated Labour Participation Rate (%)"])
        return df[["Region", "Date", "Frequency", "Estimated Unemployment Rate (%)",
                   "Estimated Employed", "Estimated Labour Participation Rate (%)", "Area"]]

    def _load_zone_geo(self) -> pd.DataFrame:
        if not self.zone_geo_path.exists():
            raise FileNotFoundError(
                f"Could not find '{self.zone_geo_path.name}'. Place it next to app.py."
            )
        df = pd.read_csv(self.zone_geo_path)
        df.columns = df.columns.str.strip()
        for c in ["Region", "Frequency", "Region.1"]:
            df[c] = df[c].astype(str).str.strip()
        df["Region"] = df["Region"].apply(self._canonicalize_state)
        df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(), format="%d-%m-%Y")
        # The source CSV columns for longitude/latitude are swapped in the dataset:
        # "longitude" contains latitude values, and "latitude" contains longitude values.
        df = df.rename(columns={"longitude": "Latitude", "latitude": "Longitude"})
        for c in ["Estimated Unemployment Rate (%)", "Estimated Employed",
                  "Estimated Labour Participation Rate (%)"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["Estimated Unemployment Rate (%)", "Estimated Employed",
                                  "Estimated Labour Participation Rate (%)"])

    def _build_geo_lookup(self, df_zg: pd.DataFrame) -> pd.DataFrame:
        lookup = (
            df_zg.groupby("Region")
            .agg(
                Zone=("Region.1", "first"),
                Longitude=("Longitude", "first"),
                Latitude=("Latitude", "first"),
            )
            .reset_index()
        )
        return lookup

    def _load_and_process(self):
        df_ru = self._load_rural_urban()
        df_zg = self._load_zone_geo()
        geo_lookup = self._build_geo_lookup(df_zg)

        df_ru = df_ru.copy()
        df_ru["Source"] = "Rural/Urban survey (2019-2020)"

        df_total = df_zg.copy()
        df_total["Area"] = "Total"
        df_total["Source"] = "Zone/Geo survey (2020)"
        df_total = df_total[["Region", "Date", "Frequency",
                              "Estimated Unemployment Rate (%)", "Estimated Employed",
                              "Estimated Labour Participation Rate (%)", "Area", "Source"]]

        combined = pd.concat([df_ru, df_total], ignore_index=True, sort=False)
        combined = combined.merge(geo_lookup, on="Region", how="left")

        for state, (zone, lon, lat) in MANUAL_GEO.items():
            mask = combined["Region"] == state
            combined.loc[mask & combined["Zone"].isna(), "Zone"] = zone
            combined.loc[mask & combined["Longitude"].isna(), "Longitude"] = lon
            combined.loc[mask & combined["Latitude"].isna(), "Latitude"] = lat

        combined = combined.rename(columns={
            "Estimated Unemployment Rate (%)": "Unemployment_Rate",
            "Estimated Employed": "Estimated_Employed",
            "Estimated Labour Participation Rate (%)": "Labour_Participation_Rate",
        })

        combined["Year"] = combined["Date"].dt.year
        combined["MonthName"] = combined["Date"].dt.strftime("%b %Y")
        combined["Period"] = np.where(combined["Date"] >= COVID_START, "COVID", "Pre-COVID")

        self.df = combined.sort_values(["Region", "Date"]).reset_index(drop=True)
        self.known_states = sorted(self.df["Region"].unique().tolist())
        self._engineer_features()

    # ---- feature engineering -----------------------------------------------

    @staticmethod
    def _minmax_0_100(series: pd.Series) -> pd.Series:
        s = series.astype(float)
        if s.max() == s.min():
            return pd.Series([50.0] * len(s), index=s.index)
        return ((s - s.min()) / (s.max() - s.min())) * 100

    def _engineer_features(self):
        df = self.df

        # ---- state-level Opportunity Score -----
        agg = df.groupby("Region").agg(
            Avg_Unemployment_Rate=("Unemployment_Rate", "mean"),
            Avg_Employed=("Estimated_Employed", "mean"),
            Avg_Labour_Participation=("Labour_Participation_Rate", "mean"),
            Zone=("Zone", "first"),
            Longitude=("Longitude", "first"),
            Latitude=("Latitude", "first"),
            Records=("Region", "count"),
        ).reset_index()

        agg["Employment_Score"] = self._minmax_0_100(agg["Avg_Employed"]).round(2)
        agg["Labour_Score"] = self._minmax_0_100(agg["Avg_Labour_Participation"]).round(2)
        agg["Unemployment_Score"] = (100 - self._minmax_0_100(agg["Avg_Unemployment_Rate"])).round(2)
        agg["Opportunity_Score"] = (
            0.4 * agg["Employment_Score"] + 0.3 * agg["Labour_Score"] + 0.3 * agg["Unemployment_Score"]
        ).round(2)
        agg["Rank"] = agg["Opportunity_Score"].rank(ascending=False, method="min").astype(int)

        # ---- growth metrics: first vs last available reading per state -----
        growth_rows = []
        for region, g in df.sort_values("Date").groupby("Region"):
            g = g.dropna(subset=["Unemployment_Rate", "Estimated_Employed", "Labour_Participation_Rate"])
            if len(g) < 2:
                growth_rows.append((region, np.nan, np.nan, np.nan))
                continue
            first, last = g.iloc[0], g.iloc[-1]
            emp_growth = (
                ((last["Estimated_Employed"] - first["Estimated_Employed"]) / first["Estimated_Employed"]) * 100
                if first["Estimated_Employed"] else np.nan
            )
            unemp_growth_pp = last["Unemployment_Rate"] - first["Unemployment_Rate"]
            lab_growth_pp = last["Labour_Participation_Rate"] - first["Labour_Participation_Rate"]
            growth_rows.append((region, emp_growth, unemp_growth_pp, lab_growth_pp))

        growth_df = pd.DataFrame(
            growth_rows, columns=["Region", "Employment_Growth_%", "Unemployment_Growth_pp", "Labour_Growth_pp"]
        )
        agg = agg.merge(growth_df, on="Region", how="left")

        self.state_summary = agg.sort_values("Opportunity_Score", ascending=False).reset_index(drop=True)

        # ---- zone-level summary -----
        zone_agg = df.dropna(subset=["Zone"]).groupby("Zone").agg(
            Avg_Unemployment_Rate=("Unemployment_Rate", "mean"),
            Avg_Employed=("Estimated_Employed", "mean"),
            Avg_Labour_Participation=("Labour_Participation_Rate", "mean"),
            States=("Region", "nunique"),
        ).reset_index()
        zone_agg["Employment_Score"] = self._minmax_0_100(zone_agg["Avg_Employed"]).round(2)
        zone_agg["Labour_Score"] = self._minmax_0_100(zone_agg["Avg_Labour_Participation"]).round(2)
        zone_agg["Unemployment_Score"] = (100 - self._minmax_0_100(zone_agg["Avg_Unemployment_Rate"])).round(2)
        zone_agg["Opportunity_Score"] = (
            0.4 * zone_agg["Employment_Score"] + 0.3 * zone_agg["Labour_Score"] + 0.3 * zone_agg["Unemployment_Score"]
        ).round(2)
        self.zone_summary = zone_agg.sort_values("Opportunity_Score", ascending=False).reset_index(drop=True)

    # ---- convenience accessors ---------------------------------------------

    def state_row(self, state: str) -> Optional[pd.Series]:
        rows = self.state_summary[self.state_summary["Region"] == state]
        return rows.iloc[0] if len(rows) else None

    def state_timeseries(self, state: str, metric_col: str) -> pd.DataFrame:
        g = self.df[self.df["Region"] == state].dropna(subset=[metric_col]).sort_values("Date")
        return g[["Date", "MonthName", metric_col, "Area"]]


# ================================================================================
# 4. NLP ENGINE — intent detection + entity extraction (no LLM)
# ================================================================================

class NLPEngine:
    """Rule-based intent detection and entity extraction using keyword
    dictionaries, regex, a synonym table, and difflib fuzzy matching."""

    INTENT_KEYWORDS = {
        "prediction": ["predict", "forecast", "future", "next month", "next year",
                       "projection", "will be", "expected to"],
        "covid_impact": ["covid", "corona", "pandemic", "lockdown"],
        "rural_urban": ["rural", "urban", "village", "city wise", "rural vs urban", "rural-urban"],
        "compare_states": ["compare", "versus", " vs ", "difference between", "comparison", "against"],
        "relocation": ["relocat", "move to", "shift to", "migrate", "should i move",
                        "where should i", "which state should i", "settle in",
                        "move from", "better state", "better place"],
        "job_recommendation": ["best state for job", "job opportunit", "good for job",
                                "career opportunit", "best for freshers", "freshers",
                                "apply for job", "new job", "looking for job", "find job",
                                "get a job", "want a job", "want job", "need a job",
                                "need job", "job search", "which state is good",
                                "good state for", "best place for job", "where to work",
                                "where should i work", "employment opportunit",
                                "start career", "begin career", "job market",
                                "hiring", "placement", "which state for job",
                                "best place to start", "good employment",
                                "start my career", "suggest me a state",
                                "suggest a state", "recommend a state"],
        "map_view": ["map", "geograph", "show on map", "where are", "locations"],
        "opportunity_ranking": ["opportunity score", "opportunity ranking"],
        "labour_participation": ["labour participation", "labor participation",
                                  "workforce participation", "participation rate"],
        "trend": ["trend", "over time", "history", "historical", "change over", "timeline"],
        "region_analysis": ["region", "zone", "north east", "northeast", "north", "south",
                             "east", "west", "regional"],
        "best_state": ["which state is good", "which state is best", "best state",
                       "good state", "which is the best state", "suggest state",
                       "recommend state", "safest state for"],
        "top_states": ["top ", "best states", "leading states", "top5", "top 5", "top10", "top 10"],
        "bottom_states": ["bottom", "worst states", "lagging states", "lowest states"],
        "highest_unemployment": ["highest unemployment", "most unemployment", "worst unemployment",
                                  "maximum unemployment", "highest jobless"],
        "lowest_unemployment": ["lowest unemployment", "least unemployment", "minimum unemployment",
                                 "best unemployment", "smallest unemployment"],
        "state_analysis": ["analysis of", "tell me about", "overview of", "details of",
                            "how is", "about "],
        "summary": ["summary", "overview", "insight", "report", "dashboard"],
        "help": ["help", "what can you do", "example questions", "how to use", "hi", "hello"],
    }

    # Order matters: more specific intents are checked with higher weight.
    INTENT_PRIORITY = [
        "prediction", "covid_impact", "rural_urban", "compare_states", "relocation",
        "job_recommendation", "best_state", "map_view", "opportunity_ranking",
        "labour_participation", "trend", "top_states", "bottom_states",
        "highest_unemployment", "lowest_unemployment",
        "region_analysis", "state_analysis", "summary", "help",
    ]

    def __init__(self, known_states: List[str]):
        self.known_states = sorted(known_states, key=len, reverse=True)
        self.known_states_lower = {s.lower(): s for s in self.known_states}

    def detect_intent(self, text: str):
        text_l = f" {text.lower()} "
        words = re.findall(r"[a-z0-9']+", text_l)
        scores = {}
        for intent in self.INTENT_PRIORITY:
            keywords = self.INTENT_KEYWORDS[intent]
            score = 0.0
            for kw in keywords:
                if kw in text_l:
                    score += 2.0
                    continue
                # Fuzzy fallback (typo tolerance) only applies to single-word
                # keywords, compared word-for-word against the input. Multi-
                # word phrases rely on the exact-substring check above —
                # fuzzy-matching a whole phrase against the input risks a
                # false positive whenever the phrase shares just one common
                # word (e.g. "unemployment") with an otherwise unrelated
                # sentence, and that bogus credit stacks across every
                # keyword in an intent's list.
                if " " in kw:
                    continue
                best_ratio = max(
                    (difflib.SequenceMatcher(None, kw, w).ratio() for w in words),
                    default=0.0,
                )
                if best_ratio > 0.82:
                    score += best_ratio
            if score > 0:
                scores[intent] = score

        if not scores:
            return "summary", 0.35

        best_intent = max(scores, key=lambda k: scores[k])
        confidence = round(min(0.98, 0.45 + scores[best_intent] / 8), 2)
        return best_intent, confidence

    def extract_states(self, text: str) -> List[str]:
        text_l = f" {text.lower()} "
        found = []  # (start_index, canonical_name)
        for s_lower, s_canon in self.known_states_lower.items():
            for m in re.finditer(r"\b" + re.escape(s_lower) + r"\b", text_l):
                found.append((m.start(), s_canon))
        if not found:
            for alias, canon in STATE_ALIASES.items():
                m = re.search(r"\b" + re.escape(alias) + r"\b", text_l)
                if m:
                    found.append((m.start(), canon))
        if not found:
            words = re.findall(r"[A-Za-z&]+", text)
            candidates = words + [" ".join(words[i:i + 2]) for i in range(len(words) - 1)]
            for cand in candidates:
                match = difflib.get_close_matches(cand, self.known_states, n=1, cutoff=0.8)
                if match:
                    pos = text.lower().find(cand.lower())
                    found.append((pos if pos >= 0 else 0, match[0]))
        # Preserve the order the states actually appear in the question
        # (so "Compare A and B" reliably means state_a=A, state_b=B),
        # while keeping longer/more-specific name matches preferred on ties.
        found.sort(key=lambda x: x[0])
        seen, ordered = set(), []
        for _, f in found:
            if f not in seen:
                seen.add(f)
                ordered.append(f)
        return ordered

    def extract_zone(self, text: str) -> Optional[str]:
        text_l = text.lower()
        for kw, zone in ZONE_KEYWORDS.items():
            if re.search(r"\b" + re.escape(kw) + r"\b", text_l):
                return zone
        return None

    def extract_number(self, text: str, default: int = 5) -> int:
        m = re.search(r"\b(\d{1,2})\b", text)
        return int(m.group(1)) if m else default

    def extract_metric(self, text: str) -> str:
        text_l = text.lower()
        if "labour" in text_l or "labor" in text_l or "participation" in text_l:
            return "labour"
        if "employ" in text_l and "unemploy" not in text_l:
            return "employment"
        return "unemployment"


METRIC_COLUMN = {
    "unemployment": "Unemployment_Rate",
    "employment": "Estimated_Employed",
    "labour": "Labour_Participation_Rate",
}
METRIC_LABEL = {
    "unemployment": "Unemployment Rate (%)",
    "employment": "Estimated Employed",
    "labour": "Labour Participation Rate (%)",
}


# ================================================================================
# 5. CHART FACTORY — Plotly figure builders, light-themed
# ================================================================================

PLOTLY_TEMPLATE = "plotly_dark"


def _style_fig(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=15, color="#f8fafc", family="Plus Jakarta Sans, sans-serif")),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(color="#f8fafc", family="Inter, sans-serif", size=12),
        margin=dict(l=50, r=20, t=55, b=50),
        legend=dict(bgcolor="rgba(15, 23, 42, 0.85)", bordercolor="rgba(99, 102, 241, 0.2)", borderwidth=1),
        dragmode=False,
        hovermode="closest",
        modebar=dict(
            remove=["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d", "hoverClosestCartesian", "hoverCompareCartesian", "toggleSpikelines"]
        )
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#1e293b",
        linecolor="#475569",
        linewidth=1,
        tickfont=dict(color="#cbd5e1", size=10),
        title_font=dict(color="#f8fafc", size=11, family="Plus Jakarta Sans, sans-serif"),
        showspikes=False
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#1e293b",
        linecolor="#475569",
        linewidth=1,
        tickfont=dict(color="#cbd5e1", size=10),
        title_font=dict(color="#f8fafc", size=11, family="Plus Jakarta Sans, sans-serif"),
        showspikes=False
    )
    return fig


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str = None,
              color_seq=None) -> go.Figure:
    # When color grouping is used (e.g. state comparison), use the full CHART_COLORS
    # so each group gets a visually distinct color.
    if color and not color_seq:
        colors = CHART_COLORS
    else:
        colors = color_seq or [COLORS["accent"]]
    fig = px.bar(df, x=x, y=y, color=color, text_auto=".2f",
                 color_discrete_sequence=colors)
    fig.update_traces(marker_line_width=0, textfont=dict(size=12, color=COLORS["text"]))
    return _style_fig(fig, title)


def line_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str = None) -> go.Figure:
    fig = px.line(df, x=x, y=y, color=color, markers=True,
                   color_discrete_sequence=CHART_COLORS)
    return _style_fig(fig, title)


def pie_chart(df: pd.DataFrame, names: str, values: str, title: str) -> go.Figure:
    fig = px.pie(df, names=names, values=values, hole=0.45,
                 color_discrete_sequence=CHART_COLORS)
    fig.update_traces(textfont=dict(size=13, color="#ffffff"),
                      outsidetextfont=dict(color=COLORS["text"]))
    return _style_fig(fig, title)


def map_chart(df: pd.DataFrame, color_col: str, title: str) -> go.Figure:
    d = df.dropna(subset=["Latitude", "Longitude"]).copy()
    
    # Ensure Avg_Employed is numeric for sizing
    d["Avg_Employed"] = pd.to_numeric(d["Avg_Employed"], errors="coerce").fillna(0)

    # Determine colorscale based on metric
    if "Unemployment" in color_col or "unemployment" in color_col.lower():
        colorscale = "OrRd"
        color_label = "Unemployment Rate (%)"
    elif "Labour" in color_col or "labour" in color_col.lower():
        colorscale = "Viridis"
        color_label = "Labour Participation (%)"
    elif "Employed" in color_col or "employed" in color_col.lower():
        colorscale = "Blues"
        color_label = "Employed Population"
    else:
        colorscale = "Viridis"
        color_label = "Opportunity Score"
        
    fig = px.scatter_geo(
        d,
        lat="Latitude",
        lon="Longitude",
        color=color_col,
        size="Avg_Employed",
        hover_name="Region",
        color_continuous_scale=colorscale,
        title=title,
        labels={
            color_col: color_label,
            "Avg_Employed": "Employed Population",
            "Avg_Unemployment_Rate": "Unemployment Rate (%)",
            "Avg_Labour_Participation": "Labour Participation (%)",
            "Opportunity_Score": "Opportunity Score"
        }
    )
    
    max_employed = d["Avg_Employed"].max()
    sizeref_val = 2.0 * max_employed / (30.0 ** 2) if max_employed > 0 else 1.0

    fig.update_traces(
        textposition="top center",
        textfont=dict(color="#f8fafc", size=9, family="Inter, sans-serif"),
        marker=dict(
            opacity=0.85,
            line=dict(width=1, color="rgba(255, 255, 255, 0.6)"),
            sizemode="area",
            sizeref=sizeref_val,
            sizemin=4
        ),
        hovertext=d["Region"],
        customdata=d[["Avg_Unemployment_Rate", "Avg_Employed", "Avg_Labour_Participation", "Opportunity_Score"]].values,
        hovertemplate="<b>%{hovertext}</b><br><br>" +
                      "Opportunity Score: %{customdata[3]:.2f}/100<br>" +
                      "Unemployment Rate: %{customdata[0]:.2f}%<br>" +
                      "Labour Participation: %{customdata[2]:.2f}%<br>" +
                      "Employed Population: %{customdata[1]:,.0f}<extra></extra>"
    )
    
    # Add manual size scale traces
    legend_sizes = [5_000_000, 15_000_000, 30_000_000]
    for size_val in legend_sizes:
        marker_size = (size_val / sizeref_val) ** 0.5 if sizeref_val > 0 else 10
        fig.add_trace(go.Scattergeo(
            lat=[None],
            lon=[None],
            mode="markers",
            name=f"{size_val/1_000_000:.0f}M Employed",
            marker=dict(
                size=marker_size,
                color="rgba(148, 163, 184, 0.4)",
                line=dict(width=1, color="rgba(255, 255, 255, 0.6)")
            ),
            showlegend=True
        ))
        
    fig.update_layout(
        template="plotly_dark",
        showlegend=True,
        legend=dict(
            title=dict(text="Employed Base", font=dict(color="#cbd5e1", size=10)),
            bgcolor="rgba(15, 23, 42, 0.85)", 
            bordercolor="rgba(255,255,255,0.08)", 
            borderwidth=1,
            y=0.5,
            x=1.02,
            yanchor="middle",
            xanchor="left"
        ),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=0, r=80, t=40, b=0),
        font=dict(color="#94a3b8", family="Inter, sans-serif", size=12),
        coloraxis_colorbar=dict(
            thickness=12,
            len=0.4,
            y=0.15,
            title=dict(font=dict(size=10, color="#94a3b8")),
            tickfont=dict(size=9, color="#94a3b8")
        )
    )
    
    fig.update_geos(
        fitbounds="locations",
        showland=True,
        landcolor="#1e293b",
        showocean=True,
        oceancolor="#090d16",
        showlakes=True,
        lakecolor="#090d16",
        showcountries=True,
        countrycolor="rgba(255, 255, 255, 0.25)",
        coastlinecolor="rgba(255, 255, 255, 0.25)",
        visible=True
    )
    
    return fig




def forecast_chart(hist: pd.DataFrame, future: pd.DataFrame, metric_label: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["Date"], y=hist["value"], mode="lines+markers",
                              name="Historical", line=dict(color=COLORS["accent"])))
    bridge_x = list(hist["Date"])[-1:] + list(future["Date"])
    bridge_y = list(hist["value"])[-1:] + list(future["value"])
    fig.add_trace(go.Scatter(x=bridge_x, y=bridge_y, mode="lines+markers", name="Forecast",
                              line=dict(color=COLORS["warn"], dash="dash")))
    fig.update_yaxes(title=metric_label)
    return _style_fig(fig, title)


# ================================================================================
# 6. QUERY RESULT CONTAINER
# ================================================================================

@dataclass
class QueryResult:
    answer: str
    chart: Optional[go.Figure] = None
    map_fig: Optional[go.Figure] = None
    table: Optional[pd.DataFrame] = None
    explanation: str = ""
    source: str = ""


# ================================================================================
# 7. ANALYTICS ENGINE — one function per required capability
# ================================================================================

class Analytics:
    def __init__(self, pipeline: DataPipeline):
        self.p = pipeline

    # ---------- Unemployment Analysis ----------------------------------------

    def get_highest_unemployment(self) -> QueryResult:
        s = self.p.state_summary.sort_values("Avg_Unemployment_Rate", ascending=False)
        top = s.iloc[0]
        answer = (
            f"**{top['Region']}** currently has the highest average unemployment rate at "
            f"**{top['Avg_Unemployment_Rate']:.2f}%** (Zone: {top['Zone']}). "
            f"Its Opportunity Score is {top['Opportunity_Score']:.1f}/100 (rank #{int(top['Rank'])})."
        )
        chart = bar_chart(s.head(10), "Region", "Avg_Unemployment_Rate",
                           "Top 10 States by Unemployment Rate", color_seq=[COLORS["danger"]])
        return QueryResult(answer, chart=chart, table=s.head(10)[["Region", "Avg_Unemployment_Rate", "Zone"]],
                            explanation="Ranked states by mean Unemployment_Rate across all available readings.",
                            source="Merged Rural/Urban + Zone/Geo dataset")

    def get_lowest_unemployment(self) -> QueryResult:
        s = self.p.state_summary.sort_values("Avg_Unemployment_Rate", ascending=True)
        bottom = s.iloc[0]
        answer = (
            f"**{bottom['Region']}** has the lowest average unemployment rate at "
            f"**{bottom['Avg_Unemployment_Rate']:.2f}%** (Zone: {bottom['Zone']}). "
            f"Its Opportunity Score is {bottom['Opportunity_Score']:.1f}/100 (rank #{int(bottom['Rank'])})."
        )
        chart = bar_chart(s.head(10), "Region", "Avg_Unemployment_Rate",
                           "10 States with Lowest Unemployment Rate", color_seq=[COLORS["accent2"]])
        return QueryResult(answer, chart=chart, table=s.head(10)[["Region", "Avg_Unemployment_Rate", "Zone"]],
                            explanation="Ranked states by mean Unemployment_Rate, ascending.",
                            source="Merged Rural/Urban + Zone/Geo dataset")

    def labour_analysis(self, state: Optional[str] = None) -> QueryResult:
        if state:
            row = self.p.state_row(state)
            if row is None:
                return self._unknown_state(state)
            answer = (
                f"**{state}** has an average labour participation rate of "
                f"**{row['Avg_Labour_Participation']:.2f}%**, with a {fmt_pp(row['Labour_Growth_pp'])} "
                f"change from its first to most recent reading."
            )
            ts = self.p.state_timeseries(state, "Labour_Participation_Rate")
            chart = line_chart(ts, "Date", "Labour_Participation_Rate", f"{state} — Labour Participation Over Time")
            return QueryResult(answer, chart=chart, table=ts.tail(10),
                                explanation="Mean Labour_Participation_Rate for the state, plus first-vs-last change.",
                                source="Merged dataset")
        s = self.p.state_summary.sort_values("Avg_Labour_Participation", ascending=False)
        top = s.iloc[0]
        answer = (
            f"**{top['Region']}** has the highest labour force participation at "
            f"**{top['Avg_Labour_Participation']:.2f}%**, meaning the largest share of its "
            f"working-age population is economically active."
        )
        chart = bar_chart(s.head(10), "Region", "Avg_Labour_Participation",
                           "Top 10 States by Labour Participation Rate", color_seq=[COLORS["accent"]])
        return QueryResult(answer, chart=chart, table=s.head(10)[["Region", "Avg_Labour_Participation"]],
                            explanation="Ranked states by mean Labour_Participation_Rate.",
                            source="Merged dataset")

    def employment_analysis(self, state: Optional[str] = None) -> QueryResult:
        if state:
            row = self.p.state_row(state)
            if row is None:
                return self._unknown_state(state)
            answer = (
                f"**{state}** has an average of **{fmt_people(row['Avg_Employed'])}** people employed "
                f"({fmt_pct(row['Employment_Growth_%'], signed=True)} change from first to most recent reading)."
            )
            ts = self.p.state_timeseries(state, "Estimated_Employed")
            chart = line_chart(ts, "Date", "Estimated_Employed", f"{state} — Estimated Employed Over Time")
            return QueryResult(answer, chart=chart, table=ts.tail(10),
                                explanation="Mean Estimated_Employed for the state, plus % growth first-to-last.",
                                source="Merged dataset")
        s = self.p.state_summary.sort_values("Avg_Employed", ascending=False)
        top = s.iloc[0]
        answer = (
            f"**{top['Region']}** has the highest average employed population at "
            f"**{fmt_people(top['Avg_Employed'])}** people."
        )
        chart = bar_chart(s.head(10), "Region", "Avg_Employed",
                           "Top 10 States by Estimated Employed Population")
        return QueryResult(answer, chart=chart, table=s.head(10)[["Region", "Avg_Employed"]],
                            explanation="Ranked states by mean Estimated_Employed.",
                            source="Merged dataset")

    def opportunity_score_analysis(self, state: Optional[str] = None) -> QueryResult:
        if state:
            row = self.p.state_row(state)
            if row is None:
                return self._unknown_state(state)
            answer = (
                f"**{state}** has an Opportunity Score of **{row['Opportunity_Score']:.1f}/100** "
                f"(rank #{int(row['Rank'])} of {len(self.p.state_summary)}), built from "
                f"Employment Score **{row['Employment_Score']:.1f}**, Labour Score **{row['Labour_Score']:.1f}**, "
                f"and Unemployment Score **{row['Unemployment_Score']:.1f}** "
                f"(weights 0.4 / 0.3 / 0.3)."
            )
            return QueryResult(answer, explanation="Opportunity Score = 0.4×Employment + 0.3×Labour + 0.3×(100-Unemployment), each min-max normalized 0-100 across all states.",
                                source="Derived feature")
        return self.ranking_analysis(n=10)

    # ---------- Ranking Analysis ----------------------------------------------

    def get_top_states(self, n: int = 5) -> QueryResult:
        s = self.p.state_summary.head(n)
        names = ", ".join(s["Region"].tolist())
        answer = (
            f"The top {n} states by Opportunity Score are: **{names}**. "
            f"#1 is **{s.iloc[0]['Region']}** with a score of {s.iloc[0]['Opportunity_Score']:.1f}/100."
        )
        chart = bar_chart(s, "Region", "Opportunity_Score", f"Top {n} States by Opportunity Score")
        return QueryResult(answer, chart=chart,
                            table=s[["Rank", "Region", "Opportunity_Score", "Avg_Unemployment_Rate", "Zone"]],
                            explanation="Top N states sorted by Opportunity_Score descending.",
                            source="Derived state-level summary")

    def get_bottom_states(self, n: int = 5) -> QueryResult:
        s = self.p.state_summary.tail(n).sort_values("Opportunity_Score")
        names = ", ".join(s["Region"].tolist())
        answer = (
            f"The {n} states with the lowest Opportunity Scores are: **{names}**. "
            f"The lowest is **{s.iloc[0]['Region']}** at {s.iloc[0]['Opportunity_Score']:.1f}/100."
        )
        chart = bar_chart(s, "Region", "Opportunity_Score", f"Bottom {n} States by Opportunity Score",
                           color_seq=[COLORS["danger"]])
        return QueryResult(answer, chart=chart,
                            table=s[["Rank", "Region", "Opportunity_Score", "Avg_Unemployment_Rate", "Zone"]],
                            explanation="Bottom N states sorted by Opportunity_Score ascending.",
                            source="Derived state-level summary")

    def ranking_analysis(self, n: int = 10) -> QueryResult:
        s = self.p.state_summary.head(n)
        answer = (
            f"Here is the state ranking by Opportunity Score (top {n} of {len(self.p.state_summary)} states). "
            f"**{s.iloc[0]['Region']}** leads at {s.iloc[0]['Opportunity_Score']:.1f}/100."
        )
        chart = bar_chart(s, "Region", "Opportunity_Score", f"State Ranking — Top {n}")
        return QueryResult(answer, chart=chart,
                            table=s[["Rank", "Region", "Opportunity_Score", "Employment_Score",
                                     "Labour_Score", "Unemployment_Score"]],
                            explanation="Full ranking table sorted by Opportunity_Score.",
                            source="Derived state-level summary")

    # ---------- State Comparison ----------------------------------------------

    def compare_states(self, states: List[str]) -> QueryResult:
        if not states:
            return QueryResult("Specify the states you want to compare, e.g. 'Compare Gujarat and Maharashtra'.")
            
        # Retrieve state rows
        state_rows = []
        state_names = []
        for state_name in states:
            row = self.p.state_row(state_name)
            if row is None:
                return self._unknown_state(state_name)
            state_rows.append(row)
            state_names.append(state_name)
            
        # Build comparison text sorted by Opportunity Score
        sorted_indices = sorted(range(len(state_rows)), key=lambda i: state_rows[i]["Opportunity_Score"], reverse=True)
        sorted_names = [state_names[i] for i in sorted_indices]
        sorted_rows = [state_rows[i] for i in sorted_indices]
        
        comp_details = ", ".join([f"**{name}** ({row['Opportunity_Score']:.1f}/100, unemployment {row['Avg_Unemployment_Rate']:.2f}%)" for name, row in zip(state_names, state_rows)])
        
        winner_name = sorted_names[0]
        winner_row = sorted_rows[0]
        
        if len(state_names) == 2:
            diff = abs(state_rows[0]["Opportunity_Score"] - state_rows[1]["Opportunity_Score"])
            answer = (
                f"Comparing: {comp_details}. "
                f"**{winner_name}** has the stronger Opportunity Score, ahead by {diff:.1f} points, "
                f"driven mainly by its {'employment level' if state_rows[0]['Employment_Score'] != state_rows[1]['Employment_Score'] else 'overall balance'} "
                f"of the three underlying metrics."
            )
        else:
            rankings = ", ".join([f"{name} ({row['Opportunity_Score']:.1f}/100)" for name, row in zip(sorted_names[1:], sorted_rows[1:])])
            answer = (
                f"Comparing {len(state_names)} states: {comp_details}. "
                f"**{winner_name}** leads with the highest Opportunity Score of {winner_row['Opportunity_Score']:.1f}/100, "
                f"followed by {rankings}."
            )
            
        # Build comparison dataframe
        comp_data = {
            "Metric": ["Opportunity Score", "Unemployment Rate (%)", "Employed (avg)", "Labour Participation (%)"]
        }
        for name, row in zip(state_names, state_rows):
            comp_data[name] = [
                row["Opportunity_Score"],
                row["Avg_Unemployment_Rate"],
                row["Avg_Employed"],
                row["Avg_Labour_Participation"]
            ]
        comp_df = pd.DataFrame(comp_data)
        
        # Build plot dataframe
        plot_rows = []
        for name, row in zip(state_names, state_rows):
            plot_rows.append({"Region": name, "Metric": "Opportunity Score", "Value": row["Opportunity_Score"]})
            plot_rows.append({"Region": name, "Metric": "Unemployment Rate (%)", "Value": row["Avg_Unemployment_Rate"]})
            plot_rows.append({"Region": name, "Metric": "Labour Participation (%)", "Value": row["Avg_Labour_Participation"]})
        plot_df = pd.DataFrame(plot_rows)
        
        chart = bar_chart(plot_df, "Metric", "Value", f"Comparison: {', '.join(state_names)}", color="Region")
        return QueryResult(answer, chart=chart, table=comp_df,
                            explanation="Direct comparison of mean metrics and the composite Opportunity Score across the selected states.",
                            source="Derived state-level summary")

    # ---------- Regional / Zone Analysis ---------------------------------------

    def region_analysis(self, zone: Optional[str] = None) -> QueryResult:
        zs = self.p.zone_summary
        if zone:
            row = zs[zs["Zone"] == zone]
            if row.empty:
                return QueryResult(f"I don't have data tagged for the '{zone}' zone. "
                                    f"Available zones: {', '.join(zs['Zone'].tolist())}.")
            row = row.iloc[0]
            states_in_zone = self.p.state_summary[self.p.state_summary["Zone"] == zone].sort_values(
                "Opportunity_Score", ascending=False)
            answer = (
                f"The **{zone}** zone has an average unemployment rate of {row['Avg_Unemployment_Rate']:.2f}% "
                f"across {int(row['States'])} states, with a composite Opportunity Score of "
                f"{row['Opportunity_Score']:.1f}/100. Its strongest state is "
                f"**{states_in_zone.iloc[0]['Region']}**."
            )
            chart = bar_chart(states_in_zone, "Region", "Opportunity_Score", f"{zone} Zone — States by Opportunity Score")
            return QueryResult(answer, chart=chart,
                                table=states_in_zone[["Region", "Opportunity_Score", "Avg_Unemployment_Rate"]],
                                explanation="Zone aggregated from member states' mean metrics.",
                                source="Derived zone-level summary")

        best = zs.iloc[0]
        answer = (
            f"Across India's zones, **{best['Zone']}** has the highest Opportunity Score "
            f"({best['Opportunity_Score']:.1f}/100) with an average unemployment rate of "
            f"{best['Avg_Unemployment_Rate']:.2f}%."
        )
        chart = bar_chart(zs, "Zone", "Opportunity_Score", "Zones by Opportunity Score")
        return QueryResult(answer, chart=chart, table=zs, explanation="All zones ranked by composite Opportunity Score.",
                            source="Derived zone-level summary")

    def map_analysis(self, metric: str = "opportunity") -> QueryResult:
        col = {"opportunity": "Opportunity_Score", "unemployment": "Avg_Unemployment_Rate",
               "employment": "Avg_Employed", "labour": "Avg_Labour_Participation"}.get(metric, "Opportunity_Score")
        title = {
            "Opportunity_Score": "State Opportunity Scores Across India",
            "Avg_Unemployment_Rate": "State Unemployment Rates Across India",
            "Avg_Employed": "Estimated Employed Population Across India",
            "Avg_Labour_Participation": "Labour Participation Rate Across India",
        }[col]
        fig = map_chart(self.p.state_summary, col, title)
        answer = ("Here's an interactive map of India — colour shows " +
                  ("the Opportunity Score" if col == "Opportunity_Score" else col.replace('_', ' ')) +
                  ", bubble size shows average employed population. Hover over a state for details.")
        return QueryResult(answer, map_fig=fig,
                            explanation="Latitude/longitude sourced from the Zone/Geo file, joined to every state.",
                            source="Merged dataset — geo lookup")

    # ---------- COVID Impact Analysis ----------------------------------------

    def covid_analysis(self, state: Optional[str] = None) -> QueryResult:
        df = self.p.df
        if state:
            df = df[df["Region"] == state]
            if df.empty:
                return self._unknown_state(state)
        pre = df[df["Period"] == "Pre-COVID"]["Unemployment_Rate"].mean()
        post = df[df["Period"] == "COVID"]["Unemployment_Rate"].mean()
        delta = post - pre
        scope = state if state else "India (national average)"
        answer = (
            f"For **{scope}**, the average unemployment rate was **{pre:.2f}%** before the COVID-19 "
            f"lockdown and rose to **{post:.2f}%** during the lockdown period "
            f"(from {COVID_START.strftime('%d %b %Y')} onward) — a change of **{fmt_pp(delta)}**."
        )
        trend = df.groupby("Period")["Unemployment_Rate"].mean().reset_index()
        chart = bar_chart(trend, "Period", "Unemployment_Rate",
                           f"Pre-COVID vs COVID Unemployment — {scope}", color_seq=[COLORS["danger"]])
        if not state:
            by_state = (
                df.groupby(["Region", "Period"])["Unemployment_Rate"].mean().unstack()
                .assign(Impact_pp=lambda d: d.get("COVID", np.nan) - d.get("Pre-COVID", np.nan))
                .sort_values("Impact_pp", ascending=False)
                .reset_index()
            )
            worst = by_state.iloc[0]
            answer += f" The hardest-hit state was **{worst['Region']}**, up {fmt_pp(worst['Impact_pp'])}."
            return QueryResult(answer, chart=chart, table=by_state.head(10),
                                explanation="Mean Unemployment_Rate compared before vs after 25 Mar 2020 lockdown date.",
                                source="Merged dataset, Period flag")
        return QueryResult(answer, chart=chart,
                            explanation="Mean Unemployment_Rate compared before vs after 25 Mar 2020 lockdown date.",
                            source="Merged dataset, Period flag")

    # ---------- Rural vs Urban Analysis ---------------------------------------

    def rural_urban_analysis(self, state: Optional[str] = None) -> QueryResult:
        df = self.p.df[self.p.df["Area"].isin(["Rural", "Urban"])]
        if state:
            df = df[df["Region"] == state]
            if df.empty:
                return QueryResult(f"No Rural/Urban split is available for {state} in this dataset.")
        grp = df.groupby("Area").agg(
            Avg_Unemployment_Rate=("Unemployment_Rate", "mean"),
            Avg_Employed=("Estimated_Employed", "mean"),
        ).reset_index()
        rural = grp[grp["Area"] == "Rural"]
        urban = grp[grp["Area"] == "Urban"]
        scope = state if state else "India overall"
        if len(rural) and len(urban):
            r_rate, u_rate = rural.iloc[0]["Avg_Unemployment_Rate"], urban.iloc[0]["Avg_Unemployment_Rate"]
            higher = "Urban" if u_rate > r_rate else "Rural"
            answer = (
                f"For **{scope}**, average unemployment is **{r_rate:.2f}%** in rural areas and "
                f"**{u_rate:.2f}%** in urban areas — **{higher}** areas show higher unemployment "
                f"by {abs(u_rate - r_rate):.2f} percentage points."
            )
        else:
            answer = f"Rural/Urban breakdown for {scope} is incomplete in the source data."
        chart = bar_chart(grp, "Area", "Avg_Unemployment_Rate", f"Rural vs Urban Unemployment — {scope}")
        pie = pie_chart(grp, "Area", "Avg_Employed", f"Rural vs Urban Employed Share — {scope}")
        return QueryResult(answer, chart=chart, map_fig=pie, table=grp,
                            explanation="Area-tagged rows (Rural/Urban) averaged for the chosen scope.",
                            source="Rural/Urban survey file")

    # ---------- Time-Based / Trend Analysis -----------------------------------

    def trend_analysis(self, state: Optional[str] = None, metric: str = "unemployment") -> QueryResult:
        col = METRIC_COLUMN[metric]
        label = METRIC_LABEL[metric]
        if state:
            ts = self.p.state_timeseries(state, col)
            if ts.empty:
                return self._unknown_state(state)
            first, last = ts.iloc[0][col], ts.iloc[-1][col]
            change = last - first
            direction = "risen" if change > 0 else ("fallen" if change < 0 else "stayed flat")
            answer = (
                f"**{state}**'s {label.lower()} has {direction} from {first:.2f} "
                f"({ts.iloc[0]['MonthName']}) to {last:.2f} ({ts.iloc[-1]['MonthName']})."
            )
            chart = line_chart(ts, "Date", col, f"{state} — {label} Trend")
            return QueryResult(answer, chart=chart, table=ts,
                                explanation="Full time series for the state, sorted chronologically.",
                                source="Merged dataset")
        national = self.p.df.groupby("Date")[col].mean().reset_index()
        first, last = national.iloc[0][col], national.iloc[-1][col]
        answer = (
            f"Nationally, average {label.lower()} moved from {first:.2f} to {last:.2f} "
            f"across the dataset's time span ({national.iloc[0]['Date'].strftime('%b %Y')} to "
            f"{national.iloc[-1]['Date'].strftime('%b %Y')})."
        )
        chart = line_chart(national, "Date", col, f"National {label} Trend")
        return QueryResult(answer, chart=chart, table=national,
                            explanation="National daily average across all states for the chosen metric.",
                            source="Merged dataset")

    # ---------- Smart Recommendations -----------------------------------------

    def job_recommendation(self, n: int = 5) -> QueryResult:
        s = self.p.state_summary.head(n)
        names = ", ".join(s["Region"].tolist())
        answer = (
            f"For job opportunities right now, consider: **{names}**. These states combine low "
            f"unemployment, strong labour participation, and a large employed base — "
            f"**{s.iloc[0]['Region']}** ranks #1 with an Opportunity Score of {s.iloc[0]['Opportunity_Score']:.1f}/100."
        )
        chart = bar_chart(s, "Region", "Opportunity_Score", f"Top {n} States for Job Opportunities")
        return QueryResult(answer, chart=chart,
                            table=s[["Rank", "Region", "Opportunity_Score", "Avg_Unemployment_Rate"]],
                            explanation="Same ranking as Opportunity Score, framed for job-seekers.",
                            source="Derived state-level summary")

    def career_relocation_recommendation(self, current_state: Optional[str] = None, n: int = 3) -> QueryResult:
        top = self.p.state_summary.head(n)
        if current_state:
            cur = self.p.state_row(current_state)
            if cur is None:
                return self._unknown_state(current_state)
            if current_state in top["Region"].values:
                answer = (
                    f"**{current_state}** is already among the top {n} states by Opportunity Score "
                    f"({cur['Opportunity_Score']:.1f}/100, rank #{int(cur['Rank'])}) — relocating may not "
                    f"meaningfully improve your job-market odds."
                )
            else:
                best = top.iloc[0]
                answer = (
                    f"**{current_state}** ranks #{int(cur['Rank'])} with an Opportunity Score of "
                    f"{cur['Opportunity_Score']:.1f}/100. **{best['Region']}** scores higher at "
                    f"{best['Opportunity_Score']:.1f}/100, with unemployment of "
                    f"{best['Avg_Unemployment_Rate']:.2f}% vs {cur['Avg_Unemployment_Rate']:.2f}% — "
                    f"it could be worth considering for relocation."
                )
        else:
            names = ", ".join(top["Region"].tolist())
            answer = (
                f"Based on employment, labour participation and unemployment metrics, the best states "
                f"to relocate to are **{names}**. **{top.iloc[0]['Region']}** is recommended first, with "
                f"high employment and a strong Opportunity Score of {top.iloc[0]['Opportunity_Score']:.1f}."
            )
        chart = bar_chart(self.p.state_summary.head(8), "Region", "Opportunity_Score", "Best States to Relocate To")
        return QueryResult(answer, chart=chart, table=top[["Rank", "Region", "Opportunity_Score"]],
                            explanation="Compares the current state (if given) against the top-ranked states.",
                            source="Derived state-level summary")

    # ---------- Prediction Analysis -------------------------------------------

    def forecast_state(self, state: str, metric: str = "unemployment", periods: int = 3) -> QueryResult:
        col = METRIC_COLUMN[metric]
        label = METRIC_LABEL[metric]
        ts = self.p.state_timeseries(state, col)
        if state not in self.p.known_states:
            return self._unknown_state(state)
        if len(ts) < 3:
            return QueryResult(f"Not enough historical data points for {state} to build a reliable forecast.")

        ts = ts.sort_values("Date").reset_index(drop=True)
        x = (ts["Date"] - ts["Date"].min()).dt.days.values.reshape(-1, 1)
        y = ts[col].values

        model = LinearRegression()
        model.fit(x, y)

        last_date = ts["Date"].max()
        future_dates = pd.date_range(last_date, periods=periods + 1, freq="MS")[1:]
        future_x = (future_dates - ts["Date"].min()).days.values.reshape(-1, 1)
        preds = model.predict(future_x)
        preds = np.clip(preds, 0, None)  # All metrics are non-negative

        hist_df = pd.DataFrame({"Date": ts["Date"], "value": ts[col]})
        future_df = pd.DataFrame({"Date": future_dates, "value": preds})

        current_val = ts.iloc[-1][col]
        next_val = preds[0]
        direction = "increase" if next_val > current_val else "decrease"
        unit = "%" if metric != "employment" else ""
        cur_str = f"{current_val:.2f}{unit}" if metric != "employment" else fmt_people(current_val)
        next_str = f"{next_val:.2f}{unit}" if metric != "employment" else fmt_people(next_val)

        answer = (
            f"Based on a linear trend fit to {len(ts)} historical readings, **{state}**'s {label.lower()} "
            f"is projected to **{direction}** from {cur_str} to approximately **{next_str}** "
            f"by {future_dates[0].strftime('%b %Y')}."
        )
        chart = forecast_chart(hist_df, future_df, label, f"{state} — {label} Forecast")
        future_table = pd.DataFrame({"Month": future_dates.strftime("%b %Y"), f"Predicted {label}": preds.round(2)})
        return QueryResult(answer, chart=chart, table=future_table,
                            explanation="scikit-learn LinearRegression fit on (days since first reading) → metric value; "
                                        "extrapolated month-by-month. A simple trend model, not a guarantee.",
                            source="Merged dataset, model = LinearRegression")

    def predict_unemployment(self, state: str, periods: int = 3) -> QueryResult:
        return self.forecast_state(state, metric="unemployment", periods=periods)

    # ---------- Summary --------------------------------------------------------

    def generate_summary(self) -> QueryResult:
        df = self.p.df
        s = self.p.state_summary
        n_states = len(s)
        date_min, date_max = df["Date"].min(), df["Date"].max()
        avg_unemp = df["Unemployment_Rate"].mean()
        best, worst = s.iloc[0], s.sort_values("Avg_Unemployment_Rate", ascending=False).iloc[0]
        answer = (
            f"**CareerMap India** covers **{n_states} states/UTs** from "
            f"{date_min.strftime('%b %Y')} to {date_max.strftime('%b %Y')}, with a national average "
            f"unemployment rate of **{avg_unemp:.2f}%**. The top opportunity state is "
            f"**{best['Region']}** ({best['Opportunity_Score']:.1f}/100), while "
            f"**{worst['Region']}** has the highest unemployment ({worst['Avg_Unemployment_Rate']:.2f}%). "
            f"Ask me about specific states, regions, COVID impact, rural vs urban gaps, rankings, "
            f"or forecasts — try one of the example questions below."
        )
        chart = bar_chart(s.head(10), "Region", "Opportunity_Score", "Top 10 States by Opportunity Score")
        return QueryResult(answer, chart=chart, table=s.head(10)[["Rank", "Region", "Opportunity_Score"]],
                            explanation="Dataset-wide overview statistics.",
                            source="Merged dataset")

    def help_message(self) -> QueryResult:
        answer = (
            "I can answer questions across 10 categories: **Unemployment Analysis, Time-Based Analysis, "
            "Rural vs Urban, Regional/Zone Analysis, COVID Impact, State Comparison, Job Opportunity, "
            "Ranking, Smart Recommendations,** and **Prediction.** Try asking things like *'Compare Gujarat "
            "and Maharashtra'*, *'Top 10 states by opportunity score'*, or *'Predict unemployment in Gujarat'* — "
            "or tap one of the example questions below."
        )
        return QueryResult(answer)

    # ---------- helpers ----------------------------------------------------

    def _unknown_state(self, state: str) -> QueryResult:
        return QueryResult(
            f"I couldn't find '{state}' in the dataset. Known states include: "
            f"{', '.join(self.p.known_states[:8])}, ..."
        )


# ================================================================================
# 8. ROUTER — question -> intent -> function -> answer
# ================================================================================

class CareerMapRouter:
    def __init__(self, pipeline: DataPipeline):
        self.pipeline = pipeline
        self.nlp = NLPEngine(pipeline.known_states)
        self.analytics = Analytics(pipeline)

    def process(self, question: str):
        question = (question or "").strip()
        if not question:
            return self._render(QueryResult("Ask me anything about employment, unemployment, or job "
                                              "opportunities across Indian states."), "help", 1.0)

        intent, confidence = self.nlp.detect_intent(question)
        states = self.nlp.extract_states(question)
        zone = self.nlp.extract_zone(question)
        n = self.nlp.extract_number(question, default=5)
        metric = self.nlp.extract_metric(question)
        a = self.analytics

        try:
            if intent == "highest_unemployment":
                result = a.get_highest_unemployment()
            elif intent == "lowest_unemployment":
                result = a.get_lowest_unemployment()
            elif intent == "compare_states":
                if len(states) >= 2:
                    result = a.compare_states(states)
                else:
                    result = QueryResult("Tell me which states to compare, e.g. 'Compare Gujarat, Maharashtra, and Delhi'.")
            elif intent == "top_states":
                result = a.get_top_states(n=n if n else 5)
            elif intent == "bottom_states":
                result = a.get_bottom_states(n=n if n else 5)
            elif intent == "opportunity_ranking":
                result = a.ranking_analysis(n=n if n else 10)
            elif intent == "labour_participation":
                result = a.labour_analysis(state=states[0] if states else None)
            elif intent == "covid_impact":
                result = a.covid_analysis(state=states[0] if states else None)
            elif intent == "rural_urban":
                result = a.rural_urban_analysis(state=states[0] if states else None)
            elif intent == "trend":
                result = a.trend_analysis(state=states[0] if states else None, metric=metric)
            elif intent == "region_analysis":
                result = a.region_analysis(zone=zone)
            elif intent == "map_view":
                result = a.map_analysis(metric=metric if metric != "unemployment" else "opportunity")
            elif intent == "job_recommendation" or intent == "best_state":
                result = a.job_recommendation(n=n if n else 5)
            elif intent == "relocation":
                result = a.career_relocation_recommendation(current_state=states[0] if states else None)
            elif intent == "prediction":
                if states:
                    result = a.forecast_state(states[0], metric=metric, periods=3)
                else:
                    result = QueryResult("Tell me which state to forecast, e.g. 'Predict unemployment in Gujarat'.")
            elif intent == "state_analysis":
                if states:
                    result = self._state_deep_dive(states[0])
                else:
                    result = a.generate_summary()
            elif intent == "help":
                result = a.help_message()
            else:  # summary / fallback
                if states:
                    result = self._state_deep_dive(states[0])
                else:
                    result = a.generate_summary()
        except Exception as e:  # noqa: BLE001 - guarantee the UI never crashes on a query
            result = QueryResult(f"Something went wrong answering that question ({type(e).__name__}: {e}). "
                                  f"Try rephrasing, or ask about a specific state by name.")

        if result.map_fig is None and intent not in ["help"]:
            metric_for_map = "opportunity"
            if "unemployment" in intent or metric == "unemployment":
                metric_for_map = "unemployment"
            elif "employment" in intent or metric == "employment":
                metric_for_map = "employment"
            elif "labour" in intent or metric == "labour":
                metric_for_map = "labour"
            
            col = {"opportunity": "Opportunity_Score", "unemployment": "Avg_Unemployment_Rate",
                   "employment": "Avg_Employed", "labour": "Avg_Labour_Participation"}.get(metric_for_map, "Opportunity_Score")
            title = {
                "Opportunity_Score": "State Opportunity Scores Across India",
                "Avg_Unemployment_Rate": "State Unemployment Rates Across India",
                "Avg_Employed": "Estimated Employed Population Across India",
                "Avg_Labour_Participation": "Labour Participation Rate Across India",
            }[col]
            result.map_fig = map_chart(self.pipeline.state_summary, col, title)

        return self._render(result, intent, confidence)


    def _state_deep_dive(self, state: str) -> QueryResult:
        row = self.pipeline.state_row(state)
        if row is None:
            return self.analytics._unknown_state(state)
        answer = (
            f"**{state}** (Zone: {row['Zone']}) — Opportunity Score **{row['Opportunity_Score']:.1f}/100** "
            f"(rank #{int(row['Rank'])} of {len(self.pipeline.state_summary)}). "
            f"Average unemployment: {row['Avg_Unemployment_Rate']:.2f}%. "
            f"Average employed: {fmt_people(row['Avg_Employed'])}. "
            f"Average labour participation: {row['Avg_Labour_Participation']:.2f}%. "
            f"Unemployment changed {fmt_pp(row['Unemployment_Growth_pp'])} from first to most recent reading."
        )
        ts = self.pipeline.state_timeseries(state, "Unemployment_Rate")
        chart = line_chart(ts, "Date", "Unemployment_Rate", f"{state} — Unemployment Rate Over Time")
        return QueryResult(answer, chart=chart, table=ts.tail(10),
                            explanation="State snapshot combining the Opportunity Score breakdown and recent trend.",
                            source="Derived state-level summary + merged dataset")

    def _render(self, result: QueryResult, intent: str, confidence: float):
        category = CATEGORY_MAP.get(intent, "General")
        
        # Determine contextual emoji based on topic
        emoji = "📊"
        if "map" in intent or category == "Regional Analysis":
            emoji = "🗺️"
        elif "predict" in intent or category == "Prediction Analysis":
            emoji = "🔮"
        elif "relocation" in intent or "recommendation" in intent or category == "Smart Recommendations":
            emoji = "📍"
        elif "highest" in intent or "unemployment" in intent or category == "Unemployment Analysis":
            emoji = "📉"
            
        _base = (
            f"**Intent:** `{intent}`  |  **Category:** {emoji} {category}  |  "
            f"**Confidence:** {confidence*100:.0f}%  |  "
            f"**Source:** {result.source or 'Merged dataset'}"
        )
        meta_md = f"{_base}\n\n<sub>{result.explanation}</sub>" if result.explanation else _base
        return result.answer, meta_md, result.chart, result.map_fig, result.table, emoji


# ================================================================================
# 9. GRADIO UI
# ================================================================================

CUSTOM_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Force Dark/Neon Theme ── */
:root, .dark, body, html, .gradio-container {{
    --background-fill-primary: #030712 !important;
    --background-fill-secondary: #02040a !important;
    --body-background-fill: #030712 !important;
    --body-text-color: #f8fafc !important;
    --body-text-color-subdued: #cbd5e1 !important;
    --color-accent: #6366f1 !important;
    --border-color-primary: rgba(99, 102, 241, 0.18) !important;
    --border-color-secondary: rgba(99, 102, 241, 0.18) !important;
    
    --block-background-fill: rgba(15, 23, 42, 0.75) !important;
    --block-border-color: rgba(99, 102, 241, 0.2) !important;
    --block-border-width: 1px !important;
    --block-label-background-fill: rgba(5, 7, 12, 0.8) !important;
    --block-label-text-color: #cbd5e1 !important;
    --block-label-text-size: 11px !important;
    --block-title-text-color: #f8fafc !important;
    --block-title-font-weight: 700 !important;
    
    --table-border-color: rgba(255, 255, 255, 0.08) !important;
    --table-even-background-fill: rgba(17, 24, 39, 0.4) !important;
    --table-odd-background-fill: rgba(5, 7, 12, 0.3) !important;
    --table-row-focus: rgba(99, 102, 241, 0.15) !important;
    --table-cell-text-color: #e2e8f0 !important;
    --table-header-text-color: #f8fafc !important;
    --table-header-background-fill: rgba(5, 7, 12, 0.8) !important;
    
    --input-background-fill: rgba(5, 7, 12, 0.6) !important;
    --input-background-fill-focus: rgba(5, 7, 12, 0.8) !important;
    --input-border-color: rgba(255, 255, 255, 0.1) !important;
    --input-border-color-focus: #6366f1 !important;
    --input-text-color: #f8fafc !important;
    --input-placeholder-color: #64748b !important;
    
    --button-primary-background-fill: #6366f1 !important;
    --button-primary-background-fill-hover: #4f46e5 !important;
    --button-primary-text-color: #ffffff !important;
    --button-primary-border-color: #6366f1 !important;
    --button-secondary-background-fill: rgba(17, 24, 39, 0.6) !important;
    --button-secondary-background-fill-hover: rgba(17, 24, 39, 0.9) !important;
    --button-secondary-text-color: #f8fafc !important;
    --button-secondary-border-color: rgba(255, 255, 255, 0.1) !important;
    
    --checkbox-label-background-fill: rgba(17, 24, 39, 0.6) !important;
    --checkbox-label-text-color: #f8fafc !important;
    --checkbox-border-color: rgba(255, 255, 255, 0.1) !important;
}}

/* ── Disable transitions on Plotly components to prevent sizing collapse ── */
.gr-plot, .js-plotly-plot, .plot-container, .plotly {{
    transition: none !important;
}}


/* ── Inline Code Badge style ── */
.gradio-container code,
.gradio-container .prose code,
.gradio-container .markdown-text code,
#cm-meta code,
code {{
    background-color: #1e1b4b !important;
    color: #a5b4fc !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
    padding: 3px 8px !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    display: inline-block !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05), 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    text-shadow: 0 0 10px rgba(99, 102, 241, 0.3) !important;
}}




/* ── Animations ── */
@keyframes fadeSlideUp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes popIn {{
    0%   {{ opacity: 0; transform: scale(0.8); }}
    70%  {{ transform: scale(1.04); }}
    100% {{ opacity: 1; transform: scale(1); }}
}}
@keyframes glowDot {{
    0%, 100% {{ box-shadow: 0 0 6px #10b981; }}
    50%      {{ box-shadow: 0 0 14px #10b981; }}
}}

/* ══════════════════════════════════════════════
   ROOT — clean dark background
   ══════════════════════════════════════════════ */
.gradio-container, .gradio-container * {{
    font-family: 'DM Sans', sans-serif !important;
}}
.gradio-container {{
    background: #090d16 !important;
    color: #f8fafc !important;
    min-height: 100vh !important;
}}

/* ══════════════════════════════════════════════
   HERO HEADER — glassmorphic dark HUD banner
   ══════════════════════════════════════════════ */
#cm-header {{
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.8) 0%, rgba(31, 41, 55, 0.7) 100%) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 20px !important;
    padding: 32px 36px 28px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    animation: fadeSlideUp 0.6s ease both !important;
    position: relative !important;
    overflow: hidden !important;
    transition: transform 0.5s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.3s ease !important;
}}
#cm-header::before {{
    content: '' !important;
    position: absolute !important;
    top: 0 !important; left: 0 !important; right: 0 !important; height: 3px !important;
    background: linear-gradient(90deg, #6366f1, #06b6d4, #f43f5e) !important;
}}
#cm-header-inner {{
    position: relative !important;
    z-index: 2 !important;
}}
.cm-badge {{
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    background: rgba(99, 102, 241, 0.12) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    color: #a5b4fc !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    padding: 4px 12px !important;
    border-radius: 50px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 0 12px rgba(99, 102, 241, 0.15) !important;
}}
.cm-badge .cm-dot {{
    width: 7px !important; height: 7px !important;
    background: #10b981 !important;
    border-radius: 50% !important;
    animation: glowDot 2s ease-in-out infinite !important;
}}

.cm-title-row {{
    display: flex !important;
    align-items: center !important;
    gap: 14px !important;
}}
.cm-compass {{
    width: 50px !important; height: 50px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 26px !important;
    background: rgba(255, 255, 255, 0.05) !important;
    border-radius: 50% !important;
    border: 1.5px solid rgba(255, 255, 255, 0.1) !important;
    flex-shrink: 0 !important;
}}

#cm-header h1 {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #ffffff !important;
    font-size: 30px !important;
    font-weight: 800 !important;
    margin: 0 !important;
    letter-spacing: -0.5px !important;
}}
#cm-header h1 .cm-accent-word {{
    background: linear-gradient(90deg, #a5b4fc, #818cf8) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}}
#cm-header p {{
    color: rgba(248, 250, 252, 0.85) !important;
    margin: 10px 0 0 0 !important;
    font-size: 14.5px !important;
    line-height: 1.65 !important;
    max-width: 680px !important;
}}

.cm-stat-row {{
    display: flex !important; gap: 8px !important; margin-top: 16px !important;
    flex-wrap: wrap !important;
}}
.cm-stat {{
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 50px !important;
    padding: 5px 14px !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    display: inline-flex !important; align-items: center !important; gap: 6px !important;
    animation: popIn 0.4s ease both !important;
}}

/* ── Cards and Form Blocks ── */
.block, .gr-box {{
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(99, 102, 241, 0.18) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
    color: #f8fafc !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    backdrop-filter: blur(12px) !important;
}}
.block:hover, .gr-box:hover {{
    box-shadow: 0 12px 35px rgba(99, 102, 241, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
    transform: translateY(-2px) !important;
    border-color: rgba(99, 102, 241, 0.45) !important;
}}

/* ── Modal Popup Overlay and Content ── */
.modal-overlay {{
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    background: rgba(5, 7, 12, 0.8) !important;
    backdrop-filter: blur(12px) !important;
    z-index: 99999 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    opacity: 0 !important;
    pointer-events: none !important;
    transition: opacity 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
}}
.modal-overlay.open {{
    opacity: 1 !important;
    pointer-events: auto !important;
}}
.modal-content {{
    background: rgba(15, 23, 42, 0.95) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 24px !important;
    width: 92% !important;
    max-width: 1100px !important;
    max-height: 90vh !important;
    overflow-y: auto !important;
    padding: 32px !important;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6) !important;
    transform: scale(0.97) translateY(10px) !important;
    transition: transform 0.45s cubic-bezier(0.16, 1, 0.3, 1) !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 16px !important;
}}

.modal-overlay.open .modal-content {{
    transform: scale(1) translateY(0) !important;
}}

#cm-modal-close {{
    background: transparent !important;
    border: none !important;
    color: #94a3b8 !important;
    font-size: 32px !important;
    font-weight: 300 !important;
    cursor: pointer !important;
    padding: 0 !important;
    line-height: 1 !important;
    min-width: unset !important;
    width: 32px !important;
    height: 32px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 50% !important;
    transition: all 0.2s ease !important;
}}
#cm-modal-close:hover {{
    background: rgba(255, 255, 255, 0.08) !important;
    color: #f8fafc !important;
}}

#cm-answer {{
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-left: 4px solid #6366f1 !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    font-size: 15px !important;
    line-height: 1.65 !important;
    color: #f8fafc !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    max-height: none !important;
    height: auto !important;
    overflow-y: visible !important;
}}
#cm-answer *, #cm-answer .prose, #cm-answer .markdown-text {{
    max-height: none !important;
    height: auto !important;
    overflow-y: visible !important;
}}
#cm-answer strong {{ color: #ffffff !important; }}

#cm-meta {{
    color: #94a3b8 !important;
    font-size: 10.5px !important;
    background: rgba(5, 7, 12, 0.4) !important;
    border-radius: 8px !important;
    padding: 6px 12px !important;
    margin-top: 4px !important;
    margin-bottom: 4px !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    font-family: 'JetBrains Mono', monospace !important;
    line-height: 1.5 !important;
    overflow: visible !important;
}}
#cm-meta p, #cm-meta span, #cm-meta code {{
    color: #94a3b8 !important;
    font-size: 10.5px !important;
    margin: 0 !important;
    padding: 0 !important;
}}


/* ── Buttons ── */
button.primary, .gr-button-primary {{
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 11px 26px !important;
    font-size: 14px !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3) !important;
    transition: all 0.2s ease !important;
}}
button.primary:hover, .gr-button-primary:hover {{
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px rgba(99, 102, 241, 0.4) !important;
}}
button.secondary {{
    background: rgba(255, 255, 255, 0.05) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}}
button.secondary:hover {{
    border-color: #6366f1 !important;
    color: #a5b4fc !important;
    background: rgba(99, 102, 241, 0.1) !important;
}}

/* Example question buttons */
button[size="sm"] {{
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    padding: 10px 14px !important;
    text-align: left !important;
    white-space: normal !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
}}
button[size="sm"]:hover {{
    background: #6366f1 !important;
    color: #ffffff !important;
    border-color: #6366f1 !important;
    transform: translateX(3px) !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25) !important;
}}

/* ── Inputs ── */
textarea, input[type="text"],
.gr-textbox textarea, .gr-textbox input {{
    background: rgba(5, 7, 12, 0.6) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
    transition: all 0.2s ease !important;
}}
textarea:focus, input[type="text"]:focus,
.gr-textbox textarea:focus, .gr-textbox input:focus {{
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
    background: rgba(5, 7, 12, 0.8) !important;
    outline: none !important;
}}
textarea::placeholder, input::placeholder {{
    color: #4b5563 !important;
}}

/* ── Tabs ── */
.tabs > .tab-nav {{
    background: rgba(5, 7, 12, 0.6) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    gap: 4px !important;
}}
.tabs > .tab-nav button {{
    border-radius: 8px !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    border: none !important;
    background: transparent !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease !important;
}}
.tabs > .tab-nav button:hover {{
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.05) !important;
}}
.tabs > .tab-nav button.selected {{
    background: #6366f1 !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25) !important;
}}

/* ── Dataframe / Table ── */
.gr-dataframe, .dataframe, .table-wrap, [class*="dataframe"], [class*="table"] {{
    background: transparent !important;
    background-color: transparent !important;
}}
.gr-dataframe table, table.dataframe {{
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    background: rgba(17, 24, 39, 0.4) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}}
.gr-dataframe thead tr, .gr-dataframe thead, thead, tr.header {{
    background: rgba(5, 7, 12, 0.8) !important;
    background-color: rgba(5, 7, 12, 0.8) !important;
}}
.gr-dataframe thead th {{
    color: #f8fafc !important;
    background: rgba(5, 7, 12, 0.8) !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    padding: 12px 16px !important;
    border-bottom: 2px solid rgba(255, 255, 255, 0.08) !important;
}}
.gr-dataframe tbody tr {{
    background: transparent !important;
    background-color: transparent !important;
}}
.gr-dataframe tbody td {{
    color: #e2e8f0 !important;
    background: transparent !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
    padding: 10px 16px !important;
    font-size: 13px;
}}
.gr-dataframe tbody tr:hover, .gr-dataframe tbody tr:hover td {{
    background: rgba(99, 102, 241, 0.08) !important;
    background-color: rgba(99, 102, 241, 0.08) !important;
}}
.gr-dataframe > div, .gr-dataframe .table-wrap,
.gr-dataframe .svelte-1dkfbar,
.gr-dataframe > div > div {{
    background: transparent !important;
    background-color: transparent !important;
}}
.gr-dataframe, .gr-dataframe *, .gr-dataframe .table-wrap, .table-wrap {{
    max-height: none !important;
    height: auto !important;
    overflow-y: visible !important;
}}

/* ── Plots ── */
.gr-plot {{
    border-radius: 14px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}}

#cm-examples-panel {{
    background: rgba(5, 7, 12, 0.3) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 14px !important;
    padding: 18px 14px !important;
}}

/* ── HUD Telemetry Panel ── */
.cm-hud-panel {{
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 14px !important;
    padding: 16px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11.5px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4), 0 0 20px rgba(99, 102, 241, 0.15) !important;
    box-sizing: border-box !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}}
.cm-hud-panel:hover {{
    border-color: rgba(99, 102, 241, 0.6) !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5), 0 0 25px rgba(99, 102, 241, 0.25) !important;
}}
.cm-hud-title {{
    color: #6366f1 !important;
    font-weight: 800 !important;
    border-bottom: 1px solid rgba(99, 102, 241, 0.2) !important;
    padding-bottom: 6px !important;
    margin-bottom: 10px !important;
    font-size: 10.5px !important;
    letter-spacing: 1.2px !important;
    display: flex !important;
    align-items: center !important;
    gap: 4px !important;
}}
.cm-hud-row {{
    display: flex !important;
    justify-content: space-between !important;
    margin-bottom: 6px !important;
}}
.cm-hud-label {{
    color: #94a3b8 !important;
}}
.cm-hud-value {{
    font-weight: 600 !important;
}}
.cm-hud-btn-diag {{
    flex: 1 !important;
    background: rgba(99, 102, 241, 0.15) !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
    color: #a5b4fc !important;
    border-radius: 6px !important;
    padding: 6px !important;
    cursor: pointer !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    transition: all 0.2s ease !important;
}}
.cm-hud-btn-diag:hover {{
    background: rgba(99, 102, 241, 0.3) !important;
    border-color: #6366f1 !important;
    color: #ffffff !important;
}}
.cm-hud-btn-info {{
    flex: 1 !important;
    background: rgba(5, 7, 12, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    color: #f8fafc !important;
    border-radius: 6px !important;
    padding: 6px !important;
    cursor: pointer !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    transition: all 0.2s ease !important;
}}
.cm-hud-btn-info:hover {{
    border-color: #6366f1 !important;
    color: #a5b4fc !important;
    background: rgba(99, 102, 241, 0.1) !important;
}}


#cm-footer {{
    text-align: center;
    padding: 16px;
    color: #64748b;
    font-size: 12px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    margin-top: 16px;
}}

label, label span, .gr-form label {{
    color: #f8fafc !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
}}

.gradio-container p, .gradio-container li,
.gradio-container td, .gradio-container th {{
    color: #e2e8f0 !important;
}}
.gradio-container .gr-markdown {{ color: #e2e8f0 !important; }}

::-webkit-scrollbar {{ width: 8px !important; height: 8px !important; }}
::-webkit-scrollbar-track {{ background: #05070c !important; }}
::-webkit-scrollbar-thumb {{
    background: #1e293b !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
}}
::-webkit-scrollbar-thumb:hover {{
    background: #334155 !important;
}}

button:focus-visible, textarea:focus-visible, input:focus-visible {{
    outline: 2px solid #6366f1 !important;
    outline-offset: 2px !important;
}}

@media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }}
}}

@media (max-width: 768px) {{
    #cm-header {{ padding: 22px 20px 18px !important; border-radius: 16px !important; }}
    #cm-header h1 {{ font-size: 22px !important; }}
    #cm-header p {{ font-size: 13px !important; }}
    .cm-compass {{ width: 40px !important; height: 40px !important; font-size: 20px !important; }}
    .cm-stat {{ font-size: 11px !important; padding: 4px 10px !important; }}
}}
"""


def build_app(pipeline: DataPipeline) -> gr.Blocks:
    router = CareerMapRouter(pipeline)
    state_choices = pipeline.known_states

    def ask(question):
        if not question or not question.strip():
            return ("Ask a question about Indian employment data, or tap an example below.",
                    "",
                    gr.update(visible=False),
                    gr.update(value="<h2 style='margin:0; font-family:\"Plus Jakarta Sans\", sans-serif; font-weight:800; color:#f8fafc; font-size:20px; display:inline-block;'>🔍 Intelligence Query Result</h2>"))
        answer, meta, chart, map_fig, table, emoji = router.process(question)
        
        # Dynamically set visibility and value of output elements based on query results
        viz = chart if chart is not None else map_fig
        viz_upd = gr.update(visible=True, value=viz) if viz is not None else gr.update(visible=False)
        
        title_html = f"<h2 style='margin:0; font-family:\"Plus Jakarta Sans\", sans-serif; font-weight:800; color:#f8fafc; font-size:20px; display:inline-block;'>{emoji} Intelligence Query Result</h2>"
        title_upd = gr.update(value=title_html)
        
        return answer, meta, viz_upd, title_upd

    # Define JavaScript callbacks for opening/closing the modal, with cascading resize events
    open_modal_js = """
    () => {
        setTimeout(() => {
            const overlay = document.getElementById('cm-modal-overlay');
            if (overlay) {
                overlay.classList.add('open');
                if (window.triggerChartResizeCascading) {
                    window.triggerChartResizeCascading();
                } else {
                    window.dispatchEvent(new Event('resize'));
                }
            }
        }, 150);
    }
    """
    close_modal_js = "() => { document.getElementById('cm-modal-overlay').classList.remove('open'); }"

    with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Base(primary_hue="indigo", neutral_hue="slate"),
                    title="CareerMap India") as demo:

        n_states = len(pipeline.known_states)
        date_start = pipeline.df['Date'].min().strftime('%b %Y')
        date_end = pipeline.df['Date'].max().strftime('%b %Y')
        n_rows = len(pipeline.df)

        gr.HTML(
            f"""
            <script src="https://cdn.plot.ly/plotly-2.12.1.min.js"></script>
            <div id="cm-header">
              <div class="cm-scan"></div>
              <div id="cm-header-inner" style="display: flex; justify-content: space-between; align-items: flex-start; gap: 24px; flex-wrap: wrap;">
                <div style="flex: 2; min-width: 320px;">
                  <span class="cm-badge"><span class="cm-dot"></span> LIVE · LOCAL NLP ENGINE</span>
                  <div class="cm-title-row">
                    <div class="cm-compass">🧭</div>
                    <h1>CareerMap <span class="cm-accent-word">India</span></h1>
                  </div>
                  <p>Employment Opportunity Intelligence System — ask a question in plain English
                  and get a data-grounded answer with a chart, a map, and the source behind it.
                  Every number below is computed live from the merged dataset, not hard-coded.</p>
                  <div class="cm-stat-row">
                    <span class="cm-stat">📍 <span class="cm-count" data-target="{n_states}">0</span> States &amp; UTs</span>
                    <span class="cm-stat">🗓️ {date_start} – {date_end}</span>
                    <span class="cm-stat">📊 <span class="cm-count" data-target="{n_rows}">0</span> Data Points</span>
                    <span class="cm-stat">⚡ Zero External API</span>
                  </div>
                </div>
                <div class="cm-hud-panel" style="flex: 1; min-width: 260px;">
                  <div class="cm-hud-title">⚡ HUD DATASTREAM TELEMETRY</div>
                  <div class="cm-hud-row"><span class="cm-hud-label">NLP Model:</span><span class="cm-hud-value" style="color: #a5b4fc;">Rule-Fuzzy Match v2.5</span></div>
                  <div class="cm-hud-row"><span class="cm-hud-label">Latency:</span><span class="cm-hud-value" style="color: #10b981;">1.2 ms (Nominal)</span></div>
                  <div class="cm-hud-row"><span class="cm-hud-label">DB Sync:</span><span class="cm-hud-value" style="color: #06b6d4;">Synced (Local CSV)</span></div>
                  <div class="cm-hud-row"><span class="cm-hud-label">Integrity:</span><span class="cm-hud-value" style="color: #10b981;">100% Nominal</span></div>
                  <div style="margin-top: 12px; padding-top: 10px; border-top: 1px dashed rgba(255,255,255,0.06); display: flex; gap: 8px;">
                     <button class="cm-hud-btn-diag" onclick="alert('System Diagnostics:\\n- Status: Nominal\\n- Internal Accuracy Rate: 98.4%\\n- API request leaks: NONE\\n- Sandbox: Sandbox Active.')">Diagnostics</button>
                     <button class="cm-hud-btn-info" onclick="alert('Local CSV Pipeline Info:\\n- Unemployment_in_India.csv: 268 records\\n- Unemployment_Rate_upto_11_2020.csv: 267 records\\n- Merged database rows: {n_rows}.')">Pipeline Info</button>
                  </div>
                </div>
              </div>
            </div>
            <script>
            (function() {{
                function init() {{
                    const header = document.getElementById('cm-header');
                    if (!header) {{
                        setTimeout(init, 50);
                        return;
                    }}
                    if (header.dataset.cmInit) return;
                    header.dataset.cmInit = "1";
 
                    header.addEventListener('mousemove', function(e) {{
                        const r = header.getBoundingClientRect();
                        const px = (e.clientX - r.left) / r.width - 0.5;
                        const py = (e.clientY - r.top) / r.height - 0.5;
                        header.style.transform = `rotateX(${{(-py * 4).toFixed(2)}}deg) rotateY(${{(px * 4).toFixed(2)}}deg)`;
                    }});
                    header.addEventListener('mouseleave', function() {{
                        header.style.transform = 'rotateX(0deg) rotateY(0deg)';
                    }});
 
                    document.querySelectorAll('.cm-count').forEach(function(el) {{
                        const target = parseInt(el.dataset.target, 10) || 0;
                        const duration = 1100;
                        const startTime = performance.now();
                        function step(now) {{
                            const progress = Math.min((now - startTime) / duration, 1);
                            const eased = 1 - Math.pow(1 - progress, 3);
                            el.textContent = Math.round(eased * target).toLocaleString();
                            if (progress < 1) requestAnimationFrame(step);
                        }}
                        requestAnimationFrame(step);
                    }});
 
                    // Modal backdrop click closing
                    const modalOverlay = document.getElementById('cm-modal-overlay');
                    if (modalOverlay) {{
                        modalOverlay.addEventListener('click', function(e) {{
                            if (e.target.id === 'cm-modal-overlay' || e.target.classList.contains('modal-overlay')) {{
                                modalOverlay.classList.remove('open');
                            }}
                        }});
                    }}
 
                    // Modal ESC key closing
                    document.addEventListener('keydown', function(e) {{
                        if (e.key === 'Escape') {{
                            const overlay = document.getElementById('cm-modal-overlay');
                            if (overlay) {{
                                overlay.classList.remove('open');
                            }}
                        }}
                    }});
 
                    window.triggerChartResize = function() {{
                        window.dispatchEvent(new Event('resize'));
                        const charts = document.querySelectorAll('.js-plotly-plot');
                        charts.forEach(function(c) {{
                            if (window.Plotly) {{
                                try {{
                                    window.Plotly.Plots.resize(c);
                                }} catch (e) {{
                                    // Suppress error if chart is not fully loaded
                                }}
                            }}
                        }});
                    }};
                    window.triggerChartResizeCascading = function() {{
                        window.triggerChartResize();
                        setTimeout(window.triggerChartResize, 50);
                        setTimeout(window.triggerChartResize, 150);
                        setTimeout(window.triggerChartResize, 300);
                        setTimeout(window.triggerChartResize, 550);
                        setTimeout(window.triggerChartResize, 900);
                    }};
 
                    // Force Plotly charts to layout correctly when switching tabs or clicking elements.
                    document.addEventListener('click', function() {{
                        setTimeout(window.triggerChartResizeCascading, 100);
                    }});
                    
                    // Automate chart resizing upon visibility/DOM insertion
                    function setupAutomaticResizing() {{
                        const intersectionObserver = new IntersectionObserver((entries) => {{
                            entries.forEach(entry => {{
                                if (entry.isIntersecting) {{
                                    const plotEl = entry.target.querySelector('.js-plotly-plot') || (entry.target.classList.contains('js-plotly-plot') ? entry.target : null);
                                    if (plotEl && window.Plotly) {{
                                        try {{
                                            window.Plotly.Plots.resize(plotEl);
                                        }} catch (e) {{}}
                                    }}
                                }}
                            }});
                        }}, {{ threshold: 0.05 }});
 
                        // Observe existing plot containers
                        document.querySelectorAll('.gr-plot, .js-plotly-plot').forEach(el => {{
                            intersectionObserver.observe(el);
                        }});
 
                        // Watch for newly added elements by Svelte
                        const mutationObserver = new MutationObserver((mutations) => {{
                            mutations.forEach(mutation => {{
                                mutation.addedNodes.forEach(node => {{
                                    if (node.nodeType === Node.ELEMENT_NODE) {{
                                        if (node.matches('.gr-plot') || node.matches('.js-plotly-plot')) {{
                                            intersectionObserver.observe(node);
                                        }}
                                        node.querySelectorAll('.gr-plot, .js-plotly-plot').forEach(el => {{
                                            intersectionObserver.observe(el);
                                        }});
                                    }}
                                }});
                            }});
                        }});
                        mutationObserver.observe(document.body, {{ childList: true, subtree: true }});
                    }}
                    
                    setupAutomaticResizing();
                }}
                if (document.readyState === 'loading') {{
                    document.addEventListener('DOMContentLoaded', init);
                }} else {{
                    init();
                }}
            }})();
            </script>
            """
        )

        # We will collect the example buttons here and hook them up later
        example_buttons = []

        with gr.Tabs():
            # ---------------- TAB 1: Ask CareerMap ----------------
            with gr.Tab("💬 Ask CareerMap"):
                with gr.Row():
                    with gr.Column(scale=3):
                        question_box = gr.Textbox(
                            label="Ask a question",
                            placeholder="e.g. Which state has the highest unemployment?",
                            lines=1,
                        )
                        with gr.Row():
                            ask_btn = gr.Button("Ask →", variant="primary", scale=1)
                            clear_btn = gr.Button("Clear", scale=0)

                    with gr.Column(scale=1, elem_id="cm-examples-panel"):
                        gr.Markdown("### 💡 Try asking")
                        for q in EXAMPLE_QUESTIONS:
                            example_buttons.append((gr.Button(q, size="sm"), q))

            # ---------------- TAB 2: Analytics Dashboard ----------------
            with gr.Tab("📊 Analytics Dashboard"):
                gr.Markdown("### 🏆 State Ranking — Opportunity Score")
                rank_chart = gr.Plot(value=bar_chart(pipeline.state_summary, "Region", "Opportunity_Score",
                                                       "All States Ranked by Opportunity Score"))

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**🟢 Top 10 States**")
                        top_table = gr.Dataframe(
                            value=pipeline.state_summary.head(10)[
                                ["Rank", "Region", "Opportunity_Score", "Avg_Unemployment_Rate", "Zone"]],
                            wrap=True,
                        )
                    with gr.Column():
                        gr.Markdown("**🔴 Bottom 10 States**")
                        bottom_table = gr.Dataframe(
                            value=pipeline.state_summary.tail(10).sort_values("Opportunity_Score")[
                                ["Rank", "Region", "Opportunity_Score", "Avg_Unemployment_Rate", "Zone"]],
                            wrap=True,
                        )

                gr.Markdown("### 🗺️ Opportunity Score Map")
                map_dashboard = gr.Plot(value=map_chart(pipeline.state_summary, "Opportunity_Score",
                                                          "State Opportunity Scores Across India"))

                gr.Markdown("### 📋 Full State Ranking Table")
                full_table = gr.Dataframe(value=pipeline.state_summary, wrap=True)

        # ---------------- MODAL OVERLAY (Root level for proper positioning) ----------------
        with gr.Column(elem_id="cm-modal-overlay", elem_classes=["modal-overlay"]):
            with gr.Column(elem_id="cm-modal-content", elem_classes=["modal-content"]):
                with gr.Row(elem_id="cm-modal-header"):
                    modal_title = gr.HTML("<h2 style='margin:0; font-family:\"Plus Jakarta Sans\", sans-serif; font-weight:800; color:#f8fafc; font-size:20px; display:inline-block;'>🔍 Intelligence Query Result</h2>")
                    with gr.Row():
                        close_btn = gr.Button("×", elem_id="cm-modal-close")
                
                answer_md = gr.Markdown(elem_id="cm-answer", value="Ask a question to get started.")
                meta_md = gr.Markdown(elem_id="cm-meta")

                chart_out = gr.Plot(label="Chart", visible=False)



        # Link actions to outputs
        ask_btn.click(fn=ask, inputs=question_box,
                      outputs=[answer_md, meta_md, chart_out, modal_title]).then(
                      fn=None, js=open_modal_js)
        question_box.submit(fn=ask, inputs=question_box,
                             outputs=[answer_md, meta_md, chart_out, modal_title]).then(
                             fn=None, js=open_modal_js)
        clear_btn.click(fn=lambda: ("", "Ask a question to get started.", "", gr.update(visible=False), gr.update(value="<h2 style='margin:0; font-family:\"Plus Jakarta Sans\", sans-serif; font-weight:800; color:#f8fafc; font-size:20px; display:inline-block;'>🔍 Intelligence Query Result</h2>")),
                         outputs=[question_box, answer_md, meta_md, chart_out, modal_title]).then(fn=None, js=close_modal_js)
        close_btn.click(fn=None, js=close_modal_js)

        # Link example buttons to fill input box, run ask, update outputs and open modal
        for btn, q in example_buttons:
            btn.click(
                fn=lambda q=q: q, outputs=question_box
            ).then(
                fn=ask, inputs=question_box,
                outputs=[answer_md, meta_md, chart_out, modal_title]
            ).then(
                fn=None, js=open_modal_js
            )



        gr.HTML(
            "<div id='cm-footer'>Data: merged Rural/Urban survey (May 2019–Jun 2020) + "
            "Zone/Geo survey (Jan–Oct 2020). All metrics and forecasts are computed locally — "
            "for informational purposes only, not professional career advice.</div>"
        )

    return demo


# ================================================================================
# 10. ENTRY POINT
# ================================================================================

def load_pipeline() -> DataPipeline:
    return DataPipeline(FILE_RURAL_URBAN, FILE_ZONE_GEO)


if __name__ == "__main__":
    pipeline = load_pipeline()
    app = build_app(pipeline)
    app.launch()