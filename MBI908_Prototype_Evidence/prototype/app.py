from pathlib import Path
import hashlib
import html
import io
import zipfile
from types import SimpleNamespace

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from pipeline import read_uploaded_files, process_raw_frames

SNAPSHOT_LABEL = "31 Dec 2024"
DEMO_FILE = "processed_demo_data.csv"
SENSITIVITY_FILE = "sensitivity_results.csv"

COLORS = {
    "blue": "#5B8CFF",
    "cyan": "#35C2C1",
    "green": "#42C98B",
    "amber": "#F5B94C",
    "red": "#FF6B7A",
    "purple": "#9B7BFF",
    "slate": "#8D99AE",
}

st.set_page_config(
    page_title="Account Health & Prioritisation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
<style>
    :root {{
        --blue: {COLORS["blue"]};
        --cyan: {COLORS["cyan"]};
        --green: {COLORS["green"]};
        --amber: {COLORS["amber"]};
        --red: {COLORS["red"]};
        --purple: {COLORS["purple"]};
        --slate: {COLORS["slate"]};
        --ink: #1F2940;
        --muted: #5C6985;
        --soft: #8A97B2;
        --line: #D9E2F3;
        --card: rgba(255,255,255,0.92);
        --card-strong: rgba(255,255,255,0.98);
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: #F6FAFF;
        background-image:
            radial-gradient(circle at 8% 8%, rgba(80, 145, 255, 0.14) 0, transparent 28%),
            radial-gradient(circle at 92% 10%, rgba(88, 211, 190, 0.12) 0, transparent 24%),
            radial-gradient(circle at 78% 88%, rgba(154, 123, 255, 0.10) 0, transparent 26%),
            linear-gradient(rgba(104, 133, 188, 0.045) 1px, transparent 1px),
            linear-gradient(90deg, rgba(104, 133, 188, 0.045) 1px, transparent 1px),
            linear-gradient(180deg, rgba(255,255,255,0.90), rgba(244,248,255,0.96));
        background-size: auto, auto, auto, 34px 34px, 34px 34px, auto;
        background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
        background: rgba(248,251,255,0.86);
        backdrop-filter: blur(6px);
    }}

    .stApp, p, label, li, div {{
        color: var(--ink);
    }}

    .block-container {{
        padding-top: 4.5rem !important;
        padding-bottom: 3rem;
        max-width: 1520px;
    }}

    h1, h2, h3 {{
        color: var(--ink) !important;
        letter-spacing: -0.02em;
    }}

    [data-testid="stSidebar"] {{
        min-width: 300px;
        max-width: 300px;
        border-right: 1px solid var(--line);
        background: linear-gradient(180deg, rgba(245,248,255,0.98), rgba(238,243,252,0.98));
    }}

    /* Keep the desktop navigation sidebar visible; hide Streamlit's
       collapse and reopen controls without changing page navigation. */
    [data-testid="stSidebarHeader"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseControl"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {{
        display: none !important;
    }}

    [data-testid="stSidebar"] .block-container {{
        padding-top: 1.4rem !important;
    }}

    [data-testid="stSidebar"] * {{
        color: #2B3651 !important;
    }}

    .hero {{
        position: relative;
        isolation: isolate;
        overflow: hidden;
        border: 1px solid #D9E6FF;
        background:
            radial-gradient(ellipse 48% 96% at 94% 12%, rgba(116,233,224,0.64), transparent 85%),
            radial-gradient(ellipse 56% 100% at 90% 88%, rgba(158,147,249,0.64), transparent 85%),
            radial-gradient(ellipse 38% 72% at 74% 65%, rgba(94,188,253,0.45), transparent 86%),
            linear-gradient(108deg, #FFFFFF 0%, #FDFEFF 48%, #EDF5FF 100%);
        box-shadow: 0 18px 42px rgba(69,96,146,0.10);
        border-radius: 26px;
        padding: 1.8rem 1.85rem 1.6rem;
        margin-bottom: 1.55rem;
    }}
    .hero::before {{
        content: "";
        position: absolute;
        z-index: -1;
        width: 65%; height: 150%;
        right: -23%; bottom: -114%;
        background: linear-gradient(120deg, rgba(79,225,212,0.22), rgba(90,180,249,0.28) 43%, rgba(166,143,249,0.35));
        border-radius: 48% 52% 44% 56%;
        transform: rotate(-22deg);
        box-shadow: -35px -5px 0 rgba(255,255,255,0.24);
        pointer-events: none;
    }}
    .hero::after {{
        content: "";
        position: absolute;
        z-index: -1;
        width: 38%; height: 90%;
        right: -17%; top: -47%;
        border-radius: 50%;
        background: linear-gradient(145deg,rgba(255,255,255,0.54),rgba(143,207,253,0.25),rgba(171,147,252,0.33));
        pointer-events: none;
    }}

    .hero-kicker {{
        color: #1EA8B0;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        margin-bottom: 0.38rem;
    }}

    .hero-title {{
        font-size: 2.25rem;
        font-weight: 800;
        line-height: 1.12;
        margin: 0;
        color: var(--ink);
    }}

    .hero-subtitle {{
        margin-top: 0.45rem;
        color: var(--muted);
        font-size: 1rem;
        max-width: 940px;
        line-height: 1.5;
    }}

    .status-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.48rem;
        margin-top: 1rem;
    }}

    .pill {{
        border: 1px solid #D8E3F6;
        background: rgba(255,255,255,0.92);
        border-radius: 999px;
        padding: 0.33rem 0.7rem;
        font-size: 0.78rem;
        color: #34415B;
        box-shadow: 0 2px 8px rgba(67, 93, 148, 0.05);
    }}

    .pill-blue {{ border-color: rgba(91,140,255,0.45); }}
    .pill-green {{ border-color: rgba(66,201,139,0.45); }}
    .pill-purple {{ border-color: rgba(155,123,255,0.45); }}

    .sidebar-brand {{
        border: 1px solid #D7E3FF;
        background: linear-gradient(145deg, rgba(255,255,255,0.98), rgba(242,246,255,0.98));
        border-radius: 18px;
        padding: 1rem 1rem 0.95rem 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 24px rgba(67, 93, 148, 0.06);
    }}

    .sidebar-brand-title {{
        font-weight: 780;
        font-size: 1.05rem;
        color: var(--ink);
    }}

    .sidebar-brand-sub {{
        font-size: 0.80rem;
        color: var(--muted);
        margin-top: 0.18rem;
    }}

    .sidebar-label {{
        font-size: 0.76rem;
        text-transform: uppercase;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #79849B;
        margin: 0.95rem 0 0.5rem 0;
    }}

    [data-testid="stSidebar"] div.stButton > button {{
        justify-content: flex-start !important;
        text-align: left !important;
        min-height: 48px;
        border-radius: 14px !important;
        padding-left: 1rem !important;
        margin-bottom: 0.32rem;
        font-weight: 650;
        border: 1px solid #D8E0F0 !important;
        background: rgba(255,255,255,0.86) !important;
        box-shadow: 0 4px 12px rgba(67, 93, 148, 0.03);
    }}

    [data-testid="stSidebar"] div.stButton > button[kind="primary"] {{
        color: white !important;
        border-color: rgba(91,140,255,0.70) !important;
        background: linear-gradient(90deg, rgba(91,140,255,0.95), rgba(132,111,255,0.90)) !important;
        box-shadow: 0 8px 22px rgba(91,140,255,0.20);
    }}

    .sidebar-help {{
        border: 1px solid #D8E1F2;
        background: rgba(255,255,255,0.74);
        border-radius: 16px;
        padding: 0.95rem 1rem;
        margin-top: 1.1rem;
        font-size: 0.84rem;
        line-height: 1.72;
        color: #4E5D78;
    }}

    .section-note {{
        border-left: 4px solid var(--blue);
        background: rgba(255,255,255,0.80);
        padding: 0.78rem 0.95rem;
        border-radius: 0 12px 12px 0;
        margin: 0.2rem 0 1.15rem 0;
        color: #46546F;
        box-shadow: 0 6px 18px rgba(67, 93, 148, 0.04);
    }}

    .section-note b, .section-note span, .section-note div {{
        color: #46546F !important;
    }}

    .metric-card {{
        position: relative;
        border: 1px solid #DCE4F2;
        border-radius: 18px;
        padding: 1rem 1.05rem 0.9rem 1.05rem;
        min-height: 162px;
        height: 100%;
        background: var(--card-strong);
        overflow: hidden;
        box-shadow: 0 10px 22px rgba(67, 93, 148, 0.05);
    }}

    /* A shared fixed minimum keeps the six account summary cards equal height. */
    .metric-card:has(.metric-icon) {{ min-height: 202px; }}

    .metric-header {{
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 0.4rem;
    }}

    .metric-icon {{
        width: 46px;
        height: 46px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: var(--metric-soft, rgba(91,140,255,0.14));
        color: var(--metric-accent, var(--blue));
        font-size: 1.35rem;
        line-height: 1;
        flex: 0 0 auto;
    }}

    .metric-label {{
        font-size: 0.79rem;
        color: var(--muted);
        margin-bottom: 0;
        font-weight: 700;
    }}

    .metric-pill {{
        display: inline-block;
        margin-top: 0.44rem;
        padding: 0.28rem 0.62rem;
        border-radius: 999px;
        background: var(--metric-soft, rgba(91,140,255,0.14));
        color: var(--metric-accent, var(--blue));
        font-size: 0.78rem;
        font-weight: 700;
        line-height: 1.2;
    }}

    .metric-value {{
        font-size: 1.82rem;
        font-weight: 780;
        line-height: 1.12;
        color: var(--ink);
    }}

    .metric-unit {{
        color: #74819A;
        font-size: 0.47em;
        font-weight: 400;
        letter-spacing: 0;
        white-space: nowrap;
    }}

    .metric-value.compact {{
        font-size: 1.10rem;
        padding-top: 0.38rem;
        line-height: 1.3;
        overflow-wrap: anywhere;
    }}

    .metric-sub {{
        font-size: 0.78rem;
        color: #74819A;
        margin-top: 0.42rem;
        line-height: 1.45;
    }}

    .decision-card {{
        border: 1px solid #DCE4F2;
        border-radius: 18px;
        min-height: 158px;
        padding: 1rem 1.08rem;
        background: var(--card-strong);
        box-shadow: 0 10px 22px rgba(67, 93, 148, 0.05);
    }}

    .decision-card.driver {{
        background: linear-gradient(142deg, rgba(255,234,238,0.75), rgba(255,255,255,0.98) 76%);
        border-color: rgba(235,112,128,0.27);
    }}
    .decision-card.action {{
        background: linear-gradient(142deg, #FFFFFF, rgba(236,255,251,0.55));
        border-color: rgba(40,175,165,0.33);
    }}
    .decision-header {{display:flex;align-items:center;gap:.7rem;margin-bottom:.75rem}}
    .decision-icon {{width:40px;height:40px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.15rem;flex:0 0 auto}}
    .driver .decision-icon {{color:#E56A77;background:#FFE1E8}}
    .action .decision-icon {{color:#159E9A;background:#DDF8F3}}
    .decision-card {{ min-height: 190px; }}

    .decision-kicker {{
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #7B879F;
        font-weight: 780;
    }}

    .decision-title {{
        font-size: 1.08rem;
        font-weight: 760;
        margin-top: 0.35rem;
        line-height: 1.36;
        color: var(--ink);
    }}

    .decision-detail {{
        font-size: 0.80rem;
        color: #66758F;
        margin-top: 0.88rem;
        line-height: 1.62;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.4rem;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: rgba(255,255,255,0.70);
        border: 1px solid #D9E2F3;
        border-radius: 12px 12px 0 0;
        color: #52617C !important;
        padding: 0.45rem 0.85rem;
    }}

    .stTabs [aria-selected="true"] {{
        background: white !important;
        color: var(--ink) !important;
        border-bottom-color: white !important;
        font-weight: 700;
    }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid #D9E2F3;
        border-radius: 14px;
        overflow: hidden;
        background: rgba(255,255,255,0.96);
    }}

    [data-testid="stExpander"] {{
        border: 1px solid #D9E2F3 !important;
        border-radius: 14px !important;
        background: rgba(255,255,255,0.82) !important;
    }}

    /* Explicit background + foreground: avoid dark-mode Streamlit summary */
    [data-testid="stExpander"] details > summary {{
        background: #EAF3FF !important;
        color: #17345C !important;
        border-radius: 12px !important;
    }}

    [data-testid="stExpander"] details > summary p,
    [data-testid="stExpander"] details > summary span,
    [data-testid="stExpander"] details > summary svg {{
        color: #17345C !important;
        fill: #17345C !important;
        font-weight: 650;
    }}

    [data-testid="stTextInput"] input::placeholder {{
        color: #64748B !important;
        opacity: 1 !important;
    }}

    /* Consistent light treatment for Streamlit/BaseWeb widgets (including portals). */
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] section,
    [data-testid="stFileUploaderDropzone"] > div {{
        background: #F0F6FF !important;
        border-color: #B7CEFA !important;
        color: #223656 !important;
    }}
    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploaderFile"] *,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p {{
        color: #243957 !important;
        opacity: 1 !important;
    }}
    [data-testid="stFileUploaderDropzone"] button {{
        background: #FFFFFF !important;
        color: #18345D !important;
        border: 1px solid #AFC7EF !important;
    }}
    [data-testid="stFileUploaderDropzone"] button * {{ color: #18345D !important; }}
    [data-testid="stExpander"] details > summary,
    [data-testid="stExpander"] summary[data-testid="stExpanderToggleIcon"] {{
        background: #EAF3FF !important;
        color: #17345C !important;
    }}
    [data-testid="stExpander"] details > summary * {{
        color: #17345C !important;
    }}
    /* Score breakdown: compact info icons and tidy floating threshold tables. */
    [data-testid="stPopover"] button {{
        width: 1.55rem !important;
        min-width: 1.55rem !important;
        height: 1.55rem !important;
        min-height: 1.55rem !important;
        padding: 0.1rem !important;
        border: 0 !important;
        box-shadow: none !important;
        background: transparent !important;
        color: #507AC0 !important;
        border-radius: 50% !important;
    }}
    [data-testid="stPopover"] button:hover {{ background: #E9F1FF !important; }}
    .rule-tooltip-body {{ padding: 7px 9px 9px; min-width: 230px; max-width: 340px; }}
    .rule-tooltip-title {{ font-weight: 700; margin-bottom: 10px; color: #223656; }}
    .rule-tooltip-table {{ width: 100%; border-collapse: collapse; table-layout: auto; }}
    .rule-tooltip-table td {{ padding: 7px 5px; font-size: .88rem; vertical-align: top; white-space: normal; }}
    .rule-tooltip-table td:first-child {{ padding-right: 18px; text-align: left; }}
    .rule-tooltip-table td:last-child {{ text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }}
    .rule-tooltip-table .rule-active td {{ font-weight: 700; color: #17345C; background: #EFF5FF; }}
    .rule-check {{ color: #3578DB; margin-left: 3px; }}
    .rule-note {{ font-size: .77rem; color: #64748B; line-height: 1.5; padding-top: 9px; margin-top: 8px; border-top: 1px solid #E4EAF4; }}
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"],
    ul[role="listbox"],
    [role="listbox"],
    [role="option"] {{
        background: #FFFFFF !important;
        color: #223656 !important;
    }}
    [role="listbox"] *, [role="option"] *,
    div[data-baseweb="menu"] * {{
        color: #223656 !important;
        opacity: 1 !important;
    }}
    [role="option"]:hover, [aria-selected="true"][role="option"] {{
        background: #E8F1FF !important;
    }}
    .stButton > button, .stDownloadButton > button,
    [data-testid="stBaseButton-secondary"],
    [data-testid="stBaseButton-primary"],
    [data-testid="stBaseButton-secondaryFormSubmit"] {{
        background: #EAF2FF !important;
        border: 1px solid #A9C4F5 !important;
        color: #17345C !important;
        opacity: 1 !important;
    }}
    .stButton > button *, .stDownloadButton > button *,
    [data-testid="stBaseButton-secondary"] *,
    [data-testid="stBaseButton-primary"] * {{
        color: #17345C !important;
        opacity: 1 !important;
    }}
    .stButton > button:disabled, .stDownloadButton > button:disabled {{
        background: #E9EDF5 !important;
        color: #596983 !important;
        opacity: 0.72 !important;
    }}
    .stButton > button:disabled *, .stDownloadButton > button:disabled * {{
        color: #596983 !important;
    }}
    [data-testid="stSidebar"] div.stButton > button[kind="primary"],
    [data-testid="stSidebar"] div.stButton > button[kind="primary"] * {{
        color: #FFFFFF !important;
    }}
    [data-testid="stDataFrame"] {{
        color-scheme: light;
    }}

    .stAlert {{
        border-radius: 14px;
    }}

    .stButton > button, .stDownloadButton > button {{
        border-radius: 12px !important;
        font-weight: 650 !important;
    }}

    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"] > div,
    input, textarea {{
        background: rgba(255,255,255,0.96) !important;
        border: 1px solid #D7E0F0 !important;
        color: var(--ink) !important;
    }}

    div[data-baseweb="select"] * {{
        color: var(--ink) !important;
    }}

    .stNumberInput label p,
    .stSelectbox label p,
    .stRadio label p,
    .stFileUploader label p,
    .stCaption, .caption-muted {{
        color: #687792 !important;
    }}

    .caption-muted {{
        font-size: 0.82rem;
        line-height: 1.55;
    }}

    .page-chip {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 42px;
        padding: 0 0.9rem;
        border-radius: 12px;
        border: 1px solid #D9E2F3;
        background: rgba(255,255,255,0.92);
        color: #52617C;
        font-weight: 650;
    }}

    .legend-row {{
        display:flex;
        flex-wrap:wrap;
        gap:1rem;
        margin-top:0.25rem;
        margin-bottom:0.35rem;
    }}
    .legend-item {{
        display:inline-flex;
        align-items:center;
        gap:0.42rem;
        color:#54627E;
        font-size:0.86rem;
        font-weight:600;
    }}
    .legend-swatch {{
        width:12px;
        height:12px;
        border-radius:999px;
        display:inline-block;
        border:1px solid rgba(31,41,64,0.08);
    }}
    .page-heading {{display:flex;align-items:center;gap:.85rem;margin:.15rem 0 1rem}}
    .page-heading-icon {{height:48px;width:48px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#E4F1FF,#E7E5FF);color:#487BE5;border-radius:14px;font-size:1.45rem;box-shadow:0 5px 16px rgba(70,110,220,.10)}}
    .page-heading-title {{font-size:2rem;font-weight:800;letter-spacing:-.025em;color:var(--ink);line-height:1.2}}
    .page-heading-kicker {{font-size:.7rem;letter-spacing:.1em;font-weight:800;color:#179FAA;text-transform:uppercase;margin-bottom:.14rem}}
    .priority-overview {{border:1px solid #DDE6F6;border-radius:17px;padding:1rem 1.1rem;margin-bottom:.9rem;background:linear-gradient(120deg,#FFFFFF,#F1F6FF)}}
    .priority-track {{height:12px;border-radius:999px;background:#E9EFF8;overflow:hidden;margin:.35rem 0 .2rem}}
    .priority-fill {{height:100%;border-radius:999px}}
    .priority-row {{padding:.65rem .15rem}}
    .priority-row-head {{display:flex;align-items:center;justify-content:space-between;gap:1rem;font-size:.91rem;font-weight:700}}
    .priority-row-note {{font-size:.78rem;color:#64728A;line-height:1.5}}
    .priority-summary {{font-size:.85rem;color:#5F6F89;margin:.4rem 0}}
    @media(max-width:850px) {{.hero {{padding:1.3rem}} .hero-title {{font-size:1.65rem}} .page-heading-title {{font-size:1.65rem}}}}
</style>
""",
    unsafe_allow_html=True,
)



def load_demo():
    base = Path(__file__).resolve().parent
    path = base / DEMO_FILE
    if not path.exists():
        return None, None, f"Bundled demo file is missing: {DEMO_FILE}"
    try:
        raw = path.read_bytes()
        return pd.read_csv(io.BytesIO(raw)), hashlib.sha256(raw).hexdigest(), None
    except Exception as exc:
        return None, None, f"Could not read {DEMO_FILE}: {exc}"


def load_sensitivity():
    base = Path(__file__).resolve().parent
    path = base / SENSITIVITY_FILE
    if not path.exists():
        return None, None
    try:
        raw = path.read_bytes()
        return pd.read_csv(io.BytesIO(raw)), hashlib.sha256(raw).hexdigest()
    except Exception:
        return None, None


def fmt_num(value, decimals=0):
    if pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f}"


def fmt_money(value):
    if pd.isna(value):
        return "—"
    return f"{value:,.0f}"


def fmt_hours(value):
    if pd.isna(value):
        return "—"
    return f"{value:,.1f} h"


DISPLAY_EVIDENCE = {
    "Scoreable": "Sufficient evidence",
    "Insufficient Exposure": "Needs Data Check",
    "Verify Data": "Needs Data Check",
}

DISPLAY_HEALTH = {
    "Limited Evidence": "Needs Data Check",
}

DISPLAY_DRIVER = {
    "Low recent engagement": "Low recent usage",
    "Recent support pressure": "Recent support pressure",
    "No material adverse driver": "No significant concern identified",
    "Low observed satisfaction": "Low satisfaction",
    "Recent escalation history": "Recent escalation history",
    "Limited Evidence — Insufficient Exposure": "Insufficient evidence",
    "Limited Evidence — Verify Data": "Insufficient evidence",
}

DEMO_CSM_OWNERS = [
    "Maya Chen",
    "Jordan Lee",
    "Priya Shah",
    "Alex Morgan",
    "Sam Rivera",
    "Taylor Kim",
]


def add_demo_csm_owner(frame):
    """Add a deterministic UI-only fictional owner without modifying source data."""
    out = frame.copy()
    def owner_for(account_id):
        digest = hashlib.sha256(str(account_id).encode("utf-8")).hexdigest()
        return DEMO_CSM_OWNERS[int(digest[:8], 16) % len(DEMO_CSM_OWNERS)]
    out["csm_owner_demo"] = out["account_id"].map(owner_for)
    return out


def evidence_label(value):
    return DISPLAY_EVIDENCE.get(str(value), str(value))


def health_label(value):
    return DISPLAY_HEALTH.get(str(value), str(value))


def driver_label(value):
    return DISPLAY_DRIVER.get(str(value), str(value))


def priority_plain_language(row):
    parts = [
        (float(row["priority_health_contribution"]), "Health concern"),
        (float(row["priority_support_contribution"]), "recent support urgency"),
        (float(row["priority_commercial_contribution"]), "commercial exposure"),
        (float(row["priority_evidence_contribution"]), "evidence concern"),
    ]
    ranked = sorted(parts, reverse=True)
    main = [f"{name} ({value:.2f} pts)" for value, name in ranked if value > 0][:2]
    if main:
        reason_text = " and ".join(main)
    else:
        reason_text = "no material adverse contribution"
    sentence = (
        f"This account is #{int(row['priority_rank'])} in the review queue, with Priority "
        f"{float(row['priority_score']):.2f} ({row['priority_band']}). "
        f"The largest Priority contributors are {reason_text}. "
        f"The main review reason is {driver_label(row['leading_driver']).lower()}."
    )
    if float(row["priority_commercial_contribution"]) > 0:
        sentence += " Commercial exposure affects review order only; it does not change Health."
    return sentence


def app_header(source_label, data_hash):
    display_source = "RavenStack (Demo)" if source_label == "Built-in demo" else "RavenStack (Uploaded)"
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">Customer Success Workspace</div>
            <div class="hero-title">Account Health &amp; Prioritisation</div>
            <div class="hero-subtitle">
                Review account health, prioritise accounts for follow-up, and understand the reasons behind each result.
            </div>
            <div class="status-row">
                <span class="pill pill-blue">Snapshot · {SNAPSHOT_LABEL}</span>
                <span class="pill">Data source · {html.escape(display_source)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


METRIC_ICON_PATHS = {
    "flag": '<path d="M5 21V4m0 0c5-3 8 3 14 0v11c-6 3-9-3-14 0"/>',
    "heart": '<path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.7l-1.1-1.1a5.5 5.5 0 0 0-7.8 7.8L12 21l8.8-8.6a5.5 5.5 0 0 0 0-7.8Z"/><path d="M3 12h4l2-3 3 7 2-4h7"/>',
    "database": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7"/>',
    "activity": '<rect x="3" y="13" width="4" height="8" rx="1"/><rect x="10" y="8" width="4" height="13" rx="1"/><rect x="17" y="3" width="4" height="18" rx="1"/>',
    "headset": '<path d="M3 14v-3a9 9 0 0 1 18 0v3"/><rect x="2" y="12" width="4" height="7" rx="2"/><rect x="18" y="12" width="4" height="7" rx="2"/><path d="M20 19c0 2-3 3-6 3h-2"/>',
    "dollar": '<path d="M12 2v20M17 6c-2-2-8-2-10 1-3 4 2 5 5 5s8 2 5 6c-2 3-8 3-10 1"/>',
    "list": '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
    "up": '<path d="m7 15 5-5 5 5M7 9l5-5 5 5"/>',
    "down": '<path d="m7 9 5 5 5-5M7 15l5 5 5-5"/>',
    "equal": '<path d="M5 9h14M5 15h14"/>',
    "heartbreak": '<path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.7l-1.1-1.1a5.5 5.5 0 0 0-7.8 7.8L12 21l8.8-8.6a5.5 5.5 0 0 0 0-7.8Z"/><path d="m12 6-2 5 4 2-2 5"/>',
    "filequestion": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6M10 13a2 2 0 1 1 3 2c-1 .5-1 1-1 2m0 2h.01"/>',
    "chart": '<path d="M4 21V11h4v10M10 21V7h4v14M16 21V3h4v18"/>',
    "ticket": '<path d="M4 5h16v5a2 2 0 0 0 0 4v5H4v-5a2 2 0 0 0 0-4V5Z"/><path d="M12 5v14" stroke-dasharray="2 3"/>',
    "inbox": '<path d="M4 4h16l2 11v5H2v-5L4 4Z"/><path d="M2 15h6l2 3h4l2-3h6"/>',
    "alert": '<circle cx="12" cy="12" r="9"/><path d="M12 7v6m0 4h.01"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l-3 2"/>',
    "star": '<path d="m12 2 3.1 6.4 7.1 1-5.1 5 .9 7-6-3.3-6 3.3 1-7-5.2-5 7.1-1Z"/>',
    "message": '<path d="M4 4h16v13H8l-4 4V4Z"/>',
}


def metric_icon_svg(icon):
    if icon not in METRIC_ICON_PATHS:
        return ""
    return (
        '<svg viewBox="0 0 24 24" width="25" height="25" fill="none" '
        'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
        'stroke-linejoin="round" aria-hidden="true">'
        + METRIC_ICON_PATHS[icon] + '</svg>'
    )


def page_heading(title, kicker, icon):
    st.markdown(
        f'<div class="page-heading"><span class="page-heading-icon">{icon}</span>'
        f'<div><div class="page-heading-kicker">{html.escape(kicker)}</div>'
        f'<div class="page-heading-title">{html.escape(title)}</div></div></div>',
        unsafe_allow_html=True,
    )


def metric_card(label, value, sub="", accent="blue", compact=False, icon=None, pill=None):
    compact_class = " compact" if compact else ""
    accent_color = COLORS.get(accent, COLORS['blue'])
    soft_map = {
        "blue": "rgba(91,140,255,0.14)",
        "cyan": "rgba(53,194,193,0.16)",
        "green": "rgba(66,201,139,0.16)",
        "amber": "rgba(245,185,76,0.18)",
        "red": "rgba(255,107,122,0.16)",
        "purple": "rgba(155,123,255,0.16)",
        "slate": "rgba(141,153,174,0.16)",
    }
    soft_color = soft_map.get(accent, "rgba(91,140,255,0.14)")
    header_html = ""
    if icon:
        header_html = (
            f'<div class="metric-header">'
            f'<div class="metric-icon" style="--metric-accent:{accent_color}; --metric-soft:{soft_color};">{metric_icon_svg(icon)}</div>'
            f'<div class="metric-label">{html.escape(str(label))}</div>'
            f'</div>'
        )
    else:
        header_html = f'<div class="metric-label">{html.escape(str(label))}</div>'
    pill_html = ""
    if pill:
        pill_html = f'<div class="metric-pill" style="--metric-accent:{accent_color}; --metric-soft:{soft_color};">{html.escape(str(pill))}</div>'
    # Render recognised physical / numerical units separately from the headline value.
    # This is display-only; no score, aggregation or threshold is recalculated.
    display_value = str(value)
    for suffix in (" accounts", " tickets", " responses", " pts", " hours", " days", " h"):
        if display_value.endswith(suffix):
            main = display_value[:-len(suffix)]
            unit = suffix.strip()
            # Coverage cards display a bold numerator with smaller regular denominator.
            if unit == "accounts" and "/" in main:
                numerator, denominator = main.split("/", 1)
                value_html = (f'{html.escape(numerator)}'
                              f'<span class="metric-unit">/{html.escape(denominator)} accounts</span>')
            else:
                value_html = f'{html.escape(main)}<span class="metric-unit"> {html.escape(unit)}</span>'
            break
    else:
        value_html = html.escape(display_value)
    st.markdown(
        f"""
        <div class="metric-card">
            {header_html}
            <div class="metric-value{compact_class}">{value_html}</div>
            <div class="metric-sub">{html.escape(str(sub))}</div>
            {pill_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def horizontal_bar_chart(frame, category, value, order, palette, height=230, x_domain=None, selection_name=None):
    d = frame.copy()
    d[category] = d[category].astype(str)

    # Use clean 50-account intervals for broad portfolio views. For small
    # filtered populations, use narrower integer intervals so bars remain readable.
    largest = max(int(d[value].max()) if len(d) else 0, int(x_domain[1]) if x_domain is not None else 0)
    if largest > 100:
        tick_step = 50
    elif largest > 40:
        tick_step = 10
    elif largest > 20:
        tick_step = 5
    elif largest > 10:
        tick_step = 2
    else:
        tick_step = 1
    tick_ceiling = max(tick_step, ((largest + tick_step - 1) // tick_step) * tick_step)
    account_ticks = list(range(0, tick_ceiling + 1, tick_step))

    bars = (
        alt.Chart(d)
        .mark_bar(cornerRadiusEnd=5, size=24)
        .encode(
            y=alt.Y(
                f"{category}:N",
                sort=order,
                axis=alt.Axis(
                    title="",
                    labelLimit=340,
                    labelFontSize=12,
                    labelPadding=8,
                ),
            ),
            x=alt.X(
                f"{value}:Q",
                axis=alt.Axis(title="Accounts", grid=True, values=account_ticks, format="d"),
                scale=alt.Scale(domain=[0, tick_ceiling], nice=False),
            ),
            color=alt.Color(
                f"{category}:N",
                scale=alt.Scale(domain=order, range=palette),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip(f"{category}:N", title="Category"),
                alt.Tooltip(f"{value}:Q", title="Accounts"),
            ],
        )
        .properties(height=height)
    )

    labels = (
        alt.Chart(d)
        .mark_text(
            align="left",
            baseline="middle",
            dx=7,
            fontSize=12,
            fontWeight="bold",
        )
        .encode(
            y=alt.Y(f"{category}:N", sort=order),
            x=alt.X(f"{value}:Q"),
            text=alt.Text(f"{value}:Q"),
        )
    )
    # Streamlit 1.50 rejects `on_select` for layered/composed charts.
    # Interactive charts MUST be single-view Altair charts. Include account
    # count alongside the category axis label; exact values remain in tooltip.
    if selection_name:
        display_col = "__display_category"
        d[display_col] = d[category] + "  ·  " + d[value].astype(int).astype(str)
        display_order = [f"{name}  ·  {int(d.loc[d[category] == name, value].iloc[0])}" for name in order if (d[category] == name).any()]
        single_view = (
            alt.Chart(d)
            .mark_bar(cornerRadiusEnd=5, size=24)
            .encode(
                y=alt.Y(f"{display_col}:N", sort=display_order,
                        axis=alt.Axis(title="", labelLimit=340, labelFontSize=12, labelPadding=8)),
                x=alt.X(f"{value}:Q",
                        axis=alt.Axis(title="Accounts", grid=True, values=account_ticks, format="d"),
                        scale=alt.Scale(domain=[0, tick_ceiling], nice=False)),
                color=alt.Color(f"{category}:N", scale=alt.Scale(domain=order, range=palette), legend=None),
                tooltip=[alt.Tooltip(f"{category}:N", title="Category"),
                         alt.Tooltip(f"{value}:Q", title="Accounts")],
            )
            .add_params(alt.selection_point(name=selection_name, fields=[category], on="click", empty=False))
            .properties(height=height)
        )
        final_chart = single_view
    else:
        final_chart = bars + labels
    return (
        final_chart
        .properties(background="#F7FAFF", padding={"left": 12, "right": 30, "top": 12, "bottom": 10})
    ).configure_view(
        strokeOpacity=0,
        fill="#F7FAFF",
    ).configure_axis(
        labelColor="#314664",
        titleColor="#314664",
        domainColor="#D9E2F3",
        tickColor="#D9E2F3",
        gridColor="#DCE8F7",
        labelFontSize=12,
        titleFontSize=12,
    )


def render_chart(chart):
    # Render Altair charts using the supported Streamlit API.
    st.altair_chart(chart, use_container_width=True)


def driver_chart(data):
    counts = (
        data["leading_driver"]
        .value_counts()
        .rename_axis("Raw driver")
        .reset_index(name="Accounts")
    )
    counts["Driver"] = counts["Raw driver"].map(driver_label)
    counts = counts.groupby("Driver", as_index=False)["Accounts"].sum()
    # Group review concerns together before escalation / neutral evidence statuses.
    # Keep descending account counts inside each group; do not change driver assignment.
    review_concerns = {"Low recent usage", "Recent support pressure", "Low satisfaction"}
    def group_order(label):
        if label in review_concerns: return 0
        if label == "Recent escalation history": return 1
        if label == "No significant concern identified": return 2
        if label == "Insufficient evidence": return 3
        return 4
    order = sorted(counts["Driver"].tolist(), key=lambda x: (
        group_order(x), -int(counts.loc[counts["Driver"] == x, "Accounts"].iloc[0]), x
    ))
    mapping = {
        "Low recent usage": COLORS["blue"],
        "Recent support pressure": COLORS["blue"],
        "Low satisfaction": COLORS["blue"],
        "Recent escalation history": "#ED9858",
        "No significant concern identified": "#98A5B9",
        "Insufficient evidence": "#CBD5E1",
    }
    palette = [mapping.get(x, COLORS["slate"]) for x in order]
    return horizontal_bar_chart(
        counts, "Driver", "Accounts", order, palette,
        height=max(300, len(order) * 46), selection_name="portfolio_reason",
    )



def unpack_ravenstack_zip(uploaded_zip):
    """Read five RavenStack CSVs from one archive without extracting to disk."""
    try:
        with zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue())) as archive:
            members = [entry for entry in archive.infolist()
                       if not entry.is_dir()
                       and not entry.filename.startswith("__MACOSX/")
                       and not Path(entry.filename).name.startswith(".")]
            if len(members) != 5 or any(not entry.filename.lower().endswith(".csv") for entry in members):
                return [], ["The ZIP must contain exactly five CSV files (one for each RavenStack table)."]
            if any(entry.file_size > 20 * 1024 * 1024 for entry in members):
                return [], ["A CSV in the archive exceeds the 20 MB safety limit."]
            if sum(entry.file_size for entry in members) > 35 * 1024 * 1024:
                return [], ["Uncompressed CSV data exceeds the 35 MB safety limit."]
            files = []
            for entry in members:
                file_name = Path(entry.filename).name
                raw = archive.read(entry)
                files.append(SimpleNamespace(name=file_name, getvalue=lambda data=raw: data))
            return files, []
    except (zipfile.BadZipFile, RuntimeError, OSError, ValueError) as exc:
        return [], [f"Could not read the ZIP archive: {exc}"]


def render_upload_setup():
    scroll_to_top_on_navigation()
    app_header("Uploaded RavenStack CSVs", "")
    st.markdown("## Import RavenStack data")
    st.markdown(
        """
        <div class="section-note">
        Import the five RavenStack source tables using one ZIP file (recommended), or select the five CSVs directly. The app validates file schemas and dates before scoring.
        </div>
        """,
        unsafe_allow_html=True,
    )

    upload_method = st.radio(
        "Upload method",
        ["One ZIP containing 5 CSV files", "Five individual CSV files"],
        horizontal=True,
        key="ravenstack_upload_method",
    )
    if upload_method == "One ZIP containing 5 CSV files":
        uploaded_zip = st.file_uploader(
            "Upload one ZIP with the 5 RavenStack CSV files",
            type=["zip"],
            accept_multiple_files=False,
            key="ravenstack_zip_upload",
            help="Add the five raw RavenStack CSV files to one ZIP. No other files are required.",
        )
        if uploaded_zip is None:
            st.info("Upload a ZIP containing all five RavenStack CSV tables.")
            return None, None, None, None
        uploaded, zip_errors = unpack_ravenstack_zip(uploaded_zip)
        if zip_errors:
            for msg in zip_errors:
                st.error(msg)
            return None, None, None, None
    else:
        uploaded = st.file_uploader(
            "Upload the 5 RavenStack CSV files",
            type=["csv"],
            accept_multiple_files=True,
            key="ravenstack_csv_upload",
            help="Select the five RavenStack CSV files. If simultaneous uploads fail on Cloud, use the one-ZIP option instead.",
        )
        if not uploaded:
            st.info("Upload all five RavenStack files to build the portfolio view.")
            return None, None, None, None

    frames, read_errors, read_warnings, source_hash = read_uploaded_files(uploaded)
    if read_errors:
        for msg in read_errors:
            st.error(msg)
        return None, None, None, None

    for msg in read_warnings:
        st.warning(msg)

    run = st.button(
        "Check files and build the portfolio view",
        type="primary",
        use_container_width=True,
    )

    if (
        "processed_source_hash" in st.session_state
        and st.session_state["processed_source_hash"] != source_hash
    ):
        for key in [
            "processed_source_hash","uploaded_app_data","upload_validation", "ui_uploaded_ticket_raw",
            "upload_stats","upload_warnings","uploaded_app_hash",
        ]:
            st.session_state.pop(key, None)

    if run:
        with st.spinner("Validating files and building the account-level scoring output..."):
            processed = process_raw_frames(frames)

        st.session_state["upload_validation"] = processed["validation"]
        st.session_state["upload_warnings"] = processed["warnings"]

        if processed["errors"]:
            st.session_state.pop("uploaded_app_data", None)
            for msg in processed["errors"]:
                st.error(msg)
        else:
            app_data = processed["app_ready"]
            csv_bytes = app_data.to_csv(index=False).encode("utf-8")
            st.session_state["ui_uploaded_ticket_raw"] = frames["support_tickets"][["account_id", "submitted_at", "closed_at"]].copy()
            st.session_state["uploaded_app_data"] = app_data
            st.session_state["upload_stats"] = processed["stats"]
            st.session_state["processed_source_hash"] = source_hash
            st.session_state["uploaded_app_hash"] = hashlib.sha256(csv_bytes).hexdigest()
            # Keep upload confirmation screen until a workspace page is explicitly chosen.
            st.session_state["upload_workspace_open"] = False

    validation = st.session_state.get("upload_validation")
    if validation is not None:
        with st.expander("Review file checks", expanded=False):
            st.dataframe(validation, use_container_width=True, hide_index=True)

    if "uploaded_app_data" in st.session_state:
        st.success("Your files passed validation and the portfolio view is ready.")
        return (
            st.session_state["uploaded_app_data"],
            st.session_state["uploaded_app_hash"],
            source_hash,
            st.session_state.get("upload_stats", {}),
        )

    return None, None, source_hash, None



# Presentation-only evidence measures; deliberately excluded from scoring.
def add_display_evidence_columns(frame, source_mode):
    out = frame.copy()
    exposure = out["tenure_days"].clip(lower=1, upper=90)
    # Reverse the original exact 90-day normalisation; never treat adjusted days as observed days.
    observed = (out["active_days_90_per_90_exposure"] * exposure / 90).round()
    out["ui_observed_active_days"] = observed.where(out["usage_events_90"].fillna(0) > 0, 0).astype("Int64")
    if source_mode == "Built-in demo":
        path = Path(__file__).resolve().parent / "sample_raw_data" / "ravenstack_support_tickets(3).csv"
        tickets = pd.read_csv(path, usecols=["account_id", "submitted_at", "closed_at"])
    else:
        tickets = st.session_state.get("ui_uploaded_ticket_raw")
    if tickets is None:
        out["ui_open_tickets"] = pd.NA
    else:
        t = tickets.copy()
        submitted = pd.to_datetime(t["submitted_at"], errors="coerce")
        closed = pd.to_datetime(t["closed_at"], errors="coerce")
        snapshot = pd.Timestamp("2024-12-31 23:59:59")
        open_mask = submitted.notna() & (submitted <= snapshot) & (closed.isna() | (closed > snapshot))
        counts = t.loc[open_mask].groupby("account_id").size()
        out["ui_open_tickets"] = out["account_id"].map(counts).fillna(0).astype(int)
    return out


def escalation_evidence(row):
    driver = str(row["leading_driver"])
    if driver == "Recent escalation history":
        n = int(row["escalated_tickets_90"]) if pd.notna(row["escalated_tickets_90"]) else 0
        return ("A support ticket was escalated during the review period." if n == 1
                else f"{n} support tickets were escalated during the review period.")
    if driver == "Low recent engagement":
        days = row["days_since_last_usage"]
        return (f"The last recorded product activity was {int(days)} days before the snapshot."
                if pd.notna(days) else "No valid recent product activity was recorded.")
    if driver == "Recent support pressure":
        count = int(row["tickets_90"])
        return (f"{count} support ticket was recorded during the 90-day review period." if count == 1
                else f"{count} support tickets were recorded during the 90-day review period.")
    if driver == "Low observed satisfaction":
        return (f"Average recorded satisfaction was {float(row['avg_satisfaction']):.2f} out of 5."
                if pd.notna(row['avg_satisfaction']) else "No satisfaction responses were recorded.")
    if driver.startswith("Limited Evidence"):
        return "Available evidence does not meet the project's Health scoring requirements."
    return "No significant adverse driver was identified under the project's scoring rules."


def display_action(row):
    action = str(row["suggested_action"])
    if str(row["leading_driver"]) == "Recent escalation history" and int(row["escalated_tickets_90"]) > 1:
        action = action.replace("the recent escalation outcome", "the recent escalation outcomes")
    return action


def priority_breakdown_html(row):
    items = [
        ("Health concern", "priority_health_contribution", 50, "#5684F5", "Low Health increases the need for review."),
        ("Recent support urgency", "priority_support_contribution", 25, "#F2B747", "Recent ticket urgency or escalation history can raise priority."),
        ("Commercial exposure", "priority_commercial_contribution", 15, "#31B9B6", "Commercial exposure affects review order, not Health."),
        ("Evidence concern", "priority_evidence_contribution", 10, "#9982E7", "Evidence gaps can add review priority."),
    ]
    bits = [f'<div class="priority-overview"><div style="font-size:.78rem;color:#64728A;font-weight:700">Overall Priority Score</div><div style="font-weight:800;font-size:1.8rem">{row["priority_score"]:.2f} <span style="font-size:.95rem;color:#6B7791">/ 100</span> <span style="font-size:.82rem;color:#D9566C">{html.escape(str(row["priority_band"]))} Priority</span></div><div class="priority-summary">A higher score means earlier CSM review, not a predicted probability of churn.</div></div>']
    for name, key, maxpoints, color, explanation in items:
        value = float(row[key]); width = max(0, min(100, 100*value/maxpoints))
        bits.append(f'<div class="priority-row"><div class="priority-row-head"><span>{name}</span><span>{value:.2f} / {maxpoints} pts</span></div><div class="priority-track"><div class="priority-fill" style="width:{width:.5f}%;background:{color}"></div></div><div class="priority-row-note">{explanation}</div></div>')
    return '<div style="margin:.45rem 0 1.15rem">' + ''.join(bits) + '</div>'

def jump_to_priority(*, band="All", health="All", evidence="All", reason="All reasons",
                     owner="All owners", industry="All industries"):
    """UI navigation only: preserve global ranks and original scoring fields."""
    st.session_state["portfolio_chart_epoch"] = st.session_state.get("portfolio_chart_epoch", 0) + 1
    st.session_state.update({
        "priority_band_filter": band,
        "priority_health_filter": health,
        "priority_evidence_filter": evidence,
        "priority_reason_filter": reason,
        "priority_owner_filter": owner,
        "priority_industry_filter": industry,
        "priority_page": 1,
        "priority_page_picker": 1,
        "nav_page": "Priority List",
        "scroll_to_top_pending": True,
    })
    st.rerun()


def clickable_portfolio_chart(frame, category, value, order, palette, *, chart_key, title,
                              height=230, x_domain=None):
    """Render a selectable Altair bar chart and return the chosen label (or None)."""
    # Chart selection is UI-only. A new key after navigation clears old selections.
    chart = horizontal_bar_chart(frame, category, value, order, palette,
                                 height=height, x_domain=x_domain, selection_name="portfolio_bar")
    key = f"{chart_key}_{st.session_state.get('portfolio_chart_epoch', 0)}"
    event = st.altair_chart(chart, use_container_width=True, on_select="rerun", key=key)
    selected = event.selection.get("portfolio_bar", []) if event is not None else []
    if selected:
        return selected[0].get(category)
    return None



# Scroll only on page/data-source navigation, not on filters or expanders.
def scroll_to_top_on_navigation():
    if st.session_state.pop("scroll_to_top_pending", False):
        import streamlit.components.v1 as components
        components.html("""<script>
        try {
          const w = window.parent;
          const root = w.document.querySelector('[data-testid="stMain"]') || w.document;
          const scroller = w.document.querySelector('[data-testid="stMain"]') || w.document.querySelector('section.main');
          if (scroller) scroller.scrollTo({top:0,behavior:'instant'});
          w.scrollTo({top:0,behavior:'instant'});
          const anchor = w.document.querySelector('[data-testid="stAppViewContainer"]');
          if (anchor) anchor.scrollTop = 0;
        } catch(e) { /* parent access may depend on Streamlit hosting */ }
        </script>""", height=0)


def scoring_rule_tooltip(title, lines, applied=None, note=None):
    """Small info icon with a consistently aligned threshold table."""
    with st.popover(":material/info:", help="View scoring thresholds"):
        safe_title = html.escape(str(title))
        rows = []
        for idx, (condition, value) in enumerate(lines):
            current = idx == applied
            label = html.escape(str(condition)) + (" <span class='rule-check'>✓</span>" if current else "")
            val = html.escape(str(value))
            rows.append(
                f"<tr class={'rule-active' if current else 'rule-normal'}>"
                f"<td>{label}</td><td>{val}</td></tr>"
            )
        note_html = f"<div class='rule-note'>{html.escape(str(note))}</div>" if note else ""
        st.markdown(
            "<div class='rule-tooltip-body'>"
            f"<div class='rule-tooltip-title'>{safe_title}</div>"
            "<table class='rule-tooltip-table'><tbody>" + "".join(rows) + "</tbody></table>"
            + note_html + "</div>", unsafe_allow_html=True
        )


def component_card(title, score, weight, contribution, evidence, calculation, meaning, tips=()):
    """Chevron-based expandable card. Collapsed on initial render per account."""
    with st.expander(f"{title}  ·  {contribution:.1f} contribution pts", expanded=False):
        st.caption(f"Base score {score:.1f} pts · Weight {weight}")
        st.markdown("**Observed evidence & applied rules**")
        for label, value, tip in evidence:
            left, right = st.columns([3.5, 2], vertical_alignment="center")
            with left:
                if tip is None:
                    st.write(label)
                else:
                    a, b = st.columns([8, 1], gap="small", vertical_alignment="center")
                    a.write(label)
                    with b:
                        scoring_rule_tooltip(**tips[tip])
            right.markdown(f"<div style='text-align:right'>{html.escape(str(value))}</div>", unsafe_allow_html=True)
        st.info(f"**Score calculation**  \n{calculation}")
        st.markdown(f"**What this indicates:** {meaning}")


def scoring_detail_cards(row):
    from pipeline import RESOLUTION_P75_HOURS, MRR_P50, MRR_P75, MRR_P90
    def num(k, d=1):
        v = row.get(k)
        return f"{float(v):.{d}f}" if pd.notna(v) else "Not available"
    rec = float(row['days_since_last_usage']) if pd.notna(row['days_since_last_usage']) else float('inf')
    act = float(row['active_days_90_per_90_exposure']) if pd.notna(row['active_days_90_per_90_exposure']) else 0
    esc = float(row['escalated_tickets_90']); hi = float(row['high_urgent_tickets_90']); tickets = float(row['tickets_90'])
    rt = row['avg_resolution_time_hours_90']; slow = pd.notna(rt) and float(rt) > RESOLUTION_P75_HOURS
    support_idx = 0 if tickets <= 0 else 1 if esc >= 1 or hi >= 2 else 2 if hi >= 1 or slow else 3
    urgency_idx = 0 if esc >= 1 else 1 if hi >= 2 else 2 if hi >= 1 or slow else 3 if tickets >= 1 else 4
    mrr = float(row['mrr_amount']) if pd.notna(row['mrr_amount']) else 0
    comm_idx = 0 if mrr <= 0 else 1 if mrr <= MRR_P50 else 2 if mrr <= MRR_P75 else 3 if mrr <= MRR_P90 else 4
    scoreable = row['evidence_status'] == 'Scoreable'
    # App-ready CSV deliberately excludes intermediate scoring columns.
    # Reconstruct display-only base components from the exported contributions;
    # do not change or recompute the pipeline's final Health/Priority scores.
    health_concern_base = float(row['priority_health_contribution']) / 0.50
    commercial_base = float(row['priority_commercial_contribution']) / 0.15
    evidence_base = float(row['priority_evidence_contribution']) / 0.10
    tips = {
      'recency':dict(title='Recency thresholds', lines=[('≤ 14 days','100 pts'),('15–30 days','65 pts'),('31–60 days','30 pts'),('> 60 days / no usage','0 pts')],applied=0 if rec<=14 else 1 if rec<=30 else 2 if rec<=60 else 3),
      'activity':dict(title='Adjusted active-day thresholds',lines=[('< 4 days','25 pts'),('4 to < 8 days','60 pts'),('≥ 8 days','100 pts')],applied=0 if act<4 else 1 if act<8 else 2,note='Activity is normalised to a 90-day equivalent.'),
      'support':dict(title='Support score rules (first matching)',lines=[('No recent tickets','100 pts'),('≥1 escalation OR ≥2 high/urgent','25 pts'),('≥1 high/urgent OR resolution > P75','50 pts'),('Other recent tickets','75 pts')],applied=support_idx,note=f'Resolution P75: {RESOLUTION_P75_HOURS:.2f} hours.'),
      'satisfaction':dict(title='Satisfaction modifier rules',lines=[('Reliability','Coverage × min(1, responses/2)'),('Raw modifier','(average − 4) × 5 × reliability'),('Bounded adjustment','−5 to +5 pts')],note='Added after Engagement and Support weighting.'),
      'health':dict(title='Health band thresholds',lines=[('Score < 50','Low'),('50 ≤ score < 70','Medium'),('Score ≥ 70','Healthy')],applied=0 if row['health_score']<50 else 1 if row['health_score']<70 else 2,note='Non-scoreable accounts show Limited Evidence.'),
      'urgency':dict(title='Support urgency (first matching)',lines=[('≥1 escalation','100 pts'),('≥2 high/urgent','85 pts'),('≥1 high/urgent OR resolution > P75','60 pts'),('≥1 recent ticket','30 pts'),('No recent tickets','0 pts')],applied=urgency_idx,note=f'Resolution P75: {RESOLUTION_P75_HOURS:.2f} hours.'),
      'commercial':dict(title='MRR exposure thresholds',lines=[('Missing or MRR ≤ 0','0 pts'),(f'0 < MRR ≤ {MRR_P50:,.2f}','25 pts'),(f'MRR ≤ {MRR_P75:,.2f}','50 pts'),(f'MRR ≤ {MRR_P90:,.2f}','75 pts'),(f'MRR > {MRR_P90:,.2f}','100 pts')],applied=comm_idx,note='MRR proxy; source currency unspecified. Cutoffs are from the frozen scoring rules.'),
      'evidence':dict(title='Evidence concern rules',lines=[('Scoreable','0 pts'),('Not scoreable','100 pts')],applied=0 if scoreable else 1,note='Limited evidence does not imply poor customer health.'),
    }
    st.markdown('#### Health Calculation')
    component_card('Engagement',float(row['engagement_score']),'60%',0.6*float(row['engagement_score']),[
      ('Days since last usage',f"{num('days_since_last_usage',0)} days · {num('recency_score',0)} pts",'recency'),
      ('Adjusted active days',f"{num('active_days_90_per_90_exposure')} days · {num('activity_score',0)} pts",'activity')],
      f"({num('recency_score',0)} × 60%) + ({num('activity_score',0)} × 40%) = {num('engagement_score')} pts; weighted Health contribution = {0.6*float(row['engagement_score']):.1f} pts.",
      'Lower recent engagement reduces the Health Score; it is a review signal, not a confirmed cause.',tips)
    component_card('Support',float(row['support_score']),'40%',.4*float(row['support_score']),[
      ('Tickets in last 90 days',f"{num('tickets_90',0)} tickets",'support'),
      ('Escalated / high-urgent',f"{num('escalated_tickets_90',0)} / {num('high_urgent_tickets_90',0)}",None),
      ('Average resolution time',f"{num('avg_resolution_time_hours_90')} hours",None)],
      f"Support score {num('support_score')} × 40% = {.4*float(row['support_score']):.1f} pts.",
      'A lower Support Score indicates greater service pressure under the defined rules.',tips)
    modifier=float(row['satisfaction_modifier'])
    component_card('Satisfaction modifier',modifier,'Additive',modifier,[
      ('Average satisfaction',f"{num('avg_satisfaction',2)} / 5",'satisfaction'),
      ('Response count',f"{num('satisfaction_response_count',0)}",None),
      ('Reliability factor',num('satisfaction_reliability',2),None)],
      f"({num('avg_satisfaction',2)} − 4) × 5 × {num('satisfaction_reliability',2)}; bounded to ±5, calculated modifier {modifier:+.2f} pts.",
      'Satisfaction provides a limited adjustment where valid responses exist.',tips)
    st.markdown(f"**Final Health Score: {float(row['health_score']):.2f} pts · {row['health_band']}**")
    st.caption('Health = 60% Engagement + 40% Support + Satisfaction modifier, clipped to 0–100. Values in cards display one decimal; final scores retain two.')
    st.divider()
    st.markdown('#### Priority Calculation')
    component_card('Health concern',health_concern_base,'50%',float(row['priority_health_contribution']),[
      ('Health Score',f"{num('health_score',2)} pts · {row['health_band']}",'health'),
      ('Evidence eligibility',str(row['evidence_status']),None)],
      f"Health concern = {'100 − Health' if scoreable else '50 (limited evidence default)'} = {health_concern_base:.1f} pts; × 50% = {float(row['priority_health_contribution']):.1f} pts.",
      'Lower Health raises review priority when evidence is scoreable.',tips)
    component_card('Support urgency',float(row['support_urgency_score']),'25%',float(row['priority_support_contribution']),[
      ('Recent escalations',f"{num('escalated_tickets_90',0)}",'urgency'),('High-urgent tickets',num('high_urgent_tickets_90',0),None),('Recent tickets',num('tickets_90',0),None)],
      f"Urgency {num('support_urgency_score')} × 25% = {float(row['priority_support_contribution']):.1f} pts.",
      'Recent ticket or escalation conditions determine operational review urgency.',tips)
    component_card('Commercial exposure',commercial_base,'15%',float(row['priority_commercial_contribution']),[
      ('MRR proxy',f"{mrr:,.2f} (currency unspecified)",'commercial')],
      f"Exposure base {commercial_base:.1f} × 15% = {float(row['priority_commercial_contribution']):.1f} pts.",
      'Commercial exposure affects review order, not the Health Score.',tips)
    component_card('Evidence concern',evidence_base,'10%',float(row['priority_evidence_contribution']),[
      ('Evidence status',str(row['evidence_status']),'evidence')],
      f"Evidence concern {evidence_base:.1f} × 10% = {float(row['priority_evidence_contribution']):.1f} pts.",
      'Limited evidence raises data-verification priority without asserting poor Health.',tips)
    st.markdown(f"**Final Priority Score: {float(row['priority_score']):.2f} pts · {row['priority_band']} Priority**")
    st.caption('Priority = 50% Health concern + 25% Support urgency + 15% Commercial exposure + 10% Evidence concern. Bands: Low <30; Medium 30–<50; High ≥50. Final score uses unrounded contributions.')

# ------------------------------------------------------------------
# Data source
# ------------------------------------------------------------------
st.sidebar.markdown('<div class="sidebar-label">Data source</div>', unsafe_allow_html=True)
source_mode = st.sidebar.radio(
    "Data source",
    ["Built-in demo", "Upload RavenStack CSVs"],
    label_visibility="collapsed",
    on_change=lambda: st.session_state.update({"scroll_to_top_pending": True}),
)

st.sidebar.markdown(
    """
    <div class="sidebar-help">
        <b style="color:#2B3651;">Quick Start Guide</b><br><br>
        <b>1. Load Data</b><br>Use the demo or upload 5 RavenStack CSVs.<br>
        <b>2. Portfolio Dashboard</b><br>Get an overview of the entire account portfolio.<br>
        <b>3. Priority List</b><br>Narrow down to accounts that need review.<br>
        <b>4. Account 360</b><br>Explore the evidence behind each account's scores.
    </div>
    """,
    unsafe_allow_html=True,
)


if source_mode == "Built-in demo":
    data, data_hash, data_error = load_demo()
    source_label = "Built-in demo"
    if data is None:
        st.error(data_error)
        st.stop()
else:
    source_label = "Uploaded RavenStack CSVs"
    if st.session_state.get("upload_workspace_open") and "uploaded_app_data" in st.session_state:
        data = st.session_state["uploaded_app_data"]
        data_hash = st.session_state["uploaded_app_hash"]
    else:
        data, data_hash, _, _ = render_upload_setup()
    if data is None:
        st.stop()

# Add a UI-only fictional owner so the review queue can demonstrate realistic portfolio ownership.
# This field is not part of RavenStack and is never used in Health or Priority scoring.
data = add_demo_csm_owner(data)
data = add_display_evidence_columns(data, source_mode)

# ------------------------------------------------------------------
# Three-page Streamlit navigation
# ------------------------------------------------------------------
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Portfolio Dashboard"

st.sidebar.markdown('<div class="sidebar-label">Review workspace</div>', unsafe_allow_html=True)

nav_items = [
    ("Portfolio Dashboard", "🏠  Portfolio Dashboard"),
    ("Priority List", "📋  Priority List"),
    ("Account 360", "🔎  Account 360"),
]

for page_name, label in nav_items:
    active = st.session_state["nav_page"] == page_name
    if st.sidebar.button(
        label,
        key=f"nav_{page_name}",
        type="primary" if active else "secondary",
        use_container_width=True,
    ):
        st.session_state["nav_page"] = page_name
        st.session_state["scroll_to_top_pending"] = True
        if source_mode == "Upload RavenStack CSVs":
            st.session_state["upload_workspace_open"] = True
        st.rerun()

page = st.session_state["nav_page"]

# In upload mode the confirmation view ends after file checks and success message.
# Never append a dashboard underneath the import screen. Navigation is intentional.
if source_mode == "Upload RavenStack CSVs":
    if not st.session_state.get("upload_workspace_open", False):
        st.stop()
    if st.sidebar.button("← Review / replace uploaded files", use_container_width=True):
        st.session_state["upload_workspace_open"] = False
        st.session_state["scroll_to_top_pending"] = True
        st.rerun()

scroll_to_top_on_navigation()
app_header(source_label, data_hash)


# ==================================================================
# 1. Portfolio Dashboard
# ==================================================================
if page == "Portfolio Dashboard":
    page_heading("Portfolio Dashboard", "Portfolio view", "🏠")
    st.markdown(
        """
        <div class="section-note">
        See the overall portfolio, identify accounts for review and check data coverage. Health shows account condition; Priority helps decide review order.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Filter portfolio", expanded=False):
        pf1, pf2 = st.columns(2)
        dashboard_owner = pf1.selectbox(
            "CSM owner",
            ["All owners"] + sorted(data["csm_owner_demo"].unique().tolist()),
            key="dashboard_owner",
            help="Demo-only fictional ownership. In production, this would come from an authorised CRM source.",
        )
        dashboard_industry = pf2.selectbox(
            "Industry",
            ["All industries"] + sorted(data["industry"].dropna().astype(str).unique().tolist()),
            key="dashboard_industry",
        )

    dashboard_data = data.copy()
    if dashboard_owner != "All owners":
        dashboard_data = dashboard_data[dashboard_data["csm_owner_demo"] == dashboard_owner]
    if dashboard_industry != "All industries":
        dashboard_data = dashboard_data[dashboard_data["industry"].astype(str) == dashboard_industry]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Accounts in view", f"{len(dashboard_data):,} accounts", "after current filters", "blue", icon="list")
    with c2:
        metric_card("High Priority", f'{int((dashboard_data["priority_band"] == "High").sum())} accounts', "review queue", "red", icon="up")
    with c3:
        metric_card("Low Health", f'{int((dashboard_data["health_band"] == "Low").sum())} accounts', "account-condition concern", "amber", icon="heartbreak")
    with c4:
        metric_card("Needs Data Check", f'{int((dashboard_data["health_band"] == "Limited Evidence").sum())} accounts', "Health not calculated", "purple", icon="filequestion")
    with c5:
        metric_card("Median Priority", f"{dashboard_data['priority_score'].median():.2f} pts" if len(dashboard_data) else "—", "median of account Priority scores", "cyan", icon="chart")

    st.markdown("### Portfolio mix")
    st.caption("Click a bar to open the matching accounts in Priority List.")
    b1, b2 = st.columns(2)

    priority = (
        dashboard_data["priority_band"]
        .value_counts()
        .reindex(["High","Medium","Low"], fill_value=0)
        .rename_axis("Band").reset_index(name="Accounts")
    )
    health = (
        dashboard_data["health_band"]
        .value_counts()
        .reindex(["Low","Medium","Healthy","Limited Evidence"], fill_value=0)
        .rename_axis("Band").reset_index(name="Accounts")
    )
    health["Band"] = health["Band"].map(health_label)
    shared_x_domain = [0, max(int(priority["Accounts"].max()), int(health["Accounts"].max()), 1)]

    with b1:
        st.caption("Priority distribution")
        clicked = clickable_portfolio_chart(
            priority, "Band", "Accounts", ["High", "Medium", "Low"],
            [COLORS["red"], COLORS["amber"], COLORS["green"]],
            chart_key="portfolio_priority_click", title="Priority distribution",
            height=230, x_domain=shared_x_domain,
        )
        if clicked in ("High", "Medium", "Low"):
            jump_to_priority(band=clicked, owner=dashboard_owner, industry=dashboard_industry)

    with b2:
        st.caption("Health snapshot")
        clicked = clickable_portfolio_chart(
            health, "Band", "Accounts", ["Low", "Medium", "Healthy", "Needs Data Check"],
            [COLORS["red"], COLORS["amber"], COLORS["green"], COLORS["purple"]],
            chart_key="portfolio_health_click", title="Health snapshot",
            height=230, x_domain=shared_x_domain,
        )
        if clicked in ("Low", "Medium", "Healthy", "Needs Data Check"):
            jump_to_priority(health=clicked, owner=dashboard_owner, industry=dashboard_industry)

    st.markdown("### Top 10 accounts to review")
    st.caption("Highest-priority accounts in the current filters. Rank numbers are from the full portfolio.")
    top10 = dashboard_data.sort_values(["priority_rank", "account_id"]).head(10).copy()
    if top10.empty:
        st.info("No accounts match the current filters.")
    else:
        top10_table = top10[["priority_rank", "account_name", "csm_owner_demo", "priority_score", "priority_band", "health_score", "health_band", "leading_driver"]].copy()
        top10_table.loc[top10_table["health_band"] == "Limited Evidence", "health_score"] = np.nan
        top10_table["health_band"] = top10_table["health_band"].map(health_label)
        top10_table["leading_driver"] = top10_table["leading_driver"].map(driver_label)
        top10_table.columns = ["Rank", "Account", "CSM Owner", "Priority", "Priority band", "Health score", "Health status", "Main reason"]
        st.dataframe(
            top10_table,
            use_container_width=True,
            hide_index=True,
            height=390,
            column_config={
                "Priority": st.column_config.NumberColumn(format="%.2f pts"),
                "Health score": st.column_config.NumberColumn(format="%.2f pts"),
                "Main reason": st.column_config.TextColumn(width="large"),
            },
        )

    st.markdown("### Evidence coverage")
    q1, q2, q3, q4 = st.columns(4)

    total_accounts = len(dashboard_data)
    usage_accounts = int(dashboard_data["days_since_last_usage"].notna().sum())
    support_accounts = int((dashboard_data["tickets_90"].fillna(0) > 0).sum())
    sat_accounts = int((dashboard_data["satisfaction_response_count"].fillna(0) > 0).sum())
    scoreable = int((dashboard_data["evidence_status"] == "Scoreable").sum())

    with q1:
        metric_card("Usage evidence", f"{usage_accounts}/{total_accounts} accounts", "valid usage available", "blue")
    with q2:
        metric_card("Recent support", f"{support_accounts}/{total_accounts} accounts", "ticket evidence in 90d", "amber")
    with q3:
        metric_card("Satisfaction", f"{sat_accounts}/{total_accounts} accounts", "observed responses", "purple")
    with q4:
        metric_card("Sufficient evidence", f"{scoreable}/{total_accounts} accounts", "Health can be calculated", "green")

    # CSM Priority Heatmap: a UI-only, clickable summary of existing priority bands.
    csm_heatmap_box = st.container(border=True)
    with csm_heatmap_box:
        st.markdown("### CSM Priority Overview")
        st.caption("Account counts by demo CSM owner and Priority band. Select a coloured cell to open the corresponding Priority List.")
        bands = ["High", "Medium", "Low"]
        matrix = (dashboard_data.groupby(["csm_owner_demo", "priority_band"])
                  .size().unstack(fill_value=0).reindex(columns=bands, fill_value=0))
        if matrix.empty:
            st.info("No accounts match the current portfolio filters.")
        else:
            matrix["Total"] = matrix[bands].sum(axis=1)
            matrix = matrix.sort_values(["High", "Total"], ascending=[False, False])
            st.markdown("""
            <style>
            [class*="st-key-csm_heat_High_"] button {background:#FCE5E8!important;color:#873342!important;border:1px solid #F3CDD4!important;font-weight:700!important;}
            [class*="st-key-csm_heat_Medium_"] button {background:#FCEACA!important;color:#815410!important;border:1px solid #F5DFAD!important;font-weight:700!important;}
            [class*="st-key-csm_heat_Low_"] button {background:#DDF5E9!important;color:#176449!important;border:1px solid #BAE9D4!important;font-weight:700!important;}
            [class*="st-key-csm_heat_"] button:hover {filter:brightness(.96);box-shadow:0 2px 10px rgba(60,90,145,.1)!important;}
            </style>
            """, unsafe_allow_html=True)
            headers = st.columns([2.3, 1, 1, 1, 1])
            for col, name in zip(headers, ["CSM Owner", "High", "Medium", "Low", "Total accounts"]):
                col.markdown(f"**{name}**")
            for idx, (owner, counts) in enumerate(matrix.iterrows()):
                cols = st.columns([2.3, 1, 1, 1, 1], vertical_alignment="center")
                cols[0].markdown(html.escape(str(owner)))
                for i, band in enumerate(bands, start=1):
                    count = int(counts[band])
                    if cols[i].button(str(count), key=f"csm_heat_{band}_{idx}",
                                      disabled=count==0, use_container_width=True,
                                      help=f"View {count} {band.lower()} priority accounts owned by {owner}"):
                        # Restore full matching cohort in Priority List: owner + band +
                        # currently selected industry; reset unrelated previous queue filters.
                        jump_to_priority(band=band, owner=owner, industry=dashboard_industry)
                cols[4].markdown(f"**{int(counts['Total'])}**")
        st.caption("CSM assignments are fictional demonstration data, not measures of staff performance and not part of scoring.")

    st.markdown("### Leading review reasons")
    st.caption("Each account is counted under one main review reason; other issues may also be present. These are not predictions of customer outcomes. Click a bar to filter Priority List.")
    reason_chart = driver_chart(dashboard_data)
    reason_key = f"portfolio_reason_click_{st.session_state.get('portfolio_chart_epoch', 0)}"
    reason_event = st.altair_chart(reason_chart, use_container_width=True,
                                   on_select="rerun", key=reason_key)
    chosen_reason = reason_event.selection.get("portfolio_reason", []) if reason_event is not None else []
    if chosen_reason:
        reason_label = chosen_reason[0].get("Driver")
        if reason_label:
            jump_to_priority(reason=reason_label, owner=dashboard_owner, industry=dashboard_industry)
    st.markdown(
        """
        <div class="legend-row">
            <span class="legend-item"><span class="legend-swatch" style="background:#5B8CFF"></span>Review concerns</span>
            <span class="legend-item"><span class="legend-swatch" style="background:#ED9858"></span>Escalation</span>
            <span class="legend-item"><span class="legend-swatch" style="background:#98A5B9"></span>No significant concern</span>
            <span class="legend-item"><span class="legend-swatch" style="background:#CBD5E1"></span>Insufficient evidence</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Industry is used for context and filtering, not scoring. CSM owner names are fictional demo assignments and are excluded from scoring.")


# ==================================================================
# 2. Priority List
# ==================================================================
elif page == "Priority List":
    page_heading("Priority List", "Review queue", "📋")
    st.markdown(
        """
        <div class="section-note">
        Find the accounts that need review. Priority shows <b>who to review first</b>; the main reason explains <b>why</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Filter accounts", expanded=True):
        f1, f2, f3, f4, f5, f6 = st.columns([1,1,1.05,1.15,1.15,0.85])
        p_filter = f1.selectbox("Priority", ["All","High","Medium","Low"], key="priority_band_filter")
        h_display_options = ["All","Low","Medium","Healthy","Needs Data Check"]
        h_filter_display = f2.selectbox("Health", h_display_options, key="priority_health_filter")
        evidence_display_options = ["All", "Sufficient evidence", "Needs Data Check"]
        e_filter_display = f3.selectbox("Data status", evidence_display_options, key="priority_evidence_filter")
        owner_filter = f4.selectbox(
            "CSM owner",
            ["All owners"] + sorted(data["csm_owner_demo"].unique().tolist()),
            key="priority_owner_filter",
            help="Demo-only fictional ownership. In production, this would come from CRM ownership data.",
        )
        industry_filter = f5.selectbox(
            "Industry",
            ["All industries"] + sorted(data["industry"].dropna().astype(str).unique().tolist()),
            key="priority_industry_filter",
        )
        page_size = int(f6.selectbox("Rows / page", [10,25,50,100], index=1))
        reason_filter = st.selectbox(
            "Main Review Reason",
            ["All reasons", "Low recent usage", "Recent support pressure", "Low satisfaction",
             "Recent escalation history", "No significant concern identified", "Insufficient evidence"],
            key="priority_reason_filter",
            help="Filter by the account's existing leading review driver; this does not alter scoring.",
        )

    filtered = data.copy()
    if p_filter != "All":
        filtered = filtered[filtered["priority_band"] == p_filter]
    if h_filter_display != "All":
        raw_h = "Limited Evidence" if h_filter_display == "Needs Data Check" else h_filter_display
        filtered = filtered[filtered["health_band"] == raw_h]
    if e_filter_display == "Sufficient evidence":
        filtered = filtered[filtered["evidence_status"] == "Scoreable"]
    elif e_filter_display == "Needs Data Check":
        filtered = filtered[filtered["evidence_status"] != "Scoreable"]
    if reason_filter != "All reasons":
        filtered = filtered[filtered["leading_driver"].map(driver_label) == reason_filter]
    if owner_filter != "All owners":
        filtered = filtered[filtered["csm_owner_demo"] == owner_filter]
    if industry_filter != "All industries":
        filtered = filtered[filtered["industry"].astype(str) == industry_filter]

    filtered = filtered.sort_values(["priority_rank", "account_id"]).reset_index(drop=True)
    total_filtered = len(filtered)
    total_pages = max(1, int(np.ceil(total_filtered / page_size))) if total_filtered else 1

    filter_signature = (
        p_filter, h_filter_display, e_filter_display,
        owner_filter, industry_filter, reason_filter, page_size,
    )
    if st.session_state.get("priority_filter_signature") != filter_signature:
        st.session_state["priority_filter_signature"] = filter_signature
        st.session_state["priority_page"] = 1
        st.session_state["priority_page_picker"] = 1

    current_page = min(max(1, int(st.session_state.get("priority_page", 1))), total_pages)
    st.session_state["priority_page"] = current_page
    if st.session_state.get("priority_page_picker") not in range(1, total_pages + 1):
        st.session_state["priority_page_picker"] = current_page

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Matching accounts", f"{total_filtered} accounts", "after current filters", "blue", icon="list")
    with c2:
        metric_card("High priority", f'{int((filtered["priority_band"] == "High").sum())} accounts', "review queue", "red", icon="up")
    with c3:
        metric_card("Medium priority", f'{int((filtered["priority_band"] == "Medium").sum())} accounts', "next in review queue", "amber", icon="equal")
    with c4:
        metric_card("Low priority", f'{int((filtered["priority_band"] == "Low").sum())} accounts', "lower review priority", "green", icon="down")
    with c5:
        metric_card("Needs Data Check", f'{int((filtered["health_band"] == "Limited Evidence").sum())} accounts', "Health not calculated", "purple", icon="filequestion")

    nav_left, nav_mid, nav_right = st.columns([1, 1.4, 1])
    # Render both navigation buttons before instantiating the page-picker widget.
    # This lets button clicks synchronise the picker in session_state without
    # mutating a widget key after Streamlit has already created that widget.
    with nav_left:
        if st.button("← Previous", disabled=current_page <= 1, use_container_width=True):
            st.session_state["priority_page"] = current_page - 1
            st.session_state["priority_page_picker"] = current_page - 1
            st.rerun()
    with nav_right:
        if st.button("Next →", disabled=current_page >= total_pages, use_container_width=True):
            st.session_state["priority_page"] = current_page + 1
            st.session_state["priority_page_picker"] = current_page + 1
            st.rerun()
    with nav_mid:
        selected_page = st.selectbox(
            "Page",
            options=list(range(1, total_pages + 1)),
            format_func=lambda x: f"Page {x} of {total_pages}",
            key="priority_page_picker",
            label_visibility="collapsed",
        )
        if int(selected_page) != current_page:
            st.session_state["priority_page"] = int(selected_page)
            current_page = int(selected_page)

    current_page = int(st.session_state.get("priority_page", 1))
    start = (current_page - 1) * page_size
    end = start + page_size
    current = filtered.iloc[start:end].copy()

    table = current[[
        "priority_rank","account_name","csm_owner_demo","industry","account_id","priority_score",
        "priority_band","health_score","health_band","evidence_status",
        "leading_driver","suggested_action","mrr_amount",
    ]].copy()

    table.loc[table["health_band"] == "Limited Evidence", "health_score"] = np.nan
    table["health_band"] = table["health_band"].map(health_label)
    table["evidence_status"] = table["evidence_status"].map(evidence_label)
    table["leading_driver"] = table["leading_driver"].map(driver_label)
    table.columns = [
        "Rank","Account","CSM owner","Industry","Account ID","Priority","Priority band",
        "Health score","Health status","Data status","Main reason","Suggested action","MRR proxy",
    ]

    priority_table_event = st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=575,
        key="priority_table_pick",
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Priority": st.column_config.NumberColumn(format="%.2f pts"),
            "Health score": st.column_config.NumberColumn(format="%.2f pts"),
            "MRR proxy": st.column_config.NumberColumn(format="%.0f", help="Monetary proxy; source does not specify the currency."),
            "Main reason": st.column_config.TextColumn(width="large"),
            "Suggested action": st.column_config.TextColumn(width="large"),
            "CSM owner": st.column_config.TextColumn(width="medium"),
            "Industry": st.column_config.TextColumn(width="small"),
        },
    )

    selected_rows = priority_table_event.selection.rows
    if selected_rows:
        chosen_index = int(selected_rows[0])
        if 0 <= chosen_index < len(current):
            chosen_row = current.iloc[chosen_index]
            if st.button(f"Open {chosen_row['account_name']} in Account 360 →", type="primary"):
                st.session_state["account_360_selected_id"] = str(chosen_row["account_id"])
                st.session_state["nav_page"] = "Account 360"
                st.session_state["scroll_to_top_pending"] = True
                st.rerun()

    d1, d2, d3 = st.columns([1,1,2])
    with d1:
        st.download_button(
            "Download this page",
            table.to_csv(index=False).encode("utf-8"),
            file_name="ravenstack_priority_page.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d2:
        full_export = filtered[[
            "priority_rank","account_name","csm_owner_demo","industry","account_id","priority_score",
            "priority_band","health_score","health_band","evidence_status",
            "leading_driver","suggested_action","mrr_amount",
        ]].copy()
        full_export.loc[full_export["health_band"] == "Limited Evidence", "health_score"] = np.nan
        full_export["health_band"] = full_export["health_band"].map(health_label)
        full_export["evidence_status"] = full_export["evidence_status"].map(evidence_label)
        full_export["leading_driver"] = full_export["leading_driver"].map(driver_label)
        full_export.columns = [
            "Rank","Account","CSM owner","Industry","Account ID","Priority","Priority band",
            "Health score","Health status","Data status","Main reason","Suggested action","MRR proxy",
        ]
        st.download_button(
            "Download filtered results",
            full_export.to_csv(index=False).encode("utf-8"),
            file_name="ravenstack_priority_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d3:
        shown_from = 0 if total_filtered == 0 else start + 1
        shown_to = min(end, total_filtered)
        st.markdown(
            f'<div class="caption-muted">Showing {shown_from:,}–{shown_to:,} of {total_filtered:,} matching accounts. '
            'Health is intentionally hidden when there is not enough reliable evidence.</div>',
            unsafe_allow_html=True,
        )

    st.caption("CSM owners are fictional demo assignments and do not affect scoring.")


# ==================================================================
# 3. Account 360
# ==================================================================
else:
    page_heading("Account 360", "Account review", "🔎")
    st.markdown(
        """
        <div class="section-note">
        Review one account's Health, Priority, supporting evidence and suggested next action.
        </div>
        """,
        unsafe_allow_html=True,
    )

    selector = data.sort_values("priority_rank").copy()
    selector["display"] = (
        "#" + selector["priority_rank"].astype(int).astype(str)
        + " · " + selector["account_name"].astype(str)
        + " · " + selector["account_id"].astype(str)
    )
    # Persist only a deliberately chosen account across page changes; filtering charts
    # must never silently change which account Account 360 shows.
    selected_id = st.session_state.get("account_360_selected_id")
    available_ids = selector["account_id"].astype(str).tolist()
    default_idx = available_ids.index(str(selected_id)) if str(selected_id) in available_ids else 0
    selected = st.selectbox("Select an account", selector["display"].tolist(),
                            index=default_idx, key="account_360_selection")
    account_id = selector.loc[selector["display"] == selected, "account_id"].iloc[0]
    st.session_state["account_360_selected_id"] = str(account_id)
    row = data.loc[data["account_id"] == account_id].iloc[0]

    st.caption(
        f"CSM owner: {row['csm_owner_demo']} (demo)  •  Industry: {row['industry']}  •  Account ID: {row['account_id']}"
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        metric_card(
            "Priority",
            f"{row['priority_score']:.2f} pts",
            "",
            "red" if row["priority_band"] == "High" else "amber" if row["priority_band"] == "Medium" else "green",
            icon="flag",
            pill=f"{row['priority_band']} · Rank #{int(row['priority_rank'])}",
        )
    with k2:
        if row["health_band"] == "Limited Evidence":
            metric_card("Health", "Needs Data Check", "Not enough evidence to calculate Health", "purple", compact=True, icon="heart")
        else:
            metric_card(
                "Health",
                f"{row['health_score']:.2f} pts",
                "",
                "red" if row["health_band"] == "Low" else "amber" if row["health_band"] == "Medium" else "green",
                icon="heart",
                pill=str(row["health_band"]),
            )
    with k3:
        metric_card("Data status", evidence_label(row["evidence_status"]), "enough data to calculate Health", "purple", compact=True, icon="database")
    with k4:
        metric_card("Engagement", f"{row['engagement_score']:.2f} pts", "recent product activity", "blue", icon="activity")
    with k5:
        metric_card("Support", f"{row['support_score']:.0f} pts", "higher means less pressure", "amber", icon="headset")
    with k6:
        metric_card("Commercial exposure", fmt_money(row["mrr_amount"]), "MRR proxy · currency unspecified", "cyan", icon="dollar")

    if row["health_band"] == "Limited Evidence":
        st.warning("Not enough evidence to calculate Health. Check the account data before drawing conclusions.")

    st.markdown("### Review summary")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(
            f"""
            <div class="decision-card driver">
                <div class="decision-header"><span class="decision-icon">⚠</span><div class="decision-kicker">Main review reason</div></div>
                <div class="decision-title">{html.escape(driver_label(row['leading_driver']))}</div>
                <div class="decision-detail">{html.escape(escalation_evidence(row))}</div>
            </div>
            """, unsafe_allow_html=True,
        )
    with d2:
        st.markdown(
            f"""
            <div class="decision-card action">
                <div class="decision-header"><span class="decision-icon">↗</span><div class="decision-kicker">Suggested next action</div></div>
                <div class="decision-title">{html.escape(display_action(row))}</div>
                <div class="decision-detail">Advisory only · CSM decides next steps</div>
            </div>
            """, unsafe_allow_html=True,
        )

    st.markdown("### Account evidence")
    tab1, tab2, tab3 = st.tabs(
        ["Usage", "Support Ticket History", "Satisfaction"]
    )

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Days since last activity", fmt_num(row["days_since_last_usage"]), "", "blue")
        with c2:
            metric_card("Active Days (Observed)", fmt_num(row["ui_observed_active_days"]), "", "cyan")
        with c3:
            metric_card("Features used (last 90 days)", fmt_num(row["distinct_features_90"]), "", "purple")
        with c4:
            metric_card("Usage events (last 90 days)", fmt_num(row["usage_events_90"]), "", "blue")
        st.caption("Observed days reflect recorded activity, not a 90-day adjusted equivalent.")

    with tab2:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            metric_card("Support tickets (90 days)", fmt_num(row["tickets_90"]), "", "blue", icon="ticket")
        with c2:
            metric_card("Open tickets", fmt_num(row["ui_open_tickets"]), "at snapshot", "amber", icon="inbox")
        with c3:
            metric_card("High-priority tickets", fmt_num(row["high_urgent_tickets_90"]), "", "amber", icon="alert")
        with c4:
            metric_card("Escalated tickets", fmt_num(row["escalated_tickets_90"]), "", "red", icon="up")
        with c5:
            metric_card("Avg. resolution time", fmt_hours(row["avg_resolution_time_hours_90"]), "", "cyan", compact=True, icon="clock")

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            csat_display = f"{float(row['avg_satisfaction']):.2f} / 5" if pd.notna(row["avg_satisfaction"]) else "—"
            metric_card("Average Satisfaction", csat_display, "", "amber", icon="star")
        with c2:
            metric_card("Satisfaction Responses", fmt_num(row["satisfaction_response_count"]), "", "blue", icon="message")
        st.caption("Satisfaction is based on available responses, which may not represent the full customer experience.")

    # Closed by default; each component offers its own rule/threshold trace.
    with st.expander("View scoring details", expanded=False):
        scoring_detail_cards(row)

