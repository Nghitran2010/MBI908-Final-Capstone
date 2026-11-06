from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import hashlib
import io

import numpy as np
import pandas as pd

SNAPSHOT = pd.Timestamp("2024-12-31")
SNAPSHOT_END = pd.Timestamp("2024-12-31 23:59:59")
WINDOW_START = SNAPSHOT - pd.Timedelta(days=89)

# Fixed dataset-relative constants. They are not recalculated from a new upload.
RESOLUTION_P75_HOURS = 51.25
MRR_P50 = 1010.50
MRR_P75 = 3184.00
MRR_P90 = 6766.00

EXPECTED_SCHEMAS = {
    "accounts": [
        "account_id","account_name","industry","country","signup_date",
        "referral_source","plan_tier","seats","is_trial","churn_flag"
    ],
    "subscriptions": [
        "subscription_id","account_id","start_date","end_date","plan_tier",
        "seats","mrr_amount","arr_amount","is_trial","upgrade_flag",
        "downgrade_flag","churn_flag","billing_frequency","auto_renew_flag"
    ],
    "feature_usage": [
        "usage_id","subscription_id","usage_date","feature_name","usage_count",
        "usage_duration_secs","error_count","is_beta_feature"
    ],
    "support_tickets": [
        "ticket_id","account_id","submitted_at","closed_at",
        "resolution_time_hours","priority","first_response_time_minutes",
        "satisfaction_score","escalation_flag"
    ],
    "churn_events": [
        "churn_event_id","account_id","churn_date","reason_code",
        "refund_amount_usd","preceding_upgrade_flag","preceding_downgrade_flag",
        "is_reactivation","feedback_text"
    ],
}

UI_COLUMNS = [
    "account_id","account_name","industry","priority_rank","priority_score",
    "priority_band","health_score","health_band","evidence_status",
    "leading_driver","suggested_action","mrr_amount","tenure_days",
    "days_since_last_usage","active_days_90_per_90_exposure",
    "distinct_features_90","usage_events_90","recency_score","activity_score",
    "engagement_score","tickets_90","high_urgent_tickets_90",
    "escalated_tickets_90","avg_resolution_time_hours_90","support_summary",
    "support_score","support_urgency_score","avg_satisfaction",
    "satisfaction_response_count","satisfaction_reliability",
    "satisfaction_modifier","health_explanation","evidence_note",
    "priority_explanation","priority_health_contribution",
    "priority_support_contribution","priority_commercial_contribution",
    "priority_evidence_contribution",
]

ACTIONS = {
    "Low recent engagement":
        "Review recent adoption and check possible usage barriers.",
    "Recent support pressure":
        "Review recent ticket severity and resolution pattern; coordinate follow-up if needed.",
    "Recent escalation history":
        "Review the recent escalation outcome and confirm whether further follow-up is needed.",
    "Low observed satisfaction":
        "Review available customer feedback and identify the issue requiring follow-up.",
    "Limited Evidence — Verify Data":
        "Verify account data completeness before interpreting Health.",
    "Limited Evidence — Insufficient Exposure":
        "Wait for or verify sufficient recent account evidence before making a confident Health judgement.",
    "No material adverse driver":
        "Continue routine monitoring; no specific adverse trigger is currently identified.",
}


def identify_table(df: pd.DataFrame) -> str | None:
    """Identify a RavenStack table by column names, independent of column order."""
    actual = set(df.columns)
    matches = [
        role for role, expected in EXPECTED_SCHEMAS.items()
        if actual == set(expected)
    ]
    return matches[0] if len(matches) == 1 else None


def _schema_diagnostic(df: pd.DataFrame) -> str:
    """Return a concise missing/unexpected-column diagnostic for the closest schema."""
    actual = set(df.columns)

    scored = []
    for role, expected in EXPECTED_SCHEMAS.items():
        expected_set = set(expected)
        missing = expected_set - actual
        extra = actual - expected_set
        overlap = len(actual & expected_set)
        distance = len(missing) + len(extra)
        scored.append((distance, -overlap, role, missing, extra))

    _, _, role, missing, extra = min(scored, key=lambda x: (x[0], x[1]))

    missing_text = ", ".join(sorted(missing)) if missing else "none"
    extra_text = ", ".join(sorted(extra)) if extra else "none"
    return (
        f"closest expected table is '{role}'. "
        f"Missing columns: {missing_text}. Unexpected columns: {extra_text}."
    )


def read_uploaded_files(
    uploaded_files
) -> Tuple[Dict[str, pd.DataFrame], List[str], List[str], str]:
    frames = {}
    errors = []
    warnings = []
    digest = hashlib.sha256()

    for uploaded in uploaded_files:
        raw = uploaded.getvalue()
        digest.update(uploaded.name.encode("utf-8"))
        digest.update(raw)
        try:
            frame = pd.read_csv(io.BytesIO(raw))
        except Exception as exc:
            errors.append(f"{uploaded.name}: could not read CSV ({exc}).")
            continue

        role = identify_table(frame)
        if role is None:
            errors.append(
                f"{uploaded.name}: schema does not match a required RavenStack table; "
                + _schema_diagnostic(frame)
            )
            continue

        if role in frames:
            errors.append(
                f"{uploaded.name}: duplicate table type '{role}'. Upload one file per table."
            )
            continue

        expected_order = EXPECTED_SCHEMAS[role]
        if list(frame.columns) != expected_order:
            warnings.append(
                f"{uploaded.name}: column order differed from the canonical '{role}' schema; "
                "columns were reordered by name before validation."
            )

        # Normalise to canonical order so downstream validation remains deterministic.
        frames[role] = frame[expected_order].copy()

    missing = [k for k in EXPECTED_SCHEMAS if k not in frames]
    if missing:
        errors.append("Missing required table(s): " + ", ".join(missing) + ".")

    return frames, errors, warnings, digest.hexdigest()


def _parse_dates(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    x = {k: v.copy() for k, v in frames.items()}
    x["accounts"]["signup_date"] = pd.to_datetime(
        x["accounts"]["signup_date"], errors="coerce"
    )
    for c in ["start_date","end_date"]:
        x["subscriptions"][c] = pd.to_datetime(
            x["subscriptions"][c], errors="coerce"
        )
    x["feature_usage"]["usage_date"] = pd.to_datetime(
        x["feature_usage"]["usage_date"], errors="coerce"
    )
    for c in ["submitted_at","closed_at"]:
        x["support_tickets"][c] = pd.to_datetime(
            x["support_tickets"][c], errors="coerce"
        )
    x["churn_events"]["churn_date"] = pd.to_datetime(
        x["churn_events"]["churn_date"], errors="coerce"
    )
    return x


def validate_frames(frames: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    errors = []
    warnings = []
    rows = []

    required_roles = list(EXPECTED_SCHEMAS)
    for role in required_roles:
        if role not in frames:
            errors.append(f"Missing required table: {role}")
            continue
        exact_schema = list(frames[role].columns) == EXPECTED_SCHEMAS[role]
        rows.append({
            "Check": f"{role}: schema",
            "Result": "PASS" if exact_schema else "FAIL",
            "Evidence": f"{len(frames[role]):,} rows × {len(frames[role].columns)} columns",
        })
        if not exact_schema:
            errors.append(f"{role}: schema mismatch")

    if errors:
        return pd.DataFrame(rows), errors, warnings

    x = _parse_dates(frames)
    accounts = x["accounts"]
    subs = x["subscriptions"]
    usage = x["feature_usage"]
    tickets = x["support_tickets"]
    churn = x["churn_events"]

    date_checks = {
        "accounts.signup_date": accounts["signup_date"].isna().sum(),
        "subscriptions.start_date": subs["start_date"].isna().sum(),
        "feature_usage.usage_date": usage["usage_date"].isna().sum(),
        "support_tickets.submitted_at": tickets["submitted_at"].isna().sum(),
        "churn_events.churn_date": churn["churn_date"].isna().sum(),
    }
    for name, invalid in date_checks.items():
        rows.append({
            "Check": f"{name}: parse",
            "Result": "PASS" if invalid == 0 else "FAIL",
            "Evidence": f"{int(invalid)} invalid/missing required date(s)",
        })
        if invalid:
            errors.append(f"{name}: {int(invalid)} invalid/missing required date(s)")

    pk_checks = {
        "accounts.account_id": accounts["account_id"].duplicated().sum(),
        "subscriptions.subscription_id": subs["subscription_id"].duplicated().sum(),
        "support_tickets.ticket_id": tickets["ticket_id"].duplicated().sum(),
        "churn_events.churn_event_id": churn["churn_event_id"].duplicated().sum(),
        "feature_usage subscription_id+usage_id":
            usage.duplicated(["subscription_id","usage_id"]).sum(),
    }
    for name, dups in pk_checks.items():
        rows.append({
            "Check": f"{name}: uniqueness",
            "Result": "PASS" if dups == 0 else "FAIL",
            "Evidence": f"{int(dups)} duplicate key row(s)",
        })
        if dups:
            errors.append(f"{name}: duplicate key rows detected")

    fk_checks = {
        "subscriptions → accounts":
            bool(subs["account_id"].isin(accounts["account_id"]).all()),
        "feature_usage → subscriptions":
            bool(usage["subscription_id"].isin(subs["subscription_id"]).all()),
        "support_tickets → accounts":
            bool(tickets["account_id"].isin(accounts["account_id"]).all()),
        "churn_events → accounts":
            bool(churn["account_id"].isin(accounts["account_id"]).all()),
    }
    for name, ok in fk_checks.items():
        rows.append({
            "Check": f"{name}: foreign key",
            "Result": "PASS" if ok else "FAIL",
            "Evidence": "No orphan rows" if ok else "Orphan rows detected",
        })
        if not ok:
            errors.append(f"{name}: foreign-key failure")

    # Every account needs one deterministic commercial proxy at or before the snapshot.
    pre_snapshot_sub_accounts = set(
        subs.loc[subs["start_date"].le(SNAPSHOT), "account_id"]
    )
    missing_proxy = sorted(
        set(accounts["account_id"]) - pre_snapshot_sub_accounts
    )
    proxy_ok = len(missing_proxy) == 0
    proxy_evidence = (
        "All accounts have at least one subscription on/before snapshot"
        if proxy_ok
        else (
            f"{len(missing_proxy)} account(s) have no subscription on/before "
            f"{SNAPSHOT.date()}; example: {', '.join(missing_proxy[:5])}"
        )
    )
    rows.append({
        "Check": "Commercial proxy coverage at snapshot",
        "Result": "PASS" if proxy_ok else "FAIL",
        "Evidence": proxy_evidence,
    })
    if not proxy_ok:
        errors.append(
            "Cannot build deterministic commercial proxy: " + proxy_evidence + "."
        )

    repeated_ids = usage.groupby("usage_id").size()
    repeated_identifier_count = int((repeated_ids > 1).sum())
    repeated_row_count = int(repeated_ids[repeated_ids > 1].sum())
    rows.append({
        "Check": "feature_usage.usage_id: repeated identifiers",
        "Result": "WARN" if repeated_identifier_count else "PASS",
        "Evidence":
            f"{repeated_identifier_count} repeated identifier(s) across {repeated_row_count} row(s); composite key remains authoritative",
    })
    if repeated_identifier_count:
        warnings.append(
            f"usage_id repeats {repeated_identifier_count} time(s) as an identifier; "
            "the project uses subscription_id + usage_id."
        )

    full_dups = sum(int(v.duplicated().sum()) for v in frames.values())
    rows.append({
        "Check": "Full-row duplicates",
        "Result": "PASS" if full_dups == 0 else "WARN",
        "Evidence": f"{full_dups} full-row duplicate(s) across five tables",
    })
    if full_dups:
        warnings.append(f"{full_dups} full-row duplicate(s) detected.")

    return pd.DataFrame(rows), errors, warnings


def build_prescore(frames: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, dict]:
    x = _parse_dates(frames)
    accounts = x["accounts"]
    subs = x["subscriptions"]
    usage = x["feature_usage"]
    tickets = x["support_tickets"]

    u = (
        usage
        .merge(
            subs[["subscription_id","account_id","start_date","end_date"]],
            on="subscription_id", how="left", validate="m:1"
        )
        .merge(
            accounts[["account_id","signup_date"]],
            on="account_id", how="left", validate="m:1"
        )
    )
    u_valid = u[
        (u["usage_date"] >= u["signup_date"])
        & (u["usage_date"] <= SNAPSHOT)
    ].copy()

    t = tickets.merge(
        accounts[["account_id","signup_date"]],
        on="account_id", how="left", validate="m:1"
    )
    t_valid = t[
        (t["submitted_at"] >= t["signup_date"])
        & (t["submitted_at"] <= SNAPSHOT)
    ].copy()

    sp = subs[subs["start_date"].le(SNAPSHOT)].copy()
    sp = sp.sort_values(["account_id","start_date","subscription_id"])
    latest = sp.groupby("account_id", as_index=False).tail(1)

    missing_proxy = set(accounts["account_id"]) - set(latest["account_id"])
    if missing_proxy:
        raise ValueError(
            f"No pre-snapshot subscription proxy for {len(missing_proxy)} account(s)."
        )

    acct = accounts.copy()
    acct["tenure_days"] = (SNAPSHOT - acct["signup_date"]).dt.days
    acct["exposure_days"] = acct["tenure_days"].clip(lower=1, upper=90)

    u90 = u_valid[
        (u_valid["usage_date"] >= WINDOW_START)
        & (u_valid["usage_date"] <= SNAPSHOT)
    ].copy()

    u_all_agg = u_valid.groupby("account_id").agg(
        valid_usage_rows=("usage_id","size"),
        first_valid_usage_date=("usage_date","min"),
        last_valid_usage_date=("usage_date","max"),
        total_usage_count_valid=("usage_count","sum"),
        total_usage_duration_secs_valid=("usage_duration_secs","sum"),
        total_usage_errors_valid=("error_count","sum"),
    ).reset_index()

    u90_agg = u90.groupby("account_id").agg(
        usage_events_90=("usage_id","size"),
        active_days_90=("usage_date","nunique"),
        distinct_features_90=("feature_name","nunique"),
        usage_count_90=("usage_count","sum"),
        usage_duration_secs_90=("usage_duration_secs","sum"),
        usage_error_count_90=("error_count","sum"),
        last_usage_date_90=("usage_date","max"),
    ).reset_index()

    t90 = t_valid[
        (t_valid["submitted_at"] >= WINDOW_START)
        & (t_valid["submitted_at"] <= SNAPSHOT)
    ].copy()

    t_all_agg = t_valid.groupby("account_id").agg(
        valid_ticket_count=("ticket_id","size"),
        high_urgent_ticket_count=("priority", lambda s:s.isin(["high","urgent"]).sum()),
        escalated_ticket_count=("escalation_flag","sum"),
        avg_resolution_time_hours=("resolution_time_hours","mean"),
        max_resolution_time_hours=("resolution_time_hours","max"),
        satisfaction_response_count=("satisfaction_score", lambda s:s.notna().sum()),
        avg_satisfaction=("satisfaction_score","mean"),
        last_ticket_date=("submitted_at","max"),
    ).reset_index()

    sat_last = (
        t_valid[t_valid["satisfaction_score"].notna()]
        .groupby("account_id")["submitted_at"].max()
        .rename("last_satisfaction_date").reset_index()
    )
    t_all_agg = t_all_agg.merge(sat_last, on="account_id", how="left")
    t_all_agg["satisfaction_response_coverage"] = (
        t_all_agg["satisfaction_response_count"] / t_all_agg["valid_ticket_count"]
    )

    t90_agg = t90.groupby("account_id").agg(
        tickets_90=("ticket_id","size"),
        high_urgent_tickets_90=("priority", lambda s:s.isin(["high","urgent"]).sum()),
        escalated_tickets_90=("escalation_flag","sum"),
        avg_resolution_time_hours_90=("resolution_time_hours","mean"),
        max_resolution_time_hours_90=("resolution_time_hours","max"),
        satisfaction_responses_90=("satisfaction_score", lambda s:s.notna().sum()),
        avg_satisfaction_90=("satisfaction_score","mean"),
    ).reset_index()
    t90_agg["satisfaction_coverage_90"] = (
        t90_agg["satisfaction_responses_90"] / t90_agg["tickets_90"]
    )

    commercial = latest[[
        "account_id","subscription_id","start_date","plan_tier","seats",
        "mrr_amount","arr_amount","is_trial","billing_frequency","auto_renew_flag",
    ]].copy()

    account_level = (
        acct
        .merge(commercial, on="account_id", how="left", suffixes=("","_commercial"))
        .merge(u_all_agg, on="account_id", how="left")
        .merge(u90_agg, on="account_id", how="left")
        .merge(t_all_agg, on="account_id", how="left")
        .merge(t90_agg, on="account_id", how="left")
    )

    count_fields = [
        "valid_usage_rows","total_usage_count_valid","total_usage_duration_secs_valid",
        "total_usage_errors_valid","usage_events_90","active_days_90",
        "distinct_features_90","usage_count_90","usage_duration_secs_90",
        "usage_error_count_90","valid_ticket_count","high_urgent_ticket_count",
        "escalated_ticket_count","satisfaction_response_count","tickets_90",
        "high_urgent_tickets_90","escalated_tickets_90","satisfaction_responses_90",
    ]
    for c in count_fields:
        account_level[c] = account_level[c].fillna(0)

    account_level["has_valid_usage"] = account_level["valid_usage_rows"] > 0
    account_level["has_valid_support"] = account_level["valid_ticket_count"] > 0
    account_level["has_satisfaction"] = account_level["satisfaction_response_count"] > 0
    account_level["no_valid_usage_or_support"] = (
        ~account_level["has_valid_usage"] & ~account_level["has_valid_support"]
    )

    account_level["days_since_last_usage"] = (
        SNAPSHOT - account_level["last_valid_usage_date"]
    ).dt.days
    account_level["days_since_last_ticket"] = (
        SNAPSHOT - account_level["last_ticket_date"]
    ).dt.days
    account_level["days_since_last_satisfaction"] = (
        SNAPSHOT - account_level["last_satisfaction_date"]
    ).dt.days

    for c in [
        "usage_events_90","active_days_90","distinct_features_90",
        "usage_count_90","usage_duration_secs_90","usage_error_count_90",
    ]:
        account_level[c+"_per_90_exposure"] = (
            account_level[c] / account_level["exposure_days"] * 90
        )

    account_level["insufficient_exposure_14"] = account_level["tenure_days"] < 14
    account_level["insufficient_exposure_30"] = account_level["tenure_days"] < 30

    stats = {
        "input_accounts": len(accounts),
        "valid_usage_rows": len(u_valid),
        "valid_usage_accounts": int(u_valid["account_id"].nunique()),
        "excluded_usage_rows": int(len(usage) - len(u_valid)),
        "valid_support_rows": len(t_valid),
        "valid_support_accounts": int(t_valid["account_id"].nunique()),
        "excluded_support_rows": int(len(tickets) - len(t_valid)),
        "satisfaction_responses": int(t_valid["satisfaction_score"].notna().sum()),
        "satisfaction_accounts": int(
            t_valid.loc[t_valid["satisfaction_score"].notna(),"account_id"].nunique()
        ),
        "accounts_no_valid_usage_support": int(
            account_level["no_valid_usage_or_support"].sum()
        ),
    }
    return account_level, stats


def score_prescore(pre: pd.DataFrame) -> pd.DataFrame:
    x = pre.copy()

    def recency(v):
        if pd.isna(v): return 0.0
        if v <= 14: return 100.0
        if v <= 30: return 65.0
        if v <= 60: return 30.0
        return 0.0

    def activity(v):
        if pd.isna(v): return 0.0
        if v < 4: return 25.0
        if v < 8: return 60.0
        return 100.0

    x["recency_score"] = x["days_since_last_usage"].apply(recency)
    x["activity_score"] = x["active_days_90_per_90_exposure"].apply(activity)
    x["engagement_score"] = (
        0.60*x["recency_score"] + 0.40*x["activity_score"]
    ).round(2)

    def support_score(row):
        if row["tickets_90"] <= 0:
            return 100.0
        if row["escalated_tickets_90"] >= 1 or row["high_urgent_tickets_90"] >= 2:
            return 25.0
        slow = (
            pd.notna(row["avg_resolution_time_hours_90"])
            and row["avg_resolution_time_hours_90"] > RESOLUTION_P75_HOURS
        )
        if row["high_urgent_tickets_90"] >= 1 or slow:
            return 50.0
        return 75.0

    x["support_score"] = x.apply(support_score, axis=1)

    coverage = x["satisfaction_response_coverage"].fillna(0).clip(0, 1)
    count_factor = np.minimum(
        1.0, x["satisfaction_response_count"].fillna(0) / 2.0
    )
    x["satisfaction_reliability"] = (coverage * count_factor).round(4)
    x["satisfaction_modifier"] = (
        (x["avg_satisfaction"].fillna(4.0) - 4.0)
        * 5.0
        * x["satisfaction_reliability"]
    ).clip(-5, 5).round(2)

    x["health_score"] = (
        0.60*x["engagement_score"]
        + 0.40*x["support_score"]
        + x["satisfaction_modifier"]
    ).clip(0, 100).round(2)

    x["evidence_status"] = np.where(
        x["tenure_days"] < 30,
        "Insufficient Exposure",
        np.where(~x["has_valid_usage"].astype(bool), "Verify Data", "Scoreable"),
    )
    x["health_band"] = np.where(
        x["evidence_status"] != "Scoreable",
        "Limited Evidence",
        np.select(
            [x["health_score"] < 50, x["health_score"] < 70],
            ["Low","Medium"],
            default="Healthy",
        ),
    )

    def urgency(row):
        if row["escalated_tickets_90"] >= 1:
            return 100.0
        if row["high_urgent_tickets_90"] >= 2:
            return 85.0
        slow = (
            pd.notna(row["avg_resolution_time_hours_90"])
            and row["avg_resolution_time_hours_90"] > RESOLUTION_P75_HOURS
        )
        if row["high_urgent_tickets_90"] >= 1 or slow:
            return 60.0
        if row["tickets_90"] >= 1:
            return 30.0
        return 0.0

    x["support_urgency_score"] = x.apply(urgency, axis=1)

    def commercial(v):
        if pd.isna(v) or v <= 0: return 0.0
        if v <= MRR_P50: return 25.0
        if v <= MRR_P75: return 50.0
        if v <= MRR_P90: return 75.0
        return 100.0

    x["commercial_score"] = x["mrr_amount"].apply(commercial)

    x["health_concern_score"] = np.where(
        x["evidence_status"] == "Scoreable",
        100.0 - x["health_score"],
        50.0,
    ).round(2)
    x["evidence_concern_score"] = np.where(
        x["evidence_status"] == "Scoreable", 0.0, 100.0
    )

    x["priority_score"] = (
        0.50*x["health_concern_score"]
        + 0.25*x["support_urgency_score"]
        + 0.15*x["commercial_score"]
        + 0.10*x["evidence_concern_score"]
    ).round(2)

    x["priority_band"] = np.select(
        [x["priority_score"] >= 50, x["priority_score"] >= 30],
        ["High","Medium"],
        default="Low",
    )

    def driver(row):
        if row["evidence_status"] == "Insufficient Exposure":
            return "Limited Evidence — Insufficient Exposure"
        if row["evidence_status"] == "Verify Data":
            return "Limited Evidence — Verify Data"
        if row["escalated_tickets_90"] >= 1:
            return "Recent escalation history"

        adverse = {
            "Low recent engagement": 100.0 - row["engagement_score"],
            "Recent support pressure": 100.0 - row["support_score"],
            "Low observed satisfaction":
                0.0 if pd.isna(row["avg_satisfaction"]) else
                max(0.0, 4.0-row["avg_satisfaction"])
                * 100.0 * row["satisfaction_reliability"],
        }
        highest = max(adverse.values())
        if highest <= 0:
            return "No material adverse driver"
        for label in [
            "Low recent engagement","Recent support pressure",
            "Low observed satisfaction",
        ]:
            if abs(adverse[label]-highest) < 1e-9:
                return label

    x["leading_driver"] = x.apply(driver, axis=1)
    x["suggested_action"] = x["leading_driver"].map(ACTIONS)

    x = x.sort_values(
        ["priority_score","account_id"], ascending=[False,True]
    ).reset_index(drop=True)
    x["priority_rank"] = np.arange(1, len(x)+1)

    return x


def build_app_ready(scored: pd.DataFrame) -> pd.DataFrame:
    x = scored.copy()

    x["priority_health_contribution"] = (
        0.50*x["health_concern_score"]
    ).round(3)
    x["priority_support_contribution"] = (
        0.25*x["support_urgency_score"]
    ).round(3)
    x["priority_commercial_contribution"] = (
        0.15*x["commercial_score"]
    ).round(3)
    x["priority_evidence_contribution"] = (
        0.10*x["evidence_concern_score"]
    ).round(3)

    def support_summary(row):
        tickets = int(row["tickets_90"]) if pd.notna(row["tickets_90"]) else 0
        if tickets <= 0:
            return "No valid support ticket recorded in the recent 90-day window."

        parts = [f"{tickets} recent ticket(s)"]
        hu = int(row["high_urgent_tickets_90"]) if pd.notna(row["high_urgent_tickets_90"]) else 0
        esc = int(row["escalated_tickets_90"]) if pd.notna(row["escalated_tickets_90"]) else 0
        if hu > 0:
            parts.append(f"{hu} high/urgent")
        if esc > 0:
            parts.append(f"{esc} with escalation history")
        if pd.notna(row["avg_resolution_time_hours_90"]):
            parts.append(f"avg. resolution {row['avg_resolution_time_hours_90']:.1f}h")
        return "; ".join(parts) + "."

    x["support_summary"] = x.apply(support_summary, axis=1)

    def evidence_note(row):
        if row["evidence_status"] == "Scoreable":
            return "Evidence status: Scoreable under the final scoring rules."
        if row["evidence_status"] == "Insufficient Exposure":
            return (
                f"Limited Evidence: account tenure is {int(row['tenure_days'])} days, "
                "below the 30-day minimum exposure rule."
            )
        if row["evidence_status"] == "Verify Data":
            return (
                "Limited Evidence: the account is old enough to score but has no valid usage "
                "evidence under the project time rule."
            )
        return f"Evidence status: {row['evidence_status']}."

    x["evidence_note"] = x.apply(evidence_note, axis=1)

    def health_explanation(row):
        if row["health_band"] == "Limited Evidence":
            return (
                "Health is not presented as a confident band because the account is "
                f"{row['evidence_status']}."
            )
        e = 0.60*row["engagement_score"]
        s = 0.40*row["support_score"]
        raw = e+s+row["satisfaction_modifier"]
        if raw > 100.0 + 1e-12 or raw < -1e-12:
            return (
                f"Raw Health {raw:.2f} = Engagement contribution {e:.2f} + "
                f"Support contribution {s:.2f} + Satisfaction modifier "
                f"{row['satisfaction_modifier']:+.2f}; clipped to "
                f"{row['health_score']:.2f} ({row['health_band']})."
            )
        return (
            f"Health {row['health_score']:.2f} ({row['health_band']}) = "
            f"Engagement contribution {e:.2f} + Support contribution {s:.2f} + "
            f"Satisfaction modifier {row['satisfaction_modifier']:+.2f}."
        )

    x["health_explanation"] = x.apply(health_explanation, axis=1)

    def priority_explanation(row):
        h = row["priority_health_contribution"]
        s = row["priority_support_contribution"]
        c = row["priority_commercial_contribution"]
        e = row["priority_evidence_contribution"]
        total = h+s+c+e
        return (
            f"Priority {row['priority_score']:.2f} ({row['priority_band']}) = "
            f"round({h:.3f} Health concern + {s:.3f} Recent support urgency + "
            f"{c:.3f} Commercial exposure + {e:.3f} Evidence concern = "
            f"{total:.3f}, 2 d.p.)."
        )

    x["priority_explanation"] = x.apply(priority_explanation, axis=1)

    out = x[UI_COLUMNS].copy()
    return out


def process_raw_frames(frames: Dict[str, pd.DataFrame]):
    validation, errors, warnings = validate_frames(frames)
    if errors:
        return {
            "validation": validation,
            "errors": errors,
            "warnings": warnings,
            "prescore": None,
            "scored": None,
            "app_ready": None,
            "stats": None,
        }

    # Safety net: user-facing upload errors should return through the normal
    # validation channel rather than exposing a raw Streamlit traceback.
    try:
        prescore, stats = build_prescore(frames)
        scored = score_prescore(prescore)
        app_ready = build_app_ready(scored)
    except Exception as exc:
        return {
            "validation": validation,
            "errors": [
                "Processing could not be completed after validation: "
                f"{type(exc).__name__}: {exc}"
            ],
            "warnings": warnings,
            "prescore": None,
            "scored": None,
            "app_ready": None,
            "stats": None,
        }

    return {
        "validation": validation,
        "errors": [],
        "warnings": warnings,
        "prescore": prescore,
        "scored": scored,
        "app_ready": app_ready,
        "stats": stats,
    }
