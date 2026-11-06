# Explainable Customer Account Health and Prioritisation

**MBI908 Master's Capstone | Python | pandas | Streamlit**

This repository contains the runnable prototype, synthetic sample data and technical evaluation evidence for the MBI908 capstone *Explainable Customer Account Health and Account Prioritisation Prototype*. The application supports B2B SaaS account review using explainable, rule-based Health and Priority scores, leading review reasons, and advisory next actions.

The demonstration uses **500 synthetic RavenStack customer accounts** and a fixed **31 December 2024** snapshot. This is an academic decision-support prototype, not a churn prediction model or a validated production system. No real-world retention improvement, practitioner usability, or commercial impact is claimed. CSM judgement remains necessary.

## 1. Application overview

- **Portfolio Dashboard:** Account distributions, health and priority summaries, evidence coverage, Top 10 review queue, CSM overview and leading review reasons.
- **Priority List:** Ranked account review queue with filters, pagination and CSV exports. Account search is **not** implemented on this page.
- **Account 360:** Search/select an account, inspect its evidence, leading reason, advisory next action, and Health/Priority score breakdown.
- **Data upload:** Use the included processed demo dataset, upload five original RavenStack CSVs individually, or upload **one ZIP containing those five CSVs**. Input schema and required dates are checked before processing.

Health represents account-condition evidence; Priority represents review order. A Needs Data Check status makes insufficient evidence visible rather than presenting a confident Health score. The application does not automatically perform customer outreach.

## 2. Repository file guide

All paths below are relative to the repository root. **Each of the 37 files in the submission ZIP is listed once**. The directory headings are organisational only.

### Root files (3)

| File | Description |
| --- | --- |
| `README.md` | This file: project overview, full inventory, setup, evaluation instructions and limitations. |
| `.gitignore` | Prevents local Python environments, caches, macOS metadata and local secret files from being added to Git. |
| `MBI908_Prototype.code-workspace` | Opens `MBI908_Prototype_Evidence` as the VS Code workspace folder. |

### VS Code configuration (3)

Location: `MBI908_Prototype_Evidence/.vscode/`

| File | Description |
| --- | --- |
| `launch.json` | VS Code launch/debug configuration for the Streamlit app using the `streamlit` Python module. |
| `settings.json` | Configures the workspace Python interpreter at `MBI908_Prototype_Evidence/.venv/bin/python`, environment activation and cache-file hiding. |
| `tasks.json` | VS Code tasks to create the environment, install dependencies, run Streamlit and execute project checks. |

### Prototype and sample data (10)

Location: `MBI908_Prototype_Evidence/prototype/`

| File | Description |
| --- | --- |
| `app.py` | Streamlit interface and interactions for Dashboard, Priority List, Account 360, file upload, filtering, exports and score explanations. |
| `pipeline.py` | RavenStack file/schema checks, date handling, aggregation, Health/Priority scoring, review reasons and advisory actions. |
| `requirements.txt` | Python package requirements for Streamlit, pandas, NumPy and Altair. |
| `processed_demo_data.csv` | Processed and ranked 500-account dataset loaded by the Built-in demo mode. |
| `RUN_APP.command` | Optional macOS shell launcher that creates a local environment in `prototype/.venv`, installs dependencies and starts Streamlit. |
| `sample_raw_data/ravenstack_accounts(3).csv` | 500 synthetic account records with account identifiers, names, industry and related account attributes. |
| `sample_raw_data/ravenstack_subscriptions(3).csv` | Subscription histories, plans, dates, seats, recurring-revenue values and subscription indicators. |
| `sample_raw_data/ravenstack_feature_usage(3).csv` | Dated product-feature usage events associated with subscriptions. |
| `sample_raw_data/ravenstack_support_tickets(3).csv` | Support tickets, submission/closure timestamps, service urgency, escalation and satisfaction fields. |
| `sample_raw_data/ravenstack_churn_events(3).csv` | Churn and reactivation event history retained in the source dataset for data handling and outcome-exclusion checks; not used as a predictor in Health/Priority scoring. |

### Processing and scoring evidence (10)

Location: `MBI908_Prototype_Evidence/evidence/01_processing_and_scoring/`

| File | Description |
| --- | --- |
| `run_pipeline_validation.py` | Re-runs raw CSV ingestion, processing and scoring checks; generates recorded validation artefacts. |
| `run_additional_validation.py` | Reproduces independent baseline score comparisons and controlled business-plausibility scenarios. |
| `pipeline_checks.csv` | Individual validation checks with PASS/FAIL status and supporting details. |
| `app_ready_validation_results.csv` | Reconstructed app-ready account output used to check the published demo dataset. |
| `scoring_validation_results.csv` | Detailed account-level intermediate indicators and scores used for independent checking and sensitivity reruns. |
| `full_38_column_reconciliation.csv` | Column-by-column comparison summary for the 38-field account output reconciliation. |
| `independent_baseline_comparison.csv` | Account-level comparison results for independently recalculated Health/Priority outputs and bands. |
| `business_plausibility_tests.csv` | Inputs, expectations and observed results of controlled scoring scenarios. |
| `report_claims_reconciliation.csv` | Recorded cross-checks of selected numerical/report claims against generated outputs. |
| `summary.json` | Machine-readable summary of the pipeline validation checks and PASS/FAIL totals. |

### Sensitivity evidence (3)

Location: `MBI908_Prototype_Evidence/evidence/02_sensitivity/`

| File | Description |
| --- | --- |
| `rerun_24.py` | Recalculates 24 sensitivity scenarios for scoring parameters and verifies results against the saved scenario output. |
| `sensitivity_baseline_results.csv` | Stored scenario-level band-retention and Top-10 overlap results, including acceptance-rule indicators. |
| `sensitivity_recalculation_comparison.csv` | Side-by-side stored and recalculated scenario outcomes, including match indicators. |

### Interface evidence (7)

Location: `MBI908_Prototype_Evidence/evidence/03_interface/`

| File | Description |
| --- | --- |
| `UI_TEST_REGISTER.csv` | Browser test register C01-C22 with requirement, input, expected result, observed result, status and evidence reference. |
| `UI_Browser_Test_Evidence.pdf` | 17 pages of screenshots supplied during manual application checks, with test IDs/captions. |
| `Evidence_Index.txt` | Maps PDF pages to test IDs and screenshot sources; records qualifications in the available evidence. |
| `run_static_review.py` | Static source-code inspection for application features and consistency checks; not a browser interaction test. |
| `source_code_checks.csv` | Results of static source-code checks against the current `app.py`. |
| `priority_page_export.csv` | C17 evidence: CSV exported for the current 10-row Priority List page under the chosen filters. |
| `priority_filtered_export.csv` | C17 evidence: CSV exported for all 16 matching accounts under the same filters. |

### Requirements mapping (1)

| File | Description |
| --- | --- |
| `MBI908_Prototype_Evidence/evidence/REQUIREMENTS_EVIDENCE_CROSSWALK_T01_T30.csv` | Crosswalk mapping technical requirements/test areas T01-T30 to expected results and available evidence, with verification status. |

## 3. Run the application locally

### Option A: VS Code workspace (macOS/Linux)

1. Open `MBI908_Prototype.code-workspace` in VS Code.
2. Open a terminal at the **repository root** and execute:

```bash
cd MBI908_Prototype_Evidence
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r prototype/requirements.txt
python -m streamlit run prototype/app.py
```

This places `.venv` where the included VS Code settings and tasks expect it. The local browser typically opens at `http://localhost:8501`. Press `Ctrl+C` to stop.

### Option B: macOS convenience launcher

The optional `MBI908_Prototype_Evidence/prototype/RUN_APP.command` uses a **separate** environment at `prototype/.venv`. If macOS reports that the file lacks execution privileges, give that local file executable permission with `chmod +x` before opening it. The VS Code/terminal option above does not require this launcher.

### Streamlit Community Cloud

Select **Python 3.12** and use the repository-relative main file path:

```text
MBI908_Prototype_Evidence/prototype/app.py
```

`requirements.txt` is in the same `prototype/` directory. Verify the deployed application and upload controls after deployment; successful offline checks alone do not prove browser or cloud functionality.

## 4. Data input and validation

Choose **Built-in demo** to explore the supplied 500-account processed dataset without uploading any files.

For raw-data processing, select **Upload RavenStack CSVs** and either:

1. Choose **One ZIP containing 5 CSV files** and upload an archive containing the five CSVs from `prototype/sample_raw_data/` (without additional CSV or unrelated files); or
2. Choose **Five individual CSV files** and upload all five files together.

Select **Check files and build the portfolio view**. The interface reports validation failures, including missing tables, invalid required columns and invalid required dates. A successful upload enables the same account-review workflow. This is a RavenStack-schema-specific importer, not a general-purpose CSV import tool.

## 5. Reproduce the technical checks

Once dependencies have been installed, open a terminal in `MBI908_Prototype_Evidence` and run:

```bash
python -B evidence/01_processing_and_scoring/run_pipeline_validation.py
python -B evidence/01_processing_and_scoring/run_additional_validation.py
python -B evidence/02_sensitivity/rerun_24.py
python -B evidence/03_interface/run_static_review.py
```

The first two scripts cover processing, reconciliation and controlled business scenarios. The third checks the stored 24 sensitivity results. The fourth examines the application source and refreshes static-check results. Some scripts overwrite generated evidence CSVs, so retain the supplied submission ZIP as an unchanged reference copy before rerunning them.

**Evidence interpretation:** `UI_TEST_REGISTER.csv` records C01-C22. `UI_Browser_Test_Evidence.pdf` contains screenshots for those checks that have direct screen evidence, not a separate screenshot for every action. C12's return-to-first-page action was user-confirmed but not separately pictured. C17's evidence consists of two exported CSVs. C21's screenshot compares one sampled account (`Company_176`) across demo/upload modes and does not establish full-portfolio equivalence by itself. Source checks and numerical reconciliation do not demonstrate real-world business benefit.

## 6. Research scope and limitations

See the accompanying MBI908 Assessment 3 report for the full research methodology, scoring rationale, evaluation criteria, limitations and AI-use declaration. In particular, the synthetic-data evaluation assesses implementation consistency, traceability and sensitivity under tested assumptions. It does **not** establish practitioner usefulness, cultural validity, causal effectiveness or real customer-retention impact. There were no external CSM interviews or practitioner usability studies in this capstone.
