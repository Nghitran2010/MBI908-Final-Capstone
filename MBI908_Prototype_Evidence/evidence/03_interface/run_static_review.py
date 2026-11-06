from pathlib import Path
import ast, csv
ROOT=Path(__file__).resolve().parent
p=ROOT.parents[1]/'prototype/app.py'
s=p.read_text()
a=ast.parse(s)
checks = [
 ('App parses as valid Python',True,'AST parse completed'),
 ('App does not display development version labels','Scoring version:' not in s and 'App version:' not in s,''),
 ('Imports shared pipeline','from pipeline import read_uploaded_files, process_raw_frames' in s,''),
 ('Built-in and upload modes',all(x in s for x in ['"Built-in demo"','"Upload RavenStack CSVs"']),''),
 ('Upload accepts 5 CSV files','accept_multiple_files=True' in s and 'Upload the 5 RavenStack CSV files' in s,''),
 ('ZIP upload option and parser available',all(x in s for x in ['def unpack_ravenstack_zip(', 'One ZIP containing 5 CSV files', 'type=["zip"]']),''),
 ('Upload validates before building',s.index('frames, read_errors, read_warnings, source_hash = read_uploaded_files(uploaded)') < s.index('processed = process_raw_frames(frames)'),''),
 ('Upload errors halt display','if read_errors:' in s and 'if processed["errors"]:' in s,''),
 ('Three nav pages',all(x in s for x in ['("Portfolio Dashboard",','("Priority List",','("Account 360",']),''),
 ('Portfolio status and evidence summary','Evidence coverage' in s and 'Leading review reasons' in s,''),
 ('Filter controls',all(x in s for x in ['"Priority", ["All"','"Health", h_display_options','"Data status"','"CSM owner"','"Industry"','"Main Review Reason"']),''),
 ('Page sizes 10,25,50,100','[10,25,50,100]' in s,''),
 ('Page change reset on filter change','priority_filter_signature' in s and 'st.session_state["priority_page"] = 1' in s,''),
 ('Previous/Next button code order',s.index('if st.button("← Previous"') < s.index('key="priority_page_picker"') and s.index('if st.button("Next →"') < s.index('key="priority_page_picker"'),''),
 ('Stable sorted queue and slicing','sort_values(["priority_rank", "account_id"])' in s and 'filtered.iloc[start:end]' in s,''),
 ('NDC Health hidden in Priority List and export',s.count('table.loc[table["health_band"] == "Limited Evidence", "health_score"] = np.nan') >= 1 and 'full_export.loc[full_export["health_band"] == "Limited Evidence", "health_score"] = np.nan' in s,''),
 ('NDC Health hidden in Account 360','metric_card("Health", "Needs Data Check"' in s,''),
 ('Same source data for all pages','data = add_demo_csm_owner(data)' in s and 'page = st.session_state["nav_page"]' in s,''),
 ('Demo CSM owner is labelled','CSM owners are fictional demo assignments and do not affect scoring.' in s,''),
 ('Action is advisory','Advisory only · CSM decides next steps' in s,''),
 ('Account 360 evidence tabs',all(x in s for x in ['"Usage", "Support Ticket History", "Satisfaction"','"Select an account"']),''),
 ('Snapshot visible','SNAPSHOT_LABEL = "31 Dec 2024"' in s,''),
]
with (ROOT/'source_code_checks.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['check','result','evidence_type','note'])
 for label,ok,note in checks:w.writerow([label,'PASS' if ok else 'FAIL','STATIC_SOURCE_ONLY',note])
print(f'Source inspection: {sum(x[1] for x in checks)}/{len(checks)} PASS; NOT an executed Streamlit UI test')
for label,ok,_ in checks:
 if not ok:print('FAIL',label)
