"""
Streamlit UI — Blockchain + Federated Learning medical record sharing prototype.

Usage (from the code/ directory, venv activated):
    streamlit run streamlit_app.py

Companion to the HTML dashboard (api.py); does not replace it.
Synthetic / public benchmark data only — no real patient records.
"""
from __future__ import annotations

from datetime import datetime
from html import escape as esc

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import integrity
from ui.runner import SimulationRunner, discover_simulation_flags
from ui.state import (
    chain_table_rows,
    events_of,
    highlight_diff,
    hospital_rows,
    load_chain_state,
    load_experiment_results,
    load_fl_state,
    rounds_dataframe,
    short_hash,
    status_label,
)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedChain FL",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
}
code, pre, .stCode {
  font-family: 'IBM Plex Mono', monospace !important;
}
.block-container {
  padding-top: 4rem;  /* clear Streamlit's fixed top toolbar (~3.75rem) */
  padding-bottom: 2rem;
  max-width: 1180px;
}
h1.app-title {
  font-size: 1.85rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: #0F3D3A;
  margin-bottom: 0.15rem;
}
p.app-sub {
  color: #4A6A66;
  margin-top: 0;
  margin-bottom: 1.1rem;
  font-size: 0.98rem;
}
.pill {
  display: inline-block;
  padding: 0.28rem 0.75rem;
  border-radius: 999px;
  font-size: 0.82rem;
  font-weight: 600;
  letter-spacing: 0.02em;
}
.pill-complete { background: #D1FAE5; color: #065F46; }
.pill-training { background: #FEF3C7; color: #92400E; }
.pill-init { background: #E0F2FE; color: #075985; }
.pill-idle { background: #E5E7EB; color: #374151; }
.pill-error { background: #FEE2E2; color: #991B1B; }
.metric-card {
  background: linear-gradient(145deg, #FFFFFF 0%, #F0F7F5 100%);
  border: 1px solid #C9DBD7;
  border-radius: 12px;
  padding: 0.95rem 1.1rem;
  height: 100%;
}
.metric-card .label {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #5B7A76;
  margin-bottom: 0.35rem;
}
.metric-card .value {
  font-size: 1.55rem;
  font-weight: 700;
  color: #0F3D3A;
  font-family: 'IBM Plex Mono', monospace;
}
.metric-card .hint {
  font-size: 0.78rem;
  color: #6B857F;
  margin-top: 0.2rem;
}
.diff { color: #B4232C; font-weight: 700; background: #FEE2E2; }
.hashline { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; word-break: break-all; color: #1A2B2A; }
.hashlabel { font-size: 0.75rem; color: #5B7A76; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.4rem; }
.badge { display: inline-block; padding: 0.3rem 0.8rem; border-radius: 8px; font-weight: 700; font-size: 0.88rem; }
.badge-ok { background: #D1FAE5; color: #065F46; }
.badge-bad { background: #FEE2E2; color: #991B1B; }
.badge-off { background: #E5E7EB; color: #374151; }
.stepno { display: inline-block; width: 2rem; height: 2rem; line-height: 2rem; text-align: center;
  border-radius: 50%; background: #0D9488; color: #FFFFFF; font-weight: 700; margin-right: 0.5rem; }
.evidence { background: #F0F7F5; border: 1px solid #C9DBD7; border-radius: 8px; padding: 0.5rem 0.8rem;
  font-size: 0.85rem; color: #1A2B2A; margin-top: 0.4rem; }
.section-note {
  background: #E8F3F0;
  border-left: 3px solid #0D9488;
  padding: 0.7rem 0.95rem;
  border-radius: 0 8px 8px 0;
  color: #2C4A46;
  font-size: 0.9rem;
  margin-bottom: 1rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _pill_class(status: str | None) -> str:
    if not status:
        return "pill-idle"
    s = status.lower()
    if s == "complete":
        return "pill-complete"
    if s == "training":
        return "pill-training"
    if s in ("initializing", "init"):
        return "pill-init"
    if s == "error":
        return "pill-error"
    return "pill-idle"


def _get_runner() -> SimulationRunner:
    if "sim_runner" not in st.session_state:
        st.session_state.sim_runner = SimulationRunner()
    return st.session_state.sim_runner


def _metric_card(label: str, value: str, hint: str = "") -> None:
    hint_html = f'<div class="hint">{hint}</div>' if hint else ""
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(fl_state: dict | None) -> None:
    with st.sidebar:
        st.markdown("### MedChain FL")
        st.caption("Hospitals train one shared model without sharing patient records. "
                   "A tamper-evident log (blockchain) records what happened.")
        status = (fl_state or {}).get("status")
        st.markdown(
            f'<span class="pill {_pill_class(status)}">{status_label(status)}</span>',
            unsafe_allow_html=True,
        )
        runner = _get_runner()
        if runner.is_running():
            st.caption(f"Demo running ({runner.kind})")
        st.markdown("---")
        st.caption("Public / synthetic data only — no real patient records.")
        st.caption("Optional: `python api.py` serves the HTML dashboard on port 5000.")


def _evidence(text: str) -> None:
    st.markdown(f'<div class="evidence">{text}</div>', unsafe_allow_html=True)


# ── tab 1: How it works ───────────────────────────────────────────────────────
def tab_how_it_works(fl_state: dict | None, chain: list | None) -> None:
    st.markdown('<h1 class="app-title">How this project works</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-sub">Several hospitals want one good medical prediction model. '
        "They are not allowed to hand over patient records. This demo shows how they can still do it, "
        "and how every step is logged so nobody can quietly change it.</p>",
        unsafe_allow_html=True,
    )
    fl = fl_state or {}
    if not fl_state:
        st.info("No run yet. Open the **Run the demo** tab and press Start, then come back: "
                "each step below will show real numbers from that run.")
    hospitals = fl.get("hospitals") or {}
    registers = events_of(chain, "register_record")
    rounds = fl.get("rounds") or []
    consent = fl.get("consent_checks") or []

    def step(n: int, title: str, text: str, evidence: str | None) -> None:
        with st.container(border=True):
            st.markdown(f'<span class="stepno">{n}</span> **{title}**', unsafe_allow_html=True)
            st.write(text)
            if evidence:
                _evidence(evidence)

    ev1 = ev2 = ev3 = ev4 = ev5 = ev6 = None
    if hospitals:
        ev1 = "Live from last run: " + " · ".join(
            f"<b>{esc(str(h))}</b> keeps {esc(str(i.get('records', '?')))} patient rows" for h, i in hospitals.items())
    if registers:
        ev2 = "Live from last run: " + " · ".join(
            f"<b>{esc(str(b['data'].get('hospital')))}</b> fingerprint "
            f"<code>{esc(str(b['data'].get('record_hash', ''))[:12])}…</code> in block #{esc(str(b.get('index')))}" for b in registers)
    if consent:
        ev3 = "Live from last run: " + " · ".join(
            f"<b>{esc(str(c.get('actor')))}</b> asked to use Hospital_A's data → "
            + ("<b>ALLOWED</b>" if c.get("allowed") else "<b>REFUSED</b>") for c in consent)
    if rounds:
        losses = rounds[0].get("client_losses") or {}
        ev4 = "Live from last run: in round 1 each hospital reported its own error ('loss'): " + ", ".join(
            esc(f"{k.replace('hospital_', 'hospital ')} = {v}") for k, v in losses.items())
        accs = [r.get("accuracy") for r in rounds if isinstance(r.get("accuracy"), (int, float))]
        if accs:
            ev5 = (f"Live from last run: shared model accuracy went from <b>{accs[0]*100:.1f}%</b> (round 1) "
                   f"to <b>{accs[-1]*100:.1f}%</b> (round {len(accs)}). Per round: "
                   + ", ".join(f"{a*100:.0f}%" for a in accs))
    if chain is not None:
        ok = fl.get("chain_valid")
        ev6 = (f"Live from last run: the log has <b>{len(chain)} blocks</b>; integrity check says "
               + ("<b>VALID</b>" if ok else ("<b>BROKEN</b>" if ok is False else "unknown")) + ".")

    step(1, "The data stays at each hospital",
         "Each hospital keeps its own patient table on its own computer. Nobody copies it to a central place.", ev1)
    step(2, "A fingerprint of each dataset goes on the blockchain",
         "A 'hash' is a short code computed from the whole file. Change even one number and the code changes completely. "
         "Each hospital writes its fingerprint (not the data) into a shared log. Later you can check the data still matches.", ev2)
    step(3, "Consent: who may use whose data",
         "Hospital A has agreed that Hospital B may use its data, but not Hospital C. The log is checked before access is given.", ev3)
    step(4, "Each hospital trains the model locally",
         "Every hospital teaches its own copy of the model using only its own patients. Only the learned numbers "
         "(model weights) leave the building, never the records.", ev4)
    step(5, "The server averages the models",
         "A coordinator takes the three models and averages them into one better shared model. "
         "This repeats for several rounds, and the shared model improves.", ev5)
    step(6, "Everything is logged and tamper-evident",
         "Each step above is added to the blockchain as a signed block that also contains the previous block's code. "
         "Editing any old block breaks the chain, and the check turns red.", ev6)

    with st.expander("Glossary (plain English)"):
        st.markdown(
            """
- **Hash**: a digital fingerprint of a file. Same file, same hash. Change one character, completely different hash.
- **Blockchain block**: one entry in a log. Each block stores the previous block's hash, so old entries cannot be edited unnoticed.
- **Federated learning**: training one model across many places without moving the data. Only the learned numbers are shared.
- **FedAvg**: "federated averaging". The server averages the models from all hospitals, weighted by how much data each has.
- **IID / non-IID**: IID means every hospital has a similar mix of patients. Non-IID means they differ (for example one sees sicker patients), which makes training harder.
- **Consent**: a recorded permission, here "Hospital A lets Hospital B use its data".
            """
        )


# ── tab 2: Run the demo ───────────────────────────────────────────────────────
@st.fragment(run_every=1)
def _run_status(runner: SimulationRunner) -> None:
    fl_state = load_fl_state()
    is_sim = runner.kind == "simulation"  # the runner is shared with the Experiments button
    running = runner.is_running() and is_sim
    # Refresh the whole page once when the process finishes so other tabs show the new run.
    was_running = st.session_state.get("_was_running", False)
    st.session_state["_was_running"] = running
    if was_running and not running:
        st.rerun()
    status = (fl_state or {}).get("status")
    if running and fl_state:
        cur = fl_state.get("current_round") or 0
        total = fl_state.get("total_rounds") or 1
        try:
            pct = min(float(cur) / float(total), 1.0)
        except (TypeError, ZeroDivisionError):
            pct = 0.0
        st.progress(pct, text=f"{status_label(status)} — round {cur} of {total}")
    elif running:
        st.progress(0.0, text="Starting…")
    elif fl_state:
        st.caption(f"Last run: {status_label(status)}.")
    logs = runner.poll_logs(max_lines=200) if is_sim else []
    if logs:
        # Constant `expanded`: if it followed `running`, Streamlit would redraw the panel
        # closed the moment the run ends, hiding the final result.
        with st.expander("Live log", expanded=True):
            st.code("\n".join(logs), language=None)


def tab_run_demo(flags: dict) -> None:
    st.subheader("Run the demo")
    st.caption("What am I looking at? Choose the settings, press Start, and watch three hospitals "
               "train a shared model while the log is written.")
    runner = _get_runner()
    col_a, col_b = st.columns(2)
    with col_a:
        dataset = st.selectbox("Dataset", flags["dataset_choices"], index=0,
                               help="pima = diabetes, heart = heart disease, mimic = synthetic ICU-style stub.")
        st.caption("Which medical dataset is shared out among the three hospitals.")
        split = st.radio("How the data is split", ["IID", "non-IID"], horizontal=True)
        st.caption("IID: every hospital gets a similar mix of patients. "
                   "non-IID: hospitals get different mixes (harder, more realistic).")
    with col_b:
        rounds = st.number_input("Training rounds", min_value=1, max_value=50, value=10, step=1)
        st.caption("One round = every hospital trains, then the server averages. More rounds, better model.")
        backend = st.selectbox("Blockchain backend", ["inprocess", "fabric"], index=0)
        st.caption("inprocess = a built-in demo ledger. fabric = a mock of Hyperledger Fabric.")
    keep = st.checkbox(
        "Train on the current (edited) hospital data",
        help="Reuse the hospital files as they are on disk (including your edits in Live Data Check) "
             "instead of creating them fresh.")
    if keep:
        st.caption("The three hospital files are used exactly as saved. The dataset must match the last run.")

    b1, b2, _ = st.columns([1, 1, 3])
    start = b1.button("Start", type="primary", disabled=runner.is_running())
    stop = b2.button("Stop", disabled=not runner.is_running())
    if start:
        if runner.start_simulation(rounds=int(rounds), dataset=dataset, non_iid=(split == "non-IID"),
                                   backend=backend, keep_data=keep):
            st.session_state["_was_running"] = True
        else:
            st.error(runner.last_error or "Failed to start.")
    if stop:
        runner.stop()
        st.warning("Stopped.")
    _run_status(runner)
    if runner.last_cmd:
        st.caption("Command: `" + " ".join(runner.last_cmd) + "`")


# ── tab 3: Live Data Check ────────────────────────────────────────────────────
_BADGES = {
    "match": ("badge-ok", "Matches blockchain"),
    "changed": ("badge-bad", "Data changed — does not match blockchain"),
    "missing": ("badge-off", "File missing"),
    "unreadable": ("badge-off", "File unreadable"),
    "not_registered": ("badge-off", "Not registered on blockchain yet"),
    "chain_invalid": ("badge-bad", "Blockchain failed verification — fingerprint can't be trusted"),
}


def _hospital_card(r: dict) -> None:
    cls, label = _BADGES.get(r["status"], ("badge-off", r["status"]))
    with st.container(border=True):
        st.markdown(f"**{r['hospital']}**")
        st.markdown(f'<span class="badge {cls}">{label}</span>', unsafe_allow_html=True)
        live, chain_h = r.get("live_hash"), r.get("chain_hash")
        st.markdown('<div class="hashlabel">Hash of the file right now</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hashline">{highlight_diff(live, chain_h) if live else "—"}</div>',
                    unsafe_allow_html=True)
        blk = r.get("chain_block_index")
        st.markdown(f'<div class="hashlabel">Hash recorded on blockchain'
                    f'{f" (block #{blk})" if blk is not None else ""}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hashline">{esc(chain_h) if chain_h else "—"}</div>', unsafe_allow_html=True)
        if r["status"] == "changed":
            if r.get("values_identical"):
                st.caption("File bytes changed but the values are identical (e.g. re-saved by Excel).")
            else:
                st.caption("Red characters are where the live hash differs.")
        rows, cols = r.get("rows"), r.get("columns")
        mod = r.get("modified_at")
        mod_txt = str(mod) if mod else "—"
        st.caption(f"Rows: {rows if rows is not None else '—'} · Columns: {cols if cols is not None else '—'} "
                   f"· Last modified: {mod_txt}")


@st.fragment(run_every=2)
def _live_cards() -> None:
    results = integrity.scan()
    cols = st.columns(len(results) or 1)
    for col, r in zip(cols, results):
        with col:
            _hospital_card(r)
    st.caption(f"Checked at {datetime.now().strftime('%H:%M:%S')} · refreshes every 2 seconds")


@st.fragment(run_every=2)
def _live_changes(h: str) -> None:
    diff = integrity.changed_cells(h)
    if diff.get("error"):
        st.caption(f"Cannot list changed cells: {diff['error']}.")
        return
    if diff.get("columns_added") or diff.get("columns_removed"):
        st.caption(f"Columns added: {diff['columns_added'] or 'none'} · removed: {diff['columns_removed'] or 'none'}")
    if diff["modified"] or diff["rows_added"] or diff["rows_removed"]:
        st.markdown(f"**Changes since the registered version of {h}**")
        if diff["modified"]:
            st.dataframe(pd.DataFrame(diff["modified"]), width="stretch", hide_index=True)
        if diff["rows_added"] or diff["rows_removed"]:
            st.caption(f"Rows added: {diff['rows_added']} · rows removed: {diff['rows_removed']}")
    elif not (diff.get("columns_added") or diff.get("columns_removed")):
        st.caption(f"No changed cells: {h} has the same values as the registered version.")


def tab_live_check() -> None:
    st.subheader("Live Data Check")
    st.caption("What am I looking at? Each hospital's data file is fingerprinted every 2 seconds and compared "
               "with the fingerprint saved on the blockchain. Edit a file in Excel or VS Code and watch a card turn red.")
    if integrity.read_meta() is None:
        st.info("No hospital data files yet. Run the demo first (Run the demo tab).")
        return
    _live_cards()

    st.markdown("---")
    st.markdown("#### Edit a hospital's data here")
    st.caption("Change a number, press Save changes, and the card above turns red. "
               "Then you can record the new version on the blockchain, or undo.")
    runner = _get_runner()
    h = st.selectbox("Hospital", integrity.HOSPITALS, key="edit_h")
    ver = st.session_state.get("edit_ver", 0)
    try:
        df = integrity.load_shard(h)
    except Exception as exc:
        st.warning(f"Cannot open {h}'s file: {exc}")
        df = None
    if df is not None:
        edited = st.data_editor(df, key=f"editor_{h}_{ver}", num_rows="fixed", width="stretch", height=260)
        c1, c2, c3 = st.columns(3)
        if c1.button("Save changes", type="primary"):
            new = integrity.save_shard(h, edited)
            st.success(f"Saved. New hash {new[:16]}…")
        if c2.button("Record new version on blockchain", disabled=runner.is_running()):
            try:
                blk = integrity.record_new_version(h)
                st.success(f"Recorded as block #{blk.get('index')}. The card should turn green.")
            except Exception as exc:  # RuntimeError carries a readable message; anything else too
                st.error(str(exc))
        if c3.button("Undo my edits"):
            try:
                integrity.restore_original(h)
                st.session_state["edit_ver"] = ver + 1
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))
        if runner.is_running():
            st.caption("Recording is disabled while the demo is running.")
        _live_changes(h)

    st.markdown("---")
    st.markdown("#### Try it yourself")
    txt = st.text_input("Type anything", value="Hospital_A patient 42: glucose 148")
    st.markdown(f'<div class="hashline">{integrity.hash_text(txt)}</div>', unsafe_allow_html=True)
    st.caption("Change just one character above and the whole fingerprint changes. "
               "That is how a tiny edit to a hospital file is caught.")


def tab_hospitals(fl_state: dict | None) -> None:
    st.subheader("Hospitals")
    st.caption("What am I looking at? One row per hospital: how many records it holds, its registered fingerprint, and its latest training error.")
    if fl_state is None:
        st.info("Run a simulation to populate hospital registration and local losses.")
        return
    rows = hospital_rows(fl_state)
    if not rows:
        st.warning("No hospital entries in fl_state.json.")
        return
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    rounds = fl_state.get("rounds") or []
    if rounds:
        st.markdown("##### Per-hospital loss (latest round)")
        losses = rounds[-1].get("client_losses") or {}
        if losses:
            chart_df = pd.DataFrame(
                {"client": list(losses.keys()), "loss": list(losses.values())}
            ).set_index("client")
            st.bar_chart(chart_df, color="#0D9488")


def tab_training(fl_state: dict | None) -> None:
    st.subheader("Training results")
    st.caption("What am I looking at? How good the shared model is after each round (accuracy: higher is better) and its average error (loss: lower is better).")
    if fl_state is None:
        st.info("No metrics yet — start a simulation first.")
        return
    df = rounds_dataframe(fl_state)
    if df.empty:
        st.warning("Round history is empty. Wait for training rounds to complete.")
        return

    c1, c2 = st.columns(2)
    with c1:
        fig_acc = go.Figure()
        fig_acc.add_trace(
            go.Scatter(
                x=df["round"],
                y=df["accuracy"],
                mode="lines+markers",
                name="Accuracy",
                line=dict(color="#0D9488", width=2.5),
                marker=dict(size=7),
            )
        )
        fig_acc.update_layout(
            title="Global accuracy vs round",
            xaxis_title="Round",
            yaxis_title="Accuracy",
            yaxis_range=[0, 1.05],
            margin=dict(l=40, r=20, t=50, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(244,247,246,0.6)",
            height=360,
        )
        st.plotly_chart(fig_acc, width="stretch")
    with c2:
        fig_loss = go.Figure()
        fig_loss.add_trace(
            go.Scatter(
                x=df["round"],
                y=df["avg_loss"],
                mode="lines+markers",
                name="Avg loss",
                line=dict(color="#0F766E", width=2.5),
                marker=dict(size=7),
            )
        )
        fig_loss.update_layout(
            title="Average client loss vs round",
            xaxis_title="Round",
            yaxis_title="Loss",
            margin=dict(l=40, r=20, t=50, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(244,247,246,0.6)",
            height=360,
        )
        st.plotly_chart(fig_loss, width="stretch")

    st.dataframe(df, width="stretch", hide_index=True)


def tab_blockchain(chain: list | None, fl_state: dict | None) -> None:
    st.subheader("Blockchain audit")
    st.caption("What am I looking at? The tamper-evident log. Each row is a block. Green rows (data_update) mean a hospital re-registered edited data.")
    if chain is None:
        st.info("No chain_state.json yet. Run the simulation to write the audit trail.")
        return

    valid = (fl_state or {}).get("chain_valid")
    if valid is True:
        st.success(f"Chain integrity VALID · {len(chain)} blocks")
    elif valid is False:
        st.error(f"Chain integrity INVALID · {len(chain)} blocks")
    else:
        st.caption(f"{len(chain)} blocks loaded (validity unknown until FL state reports it)")

    rows = chain_table_rows(chain)
    df = pd.DataFrame(rows)
    styled = df.style.apply(
        lambda r: ["background-color: #D1FAE5; color: #065F46"] * len(r) if r["Event"] == "data_update" else [""] * len(r),
        axis=1,
    )
    st.dataframe(styled, width="stretch", hide_index=True, height=420)

    with st.expander("Raw block JSON (last 3)"):
        st.json(chain[-3:] if len(chain) >= 3 else chain)


@st.fragment(run_every=2)
def _experiment_status(runner: SimulationRunner) -> None:
    was = st.session_state.get("_exp_was_running", False)
    now = runner.is_running() and runner.kind == "experiments"
    st.session_state["_exp_was_running"] = now
    if was and not now:
        st.rerun()
    if now:
        st.info("Experiments running…")
        st.code("\n".join(runner.poll_logs()[-25:]), language=None)


def tab_experiments(flags: dict) -> None:
    st.subheader("Experiments")
    st.caption("What am I looking at? Comparison runs: normal training versus attacks (a hospital sending bad updates, a tampered log) to show what the safeguards catch.")
    if not flags.get("experiments_exists"):
        st.markdown(
            '<div class="section-note"><code>run_experiments.py</code> is not present yet '
            "(roadmap Step 5). This tab will light up when baselines land. "
            "You can still inspect any results JSON if another agent adds one.</div>",
            unsafe_allow_html=True,
        )
    else:
        runner = _get_runner()
        last = load_fl_state() or {}  # default to the dataset/split of the last simulation
        choices = flags["dataset_choices"]
        last_ds = last.get("dataset") if last.get("dataset") in choices else choices[0]
        last_split = (last.get("split") or {}).get("mode", "iid")
        e1, e2 = st.columns(2)
        # keys include the last run so the defaults follow a new simulation (Streamlit keeps a
        # keyed widget's old value otherwise); a manual choice sticks until the next run.
        exp_ds = e1.selectbox("Experiment dataset", choices, index=choices.index(last_ds),
                              key=f"exp_ds_{last_ds}_{last_split}")
        exp_split = e2.radio("Experiment split", ["iid", "noniid"], horizontal=True,
                             index=1 if last_split == "noniid" else 0, key=f"exp_split_{last_ds}_{last_split}")
        if st.button("Run baselines / experiments", disabled=runner.is_running()):
            if runner.start_experiments(dataset=exp_ds, split=exp_split):
                st.session_state["_exp_was_running"] = True
            else:
                st.error(runner.last_error or "Failed to start experiments.")
        _experiment_status(runner)

    path, results = load_experiment_results()
    if results is None:
        st.caption("No experiment_results.json (or similar) found under code/.")
        return
    st.markdown(f"**Last results:** `{path}`")
    rows = results.get("results") if isinstance(results, dict) else None
    if not rows:
        st.json(results)
        return

    st.caption(
        f"Dataset **{results.get('dataset', '?')}** · split **{(results.get('split') or {}).get('mode', '?')}** · "
        f"{results.get('rounds', '?')} rounds · attacker: Hospital_C"
    )
    table = []
    for r in rows:
        acc = r.get("accuracy")
        auc = r.get("auc")
        if r.get("name") == "chain_tamper":
            acc_txt = "tamper detected" if r.get("verify_after_tamper") is False else "NOT detected"
        else:
            acc_txt = f"{acc:.3f}" if isinstance(acc, (int, float)) else "—"
        table.append({
            "Experiment": r.get("name", "?").replace("_", " "),
            "Accuracy": acc_txt,
            "AUC": f"{auc:.3f}" if isinstance(auc, (int, float)) else "—",
            "Drop vs FedAvg": f"{r['accuracy_drop_vs_fedavg']:.3f}" if isinstance(r.get("accuracy_drop_vs_fedavg"), (int, float)) else "",
        })
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)

    bars = [r for r in rows if isinstance(r.get("accuracy"), (int, float))]
    if bars:
        colors = ["#B4232C" if r["accuracy"] < 0.5 else ("#0D9488" if r["name"] == "fedavg" else "#94A9A5") for r in bars]
        fig = go.Figure(go.Bar(
            x=[r["accuracy"] for r in bars],
            y=[r["name"].replace("_", " ") for r in bars],
            orientation="h",
            marker_color=colors,
            text=[f"{r['accuracy']:.3f}" for r in bars],
            textposition="outside",
        ))
        fig.update_layout(
            height=300, margin=dict(l=10, r=40, t=30, b=10),
            xaxis=dict(range=[0, 1], title="Test accuracy"),
            yaxis=dict(autorange="reversed"),
            title="Accuracy by setup (red = attack broke the model)",
        )
        st.plotly_chart(fig, width="stretch")

    with st.expander("Raw results JSON"):
        st.json(results)


def main() -> None:
    flags = discover_simulation_flags()
    fl_state = load_fl_state()
    chain = load_chain_state()
    render_sidebar(fl_state)

    tabs = st.tabs(
        ["How it works", "Run the demo", "Live Data Check", "Hospitals",
         "Training results", "Blockchain audit", "Experiments"]
    )
    with tabs[0]:
        tab_how_it_works(fl_state, chain)
    with tabs[1]:
        tab_run_demo(flags)
    with tabs[2]:
        tab_live_check()
    with tabs[3]:
        tab_hospitals(fl_state)
    with tabs[4]:
        tab_training(fl_state)
    with tabs[5]:
        tab_blockchain(chain, fl_state)
    with tabs[6]:
        tab_experiments(flags)


if __name__ == "__main__":
    main()
