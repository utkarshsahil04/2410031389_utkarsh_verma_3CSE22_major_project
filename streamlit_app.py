"""
Streamlit UI — Blockchain + Federated Learning medical record sharing prototype.

Usage (from the code/ directory, venv activated):
    streamlit run streamlit_app.py

Companion to the HTML dashboard (api.py); does not replace it.
Synthetic / public benchmark data only — no real patient records.
"""
from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.runner import SimulationRunner, discover_simulation_flags
from ui.state import (
    chain_table_rows,
    hospital_rows,
    load_chain_state,
    load_experiment_results,
    load_fl_state,
    rounds_dataframe,
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
  padding-top: 1.4rem;
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
.section-note {
  background: #E8F3F0;
  border-left: 3px solid #0D9488;
  padding: 0.7rem 0.95rem;
  border-radius: 0 8px 8px 0;
  color: #2C4A46;
  font-size: 0.9rem;
  margin-bottom: 1rem;
}
div[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0F3D3A 0%, #145C56 55%, #0D7377 100%);
}
div[data-testid="stSidebar"] * {
  color: #E8F5F3 !important;
}
div[data-testid="stSidebar"] .stMarkdown a {
  color: #99F6E4 !important;
}
div[data-testid="stSidebar"] hr {
  border-color: rgba(255,255,255,0.18);
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
def render_sidebar(fl_state: dict | None, flags: dict) -> None:
    with st.sidebar:
        st.markdown("### MedChain FL")
        st.caption("Blockchain audit + federated learning prototype")
        st.markdown("---")
        st.markdown("**How to run**")
        st.markdown(
            """
1. Activate the venv in `code/`
2. Open this UI: `streamlit run streamlit_app.py`
3. Use **Run Simulation** to start a local FL + chain demo
4. Or CLI: `python run_simulation.py --rounds 10`
5. HTML dashboard (optional): `python api.py` → port 5000
            """
        )
        st.markdown("---")
        st.markdown("**State files**")
        st.code("fl_state.json\nchain_state.json", language=None)
        status = (fl_state or {}).get("status")
        st.markdown(
            f'<span class="pill {_pill_class(status)}">{status_label(status)}</span>',
            unsafe_allow_html=True,
        )
        runner = _get_runner()
        if runner.is_running():
            st.caption(f"Subprocess active ({runner.kind})")
        st.markdown("---")
        st.caption("Public / synthetic data only — no real patient records.")
        if flags.get("backend"):
            st.caption("Chain backends: inprocess · fabric (env CHAIN_BACKEND)")


# ── tabs ──────────────────────────────────────────────────────────────────────
def tab_overview(fl_state: dict | None, chain: list | None) -> None:
    st.markdown('<h1 class="app-title">MedChain FL Overview</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-sub">Permissioned audit trail for hospital collaboration on a shared risk model — '
        "weights only leave each node; hashes and events go on-chain.</p>",
        unsafe_allow_html=True,
    )

    if fl_state is None:
        st.markdown(
            '<div class="section-note">No <code>fl_state.json</code> yet. '
            "Open the <b>Run Simulation</b> tab (or run <code>python run_simulation.py</code>) to generate state.</div>",
            unsafe_allow_html=True,
        )
        return

    status = fl_state.get("status")
    st.markdown(
        f'<span class="pill {_pill_class(status)}">{status_label(status)}</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    cur = fl_state.get("current_round") or 0
    total = fl_state.get("total_rounds") or "—"
    acc = fl_state.get("final_accuracy")
    if acc is None and fl_state.get("rounds"):
        acc = fl_state["rounds"][-1].get("accuracy")
    acc_txt = f"{float(acc) * 100:.1f}%" if isinstance(acc, (int, float)) else "—"
    chain_ok = fl_state.get("chain_valid")
    length = fl_state.get("chain_length")
    if length is None and chain is not None:
        length = len(chain)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric_card("FL round", f"{cur} / {total}", "Current / total")
    with c2:
        _metric_card("Accuracy", acc_txt, "Held-out test set")
    with c3:
        _metric_card(
            "Chain",
            "VALID" if chain_ok else ("INVALID" if chain_ok is False else "—"),
            f"{length or '—'} blocks",
        )
    with c4:
        n_hosp = len(fl_state.get("hospitals") or {})
        _metric_card("Hospitals", str(n_hosp), "Registered nodes")

    if status in ("training", "initializing"):
        progress = 0.0
        if isinstance(total, int) and total > 0:
            progress = min(float(cur) / float(total), 1.0)
        st.progress(progress, text=f"Simulation in progress — round {cur} of {total}")


def tab_run_simulation(fl_state: dict | None, flags: dict) -> None:
    st.subheader("Run Simulation")
    st.markdown(
        '<div class="section-note">Starts <code>run_simulation.py</code> in a subprocess. '
        "Progress is read from JSON state files (auto-refresh while running).</div>",
        unsafe_allow_html=True,
    )

    runner = _get_runner()
    col_a, col_b = st.columns(2)
    with col_a:
        rounds = st.number_input("FL rounds", min_value=1, max_value=50, value=10, step=1)
        backend = st.selectbox(
            "Chain backend",
            options=["inprocess", "fabric"],
            index=0,
            help="Sets CHAIN_BACKEND (and --backend when supported).",
        )
    with col_b:
        dataset = "pima"
        if flags.get("dataset"):
            dataset = st.selectbox(
                "Dataset",
                options=flags.get("dataset_choices") or ["pima"],
                index=0,
                help="PIMA diabetes is the default smoke test.",
            )
        else:
            st.selectbox("Dataset", ["pima (default)"], disabled=True)
            st.caption("Dataset switch lands with roadmap Step 4 — UI will enable it automatically.")

        non_iid = None
        if flags.get("non_iid"):
            split = st.radio("Data split", ["IID", "non-IID"], horizontal=True)
            non_iid = split == "non-IID"
        else:
            st.radio("Data split", ["IID (equal shards)"], horizontal=True, disabled=True)
            st.caption("non-IID controls appear when Step 3 lands in run_simulation.py.")

    b1, b2, b3 = st.columns([1, 1, 2])
    with b1:
        start = st.button("Start simulation", type="primary", disabled=runner.is_running())
    with b2:
        stop = st.button("Stop", disabled=not runner.is_running())

    if start:
        ok = runner.start_simulation(
            rounds=int(rounds),
            dataset=dataset if flags.get("dataset") else None,
            non_iid=non_iid,
            backend=backend,
        )
        if ok:
            st.success("Simulation started.")
            st.rerun()
        else:
            st.error(runner.last_error or "Failed to start.")

    if stop:
        runner.stop()
        st.warning("Stopped subprocess.")
        st.rerun()

    running = runner.is_running()
    status = (fl_state or {}).get("status")
    live = running or status in ("training", "initializing")

    if live and fl_state:
        cur = fl_state.get("current_round") or 0
        total = fl_state.get("total_rounds") or 1
        try:
            pct = min(float(cur) / float(total), 1.0)
        except (TypeError, ZeroDivisionError):
            pct = 0.0
        st.progress(pct, text=f"{status_label(status)} — round {cur}/{total}")

    logs = runner.poll_logs()
    if logs:
        with st.expander("Process log", expanded=running):
            st.code("\n".join(logs[-30:]), language=None)

    if runner.last_cmd:
        st.caption("Last command: `" + " ".join(runner.last_cmd) + "`")

    if live:
        time.sleep(1.2)
        st.rerun()


def tab_hospitals(fl_state: dict | None) -> None:
    st.subheader("Hospitals")
    if fl_state is None:
        st.info("Run a simulation to populate hospital registration and local losses.")
        return
    rows = hospital_rows(fl_state)
    if not rows:
        st.warning("No hospital entries in fl_state.json.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    rounds = fl_state.get("rounds") or []
    if rounds:
        st.markdown("##### Per-hospital loss (latest round)")
        losses = rounds[-1].get("client_losses") or {}
        if losses:
            chart_df = pd.DataFrame(
                {"client": list(losses.keys()), "loss": list(losses.values())}
            ).set_index("client")
            st.bar_chart(chart_df, color="#0D9488")


def tab_fl_metrics(fl_state: dict | None) -> None:
    st.subheader("FL Metrics")
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
            margin=dict(l=40, r=20, t=50, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(244,247,246,0.6)",
            height=360,
        )
        st.plotly_chart(fig_acc, use_container_width=True)
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
        st.plotly_chart(fig_loss, use_container_width=True)

    st.dataframe(df, use_container_width=True, hide_index=True)


def tab_blockchain(chain: list | None, fl_state: dict | None) -> None:
    st.subheader("Blockchain Audit")
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
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=420)

    with st.expander("Raw block JSON (last 3)"):
        st.json(chain[-3:] if len(chain) >= 3 else chain)


def tab_experiments(flags: dict) -> None:
    st.subheader("Experiments")
    if not flags.get("experiments_exists"):
        st.markdown(
            '<div class="section-note"><code>run_experiments.py</code> is not present yet '
            "(roadmap Step 5). This tab will light up when baselines land. "
            "You can still inspect any results JSON if another agent adds one.</div>",
            unsafe_allow_html=True,
        )
    else:
        runner = _get_runner()
        if st.button("Run baselines / experiments", disabled=runner.is_running()):
            if runner.start_experiments():
                st.success("Experiments started.")
                st.rerun()
            else:
                st.error(runner.last_error or "Failed to start experiments.")
        if runner.is_running() and runner.kind == "experiments":
            st.info("Experiments running…")
            logs = runner.poll_logs()
            st.code("\n".join(logs[-25:]), language=None)
            time.sleep(1.5)
            st.rerun()

    path, results = load_experiment_results()
    if results is None:
        st.caption("No experiment_results.json (or similar) found under code/.")
        return
    st.markdown(f"**Last results:** `{path}`")
    if isinstance(results, (dict, list)):
        st.json(results)
    else:
        st.write(results)


def main() -> None:
    flags = discover_simulation_flags()
    fl_state = load_fl_state()
    chain = load_chain_state()
    render_sidebar(fl_state, flags)

    tabs = st.tabs(
        [
            "Overview",
            "Run Simulation",
            "Hospitals",
            "FL Metrics",
            "Blockchain Audit",
            "Experiments",
        ]
    )
    with tabs[0]:
        tab_overview(fl_state, chain)
    with tabs[1]:
        tab_run_simulation(fl_state, flags)
    with tabs[2]:
        tab_hospitals(fl_state)
    with tabs[3]:
        tab_fl_metrics(fl_state)
    with tabs[4]:
        tab_blockchain(chain, fl_state)
    with tabs[5]:
        tab_experiments(flags)


if __name__ == "__main__":
    main()
