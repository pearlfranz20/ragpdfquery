import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Admin Dashboard", layout="wide")
st.title("📚 PDF RAG — Admin Dashboard")

tab_ingest, tab_query, tab_eval, tab_corpus = st.tabs(
    ["📥 Ingestion", "💬 Query Tester", "📊 Evaluation", "🗂️ Corpus Browser"]
)

# --------------------------------------------------------------------------
# Ingestion tab
# --------------------------------------------------------------------------
with tab_ingest:
    st.subheader("Corpus overview")
    try:
        stats = requests.get(f"{API_BASE_URL}/admin/stats", timeout=10).json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Documents", stats["num_documents"])
        c2.metric("Chunks", stats["num_chunks"])
        c3.metric("Pages", stats["num_pages"])

        col_a, col_b = st.columns(2)
        with col_a:
            status_df = pd.DataFrame(
                {"status": list(stats["num_documents_by_status"].keys()), "count": list(stats["num_documents_by_status"].values())}
            )
            st.plotly_chart(px.pie(status_df, names="status", values="count", title="Documents by status"), use_container_width=True)
        with col_b:
            lang_df = pd.DataFrame(
                {"language": list(stats["languages"].keys()), "count": list(stats["languages"].values())}
            )
            st.plotly_chart(px.bar(lang_df, x="language", y="count", title="Documents by language"), use_container_width=True)
    except requests.RequestException as e:
        st.error(f"Could not reach API at {API_BASE_URL}: {e}")

    st.divider()
    st.subheader("Upload a PDF")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    strategy = st.selectbox("Chunking strategy", [None, "recursive", "page", "section", "semantic"], index=0)
    if uploaded and st.button("Ingest"):
        files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
        params = {"strategy": strategy} if strategy else {}
        resp = requests.post(f"{API_BASE_URL}/ingest/upload", files=files, params=params, timeout=30)
        if resp.ok:
            job = resp.json()
            st.success(f"Queued ingestion job {job['id']} (batch {job['batch_id']})")
        else:
            st.error(resp.text)

# --------------------------------------------------------------------------
# Query tester tab
# --------------------------------------------------------------------------
with tab_query:
    st.subheader("Ask a question against the corpus")
    question = st.text_area("Question", placeholder="What were the termination clauses in the 2023 vendor contracts?")
    top_k = st.slider("Top K chunks to retrieve", min_value=1, max_value=20, value=8)

    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Retrieving and generating..."):
            resp = requests.post(f"{API_BASE_URL}/query", json={"question": question, "top_k": top_k}, timeout=120)
        if resp.ok:
            data = resp.json()
            st.markdown(f"**Detected language:** `{data['detected_language']}`  |  "
                        f"**Latency:** {data['latency_ms']} ms  |  "
                        f"**Retried for faithfulness:** {data['retried']}")
            if data.get("faithfulness_score") is not None:
                st.progress(min(max(data["faithfulness_score"], 0.0), 1.0), text=f"Faithfulness score: {data['faithfulness_score']:.2f}")

            st.markdown("### Answer")
            st.write(data["answer"])

            st.markdown("### Citations")
            for i, c in enumerate(data["citations"], start=1):
                with st.expander(f"[{i}] {c['filename']} — page {c['page_number']}"):
                    st.write(c["snippet"])
        else:
            st.error(resp.text)

# --------------------------------------------------------------------------
# Evaluation tab
# --------------------------------------------------------------------------
with tab_eval:
    st.subheader("Latest DeepEval run")
    try:
        summary = requests.get(f"{API_BASE_URL}/admin/eval/latest", timeout=10).json()
        if summary:
            df = pd.DataFrame(summary)
            st.dataframe(df, use_container_width=True)
            st.plotly_chart(px.bar(df, x="metric_name", y="avg_score", title="Average score by metric"), use_container_width=True)
        else:
            st.info("No eval runs yet. Run `python scripts/run_eval.py` to populate this tab.")
    except requests.RequestException as e:
        st.error(f"Could not reach API: {e}")

    st.divider()
    st.subheader("History")
    try:
        history = requests.get(f"{API_BASE_URL}/admin/eval/history", timeout=10).json()
        if history:
            hist_df = pd.DataFrame(history)
            st.plotly_chart(
                px.line(hist_df, x="run_at", y="score", color="metric_name", title="Score trend over time"),
                use_container_width=True,
            )
            st.dataframe(hist_df, use_container_width=True)
    except requests.RequestException:
        pass

# --------------------------------------------------------------------------
# Corpus browser tab
# --------------------------------------------------------------------------
with tab_corpus:
    st.subheader("Documents")
    status_filter = st.selectbox("Filter by status", [None, "pending", "processing", "completed", "failed"])
    try:
        params = {"status": status_filter} if status_filter else {}
        docs = requests.get(f"{API_BASE_URL}/ingest/documents", params=params, timeout=10).json()
        if docs:
            df = pd.DataFrame(docs)
            st.dataframe(df, use_container_width=True)

            doc_id_to_delete = st.selectbox("Select a document to delete", [d["id"] for d in docs])
            if st.button("Delete selected document", type="secondary"):
                r = requests.delete(f"{API_BASE_URL}/ingest/documents/{doc_id_to_delete}", timeout=10)
                if r.ok:
                    st.success("Deleted. Refresh to update the list.")
                else:
                    st.error(r.text)
        else:
            st.info("No documents ingested yet.")
    except requests.RequestException as e:
        st.error(f"Could not reach API: {e}")
