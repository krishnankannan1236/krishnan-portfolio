"""
app.py — Krishnan Kannan • Portfolio + Resume RAG Chatbot
"""

import os
from pathlib import Path

import streamlit as st

from rag import build_default_index, generate_answer


# ---------- Page config ----------

st.set_page_config(
    page_title="Krishnan Kannan • IDMC Architect",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------- Styling ----------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@400;500;700;900&family=JetBrains+Mono:wght@400;500&family=Inter+Tight:wght@400;500;600&display=swap');

:root {
    --ink: #1a1a1a;
    --paper: #f5f1ea;
    --accent: #c8462c;
    --muted: #6b6b6b;
    --line: #d4cec3;
}

html, body, [class*="css"] {
    font-family: 'Inter Tight', -apple-system, sans-serif;
    color: var(--ink);
}

.stApp {
    background: var(--paper);
    background-image:
        radial-gradient(at 0% 0%, rgba(200, 70, 44, 0.04) 0%, transparent 50%),
        radial-gradient(at 100% 100%, rgba(26, 26, 26, 0.03) 0%, transparent 50%);
}

/* Header */
.hero {
    border-top: 2px solid var(--ink);
    border-bottom: 1px solid var(--line);
    padding: 2rem 0 1.5rem 0;
    margin-bottom: 2rem;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.5rem;
}
.hero-name {
    font-family: 'Fraunces', serif;
    font-weight: 900;
    font-size: clamp(2.5rem, 6vw, 4.5rem);
    line-height: 0.95;
    letter-spacing: -0.02em;
    margin: 0;
}
.hero-role {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    color: var(--muted);
    margin-top: 0.75rem;
}
.hero-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 0.5rem;
}

/* Section headers */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    border-bottom: 1px solid var(--line);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* Cards */
.exp-card {
    border-left: 2px solid var(--ink);
    padding: 0.5rem 0 0.5rem 1.25rem;
    margin-bottom: 1.5rem;
}
.exp-role {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 1.2rem;
    line-height: 1.3;
    margin: 0;
}
.exp-company {
    font-family: 'Inter Tight', sans-serif;
    font-size: 0.95rem;
    color: var(--muted);
    margin: 0.15rem 0;
}
.exp-period {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.5rem;
}
.exp-desc {
    font-size: 0.92rem;
    line-height: 1.55;
    color: var(--ink);
}

/* Skill chips */
.chip {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    background: transparent;
    border: 1px solid var(--ink);
    color: var(--ink);
    padding: 0.25rem 0.6rem;
    margin: 0.15rem 0.25rem 0.15rem 0;
    border-radius: 2px;
}
.chip-accent { background: var(--ink); color: var(--paper); }

/* Stats row */
.stat-block { padding: 1rem 0; }
.stat-num {
    font-family: 'Fraunces', serif;
    font-weight: 900;
    font-size: 2.5rem;
    line-height: 1;
    color: var(--accent);
}
.stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.4rem;
}

/* Chat */
.chat-source {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    border-left: 2px solid var(--accent);
    padding-left: 0.6rem;
    margin: 0.3rem 0;
}

/* Streamlit overrides */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid var(--line);
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 0.75rem 1.25rem;
    background: transparent;
    border-radius: 0;
    color: var(--muted);
}
.stTabs [aria-selected="true"] {
    color: var(--ink) !important;
    border-bottom: 2px solid var(--accent);
}

.stButton > button {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    background: var(--ink);
    color: var(--paper);
    border: 1px solid var(--ink);
    border-radius: 2px;
    padding: 0.5rem 1rem;
}
.stButton > button:hover {
    background: var(--accent);
    border-color: var(--accent);
    color: var(--paper);
}

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
header { visibility: hidden; }

/* Reduce default padding */
.block-container { padding-top: 2rem !important; padding-bottom: 3rem !important; max-width: 1100px; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------- Cache the index ----------

@st.cache_resource(show_spinner="Building resume index...")
def get_index():
    return build_default_index("data/resume.txt")


# ---------- Hero ----------

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">◆ Portfolio · 2026</div>
    <h1 class="hero-name">Krishnan Kannan</h1>
    <div class="hero-role">Informatica IDMC Architect — Cloud Data Integration · Snowflake · Ab Initio</div>
    <div class="hero-meta">Bengaluru, IN  ·  15 yrs  ·  krishnankannan1236@gmail.com  ·  +91 96866 99011</div>
</div>
""", unsafe_allow_html=True)


# ---------- Stats ----------

c1, c2, c3, c4 = st.columns(4)
for col, num, label in [
    (c1, "15+", "Years experience"),
    (c2, "8", "Enterprises served"),
    (c3, "4", "Industries"),
    (c4, "1", "SnowPro cert"),
]:
    with col:
        st.markdown(f"""
        <div class="stat-block">
            <div class="stat-num">{num}</div>
            <div class="stat-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)


# ---------- Tabs ----------

tab_about, tab_exp, tab_skills, tab_chat = st.tabs(
    ["About", "Experience", "Skills & Stack", "Ask the Resume ↓"]
)


# ---------- About ----------

with tab_about:
    st.markdown('<div class="section-label">01 · Profile</div>', unsafe_allow_html=True)
    st.markdown("""
Informatica IDMC Architect with 15 years designing enterprise data integration, master data management, data quality, and warehousing solutions across **pharma, smart-home, financial services, and IT services**.

Deep practitioner of **Informatica IDMC** (CDI, CAI, CDQ, Cloud MDM / MDM 360), **Informatica MDM Hub** (multi-domain, on-premise), **PowerCenter**, **Dynamic Data Masking**, and **Ab Initio** for large-scale ETL on mainframe, Teradata, and DB2.

Proven on **Snowflake** warehouse builds, **AWS** data lakes, **Veeva Network / CRM** integrations, **PowerCenter-to-IDMC** migrations, and **CI/CD on Azure DevOps**. SnowPro Advanced: Data Engineer certified.
""")

    st.markdown('<div class="section-label">02 · Currently</div>', unsafe_allow_html=True)
    st.markdown("""
**Senior Solution Architect – IDMC at Merck Group** — leading Veeva Network ↔ Veeva CRM integration, architecting a multi-account/multi-region Snowflake platform, and modernising the data estate with Data Vault 2.0, Apache Iceberg, and Snowpipe Streaming.
""")

    st.markdown('<div class="section-label">03 · Pursuing</div>', unsafe_allow_html=True)
    st.markdown("""
**Executive MBA — Artificial Intelligence & Business Analytics**, PES University, Bengaluru.
""")


# ---------- Experience ----------

with tab_exp:
    st.markdown('<div class="section-label">Roles · Reverse chronological</div>', unsafe_allow_html=True)

    roles = [
        ("Aug 2022 – Present", "Senior Solution Architect – IDMC", "Merck Group, Bengaluru",
         "Led Veeva Network ↔ Veeva CRM integration. Architected multi-account/multi-region Snowflake with RBAC, resource monitors, Streams & Tasks, Dynamic Tables. Built Snowpipe + Snowpipe Streaming ingestion from S3 and Kafka. Implemented Apache Iceberg lakehouse, Data Vault 2.0, and Tableau self-service analytics. Built CI/CD on Azure DevOps for IDMC."),
        ("Aug 2021 – Aug 2022", "IDMC Solution Architect", "Resideo Smart Home Technologies, Bengaluru",
         "Architected multi-domain on-prem Informatica MDM Hub (Customer 360, Product 360, Supplier, RDM). Designed match, trust, and survivorship logic; built Stage Mappings, Hierarchy Manager, IDD with ActiveVOS stewardship. Migrated PowerCenter to IDMC."),
        ("Sep 2019 – Jun 2021", "IDMC Solution Architect / Ab Initio Architect", "Dun & Bradstreet, Bengaluru",
         "Hybrid IDMC + Ab Initio platform for high-volume credit & business data. Implemented Cloud MDM 360 (Customer, Supplier, Product). Designed Ab Initio graphs on Co>Operating System for billions-of-records partitioned execution."),
        ("Apr 2019 – Sep 2019", "IDMC Solution Architect", "NTT DATA, Bengaluru",
         "Led PowerCenter-to-IDMC migration. CDC pipelines via PowerExchange. Integrated Snowflake, Salesforce, BigQuery sources."),
        ("May 2017 – Aug 2018", "ETL Architect – Ab Initio & Informatica", "Larsen & Toubro Infotech, Bengaluru",
         "Enterprise ETL from mainframe, SAP, Oracle into Teradata/DB2. Managed Ab Initio code base in EME with tagging and impact analysis. Led offshore team."),
        ("Jun 2015 – Apr 2017", "Senior Informatica Developer", "Cognizant Technology Solutions, Chennai",
         "Implemented Informatica Dynamic Data Masking for PII/PCI in non-prod. CDC pipelines, PowerCenter-to-IDMC migration, SCD1/2 patterns."),
        ("Oct 2014 – Jun 2015", "Senior Informatica Developer", "GE Global Services, Hyderabad",
         "DDM for sensitive customer/financial data. Salesforce → Oracle ETL. PC-to-Cloud conversion service. Automic-orchestrated jobs."),
        ("Jan 2012 – Aug 2014", "Senior Informatica / IDMC Developer", "Brillio Technologies, Bengaluru",
         "Multi-source data warehouse loads with Type 1/2, CDC, incremental strategies. Performance tuning."),
    ]
    for period, role, company, desc in roles:
        st.markdown(f"""
        <div class="exp-card">
            <div class="exp-period">{period}</div>
            <p class="exp-role">{role}</p>
            <p class="exp-company">{company}</p>
            <p class="exp-desc">{desc}</p>
        </div>
        """, unsafe_allow_html=True)


# ---------- Skills ----------

with tab_skills:
    skill_groups = {
        "Informatica IDMC": ["CDI", "CAI", "CDQ", "Cloud MDM / MDM 360", "Mass Ingestion", "Taskflows", "Secure Agent", "PC→IDMC Migration"],
        "Informatica MDM (On-Prem)": ["MDM Hub Multi-Domain", "Customer 360", "Product 360", "Supplier", "RDM", "Match & Merge", "Trust & Survivorship", "Hierarchy Manager", "IDD", "ActiveVOS", "SIF APIs"],
        "Data Security": ["Informatica DDM", "Rule Sets", "Substitution", "Shuffling", "Format-Preserving Masking", "PII / PHI / PCI"],
        "PowerCenter & Ab Initio": ["PowerCenter 10.2 → 7.x", "PowerExchange", "IDQ", "GDE", "Co>Operating System", "EME", "Conduct>It", "Continuous Flows", "BRE"],
        "Snowflake": ["Snowpipe", "Snowpipe Streaming", "Streams & Tasks", "Dynamic Tables", "Time Travel", "Zero-Copy Cloning", "Data Sharing", "Row Access Policies", "Horizon", "Apache Iceberg", "Data Vault 2.0"],
        "Cloud": ["AWS S3", "Lambda", "Glue", "IAM", "Azure", "BigQuery", "Salesforce", "Veeva Network", "Veeva CRM"],
        "Languages": ["Python", "PySpark", "SQL", "PL/SQL", "Shell", "dbt", "Streamlit", "REST APIs"],
        "Databases": ["Oracle 12c/11g/10g", "SQL Server", "Teradata", "DB2", "MySQL", "MongoDB", "Exadata"],
        "DevOps": ["Azure DevOps CI/CD", "Git", "Automic", "Jenkins", "JIRA", "Tableau", "Power BI"],
    }
    for group, items in skill_groups.items():
        st.markdown(f'<div class="section-label">{group}</div>', unsafe_allow_html=True)
        chips_html = "".join(f'<span class="chip">{s}</span>' for s in items)
        st.markdown(chips_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Certifications</div>', unsafe_allow_html=True)
    st.markdown('<span class="chip chip-accent">SnowPro Advanced · Data Engineer</span>', unsafe_allow_html=True)


# ---------- Chat (RAG) ----------

with tab_chat:
    st.markdown('<div class="section-label">Ask the resume — powered by RAG</div>', unsafe_allow_html=True)
    st.caption("Section-aware chunks → MiniLM embeddings → FAISS cosine retrieval → LLM answer grounded in retrieved chunks.")

    # API key / provider selection
    col_l, col_r = st.columns([1, 3])
    with col_l:
        provider = st.selectbox("LLM provider", ["anthropic", "openai"], index=0)
    with col_r:
        key_env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        if not os.environ.get(key_env):
            api_key = st.text_input(
                f"{key_env}",
                type="password",
                help="Stored only in this session. Get one at console.anthropic.com or platform.openai.com.",
            )
            if api_key:
                os.environ[key_env] = api_key

    # Build index (cached)
    try:
        index = get_index()
    except Exception as e:
        st.error(f"Index build failed: {e}")
        st.stop()

    # Suggested questions
    st.markdown("**Try a question:**")
    suggestions = [
        "What's Krishnan's Snowflake experience?",
        "Has he worked in pharma?",
        "Tell me about his MDM expertise.",
        "What did he do at Dun & Bradstreet?",
        "Does he know Ab Initio?",
    ]
    cols = st.columns(len(suggestions))
    for i, s in enumerate(suggestions):
        with cols[i]:
            if st.button(s, key=f"sugg_{i}", use_container_width=True):
                st.session_state["pending_q"] = s

    # Chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                for s in m["sources"]:
                    st.markdown(f'<div class="chat-source">↳ {s}</div>', unsafe_allow_html=True)

    # Input
    user_q = st.chat_input("Ask about Krishnan's experience...")
    if "pending_q" in st.session_state:
        user_q = st.session_state.pop("pending_q")

    if user_q:
        st.session_state["messages"].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            if not os.environ.get(key_env):
                msg = f"Add your {key_env} above to generate an answer. Meanwhile, here are the most relevant resume sections:"
                hits = index.search(user_q, k=4)
                src = [f"{c.section}  (similarity {s:.2f})" for c, s in hits]
                st.markdown(msg)
                for s in src:
                    st.markdown(f'<div class="chat-source">↳ {s}</div>', unsafe_allow_html=True)
                st.session_state["messages"].append({"role": "assistant", "content": msg, "sources": src})
            else:
                with st.spinner("Retrieving + generating..."):
                    hits = index.search(user_q, k=4)
                    answer = generate_answer(user_q, hits, provider=provider)
                src = [f"{c.section}  (similarity {s:.2f})" for c, s in hits]
                st.markdown(answer)
                for s in src:
                    st.markdown(f'<div class="chat-source">↳ {s}</div>', unsafe_allow_html=True)
                st.session_state["messages"].append({"role": "assistant", "content": answer, "sources": src})


# ---------- Footer ----------

st.markdown("""
---
<div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #6b6b6b; text-align: center; letter-spacing: 0.1em;">
    ◆ &nbsp; BUILT WITH STREAMLIT · FAISS · SENTENCE-TRANSFORMERS &nbsp; ◆
</div>
""", unsafe_allow_html=True)
