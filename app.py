import streamlit as st
import os
import time
from rag_chain import load_rag_chain, ask_question

st.set_page_config(
    page_title="DocMind AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #667eea11 0%, #764ba211 50%, #f093fb11 100%);
    background-color: #f4f6ff;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; }

/* Sidebar file uploader */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.08) !important;
    border: 1px dashed rgba(255,255,255,0.3) !important;
    border-radius: 12px !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton button {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.2) !important;
    transform: translateY(-1px) !important;
}

/* Main chat area */
[data-testid="stChatMessage"] {
    background: #ffffff !important;
    border-radius: 16px !important;
    border: 1px solid #e8ecff !important;
    box-shadow: 0 2px 12px rgba(102,126,234,0.08) !important;
    margin-bottom: 12px !important;
    padding: 4px 8px !important;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    border-radius: 24px !important;
    border: 2px solid #e0e4ff !important;
    background: #ffffff !important;
    box-shadow: 0 4px 20px rgba(102,126,234,0.1) !important;
    font-size: 15px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 4px 20px rgba(102,126,234,0.2) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: #f8f9ff !important;
    border: 1px solid #e0e4ff !important;
    border-radius: 12px !important;
}
.streamlit-expanderHeader {
    font-size: 13px !important;
    color: #667eea !important;
    font-weight: 600 !important;
}

/* Spinner */
.stSpinner > div { border-top-color: #667eea !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f4f6ff; }
::-webkit-scrollbar-thumb { background: #c5ceff; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #667eea; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <div style='font-size:44px;'></div>
        <h1 style='color:#ffffff; font-size:22px; margin:8px 0 4px; font-weight:700;'>DocMind AI</h1>
        <p style='color:rgba(255,255,255,0.5); font-size:12px; margin:0;'>Powered by Gemini + LangChain</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.1); margin:16px 0;'>", unsafe_allow_html=True)

    st.markdown("<p style='color:rgba(255,255,255,0.7); font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>📄 Upload Documents</p>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        for f in uploaded_files:
            save_path = os.path.join("docs", f.name)
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())
        st.markdown(f"<p style='color:#a0ffb8; font-size:12px;'>✅ {len(uploaded_files)} file(s) ready</p>", unsafe_allow_html=True)

        if st.button("⚡ Index Documents"):
            with st.spinner("Indexing..."):
                from ingest import ingest_documents
                ingest_documents()
                st.cache_resource.clear()
            st.success("Indexed successfully!")
            st.rerun()

    st.markdown("<hr style='border-color:rgba(255,255,255,0.1); margin:16px 0;'>", unsafe_allow_html=True)

    st.markdown("<p style='color:rgba(255,255,255,0.7); font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>🗂️ Chat History</p>", unsafe_allow_html=True)

    if "messages" in st.session_state and st.session_state.messages:
        user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
        for i, msg in enumerate(user_msgs[-6:]):
            preview = msg["content"][:35] + "..." if len(msg["content"]) > 35 else msg["content"]
            st.markdown(f"""
            <div style='padding:8px 12px; margin:4px 0; background:rgba(255,255,255,0.07);
            border-radius:10px; font-size:12px; color:rgba(255,255,255,0.8);
            border-left:3px solid #667eea;'>
            💬 {preview}
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:rgba(255,255,255,0.3); font-size:12px;'>No history yet</p>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.1); margin:16px 0;'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        msg_count = len([m for m in st.session_state.get("messages", []) if m["role"] == "user"])
        st.markdown(f"<div style='text-align:center; padding:8px; background:rgba(255,255,255,0.07); border-radius:10px;'><span style='color:#a0c4ff; font-size:18px; font-weight:700;'>{msg_count}</span><br><span style='color:rgba(255,255,255,0.4); font-size:11px;'>questions</span></div>", unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────
col_main, col_gap = st.columns([1, 0.001])
with col_main:

    st.markdown("""
    <div style='padding: 24px 0 8px;'>
        <h1 style='font-size:28px; font-weight:700; color:#1a1a2e; margin:0;'>
             DocMind AI
        </h1>
        <p style='color:#888; font-size:15px; margin:4px 0 0;'>
            Ask anything about your documents — powered by AI
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stats bar
    if "messages" in st.session_state and st.session_state.messages:
        q_count = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.markdown(f"""
        <div style='display:flex; gap:12px; margin:12px 0 20px; flex-wrap:wrap;'>
            <div style='background:#eef0ff; border-radius:10px; padding:8px 16px; font-size:13px; color:#534AB7;'>
                💬 <b>{q_count}</b> questions asked
            </div>
            <div style='background:#e1f5ee; border-radius:10px; padding:8px 16px; font-size:13px; color:#0F6E56;'>
                ✅ Chatbot active
            </div>
            <div style='background:#faeeda; border-radius:10px; padding:8px 16px; font-size:13px; color:#854F0B;'>
                ⚡ Gemini 2.5 Flash
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border:none; border-top:1px solid #e8ecff; margin-bottom:20px;'>", unsafe_allow_html=True)

    @st.cache_resource
    def get_chain():
        return load_rag_chain()

    chain_tuple = get_chain()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Welcome screen
    if not st.session_state.messages:
        st.markdown("""
        <div style='text-align:center; padding:50px 20px;'>
            <div style='font-size:60px; margin-bottom:16px;'>🤖</div>
            <h2 style='color:#1a1a2e; font-size:22px; font-weight:700; margin-bottom:8px;'>
                Welcome to DocMind AI!
            </h2>
            <p style='color:#888; font-size:15px; max-width:400px; margin:0 auto 24px;'>
                Upload your PDFs in the sidebar, then ask me anything about them.
            </p>
            <div style='display:flex; gap:10px; justify-content:center; flex-wrap:wrap;'>
                <div style='background:#eef0ff; border-radius:20px; padding:8px 18px; font-size:13px; color:#534AB7; border:1px solid #c5ceff;'>
                    📖 Summarize documents
                </div>
                <div style='background:#e1f5ee; border-radius:20px; padding:8px 18px; font-size:13px; color:#0F6E56; border:1px solid #9FE1CB;'>
                    ❓ Answer questions
                </div>
                <div style='background:#faeeda; border-radius:20px; padding:8px 18px; font-size:13px; color:#854F0B; border:1px solid #FAC775;'>
                    🔍 Find information
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📚 View Sources"):
                    for src in msg["sources"]:
                        file = os.path.basename(src.get("source", "unknown"))
                        page = src.get("page", "?")
                        snippet = src.get("snippet", "")
                        st.markdown(f"""
                        <div style='background:#f8f9ff; border-radius:10px; padding:12px;
                        margin:6px 0; border-left:4px solid #667eea;'>
                            <div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
                                <span style='background:#eef0ff; color:#534AB7; font-size:11px;
                                padding:2px 8px; border-radius:20px; font-weight:600;'>
                                📄 {file}
                                </span>
                                <span style='color:#aaa; font-size:12px;'>Page {page}</span>
                            </div>
                            <p style='color:#555; font-size:13px; margin:0; line-height:1.5;'>
                            {snippet}
                            </p>
                        </div>""", unsafe_allow_html=True)

    # Chat input
    if question := st.chat_input("Ask something about your documents..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Thinking..."):
                answer, sources = ask_question(chain_tuple, question)

            placeholder = st.empty()
            displayed = ""
            for char in answer:
                displayed += char
                placeholder.markdown(displayed + "▌")
                time.sleep(0.008)
            placeholder.markdown(displayed)

            source_data = []
            for doc in sources:
                source_data.append({
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "?"),
                    "snippet": doc.page_content[:150] + "..."
                })

            with st.expander("📚 View Sources"):
                for src in source_data:
                    file = os.path.basename(src["source"])
                    st.markdown(f"""
                    <div style='background:#f8f9ff; border-radius:10px; padding:12px;
                    margin:6px 0; border-left:4px solid #667eea;'>
                        <div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
                            <span style='background:#eef0ff; color:#534AB7; font-size:11px;
                            padding:2px 8px; border-radius:20px; font-weight:600;'>
                            📄 {file}
                            </span>
                            <span style='color:#aaa; font-size:12px;'>Page {src["page"]}</span>
                        </div>
                        <p style='color:#555; font-size:13px; margin:0; line-height:1.5;'>
                        {src["snippet"]}
                        </p>
                    </div>""", unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": source_data
        })