"""
Galaxy Unlearning Lab — Streamlit Frontend
Space-themed dark UI with full explanations
"""   

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import os
import psutil
from torchvision import transforms

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Galaxy Unlearning Lab",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: radial-gradient(ellipse at 20% 50%, #0d1b2a 0%, #050a0f 60%, #000000 100%);
    color: #e2e8f0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #050d1a 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0a1628;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e3a5f;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #64748b !important;
    font-weight: 600;
    font-size: 14px;
    padding: 8px 20px;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
    color: white !important;
    box-shadow: 0 2px 12px rgba(59,130,246,0.4);
}

/* ── Cards ── */
.card {
    background: linear-gradient(135deg, #0f2744 0%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 24px;
    margin: 8px 0;
    transition: border-color 0.2s;
}
.card:hover { border-color: #3b82f6; }

/* ── Metric card ── */
.metric-card {
    background: linear-gradient(135deg, #0f2744 0%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
}
.metric-value { font-size: 28px; font-weight: 800; font-family: 'Space Grotesk', sans-serif; }
.metric-label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0f2744 0%, #1e1b4b 50%, #0f2744 100%);
    border: 1px solid #312e81;
    border-radius: 20px;
    padding: 32px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}
.hero-sub { color: #94a3b8; font-size: 15px; line-height: 1.6; }

/* ── Section title ── */
.section-title {
    font-size: 11px;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 16px 0 8px 0;
}

/* ── Prediction result ── */
.pred-result {
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin: 12px 0;
}
.pred-class { font-size: 36px; font-weight: 800; font-family: 'Space Grotesk', sans-serif; }
.pred-desc { font-size: 13px; color: #94a3b8; margin-top: 6px; }

/* ── Prob bar ── */
.prob-row { margin: 8px 0; }
.prob-label { display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 13px; }
.prob-track { background: #1e3a5f; border-radius: 6px; height: 8px; overflow: hidden; }
.prob-fill { height: 8px; border-radius: 6px; transition: width 0.5s ease; }

/* ── Info box ── */
.info-box {
    background: rgba(30,58,95,0.4);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 13px;
    color: #94a3b8;
    line-height: 1.7;
    margin: 8px 0;
}
.info-box b { color: #e2e8f0; }

/* ── Badge ── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    margin: 2px;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    width: 100% !important;
    box-shadow: 0 4px 15px rgba(29,78,216,0.4) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    box-shadow: 0 6px 20px rgba(29,78,216,0.6) !important;
    transform: translateY(-1px) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0a1628;
    border: 2px dashed #1e3a5f;
    border-radius: 12px;
    padding: 20px;
}
[data-testid="stFileUploader"]:hover { border-color: #3b82f6; }

/* ── Divider ── */
hr { border-color: #1e3a5f !important; margin: 20px 0 !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #0a1628 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #1e3a5f; border-radius: 8px; }

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg, #1d4ed8, #7c3aed) !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #3b82f6 !important; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ────────────────────────────────────────────────────────────────
CLASS_NAMES  = ['Smooth', 'Featured/Disk', 'Artifact']
CLASS_COLORS = ['#60a5fa', '#34d399', '#f87171']
CLASS_ICONS  = ['🔵', '🌀', '⭐']
CLASS_DESC   = [
    'Round, featureless elliptical galaxy — no spiral arms',
    'Spiral or disk galaxy with visible arms and structure',
    'Star, imaging noise, or unclassifiable object'
]

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for key, val in [('unlearn_results', None), ('unlearn_done', False), ('prediction', None)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── HELPERS ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    from model import create_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for path in ['best_model.pth', 'corrupted_model.pth', 'baseline_model.pth']:
        if os.path.exists(path):
            try:
                from model import load_model as _lm
                m = _lm(path, device=device)
                return m, device, True, path
            except Exception:
                pass
    return create_model(device=device), device, False, None


def load_latest_unlearning_results():
    for path in ['unlearning_results.csv', 'results.csv']:
        if os.path.exists(path):
            try:
                return pd.read_csv(path), path
            except Exception:
                pass
    return None, None


def training_summary():
    if not os.path.exists('training_log.csv'):
        return None

    df = pd.read_csv('training_log.csv')
    if df.empty:
        return None

    best_idx = df['val_accuracy'].idxmax()
    best_row = df.loc[best_idx]
    return {
        'epochs': len(df),
        'best_val_accuracy': float(best_row['val_accuracy']),
        'best_balanced_accuracy': float(best_row['val_balanced_accuracy']) if 'val_balanced_accuracy' in df.columns else None,
        'best_macro_f1': float(best_row['val_macro_f1']) if 'val_macro_f1' in df.columns else None,
        'best_epoch': int(best_row['epoch'])
    }

def preprocess(img):
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return t(img).unsqueeze(0)

def predict(model, img, device):
    model.eval()
    with torch.no_grad():
        out = model(preprocess(img).to(device))
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    return int(np.argmax(probs)), probs

def sys_stats():
    s = {'cpu': psutil.cpu_percent(), 'ram': psutil.virtual_memory().percent}
    if torch.cuda.is_available():
        s['gpu'] = torch.cuda.get_device_name(0)
        s['vram_used'] = torch.cuda.memory_allocated()/1e6
        s['vram_total'] = torch.cuda.get_device_properties(0).total_memory/1e6
    return s

model, device, model_ready, model_path = load_model()
latest_results_df, latest_results_path = load_latest_unlearning_results()
train_summary = training_summary()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 10px 0;'>
        <div style='font-size:40px;'>🌌</div>
        <div style='font-family:Space Grotesk,sans-serif; font-size:18px; font-weight:700;
                    background:linear-gradient(135deg,#60a5fa,#a78bfa);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            Galaxy Unlearning Lab
        </div>
        <div style='font-size:11px; color:#475569; margin-top:4px;'>
            Machine Unlearning Research
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Model status
    st.markdown('<div class="section-title">Model Status</div>', unsafe_allow_html=True)
    if model_ready:
        st.markdown(f"""
        <div style='background:rgba(52,211,153,0.1); border:1px solid #065f46;
                    border-radius:8px; padding:10px 14px; font-size:13px;'>
            ✅ <b style='color:#34d399'>Model Ready</b><br>
            <span style='color:#475569; font-size:11px;'>{model_path}</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:rgba(248,113,113,0.1); border:1px solid #7f1d1d;
                    border-radius:8px; padding:10px 14px; font-size:13px;'>
            ❌ <b style='color:#f87171'>No Model Found</b><br>
            <span style='color:#475569; font-size:11px;'>Run training first to create best_model.pth</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Galaxy classes
    st.markdown('<div class="section-title">Galaxy Classes</div>', unsafe_allow_html=True)
    for icon, name, color, desc in zip(CLASS_ICONS, CLASS_NAMES, CLASS_COLORS, CLASS_DESC):
        st.markdown(f"""
        <div style='border-left:3px solid {color}; padding:8px 12px; margin:6px 0;
                    background:rgba(15,39,68,0.6); border-radius:0 8px 8px 0;'>
            <div style='font-weight:600; color:{color}; font-size:13px;'>{icon} {name}</div>
            <div style='color:#475569; font-size:11px; margin-top:2px;'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # System stats
    st.markdown('<div class="section-title">System</div>', unsafe_allow_html=True)
    stats = sys_stats()
    c1, c2 = st.columns(2)
    c1.metric("CPU", f"{stats['cpu']:.0f}%")
    c2.metric("RAM", f"{stats['ram']:.0f}%")
    if 'gpu' in stats:
        st.markdown(f"""
        <div style='background:rgba(15,39,68,0.6); border:1px solid #1e3a5f;
                    border-radius:8px; padding:10px; font-size:12px; margin-top:8px;'>
            🎮 <b>{stats['gpu']}</b><br>
            <div style='background:#1e3a5f; border-radius:4px; height:6px; margin:6px 0;'>
                <div style='background:linear-gradient(90deg,#3b82f6,#7c3aed);
                            width:{min(stats["vram_used"]/stats["vram_total"]*100,100):.0f}%;
                            height:6px; border-radius:4px;'></div>
            </div>
            <span style='color:#475569;'>{stats["vram_used"]:.0f} / {stats["vram_total"]:.0f} MB VRAM</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">⚠️ Running on <b>CPU</b></div>', unsafe_allow_html=True)

# ── MAIN TABS ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔭  Classify Galaxy", "🧠  Unlearning Dashboard", "📊  Training Metrics"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — GALAXY CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-title">🔭 Galaxy Morphology Classifier</div>
        <div class="hero-sub">
            Upload any galaxy image and our ResNet34 model
            """ + (f"""— best validation accuracy <b style='color:#34d399'>{train_summary['best_val_accuracy']:.2f}%</b>
            at epoch <b style='color:#60a5fa'>{train_summary['best_epoch']}</b> —""" if train_summary else "— trained on 61,578 real Galaxy Zoo 2 images —") + """
            will classify it into one of three morphological types with confidence scores.
        </div>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="section-title">Step 1 — Upload Galaxy Image</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            📁 Use any galaxy image from:<br>
            &nbsp;&nbsp;• <b>Galaxy Zoo test set</b> → <code>data/images_test_rev1/</code><br>
            &nbsp;&nbsp;• <b>NASA Hubble</b> → hubblesite.org<br>
            &nbsp;&nbsp;• <b>Any JPG/PNG</b> from the internet
        </div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader("", type=['jpg','jpeg','png'], label_visibility="collapsed")

        if uploaded:
            img = Image.open(uploaded).convert('RGB')
            st.image(img, use_column_width=True, caption="Your galaxy image")

            st.markdown('<div class="section-title" style="margin-top:16px;">Step 2 — Run Classification</div>', unsafe_allow_html=True)
            if st.button("🚀 Classify this Galaxy", disabled=not model_ready):
                with st.spinner("Analyzing morphology..."):
                    idx, probs = predict(model, img, device)
                    st.session_state.prediction = (idx, probs)

    with right:
        if not uploaded:
            st.markdown("""
            <div class="card" style="margin-top:48px; text-align:center;">
                <div style='font-size:64px; margin-bottom:16px;'>🌌</div>
                <div style='font-size:18px; font-weight:600; color:#60a5fa; margin-bottom:8px;'>
                    Upload a galaxy image to begin
                </div>
                <div style='color:#475569; font-size:13px;'>
                    The AI will classify it as Smooth, Featured/Disk, or Artifact
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-title" style="margin-top:24px;">Galaxy Type Reference</div>', unsafe_allow_html=True)
            for icon, name, color, desc in zip(CLASS_ICONS, CLASS_NAMES, CLASS_COLORS, CLASS_DESC):
                st.markdown(f"""
                <div class="card" style="padding:14px 18px; margin:6px 0;">
                    <span style='font-size:20px;'>{icon}</span>
                    <span style='font-weight:700; color:{color}; margin-left:8px;'>{name}</span><br>
                    <span style='color:#64748b; font-size:12px;'>{desc}</span>
                </div>""", unsafe_allow_html=True)

        elif st.session_state.prediction:
            idx, probs = st.session_state.prediction
            name  = CLASS_NAMES[idx]
            color = CLASS_COLORS[idx]
            icon  = CLASS_ICONS[idx]
            conf  = probs[idx] * 100

            st.markdown(f"""
            <div class="pred-result" style="background:linear-gradient(135deg,{color}18,{color}08);
                         border:2px solid {color}55;">
                <div style='font-size:48px;'>{icon}</div>
                <div class="pred-class" style="color:{color};">{name}</div>
                <div class="pred-desc">{CLASS_DESC[idx]}</div>
                <div style='margin-top:12px; font-size:32px; font-weight:800; color:white;'>
                    {conf:.1f}% confident
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-title">Confidence Breakdown</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-box">How confident the model is for each galaxy type. The highest bar is the predicted class.</div>', unsafe_allow_html=True)

            for i, (n, p, c) in enumerate(zip(CLASS_NAMES, probs, CLASS_COLORS)):
                bold = "font-weight:700; color:white;" if i == idx else "color:#64748b;"
                st.markdown(f"""
                <div class="prob-row">
                    <div class="prob-label">
                        <span style='{bold}'>{CLASS_ICONS[i]} {n}</span>
                        <span style='color:{c}; font-weight:700;'>{p*100:.1f}%</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill" style="width:{p*100:.1f}%; background:{c};"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

            fig = go.Figure(go.Bar(
                x=CLASS_NAMES, y=probs*100,
                marker_color=CLASS_COLORS,
                text=[f"{p*100:.1f}%" for p in probs],
                textposition='outside', textfont=dict(color='white')
            ))
            fig.update_layout(
                template='plotly_dark', paper_bgcolor='#0a1628', plot_bgcolor='#0a1628',
                height=280, yaxis_title="Confidence (%)", showlegend=False,
                margin=dict(t=30, b=10, l=10, r=10),
                yaxis=dict(gridcolor='#1e3a5f'), xaxis=dict(gridcolor='#1e3a5f')
            )
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — UNLEARNING DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="hero">
        <div class="hero-title">🧠 Machine Unlearning Dashboard</div>
        <div class="hero-sub">
            Latest experiment results are loaded from the most recent unlearning CSV.
            Compare Gradient Ascent, Fisher Forgetting, and Full Retrain using your current trained model.
        </div>
    </div>""", unsafe_allow_html=True)

    # Explainer cards
    c1, c2, c3 = st.columns(3)
    time_lookup = {}
    if latest_results_df is not None and 'computation_time_seconds' in latest_results_df.columns:
        for _, row in latest_results_df.iterrows():
            time_lookup[str(row['method'])] = f"{float(row['computation_time_seconds']):.1f}s"

    for col, icon, title, desc, color, time in zip(
        [c1, c2, c3],
        ['⚡', '🎲', '🔄'],
        ['GradientAscent', 'FisherForgetting', 'FullRetrain'],
        [
            'Maximizes loss on bad samples — forces model to forget wrong patterns',
            'Adds calibrated noise to weights based on Fisher Information Matrix',
            'Retrains from scratch without mislabeled data — gold standard'
        ],
        ['#60a5fa', '#a78bfa', '#34d399'],
        [
            time_lookup.get('GradientAscent', '~100s'),
            time_lookup.get('FisherForgetting', '~214s'),
            time_lookup.get('FullRetrain', '~2649s')
        ]
    ):
        col.markdown(f"""
        <div class="card" style="border-top:3px solid {color}; text-align:center;">
            <div style='font-size:32px;'>{icon}</div>
            <div style='font-weight:700; color:{color}; font-size:15px; margin:8px 0;'>{title.replace('GradientAscent','Gradient Ascent').replace('FisherForgetting','Fisher Forgetting').replace('FullRetrain','Full Retrain')}</div>
            <div style='color:#64748b; font-size:12px; line-height:1.6;'>{desc}</div>
            <div style='margin-top:10px; background:{color}22; border-radius:6px; padding:4px 10px;
                        display:inline-block; font-size:12px; color:{color}; font-weight:600;'>
                ⏱ {time}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Metrics explainer
    st.markdown("""
    <div class="info-box">
        <b>📖 How to read the results:</b><br>
        🎯 <b>Test Accuracy</b> — Overall accuracy on clean test images. Higher = better.<br>
        🗑️ <b>Forget Accuracy</b> — Accuracy on mislabeled samples. <b style='color:#f87171'>Lower = better</b>
        (means the model forgot the wrong labels!)<br>
        💾 <b>Retain Accuracy</b> — Accuracy on correct samples. <b style='color:#34d399'>Higher = better</b>
        (means correct knowledge is preserved!)<br>
        ⏱️ <b>Time</b> — How long the unlearning took.
    </div>""", unsafe_allow_html=True)

    # Load existing results
    if latest_results_df is not None and not st.session_state.unlearn_done:
        st.session_state.unlearn_results = latest_results_df
        st.session_state.unlearn_done = True

    btn_col, info_col = st.columns([1, 2])
    with btn_col:
        run = st.button("🔄 Run Unlearning Pipeline", disabled=not model_ready)
    with info_col:
        if not model_ready:
            st.markdown('<div class="info-box">⚠️ Run <code>python main.py</code> first to train the model</div>', unsafe_allow_html=True)
        elif st.session_state.unlearn_done:
            path_label = latest_results_path if latest_results_path else 'results file'
            st.markdown(f'<div class="info-box" style="border-color:#065f46;">✅ Latest results loaded from <code>{path_label}</code> — click button to re-run</div>', unsafe_allow_html=True)

    if run and model_ready:
        bar = st.progress(0)
        status = st.empty()
        try:
            from data_loader import get_data_loaders, inject_mislabels
            from unlearn import UnlearningMethods, create_forget_retain_loaders
            from evaluate import comprehensive_evaluation

            status.info("📂 Loading dataset...")
            bar.progress(10)
            train_loader, val_loader, test_loader, dataset, train_idx, class_weights = get_data_loaders(batch_size=16)

            status.info("💥 Injecting mislabels (12%)...")
            bar.progress(20)
            mis_idx, _ = inject_mislabels(dataset, mislabel_ratio=0.12)
            forget_loader, retain_loader = create_forget_retain_loaders(dataset, train_idx, mis_idx)

            unlearner = UnlearningMethods(model, device)
            results = []

            status.info("📊 Evaluating baseline (no unlearning)...")
            bar.progress(30)
            results.append(comprehensive_evaluation(model, test_loader, forget_loader, retain_loader, device, "No Unlearning", 0.0))

            status.info("⚡ Running Gradient Ascent Unlearning...")
            bar.progress(50)
            ga_m, ga_t = unlearner.gradient_ascent_unlearning(forget_loader, num_epochs=3)
            results.append(comprehensive_evaluation(ga_m, test_loader, forget_loader, retain_loader, device, "Gradient Ascent", ga_t))

            status.info("🎲 Running Fisher Forgetting...")
            bar.progress(75)
            ff_m, ff_t = unlearner.fisher_forgetting(retain_loader)
            results.append(comprehensive_evaluation(ff_m, test_loader, forget_loader, retain_loader, device, "Fisher Forgetting", ff_t))

            bar.progress(100)
            status.success("✅ All unlearning methods complete!")

            df = pd.DataFrame(results)
            df.to_csv('unlearning_results.csv', index=False)
            st.session_state.unlearn_results = df
            st.session_state.unlearn_done = True
        except Exception as e:
            st.error(f"Error: {e}")

    # Show results
    if st.session_state.unlearn_done and st.session_state.unlearn_results is not None:
        df = st.session_state.unlearn_results
        st.markdown("---")
        st.markdown('<div class="section-title">Results Summary</div>', unsafe_allow_html=True)

        cols = st.columns(len(df))
        for col, (_, row) in zip(cols, df.iterrows()):
            pretty_method = str(row['method']).replace('GradientAscent','Gradient Ascent').replace('FisherForgetting','Fisher Forgetting').replace('FullRetrain','Full Retrain')
            color = {'No Unlearning':'#64748b','Gradient Ascent':'#60a5fa',
                     'Fisher Forgetting':'#a78bfa','Full Retrain':'#34d399'}.get(pretty_method,'#60a5fa')
            col.markdown(f"""
            <div class="metric-card" style="border-top:3px solid {color};">
                <div style='font-weight:700; color:{color}; font-size:13px; margin-bottom:12px;'>{pretty_method}</div>
                <div class="metric-value" style="color:white;">{row['test_accuracy']:.1f}%</div>
                <div class="metric-label">Test Accuracy</div>
                <hr style='border-color:#1e3a5f; margin:10px 0;'>
                <div style='font-size:20px; font-weight:700; color:#f87171;'>{row['forget_accuracy']:.1f}%</div>
                <div class="metric-label">Forget Acc ↓</div>
                <div style='font-size:20px; font-weight:700; color:#34d399; margin-top:6px;'>{row['retain_accuracy']:.1f}%</div>
                <div class="metric-label">Retain Acc ↑</div>
                <div style='font-size:16px; font-weight:600; color:#e3b341; margin-top:6px;'>{row['computation_time_seconds']:.1f}s</div>
                <div class="metric-label">Time</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="margin-top:24px;">Accuracy Comparison</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for metric, label, color in [
            ('test_accuracy','Test Accuracy','#60a5fa'),
            ('forget_accuracy','Forget Accuracy (lower=better)','#f87171'),
            ('retain_accuracy','Retain Accuracy','#34d399')
        ]:
            fig.add_trace(go.Bar(
                name=label,
                x=[str(m).replace('GradientAscent','Gradient Ascent').replace('FisherForgetting','Fisher Forgetting').replace('FullRetrain','Full Retrain') for m in df['method']],
                y=df[metric],
                marker_color=color,
                text=[f"{v:.1f}%" for v in df[metric]], textposition='outside',
                textfont=dict(color='white', size=11)
            ))
        fig.update_layout(
            barmode='group', template='plotly_dark',
            paper_bgcolor='#0a1628', plot_bgcolor='#0a1628',
            height=420, yaxis_title="Accuracy (%)",
            legend=dict(orientation='h', y=-0.25, font=dict(color='#94a3b8')),
            margin=dict(t=20, b=80),
            yaxis=dict(gridcolor='#1e3a5f'), xaxis=dict(gridcolor='#1e3a5f')
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Computation Time</div>', unsafe_allow_html=True)
        fig2 = go.Figure(go.Bar(
            x=[str(m).replace('GradientAscent','Gradient Ascent').replace('FisherForgetting','Fisher Forgetting').replace('FullRetrain','Full Retrain') for m in df['method']],
            y=df['computation_time_seconds'],
            marker_color=['#64748b','#60a5fa','#a78bfa','#34d399'][:len(df)],
            text=[f"{v:.1f}s" for v in df['computation_time_seconds']],
            textposition='outside', textfont=dict(color='white')
        ))
        fig2.update_layout(
            template='plotly_dark', paper_bgcolor='#0a1628', plot_bgcolor='#0a1628',
            height=300, yaxis_title="Seconds", showlegend=False,
            margin=dict(t=20, b=20),
            yaxis=dict(gridcolor='#1e3a5f'), xaxis=dict(gridcolor='#1e3a5f')
        )
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("📋 Full Results Table"):
            st.dataframe(df, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRAINING METRICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="hero">
        <div class="hero-title">📊 Training Metrics</div>
        <div class="hero-sub">
            """ + (f"""Latest training run reached <b style='color:#34d399'>{train_summary['best_val_accuracy']:.2f}%</b>
            validation accuracy in <b style='color:#60a5fa'>{train_summary['epochs']}</b> epochs.""" if train_summary else "See how the ResNet34 model learned to classify galaxies over time.") + """
        </div>
    </div>""", unsafe_allow_html=True)

    if not os.path.exists('training_log.csv'):
        st.markdown("""
        <div class="card" style="text-align:center; padding:40px;">
            <div style='font-size:48px;'>📭</div>
            <div style='font-size:18px; font-weight:600; color:#60a5fa; margin:12px 0;'>No training data yet</div>
            <div style='color:#475569;'>Run <code>python main.py</code> to train the model first</div>
        </div>""", unsafe_allow_html=True)
    else:
        df = pd.read_csv('training_log.csv')

        # Summary row
        c1, c2, c3, c4 = st.columns(4)
        balanced_label = f"{df['val_balanced_accuracy'].max():.1f}%" if 'val_balanced_accuracy' in df.columns else "N/A"
        macro_f1_label = f"{df['val_macro_f1'].max():.1f}%" if 'val_macro_f1' in df.columns else "N/A"
        for col, val, label, color in zip(
            [c1, c2, c3, c4],
            [f"{df['val_accuracy'].max():.1f}%", balanced_label, macro_f1_label, str(len(df))],
            ["Best Val Accuracy", "Best Balanced Acc", "Best Macro-F1", "Total Epochs"],
            ["#34d399", "#60a5fa", "#a78bfa", "#e3b341"]
        ):
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color};">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Loss curve
        st.markdown('<div class="section-title">Training & Validation Loss</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            📉 <b>Loss</b> = how wrong the model is. Both lines should go <b>DOWN</b> over time.<br>
            If val loss rises while train loss drops → <b style='color:#f87171'>overfitting</b> (memorizing instead of learning).
        </div>""", unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['epoch'], y=df['train_loss'], name='Train Loss',
            mode='lines+markers', line=dict(color='#60a5fa', width=2),
            marker=dict(size=6, color='#60a5fa'),
            fill='tozeroy', fillcolor='rgba(96,165,250,0.05)'
        ))
        fig.add_trace(go.Scatter(
            x=df['epoch'], y=df['val_loss'], name='Validation Loss',
            mode='lines+markers', line=dict(color='#f87171', width=2),
            marker=dict(size=6, color='#f87171'),
            fill='tozeroy', fillcolor='rgba(248,113,113,0.05)'
        ))
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='#0a1628', plot_bgcolor='#0a1628',
            height=360, xaxis_title="Epoch", yaxis_title="Loss",
            legend=dict(orientation='h', y=-0.2, font=dict(color='#94a3b8')),
            margin=dict(t=10, b=60),
            yaxis=dict(gridcolor='#1e3a5f'), xaxis=dict(gridcolor='#1e3a5f', dtick=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Accuracy curve
        st.markdown('<div class="section-title">Validation Accuracy</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            📈 <b>Accuracy</b> = % of galaxies correctly classified. Should go <b>UP</b> over epochs.
        </div>""", unsafe_allow_html=True)

        fig2 = go.Figure(go.Scatter(
            x=df['epoch'], y=df['val_accuracy'], name='Val Accuracy',
            mode='lines+markers', line=dict(color='#34d399', width=2),
            marker=dict(size=6, color='#34d399'),
            fill='tozeroy', fillcolor='rgba(52,211,153,0.08)'
        ))
        fig2.update_layout(
            template='plotly_dark', paper_bgcolor='#0a1628', plot_bgcolor='#0a1628',
            height=320, xaxis_title="Epoch", yaxis_title="Accuracy (%)",
            showlegend=False, margin=dict(t=10, b=20),
            yaxis=dict(gridcolor='#1e3a5f'), xaxis=dict(gridcolor='#1e3a5f', dtick=1)
        )
        st.plotly_chart(fig2, use_container_width=True)

        if 'val_balanced_accuracy' in df.columns or 'val_macro_f1' in df.columns:
            st.markdown('<div class="section-title">Minority-Aware Metrics</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
                🎯 <b>Balanced Accuracy</b> and <b>Macro-F1</b> matter here because the Artifact class is tiny.
                They show whether the model is improving fairly across classes, not just on the majority labels.
            </div>""", unsafe_allow_html=True)

            fig3 = go.Figure()
            if 'val_balanced_accuracy' in df.columns:
                fig3.add_trace(go.Scatter(
                    x=df['epoch'], y=df['val_balanced_accuracy'], name='Balanced Accuracy',
                    mode='lines+markers', line=dict(color='#60a5fa', width=2),
                    marker=dict(size=6, color='#60a5fa')
                ))
            if 'val_macro_f1' in df.columns:
                fig3.add_trace(go.Scatter(
                    x=df['epoch'], y=df['val_macro_f1'], name='Macro-F1',
                    mode='lines+markers', line=dict(color='#a78bfa', width=2),
                    marker=dict(size=6, color='#a78bfa')
                ))
            fig3.update_layout(
                template='plotly_dark', paper_bgcolor='#0a1628', plot_bgcolor='#0a1628',
                height=320, xaxis_title="Epoch", yaxis_title="Score (%)",
                legend=dict(orientation='h', y=-0.2, font=dict(color='#94a3b8')),
                margin=dict(t=10, b=60),
                yaxis=dict(gridcolor='#1e3a5f'), xaxis=dict(gridcolor='#1e3a5f', dtick=1)
            )
            st.plotly_chart(fig3, use_container_width=True)

        # Confusion matrices
        st.markdown('<div class="section-title">Confusion Matrices</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            🔲 <b>Confusion Matrix</b> = shows where the model gets confused.<br>
            <b>Diagonal</b> = correct predictions ✅ &nbsp;|&nbsp; <b>Off-diagonal</b> = mistakes ❌<br>
            After unlearning, the matrix should improve (more correct predictions).
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists('plots/confusion_matrix_before.png'):
                st.image('plots/confusion_matrix_before.png',
                         caption="Before Unlearning — affected by mislabeled data",
                         use_column_width=True)
            else:
                st.markdown('<div class="card" style="text-align:center; color:#475569;">Run main.py to generate</div>', unsafe_allow_html=True)
        with col2:
            if os.path.exists('plots/confusion_matrix_after.png'):
                st.image('plots/confusion_matrix_after.png',
                         caption="After Unlearning — corrected model",
                         use_column_width=True)
            else:
                st.markdown('<div class="card" style="text-align:center; color:#475569;">Run main.py to generate</div>', unsafe_allow_html=True)

        with st.expander("📋 Full Training Log"):
            st.dataframe(df, use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:30px 0 10px 0; color:#1e3a5f; font-size:12px;'>
    Machine Unlearning in Galaxy Morphology Classification &nbsp;|&nbsp;
    ResNet34 + Galaxy Zoo 2 &nbsp;|&nbsp; Latest training + unlearning results loaded
</div>""", unsafe_allow_html=True)
