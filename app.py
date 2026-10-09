import json
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps


# =====================================================
# 1. APP CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Kisan Mitra | Smart Farming",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "kisan_mitra_model.keras"
CLASSES_PATH = BASE_DIR / "model" / "class_names.json"


# =====================================================
# 2. SESSION STATE
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# =====================================================
# 3. PREMIUM DASHBOARD CSS
# =====================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #F4F7F2;
        color: #183B2A;
    }

    [data-testid="stHeader"] {
        background: rgba(244,247,242,0.96);
        height: 3.5rem;
    }

    [data-testid="stMainBlockContainer"] {
        padding-top: 1.8rem;
        max-width: 1550px;
    }

    [data-testid="stSidebar"] {
        background: #123B2A;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    [data-testid="stSidebar"] hr {
        border-color: #315B43;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
    }

    .brand-logo {
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        border-radius: 15px;
        background: #DDF3D9;
        font-size: 27px;
    }

    .brand-name {
        font-size: 23px;
        line-height: 1.3;
        font-weight: 800;
        color: #FFFFFF;
    }

    .brand-tag {
        font-size: 11px;
        color: #C7DFCC;
        margin-top: 3px;
    }

    .page-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 22px;
    }

    .page-brand h1 {
        font-size: 25px;
        color: #173E2B;
        font-weight: 800;
        margin: 0;
    }

    .page-brand p {
        font-size: 12px;
        color: #728276;
        margin: 4px 0 0 0;
    }

    .hero {
        background: linear-gradient(115deg, #16452F, #28784A);
        border-radius: 22px;
        padding: 30px;
        margin-bottom: 24px;
        box-shadow: 0 9px 24px rgba(25,75,45,0.10);
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.15);
        padding: 6px 11px;
        border-radius: 30px;
        color: #F0FFF2;
        font-size: 11px;
        margin-bottom: 13px;
    }

    .hero h2 {
        color: #FFFFFF;
        font-size: clamp(25px, 3vw, 34px);
        font-weight: 800;
        margin: 0 0 10px 0;
    }

    .hero p {
        color: #E1F0E3;
        font-size: 13px;
        line-height: 1.8;
        margin: 0;
        max-width: 720px;
    }

    .section-title {
        color: #183E2B;
        font-size: 21px;
        font-weight: 800;
        margin: 6px 0 4px 0;
    }

    .section-caption {
        color: #778579;
        font-size: 12px;
        margin-bottom: 17px;
    }

    .metric {
        background: #FFFFFF;
        border: 1px solid #E4EBE1;
        border-radius: 17px;
        padding: 18px;
        min-height: 115px;
        box-shadow: 0 4px 15px rgba(23,65,39,0.035);
    }

    .metric-label {
        font-size: 12px;
        color: #718074;
    }

    .metric-number {
        font-size: 27px;
        font-weight: 800;
        color: #1B4B31;
        margin-top: 12px;
    }

    .metric-note {
        color: #849087;
        font-size: 10px;
        margin-top: 5px;
    }

    .panel {
        background: #FFFFFF;
        border: 1px solid #E3EAE1;
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 17px;
        box-shadow: 0 4px 15px rgba(23,65,39,0.035);
    }

    .panel-title {
        color: #19442D;
        font-size: 18px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .panel-caption {
        color: #778579;
        font-size: 12px;
        line-height: 1.7;
        margin-bottom: 14px;
    }

    .upload-hint {
        background: #F1F8F0;
        border: 1px dashed #82B28C;
        border-radius: 14px;
        padding: 17px 10px;
        margin: 10px 0;
        text-align: center;
        color: #275C39;
        font-size: 13px;
        line-height: 1.9;
    }

    .advice {
        background: #F2F7F0;
        border-radius: 11px;
        padding: 12px;
        margin: 9px 0;
        color: #31553B;
        font-size: 12px;
        line-height: 1.8;
    }

    .status-healthy {
        background: #EAF7ED;
        border-left: 4px solid #36965A;
        border-radius: 10px;
        padding: 15px;
        margin: 12px 0;
        color: #215D35;
        line-height: 1.8;
    }

    .status-review {
        background: #FFF5E6;
        border-left: 4px solid #E2A03A;
        border-radius: 10px;
        padding: 15px;
        margin: 12px 0;
        color: #80541B;
        line-height: 1.8;
    }

    div.stButton > button[kind="primary"] {
        background: #237447;
        border: 1px solid #237447;
        color: #FFFFFF;
        font-weight: 700;
        min-height: 45px;
        border-radius: 11px;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #175B35;
        border-color: #175B35;
    }

    [data-testid="stFileUploader"] {
        background: #F8FBF7;
        border: 1px solid #DFE9DD;
        border-radius: 13px;
        padding: 10px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #F8FBF7;
        border: 1px dashed #8AB795;
        border-radius: 10px;
    }

    .footer {
        text-align: center;
        color: #839084;
        font-size: 11px;
        line-height: 1.8;
        padding: 25px 0 8px 0;
    }

    @media(max-width: 700px) {
        .hero { padding: 22px 18px; }
        .panel { padding: 15px; }
        .metric { padding: 13px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# 4. MODEL AND CLASS LABELS
# =====================================================
@st.cache_resource(show_spinner=False)
def load_model():
    import tensorflow as tf

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file nahi mili: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )


@st.cache_data(show_spinner=False)
def load_class_names():
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file nahi mili: {CLASSES_PATH}"
        )

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        names = json.load(f)

    if isinstance(names, dict):
        try:
            names = [
                name
                for name, index in sorted(
                    names.items(),
                    key=lambda pair: int(pair[1])
                )
            ]
        except (ValueError, TypeError):
            raise ValueError(
                "class_names.json ka format verify karein."
            )

    if not isinstance(names, list) or not names:
        raise ValueError(
            "class_names.json mein valid class list nahi hai."
        )

    return names


# =====================================================
# 5. DISEASE ADVISORY
# =====================================================
def get_advisory(disease):
    name = disease.lower().replace("_", " ").replace("-", " ")

    if "healthy" in name:
        return {
            "healthy": True,
            "heading": "Leaf healthy classify hua hai",
            "message": (
                "Model ne healthy class ko sabse zyada score diya. "
                "Yeh guarantee nahi hai ki paudha bilkul disease-free hai."
            ),
            "steps": [
                "Patton ko regularly inspect karein.",
                "Paani aur nutrients crop ki zaroorat ke hisaab se dein.",
                "Naye spots ya unusual changes dikhein to dobara check karein.",
            ],
        }

    if "powdery mildew" in name:
        steps = [
            "Fasal mein hawa ka circulation achha rakhein.",
            "Affected leaves ko identify karke spread monitor karein.",
            "Local agriculture expert se crop-specific treatment confirm karein.",
        ]
    elif "late blight" in name:
        steps = [
            "Affected plants ko jaldi inspect karein.",
            "Infection ke spread ko monitor karein aur tools clean rakhein.",
            "Late blight ka shak ho to local agriculture officer se jaldi salah lein.",
        ]
    elif "early blight" in name:
        steps = [
            "Patton par gol spots aur unke spread ko observe karein.",
            "Infected plant material ko khet mein jama na hone dein.",
            "Treatment ko crop aur disease verify karne ke baad hi choose karein.",
        ]
    elif "leaf mold" in name:
        steps = [
            "Crop area mein humidity aur ventilation check karein.",
            "Patton par symptoms ka spread monitor karein.",
            "Local expert se disease aur approved treatment confirm karein.",
        ]
    elif "rust" in name:
        steps = [
            "Patton par orange ya brown spots ko monitor karein.",
            "Crop hygiene aur air circulation par dhyan dein.",
            "Crop-specific advice ke liye agriculture expert se salah lein.",
        ]
    elif "bacterial" in name:
        steps = [
            "Wet plants ko handle karne se bachhein.",
            "Gardening tools ko saaf rakhein.",
            "Symptoms verify karne ke liye local expert se contact karein.",
        ]
    else:
        steps = [
            "Affected leaves ki clear photos record karein.",
            "Symptoms aur disease ke spread ko monitor karein.",
            "Sahi diagnosis ke liye agriculture expert se salah lein.",
        ]

    return {
        "healthy": False,
        "heading": "Possible disease class detected",
        "message": (
            "Yeh model prediction initial guidance ke liye hai. "
            "Disease confirm kiye bina pesticide ya chemical ka use na karein."
        ),
        "steps": steps,
    }


# =====================================================
# 6. PREDICTION FUNCTION
# =====================================================
def predict_leaf(image, model, class_names):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((224, 224))

    # Existing model is expected to contain its own rescaling layer.
    # Do not normalize the pixels a second time.
    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    output = np.asarray(
        model.predict(array, verbose=0)
    ).reshape(-1)

    if len(output) != len(class_names):
        raise ValueError(
            f"Model output mein {len(output)} classes hain, "
            f"lekin class_names.json mein {len(class_names)} hain. "
            "Model aur class mapping match karni hogi."
        )

    if not np.all(np.isfinite(output)):
        raise ValueError("Model ne invalid prediction values di hain.")

    # Convert logits to softmax scores if needed.
    if np.any(output < 0) or not np.isclose(
        np.sum(output), 1.0, atol=0.02
    ):
        shifted = output - np.max(output)
        exp_output = np.exp(shifted)
        output = exp_output / np.sum(exp_output)

    indexes = np.argsort(output)[::-1][:3]

    results = [
        {
            "name": str(class_names[int(i)]),
            "score": float(output[int(i)]),
        }
        for i in indexes
    ]

    return results


# =====================================================
# 7. SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-logo">🌱</div>
            <div>
                <div class="brand-name">Kisan Mitra</div>
                <div class="brand-tag">Smart Farming Assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption("NAVIGATION")

    page = st.radio(
        "Menu",
        [
            "Dashboard",
            "Disease Detection",
            "Crop Guide",
            "History",
            "About",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### 🌾 Better Farming")
    st.caption(
        "Leaf analysis aur initial crop guidance ek hi jagah."
    )


# =====================================================
# 8. PAGE HEADER
# =====================================================
st.markdown(
    """
    <div class="page-brand">
        <div class="brand-logo">🌿</div>
        <div>
            <h1>Kisan Mitra</h1>
            <p>Better Decisions for Healthier Crops</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def render_hero(title, description, badge="AI-POWERED AGRICULTURE"):
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-badge">🌱 {badge}</div>
            <h2>{title}</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_latest_result():
    result = st.session_state.last_result

    if not result:
        return

    st.markdown(
        '<div class="section-title">🔬 Detection Result</div>',
        unsafe_allow_html=True,
    )

    left, middle, right = st.columns([1, 1, 1])

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Predicted class**")
        st.subheader(result["results"][0]["name"].replace("_", " "))
        st.caption("Top model prediction")
        st.markdown("</div>", unsafe_allow_html=True)

    with middle:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Model score**")
        st.metric(
            "Top prediction",
            f'{result["results"][0]["score"] * 100:.2f}%',
        )
        st.caption(
            "Score is not a guarantee of diagnostic accuracy."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Initial guidance**")
        st.write(result["advisory"]["heading"])
        st.markdown("</div>", unsafe_allow_html=True)

    advice_col, predictions_col = st.columns([1.1, 0.9])

    with advice_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🌾 Farmer Advisory</div>',
                    unsafe_allow_html=True)

        advisory = result["advisory"]
        css_class = (
            "status-healthy" if advisory["healthy"]
            else "status-review"
        )

        st.markdown(
            f'<div class="{css_class}"><b>{advisory["heading"]}</b><br>'
            f'{advisory["message"]}</div>',
            unsafe_allow_html=True,
        )

        for step in advisory["steps"]:
            st.markdown(
                f'<div class="advice">✓ {step}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with predictions_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Top 3 Predictions</div>',
                    unsafe_allow_html=True)

        for item in result["results"]:
            st.write(
                f'**{item["name"].replace("_", " ")}**'
            )
            st.progress(
                min(max(item["score"], 0.0), 1.0)
            )
            st.caption(f'{item["score"] * 100:.2f}%')

        st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# 9. DASHBOARD
# =====================================================
if page == "Dashboard":
    render_hero(
        "Welcome to Kisan Mitra",
        "Detect possible crop leaf diseases, review the model's "
        "prediction and get simple guidance to decide your next step.",
    )

    total = len(st.session_state.history)
    healthy = sum(
        item["healthy"] for item in st.session_state.history
    )
    other = total - healthy

    st.markdown(
        '<div class="section-title">Your Dashboard</div>'
        '<div class="section-caption">'
        'Your current-session activity at a glance.'
        '</div>',
        unsafe_allow_html=True,
    )

    metrics = [
        ("📷", "Total Predictions", total, "Current session"),
        ("🌿", "Healthy Results", healthy, "Model classifications"),
        ("🔎", "Other Results", other, "Require further checking"),
        (
            "🤖",
            "Model Status",
            "Ready" if MODEL_PATH.exists() else "Missing",
            "Local model file",
        ),
    ]

    cols = st.columns(4)

    for col, metric in zip(cols, metrics):
        icon, label, value, note = metric
        with col:
            st.markdown(
                f"""
                <div class="metric">
                    <div class="metric-label">{icon} {label}</div>
                    <div class="metric-number">{value}</div>
                    <div class="metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    st.markdown(
        '<div class="section-title">🌿 Detect Plant Disease</div>'
        '<div class="section-caption">'
        'Upload a clear leaf image to start your analysis.'
        '</div>',
        unsafe_allow_html=True,
    )

    upload_col, guide_col = st.columns([1.15, 0.85], gap="large")

    with upload_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.markdown(
            '<div class="panel-title">📤 Upload Leaf Image</div>'
            '<div class="panel-caption">'
            'Select a JPG, JPEG or PNG image from your device.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="upload-hint">
                <span style="font-size:28px;">🍃</span><br>
                <b>Choose your leaf image</b><br>
                Drag and drop a file or use Browse files below.<br>
                <small>Clear lighting and one visible leaf work best.</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # This is the real, interactive upload component.
        uploaded = st.file_uploader(
            "Browse files",
            type=["jpg", "jpeg", "png"],
            key="dashboard_upload",
            help="Choose a JPG, JPEG or PNG image.",
        )

        if uploaded is not None:
            try:
                image = ImageOps.exif_transpose(
                    Image.open(uploaded)
                ).convert("RGB")

                st.image(
                    image,
                    caption="Selected leaf image",
                    use_container_width=True,
                )

                if st.button(
                    "🔍 Detect Disease",
                    type="primary",
                    use_container_width=True,
                    key="dashboard_predict",
                ):
                    try:
                        with st.spinner(
                            "Analyzing leaf image. Please wait..."
                        ):
                            model = load_model()
                            classes = load_class_names()
                            results = predict_leaf(
                                image, model, classes
                            )
                            advisory = get_advisory(
                                results[0]["name"]
                            )

                        st.session_state.last_result = {
                            "results": results,
                            "advisory": advisory,
                        }

                        st.session_state.history.insert(
                            0,
                            {
                                "time": datetime.now().strftime(
                                    "%d %b %Y, %I:%M %p"
                                ),
                                "disease": results[0]["name"],
                                "score": results[0]["score"],
                                "healthy": advisory["healthy"],
                            },
                        )

                        st.success("Analysis completed.")
                        st.rerun()

                    except Exception as error:
                        st.error(f"Prediction failed: {error}")

            except Exception as error:
                st.error(f"Image could not be opened: {error}")

        st.markdown("</div>", unsafe_allow_html=True)

    with guide_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.markdown(
            '<div class="panel-title">✨ How it works</div>'
            '<div class="panel-caption">'
            'Three simple steps to review crop health.'
            '</div>',
            unsafe_allow_html=True,
        )

        for title, description in [
            (
                "01 · Upload",
                "Choose a clear photo of the plant leaf.",
            ),
            (
                "02 · Analyze",
                "The trained model predicts one of its learned classes.",
            ),
            (
                "03 · Review",
                "Read the top predictions and initial farming guidance.",
            ),
        ]:
            st.markdown(
                f'<div class="advice"><b>{title}</b><br>'
                f'{description}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.last_result:
        st.divider()
        render_latest_result()


# =====================================================
# 10. DISEASE DETECTION PAGE
# =====================================================
elif page == "Disease Detection":
    render_hero(
        "Disease Detection",
        "Upload a leaf image to review the predicted class and advisory.",
        "LEAF ANALYSIS",
    )

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 📤 Upload Leaf Image")

        detection_file = st.file_uploader(
            "Choose a leaf image",
            type=["jpg", "jpeg", "png"],
            key="detection_upload",
        )

        if detection_file:
            try:
                image = ImageOps.exif_transpose(
                    Image.open(detection_file)
                ).convert("RGB")

                st.image(
                    image,
                    caption="Leaf preview",
                    use_container_width=True,
                )

                if st.button(
                    "Analyze Leaf",
                    type="primary",
                    use_container_width=True,
                    key="detection_predict",
                ):
                    try:
                        with st.spinner("Analyzing image..."):
                            model = load_model()
                            classes = load_class_names()
                            results = predict_leaf(image, model, classes)
                            advisory = get_advisory(results[0]["name"])

                        st.session_state.last_result = {
                            "results": results,
                            "advisory": advisory,
                        }

                        st.session_state.history.insert(
                            0,
                            {
                                "time": datetime.now().strftime(
                                    "%d %b %Y, %I:%M %p"
                                ),
                                "disease": results[0]["name"],
                                "score": results[0]["score"],
                                "healthy": advisory["healthy"],
                            },
                        )
                        st.rerun()

                    except Exception as error:
                        st.error(f"Analysis failed: {error}")

            except Exception as error:
                st.error(f"Image error: {error}")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 🔬 Latest Result")

        if st.session_state.last_result:
            result = st.session_state.last_result
            best = result["results"][0]

            st.subheader(best["name"].replace("_", " "))
            st.metric("Top model score", f'{best["score"] * 100:.2f}%')

            st.markdown("**Top predictions**")
            for item in result["results"]:
                st.write(
                    f'{item["name"].replace("_", " ")} — '
                    f'{item["score"] * 100:.2f}%'
                )

            st.markdown("**Farmer guidance**")
            st.write(result["advisory"]["message"])

            for step in result["advisory"]["steps"]:
                st.markdown(f"- {step}")
        else:
            st.info(
                "Upload a leaf image and click Analyze Leaf to see results."
            )

        st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# 11. CROP GUIDE PAGE
# =====================================================
elif page == "Crop Guide":
    render_hero(
        "Crop Care Guide",
        "Simple crop-health practices to help you monitor your plants.",
        "FARMER RESOURCES",
    )

    guide_columns = st.columns(3)

    guide_cards = [
        (
            "🍃 Leaf Inspection",
            "Inspect leaves regularly and note spots, yellowing, holes or unusual growth.",
        ),
        (
            "💧 Water Management",
            "Adjust watering to crop needs and avoid unnecessary prolonged leaf wetness.",
        ),
        (
            "🧑‍🌾 Expert Guidance",
            "Confirm disease and treatment with a local agricultural expert or KVK.",
        ),
    ]

    for col, (title, text) in zip(guide_columns, guide_cards):
        with col:
            st.markdown(
                f"""
                <div class="metric" style="min-height:180px;">
                    <div class="metric-label">{title}</div>
                    <div style="font-size:13px;line-height:1.9;margin-top:16px;">
                        {text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.warning(
        "AI prediction ko final diagnosis na maanein. "
        "Treatment aur pesticide use se pehle crop-specific advice verify karein."
    )


# =====================================================
# 12. HISTORY PAGE
# =====================================================
elif page == "History":
    render_hero(
        "Prediction History",
        "Review the predictions made during the current app session.",
        "YOUR ACTIVITY",
    )

    if st.session_state.history:
        st.caption(
            "History session-based hai; app restart hone par clear ho sakti hai."
        )

        for item in st.session_state.history:
            with st.container(border=True):
                c1, c2, c3 = st.columns([1.1, 2, 1])

                with c1:
                    st.caption(item["time"])

                with c2:
                    st.markdown(
                        f"**{item['disease'].replace('_', ' ')}**"
                    )
                    st.caption(
                        "Healthy classification"
                        if item["healthy"]
                        else "Further checking recommended"
                    )

                with c3:
                    st.metric(
                        "Model score",
                        f'{item["score"] * 100:.1f}%',
                    )

        if st.button("Clear History"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()
    else:
        st.info(
            "No predictions yet. Go to Dashboard and upload a leaf image."
        )


# =====================================================
# 13. ABOUT PAGE
# =====================================================
elif page == "About":
    render_hero(
        "About Kisan Mitra",
        "An AI-powered crop-health prototype for initial leaf image analysis.",
        "OUR PROJECT",
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown(
        '<div class="panel-title">🎯 Our Goal</div>'
        '<div class="panel-caption">'
        'Make preliminary crop-health information easier to review '
        'through image classification and simple guidance.'
        '</div>',
        unsafe_allow_html=True,
    )

    for title, description in [
        (
            "Image Analysis",
            "Uses the trained local image-classification model.",
        ),
        (
            "Farmer Guidance",
            "Provides basic next-step suggestions based on the predicted label.",
        ),
        (
            "Responsible Use",
            "Results can be incorrect and should be verified before treatment.",
        ),
    ]:
        st.markdown(
            f'<div class="advice"><b>{title}</b><br>'
            f'{description}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# 14. FOOTER
# =====================================================
st.markdown(
    """
    <div class="footer">
        🌱 Kisan Mitra · AI-powered crop-health prototype<br>
        Predictions are preliminary guidance, not a confirmed diagnosis.
    </div>
    """,
    unsafe_allow_html=True,
)