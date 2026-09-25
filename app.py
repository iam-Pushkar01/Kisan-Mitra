import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json


# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="Kisan Mitra",
    page_icon="🌱",
    layout="centered"
)


# =====================================================
# SIMPLE CSS
# =====================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #f1f8e9,
            #ffffff,
            #fffde7
        );
    }

    .title {
        text-align: center;
        color: #2e7d32;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #607d68;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        margin-top: 20px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e8f5e9;
    }

    .result {
        background: #e8f5e9;
        padding: 20px;
        border-radius: 15px;
        margin-top: 15px;
        border-left: 6px solid #43a047;
    }

    .disease {
        color: #1b5e20;
        font-size: 25px;
        font-weight: 800;
    }

    .confidence {
        color: #45634d;
        font-size: 18px;
        margin-top: 10px;
    }

    .advisory {
        background: #fff8e1;
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        border-left: 6px solid #f9a825;
    }

    .footer {
        text-align: center;
        color: #718078;
        font-size: 13px;
        margin-top: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# MODEL PATH
# =====================================================

MODEL_PATH = "model/kisan_mitra_model.keras"
CLASS_NAMES_PATH = "model/class_names.json"


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


model = load_model()


# =====================================================
# LOAD CLASS NAMES
# =====================================================

@st.cache_data
def load_class_names():

    with open(
        CLASS_NAMES_PATH,
        "r"
    ) as file:

        return json.load(file)


class_names = load_class_names()


# =====================================================
# ADVISORY
# =====================================================

advisories = {

    "Tomato___Early_blight": {
        "title": "🍅 Tomato Early Blight",
        "hindi": "Prabhavit patton ko hata kar field ko clean rakhein.",
        "english": "Remove affected leaves and keep the field clean.",
        "extra": "Crop ko regularly monitor karein."
    },

    "Tomato___Late_blight": {
        "title": "🍅 Tomato Late Blight",
        "hindi": "Prabhavit patton ko jaldi hata dein.",
        "english": "Remove affected leaves early.",
        "extra": "Severe condition mein agriculture expert se consult karein."
    },

    "Tomato___Bacterial_spot": {
        "title": "🍅 Tomato Bacterial Spot",
        "hindi": "Prabhavit leaves ko remove karein aur field sanitation maintain karein.",
        "english": "Remove affected leaves and maintain good field sanitation.",
        "extra": "Leaves ko unnecessarily wet rakhne se bachein."
    },

    "Tomato___Leaf_Mold": {
        "title": "🍅 Tomato Leaf Mold",
        "hindi": "Prabhavit patton ko remove karein aur plants ke beech proper airflow maintain karein.",
        "english": "Remove affected leaves and maintain proper airflow.",
        "extra": "Leaves ko unnecessarily wet rakhne se bachein."
    },

    "Tomato___healthy": {
        "title": "🍅 Healthy Tomato Leaf",
        "hindi": "Leaf healthy lag raha hai. Regular crop monitoring continue rakhein.",
        "english": "The leaf appears healthy. Continue regular crop monitoring.",
        "extra": "Good crop hygiene maintain karein."
    },

    "Potato___Early_blight": {
        "title": "🥔 Potato Early Blight",
        "hindi": "Affected leaves ko remove karein aur crop area clean rakhein.",
        "english": "Remove affected leaves and keep the crop area clean.",
        "extra": "Plant ko regularly monitor karein."
    },

    "Potato___Late_blight": {
        "title": "🥔 Potato Late Blight",
        "hindi": "Affected leaves ko remove karein aur disease spread ko monitor karein.",
        "english": "Remove affected leaves and monitor disease spread.",
        "extra": "Severe condition mein agriculture expert se advice lein."
    },

    "Potato___healthy": {
        "title": "🥔 Healthy Potato Leaf",
        "hindi": "Leaf healthy lag raha hai. Regular monitoring continue rakhein.",
        "english": "The leaf appears healthy. Continue regular monitoring.",
        "extra": "Good crop hygiene maintain karein."
    }
}


# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="title">🌱 Kisan Mitra</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Crop Disease Detection & Advisory</div>',
    unsafe_allow_html=True
)


# =====================================================
# INTRO
# =====================================================

st.markdown(
    """
    <div class="card">

    <h3 style="color:#2e7d32;">
    📷 Upload Crop Leaf
    </h3>

    <p>
    Upload a clear image of a crop leaf.
    Kisan Mitra will analyse the image and
    identify the possible disease.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# IMAGE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Choose a crop leaf image",
    type=["jpg", "jpeg", "png"]
)


# =====================================================
# IMAGE + DETECTION
# =====================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.image(
        image,
        caption="🌿 Uploaded Crop Leaf",
        use_container_width=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    st.write("")


    # =================================================
    # DETECT BUTTON
    # =================================================

    if st.button(
        "🔍 Detect Disease",
        use_container_width=True
    ):

        with st.spinner(
            "AI is analysing the crop leaf..."
        ):

            # Resize
            image_resized = image.resize(
                (224, 224)
            )

            # Convert to numpy
            image_array = np.array(
                image_resized
            )

            # Add batch dimension
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # MobileNetV2 preprocessing
            image_array = (
                tf.keras.applications.mobilenet_v2
                .preprocess_input(
                    image_array
                )
            )

            # Prediction
            prediction = model.predict(
                image_array,
                verbose=0
            )

            # Best class
            predicted_index = np.argmax(
                prediction[0]
            )

            # Confidence
            confidence = (
                float(
                    np.max(
                        prediction[0]
                    )
                ) * 100
            )

            # Disease
            disease = class_names[
                predicted_index
            ]


        # =================================================
        # READABLE DISEASE NAME
        # =================================================

        if "___" in disease:

            crop_part, disease_part = disease.split(
                "___",
                1
            )

            crop_name = crop_part.replace(
                "_",
                " "
            )

            disease_name = disease_part.replace(
                "_",
                " "
            )

        else:

            crop_name = "Crop"

            disease_name = disease.replace(
                "_",
                " "
            )


        # =================================================
        # CONFIDENCE LEVEL
        # =================================================

        if confidence >= 70:

            confidence_level = "🟢 High Confidence"

        elif confidence >= 40:

            confidence_level = "🟡 Moderate Confidence"

        else:

            confidence_level = "🔴 Low Confidence"


        # =================================================
        # RESULT
        # =================================================

        st.markdown(
            """
            <div class="card">

            <h3 style="color:#2e7d32;">
            🌿 Detection Result
            </h3>

            """,
            unsafe_allow_html=True
        )


        st.markdown(
            f"""
            <div class="result">

                <div style="color:#607d68;">
                    DETECTED CROP
                </div>

                <div style="
                    font-size:22px;
                    font-weight:700;
                    color:#1b5e20;
                    margin-bottom:15px;
                ">
                    🌿 {crop_name}
                </div>

                <div style="color:#607d68;">
                    POSSIBLE CONDITION
                </div>

                <div class="disease">
                    {disease_name}
                </div>

                <div class="confidence">
                    <b>Confidence:</b>
                    {confidence:.2f}%
                </div>

                <div style="
                    margin-top:10px;
                    font-weight:700;
                ">
                    {confidence_level}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.progress(
            min(
                confidence / 100,
                1.0
            )
        )


        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # =================================================
        # ADVISORY
        # =================================================

        st.markdown(
            """
            <div class="advisory">

            <h3 style="color:#8a5a00;">
            💡 Kisan Mitra Advisory
            </h3>

            """,
            unsafe_allow_html=True
        )


        if disease in advisories:

            advice = advisories[disease]


            st.write(
                f"### {advice['title']}"
            )


            st.write(
                "**🇮🇳 Hindi:**"
            )

            st.write(
                advice["hindi"]
            )


            st.write(
                "**🇬🇧 English:**"
            )

            st.write(
                advice["english"]
            )


            st.warning(
                "⚠️ " + advice["extra"]
            )


        else:

            st.info(
                "AI prediction ko initial guidance "
                "ke roop mein use karein."
            )


        st.markdown(
            """
            <p style="
                font-size:12px;
                color:#6d756f;
                margin-top:15px;
            ">
            AI result is an initial assessment and should
            not replace professional agricultural advice.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


# =====================================================
# FOOTER
# =====================================================

st.markdown(
    """
    <div class="footer">

    🌱 Kisan Mitra • Supporting Farmers with AI

    <br>

    Crop Disease Detection • Farmer Advisory

    </div>
    """,
    unsafe_allow_html=True
)