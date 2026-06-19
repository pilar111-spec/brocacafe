import streamlit as st
import supervision as sv
from inference_sdk import InferenceHTTPClient
from PIL import Image
import numpy as np
import io
import time

st.set_page_config(
    page_title="BrocaCafe - Detección de Broca",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', -apple-system, sans-serif; }

    .stApp {
        background: #0d1117;
    }

    .main > div {
        background: #0d1117;
    }

    .header-container {
        background: linear-gradient(135deg, #1a120b 0%, #2d1a0e 50%, #1a120b 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid #2d1a0e;
        position: relative;
        overflow: hidden;
    }

    .header-container::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #f0a500, transparent);
    }

    .header-title {
        color: #ffffff;
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.3;
    }

    .header-subtitle {
        color: #b0a090;
        font-size: 0.9rem;
        font-weight: 400;
        margin-top: 0.3rem;
    }

    .card {
        background: #161b22;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #21262d;
        transition: border-color 0.2s;
    }

    .card:hover {
        border-color: #30363d;
    }

    .card h4 {
        color: #f0f0f0;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }

    .metric-card {
        background: #161b22;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border: 1px solid #21262d;
        border-left: 3px solid #f0a500;
    }

    .metric-label {
        color: #8b949e;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        color: #f0f0f0;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 0.15rem;
    }

    .footer {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
        color: #484f58;
        font-size: 0.8rem;
        border-top: 1px solid #21262d;
        margin-top: 2.5rem;
    }

    .footer strong { color: #8b949e; }

    .stSpinner > div { border-color: #f0a500 !important; }

    .stButton > button {
        background: linear-gradient(135deg, #f0a500 0%, #c77d00 100%);
        color: #000000;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        font-size: 0.85rem;
        transition: opacity 0.2s;
    }

    .stButton > button:hover {
        opacity: 0.9;
        color: #000000;
    }

    section[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #21262d;
    }

    section[data-testid="stSidebar"] .stMarkdown p {
        color: #c9d1d9;
    }

    section[data-testid="stSidebar"] h2 {
        color: #f0a500;
        font-weight: 700;
        font-size: 1.5rem;
    }

    section[data-testid="stSidebar"] h3 {
        color: #f0f0f0;
        font-weight: 600;
        font-size: 1rem;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #21262d;
    }

    section[data-testid="stSidebar"] .stCaption {
        color: #484f58;
    }

    .stFileUploader > div {
        background: #161b22 !important;
        border: 1px dashed #30363d !important;
        border-radius: 10px !important;
        color: #c9d1d9 !important;
    }

    .stFileUploader > div:hover {
        border-color: #f0a500 !important;
    }

    .stFileUploader > div > div > small {
        color: #8b949e !important;
    }

    .stAlert {
        border-radius: 10px;
        border: none;
    }

    .stAlert.st-error {
        background: rgba(248, 81, 73, 0.1) !important;
        border-left: 4px solid #f85149 !important;
        color: #f85149 !important;
    }

    .stAlert.st-success {
        background: rgba(63, 185, 80, 0.1) !important;
        border-left: 4px solid #3fb950 !important;
        color: #3fb950 !important;
    }

    .element-container .stText {
        color: #c9d1d9;
    }

    .st-emotion-cache-1avcm0n {
        background: #0d1117;
    }

    button[title="Main menu"] { display: none !important; }
    .stMainMenu { display: none !important; }
    footer { display: none !important; }
    #MainMenu { visibility: hidden !important; }
    header[data-testid="stHeader"] { display: none !important; }

    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
    }

    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.6;
    }

    .empty-state h3 {
        color: #c9d1d9;
        font-weight: 600;
    }

    .empty-state p {
        color: #8b949e;
    }

    @media (max-width: 768px) {
        .header-title { font-size: 1.2rem; }
        .header-container { padding: 1.2rem; }
    }
</style>
""", unsafe_allow_html=True)

LABEL_MAP = {
    "Coffee Berry Borer": "Broca Encontrada",
    "Coffee Berry Borer - v2 2024-02-05 9-24pm": "Broca Encontrada",
}

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=st.secrets["ROBOFLOW_API_KEY"]
)

with st.sidebar:
    st.markdown("## ☕ BrocaCafe")
    st.markdown("---")
    st.markdown("### ¿Cómo funciona?")
    st.markdown(
        "1. Sube una foto de un grano de café\n"
        "2. El modelo analiza la imagen\n"
        "3. Si hay broca, se marca con un recuadro\n"
        "4. Revisa los resultados en segundos"
    )
    st.markdown("---")
    st.markdown("### Estado del modelo")
    st.markdown("🟢 **En línea**")
    st.markdown("---")
    st.markdown("### Acerca de")
    st.markdown(
        "Sistema de detección de broca del café "
        "usando visión por computadora"

    )
    st.caption("v1.0.0")

st.markdown(
    """
    <div class="header-container">
        <div style="position: relative; z-index: 1;">
            <div class="header-title">"MODELO INTELIGENTE DE DETECCIÓN DE LA BROCA EN EL GRANO DE CAFÉ EMPLEANDO VISIÓN ARTIFICIAL"</div>
            <div style="margin-top: 0.8rem; border-top: 1px solid rgba(240,165,0,0.15); padding-top: 0.7rem;">
                <div style="color: #c9d1d9; font-size: 1.1rem; font-weight: 600;">Postulante: Evelyn Pilar Gonzales Poma</div>
                <div style="color: #c9d1d9; font-size: 1.1rem; font-weight: 600; margin-top: 0.2rem;">Tutor Especialista: Ph.D. Rogelio Mamani Ramos</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "\U0001f4e4 Sube una imagen del grano de café",
    type=["jpg", "jpeg", "png"],
    help="Formatos aceptados: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image.save("temp.jpg")

    with st.spinner('🤖 La inteligencia artificial está analizando la imagen...'):
        time.sleep(0.5)
        result = CLIENT.infer("temp.jpg", model_id="proyecto-deteccion-broca-de-cafe/2")
        detections = sv.Detections.from_inference(result)

    scene = np.array(image)

    box_annotator = sv.BoxAnnotator(thickness=3, color=sv.Color.from_hex("#f85149"))
    label_annotator = sv.LabelAnnotator(
        text_scale=1.2,
        text_thickness=2,
        text_color=sv.Color.from_hex("#FFFFFF"),
        color=sv.Color.from_hex("#f85149"),
        color_lookup=sv.ColorLookup.CLASS
    )

    labels = [
        f"{LABEL_MAP.get(class_name, class_name)} ({confidence:.0%})"
        for class_name, confidence in zip(detections['class_name'], detections.confidence)
    ]

    annotated_image = box_annotator.annotate(scene=scene.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        
        st.markdown("#### \U0001f4f8 Imagen Original")
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
       
        st.markdown("#### \U0001f50d Resultado de Deteccio\u0301n")
        st.image(annotated_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    cantidad = len(detections)
    col_m1, col_m2, _ = st.columns([1, 1, 2])

    with col_m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Broca Detectada</div>
                <div class="metric-value">{'Sí' if cantidad > 0 else 'No'}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Cantidad</div>
                <div class="metric-value">{cantidad}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if cantidad > 0:
        st.error(
            f"⚠️ **Se detectaron {cantidad} broca(s) en el grano.** "
            "Se recomienda tomar medidas de control."
        )
    else:
        st.success(
            "✅ **No se detecto\u0301 presencia de broca en esta muestra.** "
            "El grano se encuentra en buen estado."
        )

    annotated_pil = Image.fromarray(annotated_image)
    buf = io.BytesIO()
    annotated_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    col_d1, col_d2 = st.columns([1, 5])
    with col_d1:
        st.download_button(
            label="\U0001f4e5 Descargar imagen anotada",
            data=byte_im,
            file_name="broca_deteccion.png",
            mime="image/png",
        )

else:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-state-icon">☕</div>
            <h3>Sube una imagen para comenzar</h3>
            <p>
                Sube una foto de un grano de café y el modelo entrenado detectará
                automáticamente si hay presencia de broca.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div class="footer">
        <strong>BrocaCafe</strong> · Sistema de Monitoreo Inteligente v1.0.0<br>
    
    </div>
    """,
    unsafe_allow_html=True
)
