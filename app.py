import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
import io
from pathlib import Path
from urllib.request import urlretrieve
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Classifier | AI Medical Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Modern Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    * { box-sizing: border-box; }
    
    /* Main App Background - Modern Gradient */
    .stApp { 
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        color: #e0e6f0;
    }
    
    /* Typography */
    h1 { 
        color: #ffffff !important; 
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #4fc3f7, #b98aff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    h2 { 
        color: #ffffff !important; 
        font-weight: 700;
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }
    h3, h4 { 
        color: #e0e6f0 !important;
        font-weight: 600;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #13171f 0%, #0f1419 100%);
        border-right: 1px solid #2a3045;
    }
    [data-testid="stSidebar"] * { color: #c5cee0 !important; }
    
    /* Enhanced Result Card */
    .result-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #262d3c 100%);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #2a3045;
        margin-top: 12px;
        box-shadow: 0 4px 20px rgba(79, 195, 247, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .result-card:hover {
        border-color: #4fc3f7;
        box-shadow: 0 8px 32px rgba(79, 195, 247, 0.2);
    }
    
    /* Info Cards */
    .info-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #262d3c 100%);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #2a3045;
        margin: 8px 0;
        transition: all 0.2s ease;
    }
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 195, 247, 0.15);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        color: #4fc3f7 !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 14px !important;
        color: #8a94a6 !important;
    }
    
    /* Confidence Label */
    .conf-label { 
        font-size: 13px; 
        color: #8a94a6; 
        margin-bottom: 5px;
        font-weight: 600;
    }
    
    /* Badges */
    .badge-glioma     { background: rgba(255, 107, 107, 0.1); color: #ff6b6b; border: 2px solid #ff6b6b; }
    .badge-meningioma { background: rgba(255, 209, 102, 0.1); color: #ffd166; border: 2px solid #ffd166; }
    .badge-pituitary  { background: rgba(185, 138, 255, 0.1); color: #b98aff; border: 2px solid #b98aff; }
    .badge-notumor    { background: rgba(6, 214, 160, 0.1); color: #06d6a0; border: 2px solid #06d6a0; }
    .badge {
        display: inline-block;
        padding: 10px 20px;
        border-radius: 24px;
        font-weight: 700;
        font-size: 16px;
        margin-bottom: 12px;
    }
    
    /* Progress Bars */
    .prob-bar-wrap { 
        margin: 12px 0;
        animation: fadeIn 0.5s ease;
    }
    .prob-bar-bg {
        background: #1e2433;
        border-radius: 8px;
        height: 24px;
        position: relative;
        overflow: hidden;
        border: 1px solid #2a3045;
    }
    .prob-bar-fill { 
        height: 100%;
        border-radius: 8px;
        transition: width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
        box-shadow: 0 0 10px currentColor;
    }
    .prob-bar-text {
        position: absolute;
        right: 10px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 12px;
        color: #fff;
        font-weight: 700;
        text-shadow: 0 1px 3px rgba(0,0,0,0.5);
    }
    
    /* Disclaimer Box */
    .disclaimer {
        background: linear-gradient(135deg, rgba(240, 165, 0, 0.1), rgba(255, 209, 102, 0.05));
        border-left: 4px solid #f0a500;
        border-radius: 8px;
        padding: 16px;
        color: #d4b86a;
        font-size: 14px;
        margin-top: 20px;
        border: 1px solid rgba(240, 165, 0, 0.2);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #4a5568;
        font-size: 12px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #1e2433;
        opacity: 0.8;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideIn {
        from { transform: translateY(10px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4fc3f7, #b98aff) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(79, 195, 247, 0.4) !important;
    }
    
    /* File Uploader */
    [data-testid="stFileUploadDropzone"] {
        background: linear-gradient(135deg, rgba(79, 195, 247, 0.05), rgba(185, 138, 255, 0.05)) !important;
        border: 2px dashed #4fc3f7 !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES     = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
IMG_SIZE        = 224
WEIGHTS_FILE    = 'resnet_weights.weights.h5'
APP_DIR         = Path(__file__).resolve().parent
MODEL_CACHE_DIR = APP_DIR / ".streamlit" / "model_cache"
LOW_CONF_THRESH = 60.0


def get_secret_value(name: str) -> str:
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


def download_weights_file(url: str) -> Path:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached_path = MODEL_CACHE_DIR / WEIGHTS_FILE
    if not cached_path.exists():
        with st.spinner("Downloading model weights for deployment..."):
            urlretrieve(url, cached_path)
    return cached_path


def resolve_weights_file() -> tuple[Path | None, str | None]:
    local_path = APP_DIR / WEIGHTS_FILE
    if local_path.exists():
        return local_path, None

    configured_path = (
        os.environ.get("MODEL_WEIGHTS_PATH")
        or os.environ.get("STREAMLIT_MODEL_PATH")
        or os.environ.get("RESNET_WEIGHTS_PATH")
        or get_secret_value("MODEL_WEIGHTS_PATH")
    )
    if configured_path:
        external_path = Path(configured_path).expanduser()
        if external_path.exists():
            return external_path, None
        return None, f"Configured model path does not exist: `{external_path}`"

    weights_url = (
        os.environ.get("MODEL_WEIGHTS_URL")
        or get_secret_value("MODEL_WEIGHTS_URL")
    )
    if weights_url:
        try:
            return download_weights_file(weights_url), None
        except Exception as exc:
            return None, f"Unable to download model weights from the configured URL: {exc}"

    return None, (
        "Model weights were not found. Add `resnet_weights.weights.h5` to the app folder, "
        "set `MODEL_WEIGHTS_PATH`, or configure `MODEL_WEIGHTS_URL` in Streamlit secrets."
    )

CLASS_INFO = {
    'Glioma': {
        'icon': '🔴',
        'badge_class': 'badge-glioma',
        'desc': 'A tumor originating in the glial (support) cells of the brain or spinal cord. '
                'Can be low-grade (slow growing) or high-grade (aggressive).',
    },
    'Meningioma': {
        'icon': '🟠',
        'badge_class': 'badge-meningioma',
        'desc': 'A tumor arising from the meninges — the membranes surrounding the brain and '
                'spinal cord. Usually benign and slow-growing.',
    },
    'Pituitary': {
        'icon': '🟣',
        'badge_class': 'badge-pituitary',
        'desc': 'A tumor in the pituitary gland at the base of the brain. Most are benign '
                'adenomas that can affect hormone production.',
    },
    'No Tumor': {
        'icon': '✅',
        'badge_class': 'badge-notumor',
        'desc': 'No signs of a brain tumor were detected in this MRI scan.',
    },
}

BAR_COLORS = {
    'Glioma':     '#ff6b6b',
    'Meningioma': '#ffd166',
    'Pituitary':  '#b98aff',
    'No Tumor':   '#06d6a0',
}

# ── Detailed Report Content per Class ─────────────────────────────────────────
REPORT_INFO = {
    'Glioma': {
        'overview': (
            'Gliomas are tumors that arise from glial cells in the brain or spinal cord. '
            'They are classified by grade (I-IV), with Grade IV (Glioblastoma) being the most '
            'aggressive. Early diagnosis and specialist referral are critical for optimal outcomes.'
        ),
        'next_steps': [
            'Consult a Neurologist or Neuro-oncologist immediately for further evaluation.',
            'Schedule an advanced MRI with contrast (Gadolinium) and possibly an MR Spectroscopy.',
            'A biopsy or surgical resection may be recommended to confirm tumor grade.',
            'Discuss treatment options: surgery, radiation therapy, and/or chemotherapy (e.g., Temozolomide).',
            'Get a second opinion from a specialist at a tertiary cancer center.',
            'Undergo blood tests and neurological assessments as advised by your doctor.',
        ],
        'precautions': [
            'Avoid driving or operating heavy machinery until cleared by your doctor.',
            'Do not ignore symptoms like persistent headaches, seizures, or vision changes.',
            'Avoid stress and ensure adequate sleep to support neurological health.',
            'Follow a nutritious diet rich in antioxidants and omega-3 fatty acids.',
            'Limit alcohol consumption and avoid smoking entirely.',
            'Keep all follow-up appointments and monitor for any new or worsening symptoms.',
            'Inform family members so they can provide support and watch for behavioral changes.',
        ],
        'lifestyle': [
            'Engage in light physical activity as tolerated and approved by your doctor.',
            'Consider joining a brain tumor support group for mental and emotional well-being.',
            'Maintain open communication with your healthcare team about all medications.',
            'Practice stress-reduction techniques such as meditation or gentle yoga.',
        ],
        'warning_signs': [
            'Sudden severe headache (thunderclap headache)',
            'New onset seizures or worsening of existing seizures',
            'Sudden weakness or numbness in limbs',
            'Sudden vision or speech problems',
            'Loss of consciousness - call emergency services immediately',
        ],
    },
    'Meningioma': {
        'overview': (
            'Meningiomas arise from the meninges - the protective membranes surrounding the brain '
            'and spinal cord. The majority (over 90%) are benign (Grade I) and slow-growing. '
            'Many are discovered incidentally and may only require observation ("watchful waiting").'
        ),
        'next_steps': [
            'Consult a Neurosurgeon or Neurologist for a clinical evaluation.',
            'An MRI with contrast is recommended to assess tumor size, location, and involvement.',
            'Small, asymptomatic meningiomas may be managed with regular MRI monitoring (every 6-12 months).',
            'Larger or symptomatic tumors may require surgical removal or stereotactic radiosurgery (Gamma Knife).',
            'Discuss the risks and benefits of each treatment option with your specialist.',
            'Hormonal evaluation may be advised, as meningiomas can be hormone-sensitive.',
        ],
        'precautions': [
            'Avoid prolonged use of oral contraceptives or hormone replacement therapy without medical advice.',
            'Protect yourself from head injuries; wear helmets during sports activities.',
            'Do not self-medicate with steroids or anti-inflammatory drugs without a prescription.',
            'Avoid overexertion and activities that significantly raise intracranial pressure.',
            'Limit exposure to ionizing radiation (e.g., unnecessary CT scans).',
            'Adhere strictly to the monitoring schedule recommended by your doctor.',
        ],
        'lifestyle': [
            'Maintain a healthy weight; obesity may influence meningioma growth.',
            'Eat a balanced diet rich in vegetables, fruits, and whole grains.',
            'Stay physically active with low-impact exercises like walking or swimming.',
            'Manage stress effectively, as chronic stress can suppress immune function.',
        ],
        'warning_signs': [
            'Increasing or new onset headaches, especially in the morning',
            'Vision or hearing changes',
            'Memory loss or personality changes',
            'Weakness or numbness in arms or legs',
            'Balance problems or difficulty walking - seek medical attention promptly',
        ],
    },
    'Pituitary': {
        'overview': (
            'Pituitary tumors (adenomas) develop in the pituitary gland at the base of the brain. '
            'Most are benign and non-cancerous. They are classified as functioning (hormone-secreting) '
            'or non-functioning. Treatment outcomes are generally very favorable with proper management.'
        ),
        'next_steps': [
            'Consult an Endocrinologist and Neurosurgeon as a first step.',
            'A dedicated pituitary MRI (with thin-slice gadolinium contrast) is recommended.',
            'Comprehensive hormonal blood panel: Prolactin, GH, IGF-1, ACTH, Cortisol, TSH, FSH, LH.',
            'Visual field testing (perimetry) to check for optic nerve compression.',
            'Treatment options include medication (e.g., dopamine agonists for prolactinomas), '
            'trans-sphenoidal surgery, or radiation therapy.',
            'Work with a multidisciplinary team including endocrinology, neurosurgery, and ophthalmology.',
        ],
        'precautions': [
            'Do not abruptly stop any hormone replacement medications without doctor guidance.',
            'Monitor for symptoms of hormonal imbalance: unusual weight changes, fatigue, or mood swings.',
            'Avoid activities with high risk of head trauma.',
            'Inform all healthcare providers about your pituitary condition before any procedures.',
            'Women should discuss the impact on fertility and pregnancy with their endocrinologist.',
            'Avoid stress, as it can exacerbate hormonal fluctuations.',
        ],
        'lifestyle': [
            'Monitor blood pressure regularly, especially if cortisol or growth hormone levels are abnormal.',
            'Maintain a balanced diet tailored to your hormonal status (e.g., low-sodium if Cushings suspected).',
            'Prioritize sleep; pituitary hormones are closely tied to circadian rhythm.',
            'Engage in moderate exercise; avoid extreme endurance sports without clearance.',
        ],
        'warning_signs': [
            'Sudden severe headache with visual loss (pituitary apoplexy - EMERGENCY)',
            'Rapid vision deterioration or double vision',
            'Extreme fatigue, nausea, or low blood pressure (adrenal crisis)',
            'Unusual rapid weight gain or loss',
            'Cessation of menstruation or new erectile dysfunction - report to your doctor',
        ],
    },
    'No Tumor': {
        'overview': (
            'The AI model did not detect signs of a brain tumor in this MRI scan. '
            'This is a positive finding; however, it is important to remember that this tool '
            'is for educational purposes only and should not replace a professional radiological assessment.'
        ),
        'next_steps': [
            'Share these results with your treating physician for professional interpretation.',
            'If you have ongoing symptoms, request a formal radiologist review of the original MRI.',
            'Continue with any recommended routine health check-ups.',
            'If symptoms persist (headaches, dizziness, vision changes), investigate other possible causes.',
            'Follow your doctor\'s advice regarding any further investigations.',
        ],
        'precautions': [
            'Do not use this AI result as a definitive clinical diagnosis.',
            'Always have MRI scans reviewed by a qualified radiologist.',
            'Do not ignore persistent neurological symptoms, even with a negative AI result.',
            'Maintain regular health screenings as advised by your physician.',
        ],
        'lifestyle': [
            'Maintain a brain-healthy lifestyle: regular exercise, balanced diet, and adequate sleep.',
            'Manage cardiovascular risk factors (blood pressure, cholesterol, diabetes).',
            'Avoid smoking and limit alcohol consumption.',
            'Engage in mentally stimulating activities to support long-term brain health.',
        ],
        'warning_signs': [
            'New or worsening headaches that are different in character',
            'Seizures of any kind',
            'Sudden weakness, numbness, or speech difficulty',
            'Unexplained personality or memory changes - consult a doctor promptly',
        ],
    },
}

# ── MRI Validation ────────────────────────────────────────────────────────────
def is_mri_scan(image: Image.Image, color_threshold: float = 18.0) -> tuple[bool, float]:
    img_array   = np.array(image.resize((128, 128)), dtype=np.float32)
    r, g, b     = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    color_score = (np.mean(np.abs(r - g)) + np.mean(np.abs(r - b)) + np.mean(np.abs(g - b))) / 3.0
    return color_score <= color_threshold, color_score

# ── PDF Report Generator ──────────────────────────────────────────────────────
def generate_pdf_report(
    predicted_class: str,
    confidence: float,
    all_probs: dict,
    filename: str,
    img_dims: tuple,
    patient_name: str = "Not Provided"
) -> bytes:
    """Generate a professional PDF diagnostic report using ReportLab."""

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=20*mm, leftMargin=20*mm,
        topMargin=18*mm, bottomMargin=18*mm,
    )

    # Colour palette
    DARK_BLUE   = colors.HexColor('#0d1b2a')
    MID_BLUE    = colors.HexColor('#1b3a5c')
    LIGHT_GREY  = colors.HexColor('#f0f4f8')
    MID_GREY    = colors.HexColor('#d0d8e4')
    TEXT_DARK   = colors.HexColor('#1a202c')
    TEXT_MED    = colors.HexColor('#4a5568')
    WARNING_RED = colors.HexColor('#c53030')

    CLASS_ACCENT = {
        'Glioma':     colors.HexColor('#c53030'),
        'Meningioma': colors.HexColor('#b7791f'),
        'Pituitary':  colors.HexColor('#6b46c1'),
        'No Tumor':   colors.HexColor('#276749'),
    }
    accent_color = CLASS_ACCENT.get(predicted_class, MID_BLUE)

    # Styles
    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    style_h1      = S('H1',  fontSize=20, textColor=colors.white,   fontName='Helvetica-Bold', alignment=TA_CENTER)
    style_sub     = S('Sub', fontSize=9,  textColor=colors.HexColor('#a0aec0'), fontName='Helvetica', alignment=TA_CENTER)
    style_section = S('Sec', fontSize=12, textColor=MID_BLUE,        fontName='Helvetica-Bold', spaceBefore=8, spaceAfter=3)
    style_body    = S('Bod', fontSize=9.5, textColor=TEXT_DARK,      fontName='Helvetica', leading=14, spaceAfter=3, alignment=TA_JUSTIFY)
    style_bullet  = S('Bul', fontSize=9.5, textColor=TEXT_DARK,      fontName='Helvetica', leading=14, leftIndent=10, spaceAfter=2)
    style_warning = S('War', fontSize=9,  textColor=WARNING_RED,     fontName='Helvetica-Bold', leading=13, spaceAfter=2)
    style_disc    = S('Dis', fontSize=8,  textColor=TEXT_MED,        fontName='Helvetica-Oblique', leading=11, alignment=TA_CENTER)
    style_label   = S('Lab', fontSize=9,  textColor=TEXT_MED,        fontName='Helvetica')
    style_value   = S('Val', fontSize=9,  textColor=TEXT_DARK,       fontName='Helvetica-Bold')
    style_res     = S('Res', fontSize=16, textColor=colors.white,    fontName='Helvetica-Bold', alignment=TA_CENTER)
    style_conf    = S('Cnf', fontSize=11, textColor=colors.HexColor('#e2e8f0'), fontName='Helvetica', alignment=TA_CENTER)

    info  = REPORT_INFO[predicted_class]
    rinfo = CLASS_INFO[predicted_class]
    story = []
    W     = A4[0] - 40*mm  # usable width

    # ── Header ────────────────────────────────────────────────────────────────
    header_table = Table(
        [[Paragraph('Brain Tumor MRI - Diagnostic Report', style_h1)]],
        colWidths=[W]
    )
    header_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), DARK_BLUE),
        ('TOPPADDING',    (0,0), (-1,-1), 14),
        ('BOTTOMPADDING', (0,0), (-1,-1), 14),
        ('LEFTPADDING',   (0,0), (-1,-1), 12),
        ('RIGHTPADDING',  (0,0), (-1,-1), 12),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 5*mm))

    # ── Meta Info ─────────────────────────────────────────────────────────────
    now = datetime.now()
    meta_rows = [
        [Paragraph('Patient Name', style_label), Paragraph(patient_name, style_value),
         Paragraph('Report Date', style_label), Paragraph(now.strftime('%d %B %Y'), style_value)],
        [Paragraph('Report Time',  style_label), Paragraph(now.strftime('%I:%M %p'),  style_value),
         Paragraph('Image File',  style_label), Paragraph(filename,                  style_value)],
        [Paragraph('Image Size',  style_label), Paragraph(f'{img_dims[0]} x {img_dims[1]} px', style_value),
         Paragraph('Model Used',  style_label), Paragraph('ResNet50 Transfer Learning', style_value)],
    ]
    meta_table = Table(meta_rows, colWidths=[W*0.18, W*0.32, W*0.18, W*0.32])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), LIGHT_GREY),
        ('GRID',          (0,0), (-1,-1), 0.4, MID_GREY),
        ('TOPPADDING',    (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING',   (0,0), (-1,-1), 7),
        ('RIGHTPADDING',  (0,0), (-1,-1), 7),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 5*mm))

    # ── Result Banner ─────────────────────────────────────────────────────────
    result_label = 'No Tumor Detected' if predicted_class == 'No Tumor' else f'Tumor Detected: {predicted_class}'
    result_table = Table(
        [[Paragraph(result_label, style_res),
          Paragraph(f'Model Confidence: {confidence:.1f}%', style_conf)]],
        colWidths=[W*0.65, W*0.35]
    )
    result_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), accent_color),
        ('TOPPADDING',    (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('LEFTPADDING',   (0,0), (-1,-1), 12),
        ('RIGHTPADDING',  (0,0), (-1,-1), 12),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 4*mm))

    if confidence < LOW_CONF_THRESH:
        story.append(Paragraph(
            f'WARNING: Low Confidence ({confidence:.1f}%) - The model prediction confidence is below '
            f'the recommended threshold of {LOW_CONF_THRESH:.0f}%. Please interpret this result with '
            'additional caution and consult a qualified medical professional.',
            style_warning
        ))
        story.append(Spacer(1, 3*mm))

    # ── Probability Table ─────────────────────────────────────────────────────
    story.append(Paragraph('Class Probability Breakdown', style_section))
    story.append(HRFlowable(width=W, thickness=1, color=MID_BLUE, spaceAfter=4))

    prob_header = [
        Paragraph('<b>Tumor Class</b>', style_value),
        Paragraph('<b>Probability</b>', style_value),
        Paragraph('<b>Confidence Bar</b>', style_value),
    ]
    prob_rows = [prob_header]
    for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        filled   = int((prob / 100) * 30)
        bar_str  = ('|' * filled) + ('.' * (30 - filled))
        is_top   = cls == predicted_class
        fn       = 'Helvetica-Bold' if is_top else 'Helvetica'
        prefix   = '>> ' if is_top else '   '
        prob_rows.append([
            Paragraph(f'{prefix}{cls}', S('pr',  fontSize=9, fontName=fn, textColor=TEXT_DARK)),
            Paragraph(f'{prob:.2f}%',   S('pr2', fontSize=9, fontName=fn, textColor=TEXT_DARK, alignment=TA_CENTER)),
            Paragraph(bar_str,          S('bar', fontSize=7, fontName='Courier',
                                          textColor=CLASS_ACCENT.get(cls, MID_BLUE))),
        ])

    prob_table = Table(prob_rows, colWidths=[W*0.30, W*0.20, W*0.50])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND',     (0,0), (-1,0),  MID_BLUE),
        ('TEXTCOLOR',      (0,0), (-1,0),  colors.white),
        ('GRID',           (0,0), (-1,-1), 0.4, MID_GREY),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_GREY]),
        ('TOPPADDING',     (0,0), (-1,-1), 5),
        ('BOTTOMPADDING',  (0,0), (-1,-1), 5),
        ('LEFTPADDING',    (0,0), (-1,-1), 8),
        ('RIGHTPADDING',   (0,0), (-1,-1), 8),
        ('VALIGN',         (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 5*mm))

    # Helper: section block
    def section_block(title, items, bullet='numbered', color=MID_BLUE):
        block = [
            Paragraph(title, style_section),
            HRFlowable(width=W, thickness=1, color=color, spaceAfter=4),
        ]
        if isinstance(items, str):
            block.append(Paragraph(items, style_body))
        else:
            for i, item in enumerate(items):
                prefix = f'{i+1}.  ' if bullet == 'numbered' else '  *  '
                st_use = style_warning if color == WARNING_RED else style_bullet
                block.append(Paragraph(f'{prefix}{item}', st_use))
        block.append(Spacer(1, 3*mm))
        return KeepTogether(block)

    # ── Clinical Overview ─────────────────────────────────────────────────────
    story.append(section_block('Clinical Overview', info['overview']))

    # ── Next Steps ────────────────────────────────────────────────────────────
    story.append(section_block('Recommended Next Steps', info['next_steps'], bullet='numbered'))

    # ── Precautions ───────────────────────────────────────────────────────────
    story.append(section_block('Precautions to Follow', info['precautions'], bullet='bullet'))

    # ── Lifestyle ─────────────────────────────────────────────────────────────
    story.append(section_block('Lifestyle Recommendations', info['lifestyle'], bullet='bullet'))

    # ── Warning Signs ─────────────────────────────────────────────────────────
    story.append(section_block('Warning Signs — Seek Immediate Medical Attention',
                               info['warning_signs'], bullet='bullet', color=WARNING_RED))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width=W, thickness=0.5, color=MID_GREY, spaceAfter=4))
    disc_text = (
        'IMPORTANT DISCLAIMER: This report is generated by an AI-based educational tool and is '
        'NOT a substitute for professional medical advice, diagnosis, or treatment. '
        'The predictions made by this model may not be accurate and must be reviewed by a '
        'qualified medical professional (neurologist, neurosurgeon, or radiologist) before '
        'any clinical decision is made. Always seek the advice of your physician with any '
        'questions you may have regarding a medical condition.'
    )
    disc_table = Table([[Paragraph(disc_text, style_disc)]], colWidths=[W])
    disc_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), colors.HexColor('#fff3cd')),
        ('BOX',           (0,0), (-1,-1), 1, colors.HexColor('#f0a500')),
        ('TOPPADDING',    (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
        ('RIGHTPADDING',  (0,0), (-1,-1), 10),
    ]))
    story.append(disc_table)

    doc.build(story)
    return buffer.getvalue()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Brain Tumor Classifier")
    st.markdown("**AI-Powered MRI Analysis Tool**")
    st.markdown("---")
    
    # Sidebar tabs for better organization
    sidebar_tab = st.radio(
        "Navigate:",
        ["📋 Overview", "ℹ️ About Model", "🎯 How It Works", "❓ FAQ", "⚠️ Disclaimer"],
        label_visibility="collapsed"
    )
    
    if sidebar_tab == "📋 Overview":
        st.markdown("""
### What This App Does
This tool uses **advanced AI** powered by a **ResNet50 Deep Learning Model** 
to analyze brain MRI scans and classify them into four categories.

### Detectable Classes:
- 🔴 **Glioma** — Brain tumor from glial cells
- 🟠 **Meningioma** — Tumor from brain membranes
- 🟣 **Pituitary** — Tumor from pituitary gland
- ✅ **No Tumor** — Healthy scan

### Key Features:
- 🔍 Instant MRI analysis
- 📊 Confidence scoring
- 📄 Detailed PDF reports
- ⚕️ Clinical guidance included
        """)
    
    elif sidebar_tab == "ℹ️ About Model":
        st.markdown("""
### Model Architecture
**Base Model:** ResNet50 (Transfer Learning)
- Pre-trained on ImageNet weights
- 50 deep residual layers
- Optimized for medical imaging

### Model Stack:
1. ResNet50 backbone (feature extraction)
2. Global Average Pooling
3. Dense Layer (256 units, ReLU)
4. Dropout (40% regularization)
5. Output Layer (4 classes, Softmax)

### Training Details
- **Dataset:** Brain Tumor MRI Dataset (Kaggle)
- **Image Size:** 224×224 pixels
- **Classes:** 4 tumor types + no tumor
- **Optimization:** Data augmentation & regularization
- **Purpose:** Educational & Research Only

### Important Notes:
✓ Model trained on curated medical data
✓ Optimized for brain MRI images
⚠️ NOT for clinical diagnosis
⚠️ Results MUST be verified by radiologist
        """)
    
    elif sidebar_tab == "🎯 How It Works":
        st.markdown("""
### Step-by-Step Process:

**1. Image Upload**
- Upload a brain MRI scan (JPG/PNG)
- Supported format: Grayscale medical images

**2. Validation Check**
- System verifies it's a valid MRI scan
- Rejects colorful or non-medical images

**3. Image Processing**
- Resize to 224×224 pixels
- Apply ResNet50 preprocessing
- Normalize pixel values

**4. AI Analysis**
- Feed through trained neural network
- Extract deep features
- Calculate probabilities

**5. Results Display**
- Show predicted tumor type
- Display confidence percentage
- Show probabilities for all classes

**6. Report Generation**
- Optional: Generate detailed PDF
- Includes clinical guidance
- Professional medical advice reference

### Processing Time:
⚡ ~2-3 seconds per MRI scan
        """)
    
    elif sidebar_tab == "❓ FAQ":
        st.markdown("""
### Frequently Asked Questions

**Q: Is this a medical diagnostic tool?**
A: No, this is an educational AI tool. Always consult a 
radiologist or doctor for actual diagnosis.

**Q: What file formats are supported?**
A: JPG, JPEG, and PNG. Images must be brain MRI scans 
(grayscale medical images).

**Q: Why was my image rejected?**
A: The system detected it's not a brain MRI. Upload 
actual medical MRI scan images.

**Q: How accurate is the model?**
A: The model achieves good performance on the training 
dataset. However, real-world performance may vary.

**Q: Can I download the report?**
A: Yes! Generate a detailed PDF report with clinical 
guidance after each analysis.

**Q: Is my data stored?**
A: No. All images are processed locally and not stored 
on our servers.

**Q: Should I trust the results?**
A: No medical diagnosis should rely solely on this tool. 
Always consult healthcare professionals.
        """)
    
    elif sidebar_tab == "⚠️ Disclaimer":
        st.markdown("""
### ⚠️ IMPORTANT MEDICAL DISCLAIMER

**This application is for educational and research 
purposes only.**

✗ NOT a substitute for professional medical advice
✗ NOT for clinical diagnosis or treatment decisions
✗ Results must be verified by qualified professionals

### By Using This Tool, You Agree:
- Results are AI predictions, not medical diagnoses
- You will consult a doctor/radiologist for actual diagnosis
- The developers are NOT liable for misuse
- This tool has NOT been clinically validated

### Seek Professional Help If:
- You have neurological symptoms
- You need a medical diagnosis
- Your doctor recommended MRI analysis
- You experience severe headaches or other symptoms

**Always consult a qualified neurologist, neurosurgeon, 
or radiologist for proper medical evaluation.**

---
*Developed for educational purposes*
        """)

# ── Modern Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align: center; margin-bottom: 30px; animation: slideIn 0.8s ease;'>
    <h1 style='text-align: center; margin-bottom: 5px;'>🧠 Brain Tumor MRI Classifier</h1>
    <p style='text-align: center; color: #8a94a6; font-size: 18px; margin: 0;'>
        <strong>Advanced AI-Powered Medical Image Analysis</strong>
    </p>
    <p style='text-align: center; color: #6a7280; font-size: 14px; margin: 8px 0 0 0;'>
        Powered by ResNet50 Deep Learning • Real-time Analysis • Educational Purpose
    </p>
</div>
""", unsafe_allow_html=True)

# ── Feature Cards with Hover Effects ────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4, gap="small")
with col1:
    st.markdown("""
    <div class='info-card' style='text-align: center; padding: 20px;'>
        <div style='font-size: 32px; margin-bottom: 8px;'>🤖</div>
        <p style='color: #c5cee0; font-weight: 600; font-size: 14px; margin: 0;'>AI-Powered</p>
        <p style='color: #8a94a6; font-size: 12px; margin: 4px 0 0 0;'>ResNet50 Model</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='info-card' style='text-align: center; padding: 20px;'>
        <div style='font-size: 32px; margin-bottom: 8px;'>⚡</div>
        <p style='color: #ffd166; font-weight: 600; font-size: 14px; margin: 0;'>Instant Results</p>
        <p style='color: #8a94a6; font-size: 12px; margin: 4px 0 0 0;'>2-3 Seconds</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='info-card' style='text-align: center; padding: 20px;'>
        <div style='font-size: 32px; margin-bottom: 8px;'>📊</div>
        <p style='color: #b98aff; font-weight: 600; font-size: 14px; margin: 0;'>Detailed Reports</p>
        <p style='color: #8a94a6; font-size: 12px; margin: 4px 0 0 0;'>PDF Export</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class='info-card' style='text-align: center; padding: 20px;'>
        <div style='font-size: 32px; margin-bottom: 8px;'>🔒</div>
        <p style='color: #06d6a0; font-weight: 600; font-size: 14px; margin: 0;'>100% Private</p>
        <p style='color: #8a94a6; font-size: 12px; margin: 4px 0 0 0;'>No Data Stored</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(weights_path: str):
    base_model = ResNet50(
        weights=None,
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
    model.build((None, IMG_SIZE, IMG_SIZE, 3))
    model.load_weights(weights_path)
    return model

weights_path, weights_error = resolve_weights_file()
if weights_path is None:
    st.error(weights_error)
    st.stop()

try:
    with st.spinner('Loading AI model...'):
        model = load_model(str(weights_path))
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ── Upload Image ──────────────────────────────────────────────────────────────
st.markdown("### 📤 Upload Brain MRI Image")
st.markdown("""
<div style='background: #1a1f2e; border-left: 4px solid #4fc3f7; border-radius: 6px; padding: 12px;'>
<p style='color: #8a94a6; margin: 0; font-size: 13px;'>
<strong>Supported formats:</strong> JPG, JPEG, PNG (grayscale medical images)<br>
<strong>Recommended size:</strong> 224×224 pixels or larger<br>
<strong>Processing time:</strong> Usually 2-3 seconds
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("")

uploaded_file = st.file_uploader(
    "Choose an MRI image file",
    type=["jpg", "jpeg", "png"],
    help="Upload a brain MRI scan in grayscale format"
)

# ── Prediction ────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
    except Exception:
        st.error("Could not open the uploaded file. Please upload a valid image.")
        st.stop()

    # MRI Validation Gate
    valid_mri, color_score = is_mri_scan(image)
    if not valid_mri:
        col_img, col_msg = st.columns([1, 1], gap="large")
        with col_img:
            st.markdown("#### 🖼️ Uploaded Image")
            st.image(image, use_container_width=True)
            st.caption(f"File: `{uploaded_file.name}` — {image.width}×{image.height} px")
        with col_msg:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(
                "<div style='background:#1e0f0f; border:2px solid #ff4b4b; border-radius:12px;"
                "padding:24px; margin-top:10px;'>"
                "<h3 style='color:#ff4b4b; margin-top:0;'>🚫 Invalid Image</h3>"
                "<p style='color:#e0c0c0; font-size:15px;'>"
                "The uploaded image does not appear to be a <strong>Brain MRI scan</strong>.</p>"
                "<p style='color:#c09090; font-size:13px;'>Please upload a valid grayscale "
                "Brain MRI image in <strong>JPG, JPEG, or PNG</strong> format.</p>"
                "<hr style='border-color:#3a2020;'>"
                "<p style='color:#8a6060; font-size:12px; margin-bottom:0;'>"
                "MRI scans are grayscale medical images. Colorful photos, selfies, "
                "or screenshots are not accepted.</p></div>",
                unsafe_allow_html=True
            )
        st.stop()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### 🖼️ Uploaded MRI Image")
        st.image(image, use_container_width=True)
        st.caption(f"File: `{uploaded_file.name}` — {image.width}×{image.height} px")

    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array   = np.array(img_resized, dtype=np.float32)
    img_array   = np.expand_dims(img_array, axis=0)
    img_array   = preprocess_input(img_array)

    with st.spinner('🔍 Analyzing MRI...'):
        predictions     = model.predict(img_array, verbose=0)
        predicted_idx   = int(np.argmax(predictions[0]))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence      = float(predictions[0][predicted_idx]) * 100
        all_probs       = {cls: float(predictions[0][i]) * 100
                           for i, cls in enumerate(CLASS_NAMES)}

    with col2:
        st.markdown("#### 🎯 AI Diagnosis Result")
        info = CLASS_INFO[predicted_class]

        st.markdown(
            f"<div class='result-card'>"
            f"<span class='badge {info['badge_class']}'>"
            f"{info['icon']}  {predicted_class}</span><br><br>"
            f"<p style='color:#c5cee0; font-size:15px; margin: 0;'>{info['desc']}</p>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Confidence metric with color coding
        if confidence >= 85:
            conf_color = "#06d6a0"
            conf_label = "🟢 Excellent"
            conf_emoji = "✨"
        elif confidence >= 70:
            conf_color = "#ffd166"
            conf_label = "🟡 Good"
            conf_emoji = "👍"
        else:
            conf_color = "#ff6b6b"
            conf_label = "🔴 Low"
            conf_emoji = "⚠️"
        
        st.markdown(f"""
<div style='background: linear-gradient(135deg, #1a1f2e 0%, #262d3c 100%); border: 1px solid #2a3045; border-radius: 12px; padding: 20px; text-align: center;'>
<p style='color: #8a94a6; margin: 0; font-size: 12px; font-weight: 600;'>CONFIDENCE SCORE</p>
<h2 style='color: {conf_color}; margin: 12px 0 8px 0; font-size: 48px;'>{confidence:.1f}%</h2>
<p style='color: {conf_color}; margin: 0; font-size: 14px; font-weight: 600;'>{conf_label} {conf_emoji}</p>
</div>""", unsafe_allow_html=True)

        if confidence < LOW_CONF_THRESH:
            st.warning(
                f"⚠️ **Uncertain Result** — Confidence is {confidence:.1f}%. "
                "Professional medical evaluation is essential.",
                icon="⚠️"
            )

        # Interactive Probability Chart with Plotly
        st.markdown("#### 📊 Prediction Confidence Distribution")
        
        sorted_classes = sorted(CLASS_NAMES, key=lambda x: predictions[0][CLASS_NAMES.index(x)], reverse=True)
        sorted_probs = [float(predictions[0][CLASS_NAMES.index(cls)]) * 100 for cls in sorted_classes]
        colors_map = [BAR_COLORS[cls] for cls in sorted_classes]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=sorted_classes,
            x=sorted_probs,
            orientation='h',
            marker=dict(
                color=colors_map,
                line=dict(color=['white' if cls == predicted_class else 'rgba(255,255,255,0.3)' 
                               for cls in sorted_classes], width=2)
            ),
            text=[f'{p:.1f}%' for p in sorted_probs],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2f}%<extra></extra>',
        ))
        
        fig.update_layout(
            xaxis_title="Confidence (%)",
            yaxis_title="Tumor Type",
            template="plotly_dark",
            hovermode='closest',
            margin=dict(l=0, r=50, t=30, b=0),
            height=320,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            font=dict(color='#c5cee0', family='Segoe UI'),
            xaxis=dict(
                range=[0, 100],
                gridcolor='rgba(42, 48, 69, 0.3)',
                showgrid=True,
            ),
            yaxis=dict(gridcolor='rgba(42, 48, 69, 0.3)'),
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # ── Success Message ───────────────────────────────────────────────────────
    st.success(
        f"✅ **Analysis Complete!** Prediction: **{predicted_class}** ({confidence:.1f}% confidence)",
        icon="✅"
    )

    if confidence < LOW_CONF_THRESH:
        st.warning(
            f"⚠️ **Uncertain Result** — Confidence is {confidence:.1f}%. "
            "Professional medical evaluation is essential.",
            icon="⚠️"
        )

    # Detailed probability display
    st.markdown("#### 📋 Detailed Class Breakdown")
    for i, cls in enumerate(CLASS_NAMES):
        prob        = float(predictions[0][i]) * 100
        color       = BAR_COLORS[cls]
        is_top      = (cls == predicted_class)
        label_style = "font-weight:bold; color:#fff;" if is_top else "color:#8a94a6;"
        st.markdown(
            f"<div class='prob-bar-wrap'>"
            f"<div class='conf-label' style='{label_style}'>{cls}</div>"
            f"<div class='prob-bar-bg'>"
            f"<div class='prob-bar-fill' style='width:{prob:.1f}%; background:{color};'></div>"
            f"<span class='prob-bar-text'>{prob:.1f}%</span>"
            f"</div></div>",
            unsafe_allow_html=True
        )
    
    # ── Download Report Section ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📄 Generate Detailed PDF Report")
    st.markdown("""
This comprehensive report includes:
- ✅ AI prediction with confidence score
- 📋 Clinical overview of diagnosis
- 🏥 Recommended next medical steps
- ⚠️ Important precautions to follow
- 💪 Lifestyle recommendations
- 🚨 Warning signs requiring emergency care
    """)
    
    # Patient name input
    patient_name = st.text_input(
        "Patient Name (Optional)",
        placeholder="Enter patient name or leave blank",
        help="Include patient information in the PDF report"
    )
    
    st.markdown("")

    with st.spinner('Generating PDF report...'):
        pdf_bytes = generate_pdf_report(
            predicted_class = predicted_class,
            confidence      = confidence,
            all_probs       = all_probs,
            filename        = uploaded_file.name,
            img_dims        = (image.width, image.height),
            patient_name    = patient_name if patient_name else "Not Provided",
        )

    st.download_button(
        label               = "⬇️ Download PDF Report",
        data                = pdf_bytes,
        file_name           = f"brain_tumor_report_{predicted_class.lower().replace(' ', '_')}.pdf",
        mime                = "application/pdf",
        use_container_width = True,
    )

    # ── Medical Disclaimer ────────────────────────────────────────────────────
    st.markdown(
        "<div class='disclaimer'>"
        "⚠️ <strong>Medical Disclaimer:</strong> This tool is for <strong>educational "
        "purposes only</strong> and is not a substitute for professional medical advice, "
        "diagnosis, or treatment. Always consult a qualified doctor or radiologist for "
        "proper medical evaluation."
        "</div>",
        unsafe_allow_html=True
    )

    # ── Medical Disclaimer ────────────────────────────────────────────────────
    st.markdown(
        "<div class='disclaimer'>"
        "⚠️ <strong>Medical Disclaimer:</strong> This tool is for <strong>educational "
        "purposes only</strong> and is not a substitute for professional medical advice, "
        "diagnosis, or treatment. Always consult a qualified doctor or radiologist for "
        "proper medical evaluation."
        "</div>",
        unsafe_allow_html=True
    )

# ── App Information & Footer ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("### ℹ️ About This Application")

info_col1, info_col2, info_col3 = st.columns(3, gap="medium")

with info_col1:
    st.markdown("""
**🛠️ Technology Stack**
- Framework: Streamlit
- Model: ResNet50 (Transfer Learning)
- Dataset: Brain Tumor MRI (Kaggle)
- Backend: TensorFlow/Keras
    """)

with info_col2:
    st.markdown("""
**🎓 Model Performance**
- Classes: 4 tumor types
- Input Size: 224×224 pixels
- Architecture: Deep CNN
- Optimization: Data augmentation
    """)

with info_col3:
    st.markdown("""
**📊 Detection Classes**
- 🔴 Glioma
- 🟠 Meningioma
- 🟣 Pituitary
- ✅ No Tumor
    """)

st.markdown("---")

st.markdown("""
### 🔒 Privacy & Security
- ✅ No data is stored or transmitted to external servers
- ✅ All processing happens locally on your device
- ✅ Images are deleted after analysis
- ✅ Completely anonymous analysis

### ⚠️ Important Limitations
- ❌ NOT a replacement for professional radiologist review
- ❌ NOT suitable for clinical diagnosis
- ❌ Results must be verified by qualified medical professionals
- ❌ Model trained on specific dataset; real-world performance may vary
- ❌ NOT FDA-approved or clinically validated

### 📞 When to Seek Professional Help
If you experience any of these symptoms, consult a doctor immediately:
- Persistent headaches or migraines
- Seizures or convulsions
- Vision or hearing changes
- Weakness or numbness in limbs
- Memory problems or confusion
- Loss of balance or coordination
- Nausea or vomiting
- Any neurological symptoms

**Always prioritize professional medical consultation over AI results.**
""")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='footer'>🧠 Brain Tumor MRI Classifier • Powered by Deep Learning • Educational Tool Only</div>",
    unsafe_allow_html=True
)
