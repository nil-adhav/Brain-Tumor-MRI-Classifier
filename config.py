"""
Configuration for Brain Tumor MRI Classifier
Easy switching between different model types and inference methods
"""

import os

# ── Model Configuration ───────────────────────────────────────────────────────

# Available models: 'single_resnet', 'ensemble'
MODEL_TYPE = 'single_resnet'  # Change to 'ensemble' for better accuracy

# Class names (must match training order)
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Image size for model input
IMG_SIZE = 224

# Confidence thresholds
LOW_CONF_THRESH = 60.0        # Warning if below this
CRITICAL_THRESH = 50.0        # Very low confidence threshold

# ── Model Weights Configuration ───────────────────────────────────────────────

# Single Model Weights
SINGLE_MODEL_WEIGHTS = {
    'resnet': 'resnet_weights.weights.h5',
    'efficientnet': 'efficientnet_weights.weights.h5',
    'densenet': 'densenet_weights.weights.h5',
}

# Ensemble Weights (if you have all three models trained)
ENSEMBLE_WEIGHTS = {
    'resnet': 'resnet_weights_improved.weights.h5',
    'efficientnet': 'efficientnet_weights.weights.h5',
    'densenet': 'densenet_weights.weights.h5',
}

# ── UI Configuration ─────────────────────────────────────────────────────────

# Color scheme
COLORS = {
    'primary': '#4fc3f7',
    'secondary': '#b98aff',
    'success': '#06d6a0',
    'warning': '#ffd166',
    'danger': '#ff6b6b',
}

# Gradient backgrounds
GRADIENTS = {
    'header': 'linear-gradient(135deg, #4fc3f7, #b98aff)',
    'card': 'linear-gradient(135deg, #1a1f2e 0%, #262d3c 100%)',
}

# ── Inference Configuration ──────────────────────────────────────────────────

# Ensemble prediction method: 'average', 'voting', 'weighted'
ENSEMBLE_METHOD = 'weighted'

# Enable confidence calibration
ENABLE_CALIBRATION = True

# Temperature for confidence scaling (only if calibration enabled)
TEMPERATURE = 1.2

# ── Feature Flags ────────────────────────────────────────────────────────────

# Features
ENABLE_PDF_EXPORT = True
ENABLE_CHARTS = True
ENABLE_LIVE_DEMO = False
ENABLE_MODEL_STATS = False

# Validation
VALIDATE_MRI = True            # Check if image is grayscale medical image
MRI_COLOR_THRESHOLD = 18.0     # Color score threshold for MRI validation

# ── Model-specific Settings ──────────────────────────────────────────────────

# Cache model in memory (faster, uses more RAM)
CACHE_MODEL = True

# Show model loading spinner
SHOW_LOADING_SPINNER = True

# Enable model statistics display
SHOW_MODEL_STATS = False

# ── Utility Functions ────────────────────────────────────────────────────────

def is_model_available(model_name: str) -> bool:
    """Check if model weights file exists"""
    if model_name == 'resnet':
        return os.path.exists(SINGLE_MODEL_WEIGHTS['resnet'])
    elif model_name == 'ensemble':
        return all(os.path.exists(path) for path in ENSEMBLE_WEIGHTS.values())
    return False

def get_available_models() -> list:
    """Get list of available models"""
    available = []
    if is_model_available('resnet'):
        available.append('ResNet50')
    if is_model_available('ensemble'):
        available.append('Ensemble (Best Accuracy)')
    return available

def get_confidence_category(confidence: float) -> tuple:
    """
    Get confidence category and emoji
    
    Args:
        confidence: Confidence percentage (0-100)
    
    Returns:
        Tuple of (category_name, emoji, color)
    """
    if confidence >= 90:
        return "Excellent", "🟢", COLORS['success']
    elif confidence >= 75:
        return "Good", "🟡", COLORS['warning']
    elif confidence >= 60:
        return "Moderate", "🟠", COLORS['warning']
    elif confidence >= 50:
        return "Low", "🟠", COLORS['danger']
    else:
        return "Very Low", "🔴", COLORS['danger']

# ── Display Utilities ────────────────────────────────────────────────────────

class UIStrings:
    """UI text strings"""
    
    TITLE = "🧠 Brain Tumor MRI Classifier"
    SUBTITLE = "Advanced AI-Powered Medical Image Analysis"
    TAGLINE = "Powered by ResNet50 Deep Learning • Real-time Analysis • Educational Purpose"
    
    FEATURE_ONE = ("🤖", "AI-Powered", "ResNet50 Model")
    FEATURE_TWO = ("⚡", "Instant Results", "2-3 Seconds")
    FEATURE_THREE = ("📊", "Detailed Reports", "PDF Export")
    FEATURE_FOUR = ("🔒", "100% Private", "No Data Stored")
    
    UPLOAD_INSTRUCTION = "📤 Upload Brain MRI Image"
    UPLOAD_HELP = "Supported formats: JPG, JPEG, PNG (grayscale medical images)"
    
    RESULT_TITLE = "🎯 AI Diagnosis Result"
    CONFIDENCE_TITLE = "CONFIDENCE SCORE"
    PROBABILITIES_TITLE = "📊 Prediction Confidence Distribution"
    BREAKDOWN_TITLE = "📋 Detailed Class Breakdown"
    
    PDF_TITLE = "📄 Generate Detailed PDF Report"
    PATIENT_NAME = "Patient Name (Optional)"
    
    DISCLAIMER_SHORT = "⚠️ Educational tool only - consult medical professionals"
    DISCLAIMER_FULL = "This tool is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment."

if __name__ == "__main__":
    print("Configuration Module")
    print(f"Model Type: {MODEL_TYPE}")
    print(f"Available Models: {get_available_models()}")
    print(f"Confidence Threshold: {LOW_CONF_THRESH}%")
