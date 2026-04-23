<<<<<<< HEAD
# 🧠 Brain Tumor MRI Classifier

An AI-powered web application for analyzing brain MRI scans and classifying tumor types using deep learning (ResNet50 transfer learning).

## 🎯 Features

- **Instant MRI Analysis** - Process brain MRI scans in 2-3 seconds
- **4-Class Classification**:
  - 🔴 **Glioma** - Tumors from glial cells
  - 🟠 **Meningioma** - Tumors from brain membranes
  - 🟣 **Pituitary** - Tumors from pituitary gland
  - ✅ **No Tumor** - Healthy scans
- **Confidence Scoring** - See model confidence percentages
- **PDF Reports** - Generate detailed diagnostic reports with clinical guidance
- **Privacy-Focused** - All processing is local; no data stored on servers

## ⚠️ Important Disclaimer

**This application is for educational and research purposes only.**
- ❌ NOT a substitute for professional medical advice
- ❌ NOT for clinical diagnosis or treatment decisions
- ❌ Results MUST be verified by qualified medical professionals
- ❌ NOT clinically validated or FDA-approved

**Always consult a qualified neurologist, neurosurgeon, or radiologist for proper medical evaluation.**

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/nil-adhav/Brain-Tumor-MRI-Classifier.git
   cd Brain-Tumor-MRI-Classifier
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pre-trained model weights**
   - Place `resnet_weights.weights.h5` in the project root directory
   - This file is required to run the app

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

The app will open at `http://localhost:8501`

## 📦 Project Structure

```
Brain-Tumor-MRI-Classifier/
├── app.py                              # Main Streamlit application
├── run_app.py                          # Script to run the app
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore rules
├── .streamlit/
│   └── config.toml                    # Streamlit configuration
├── resnet_weights.weights.h5          # Pre-trained model weights (not in repo)
├── Brain_Tumor_MRI_Classifier.ipynb   # Training notebook
├── Image Dataset/                      # Training/testing datasets (not in repo)
│   ├── Training/
│   └── Testing/
└── README.md                          # This file
```

## 🤖 Model Architecture

**Base Model:** ResNet50 (Transfer Learning)
- Pre-trained on ImageNet
- 50 deep residual layers
- Optimized for medical image analysis

**Custom Head:**
1. Global Average Pooling
2. Dense Layer (256 units, ReLU activation)
3. Dropout (40% regularization)
4. Output Layer (4 classes, Softmax)

## 📊 Model Performance

- **Dataset:** Brain Tumor MRI Dataset (Kaggle)
- **Image Size:** 224×224 pixels
- **Training:** Data augmentation & regularization
- **Optimization:** Adam optimizer with early stopping

## 🌐 Deploy on Streamlit Community Cloud

### Prerequisites
- GitHub account
- Code pushed to GitHub repository

### Deployment Steps

1. **Push code to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit: Brain Tumor MRI Classifier"
   git push origin main
   ```

2. **Visit Streamlit Cloud**
   - Go to
   - Click "New app"

3. **Configure Deployment**
   - Repository: `nil-adhav/Brain-Tumor-MRI-Classifier`
   - Branch: `main`
   - Main file path: `app.py`

4. **Add Secrets** (if needed)
   - Settings → Secrets management

5. **Deploy**
   - Click "Deploy"
   - Streamlit will install dependencies from `requirements.txt`

⚠️ **Important:** Large model weights files (`.h5`, `.keras`) should be:
- Stored separately or uploaded to Streamlit Cloud's file storage
- NOT committed to GitHub
- Downloaded during app initialization

## 📋 Usage Instructions

1. **Upload MRI Image**
   - Supported formats: JPG, JPEG, PNG
   - Image must be a grayscale brain MRI scan
   - Recommended size: 224×224 pixels or larger

2. **View Results**
   - AI prediction with confidence percentage
   - Probability scores for all classes
   - Color-coded results

3. **Generate Report**
   - Optional: Enter patient name
   - Download comprehensive PDF report
   - Includes clinical guidance and precautions

## 🔧 Technologies Used

- **Streamlit** - Web framework for ML/AI apps
- **TensorFlow/Keras** - Deep learning framework
- **ResNet50** - Transfer learning model
- **NumPy** - Numerical computing
- **Pillow** - Image processing
- **ReportLab** - PDF generation

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚖️ Legal Notice

**Medical Disclaimer:**
This software is provided for educational purposes only. It is not intended to be used for medical diagnosis, treatment, or decision-making. Users assume full responsibility for the use of this tool. The creators and contributors are not liable for any misuse or incorrect diagnoses resulting from this application.

**Always consult with qualified healthcare professionals for medical concerns.**

---

**Developed with ❤️ for educational purposes**
=======
# nil-adhav-Brain-Tumor-MRI-Classifier
>>>>>>> 2f0e21999333cd586da6c305a7f95551b05e54fc
