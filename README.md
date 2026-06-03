# Phishing URL Detector

Rule based phishing URL detection system

## Architecture

```
Input URL
    ↓
Preprocessing & Tokenization
    ↓
Multi-Embedding Layer:
├─ Word2Vec (100-dim)
├─ FastText (100-dim)
├─ GloVe (100-dim)
├─ BERT (768-dim)
├─ RoBERTa (768-dim)
└─ GPT-2 (768-dim)
    ↓
Embedding Fusion (Concatenation/Attention)
    ↓
Attention Layer
    ↓
1D CNN with Multiple Kernel Sizes
    ↓
Dense Layers with Dropout & Batch Norm
    ↓
Binary Classification Output (Sigmoid)
```

## Clone

```bash
git clone https://github.com/Daksh1685/Phishing-URL-Detector.git
cd Phishing-URL-Detector
```

## Features

- **Single URL Detection** - Check individual URLs for phishing
- **Batch Processing** - Upload CSV files to analyze multiple URLs
- **Brand Spoofing Detection** - Identifies homograph attacks (paypa1, amaz0n, etc.)
- **Pattern Matching** - Detects admin panels, dev folders, confirmation pages
- **Detection History** - Track all analyzed URLs with timestamps
- **CSV Export** - Download analysis results
- **Privacy First** - All analysis happens locally

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Usage

Visit app at: https://url-phising-detection.streamlit.app/
