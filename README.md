# Phishing URL Detector

Rule-based phishing URL detection system with web interface.

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
- **Rule-Based** - No ML models, lightweight and fast
- **Privacy First** - All analysis happens locally

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Usage

Visit app at: https://url-phising-detection.streamlit.app/