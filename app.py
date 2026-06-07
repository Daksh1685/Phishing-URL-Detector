import streamlit as st
import pandas as pd
from datetime import datetime
from ensemble_detector import PhishingRuleDetector

@st.cache_resource
def load_detector():
    return PhishingRuleDetector()

detector = load_detector()
st.set_page_config(
    page_title="Phishing URL Detector",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if 'history' not in st.session_state:
    st.session_state.history = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = pd.DataFrame()

st.title("Phishing URL Detector")

tab1, tab2, tab3, tab4 = st.tabs(["Check URL", "Batch Check", "History", "About"])


with tab1:
    st.header("Check Single URL")
    
    with st.form(key="url_check_form"):
        url_input = st.text_input(
            "Enter URL to check:",
            placeholder="e.g., http://example.com",
            key="single_url"
        )
        check_button = st.form_submit_button("Check", use_container_width=True)
    
    if check_button and url_input:
        try:

            prediction, confidence, reason = detector.detect(url_input)
            

            st.session_state.history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'url': url_input,
                'prediction': prediction,
                'confidence': confidence,
                'reason': reason
            })
            

            st.markdown("---")
            
            if prediction == 'phishing':
                st.error(f"PHISHING DETECTED")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prediction", prediction.upper())
                with col2:
                    st.metric("Confidence", f"{confidence:.1%}")
                with col3:
                    st.metric("Risk Level", "HIGH")
            else:
                st.success(f"LEGITIMATE URL")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prediction", prediction.upper())
                with col2:
                    st.metric("Confidence", f"{confidence:.1%}")
                with col3:
                    st.metric("Risk Level", "LOW")
            
            st.info(f"**Reason:** {reason}")
            
        except Exception as e:
            st.error(f"Error analyzing URL: {str(e)}")
    elif check_button:
        st.warning("Please enter a URL to check")


with tab2:
    st.header("Batch URL Analysis")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV with URLs (column: 'url')",
            type=['csv'],
            key="batch_upload"
        )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'url' not in df.columns:
                st.error("CSV must contain a 'url' column")
            else:
                if st.button("Analyze All URLs", use_container_width=True):
                    progress_bar = st.progress(0)
                    results = []
                    
                    for i, url in enumerate(df['url'].values):
                        try:
                            prediction, confidence, reason = detector.detect(url)
                            results.append({
                                'url': url,
                                'prediction': prediction,
                                'confidence': f"{confidence:.1%}",
                                'reason': reason
                            })
                        except Exception as e:
                            results.append({
                                'url': url,
                                'prediction': 'error',
                                'confidence': 'N/A',
                                'reason': str(e)
                            })
                        
                        progress_bar.progress((i + 1) / len(df))
                    
                    st.session_state.batch_results = pd.DataFrame(results)
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    if not st.session_state.batch_results.empty:
        st.markdown("---")
        st.subheader("Results")
        

        phishing_count = len(st.session_state.batch_results[st.session_state.batch_results['prediction'] == 'phishing'])
        legitimate_count = len(st.session_state.batch_results[st.session_state.batch_results['prediction'] == 'legitimate'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total URLs", len(st.session_state.batch_results))
        with col2:
            st.metric("Phishing", phishing_count)
        with col3:
            st.metric("Legitimate", legitimate_count)
        
        st.markdown("---")
        st.dataframe(st.session_state.batch_results, use_container_width=True)
        

        csv = st.session_state.batch_results.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"phishing_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


with tab3:
    st.header("Detection History")
    
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Checked", len(history_df))
        with col2:
            phishing_in_history = len(history_df[history_df['prediction'] == 'phishing'])
            st.metric("Phishing Found", phishing_in_history)
        with col3:
            legitimate_in_history = len(history_df[history_df['prediction'] == 'legitimate'])
            st.metric("Legitimate", legitimate_in_history)
        
        st.markdown("---")
        st.dataframe(history_df, use_container_width=True)
        

        csv_history = history_df.to_csv(index=False)
        st.download_button(
            label="Export History as CSV",
            data=csv_history,
            file_name=f"detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        if st.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No detection history yet. Start checking URLs to build history.")


with tab4:
    st.header("About This Application")
    






































