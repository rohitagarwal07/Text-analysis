import streamlit as st
import speech_recognition as sr
import io
import wave
import numpy as np
from textblob import TextBlob
import pandas as pd
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import tempfile
import os
from st_audiorec import st_audiorec
import nltk
from fpdf import FPDF

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Page configuration
st.set_page_config(
    page_title="Speech-to-Text Analyzer",
    page_icon="🎤",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #A23B72;
        border-bottom: 2px solid #A23B72;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
</style>
""", unsafe_allow_html=True)


def transcribe_audio(audio_file):
    """Convert audio to text using speech recognition"""
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text, None
    except sr.UnknownValueError:
        return None, "Could not understand the audio"
    except sr.RequestError as e:
        return None, f"Error with speech recognition service: {e}"
    except Exception as e:
        return None, f"Error processing audio: {e}"


def analyze_text(text):
    """Perform comprehensive text analysis"""
    try:
        blob = TextBlob(text)

        # Basic metrics
        word_count = len(text.split())
        char_count = len(text)

        # Try to get sentence count using TextBlob, fallback to simple method
        try:
            sentence_count = len(blob.sentences)
        except:
            # Fallback: count sentences by periods, exclamation marks, and question marks
            sentence_count = len(re.findall(r'[.!?]+', text))
            if sentence_count == 0:
                sentence_count = 1

        # Sentiment analysis
        try:
            sentiment = blob.sentiment
            sentiment_polarity = sentiment.polarity
            sentiment_subjectivity = sentiment.subjectivity
        except:
            # Fallback: basic sentiment analysis
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like',
                              'happy', 'joy']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'disappointed', 'horrible',
                              'worst']

            words_lower = text.lower().split()
            pos_count = sum(1 for word in words_lower if word in positive_words)
            neg_count = sum(1 for word in words_lower if word in negative_words)
            total_sentiment_words = pos_count + neg_count

            if total_sentiment_words > 0:
                sentiment_polarity = (pos_count - neg_count) / total_sentiment_words
                sentiment_subjectivity = total_sentiment_words / len(words_lower)
            else:
                sentiment_polarity = 0.0
                sentiment_subjectivity = 0.0

        sentiment_label = "Positive" if sentiment_polarity > 0.1 else "Negative" if sentiment_polarity < -0.1 else "Neutral"

        # Keywords extraction (simple approach)
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is',
                      'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                      'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
                      'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        keyword_freq = Counter(keywords).most_common(10)

        # Summary (first sentence or first 150 chars)
        try:
            sentences = blob.sentences
            summary = str(sentences[0]) if len(sentences) > 0 else text[:150] + "..."
            if len(sentences) > 1 and len(summary) < 200:
                summary += " " + str(sentences[1])
        except:
            # Fallback: simple summary by taking first 150 characters
            sentences = re.split(r'[.!?]+', text)
            summary = sentences[0] if sentences else text[:150]
            if len(summary) < 150 and len(sentences) > 1:
                summary += ". " + sentences[1]
            summary = summary.strip() + ("..." if len(text) > len(summary) else "")

        return {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'sentiment_polarity': round(sentiment_polarity, 3),
            'sentiment_subjectivity': round(sentiment_subjectivity, 3),
            'sentiment_label': sentiment_label,
            'keywords': keyword_freq,
            'summary': summary
        }

    except Exception as e:
        st.error(f"Error in text analysis: {e}")
        # Return basic analysis if everything fails
        return {
            'word_count': len(text.split()),
            'char_count': len(text),
            'sentence_count': 1,
            'sentiment_polarity': 0.0,
            'sentiment_subjectivity': 0.0,
            'sentiment_label': 'Neutral',
            'keywords': [],
            'summary': text[:150] + "..." if len(text) > 150 else text
        }


def create_pdf_report(text, analysis):
    """Generate PDF report using FPDF"""

    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Speech-to-Text Analysis Report', 0, 1, 'C')
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', size=12)

    # Original Text Section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Original Text:', 0, 1)
    pdf.set_font('Arial', size=10)

    # Handle text encoding and length
    clean_text = text.encode('latin-1', 'ignore').decode('latin-1')
    # Split text into lines that fit the page
    lines = []
    words = clean_text.split(' ')
    current_line = ""
    for word in words:
        if len(current_line + word) < 80:  # Approximate character limit per line
            current_line += word + " "
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    if current_line:
        lines.append(current_line.strip())

    for line in lines[:20]:  # Limit to first 20 lines
        pdf.cell(0, 6, line, 0, 1)

    pdf.ln(5)

    # Summary Section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Summary:', 0, 1)
    pdf.set_font('Arial', size=10)
    summary_clean = analysis['summary'].encode('latin-1', 'ignore').decode('latin-1')
    summary_lines = []
    words = summary_clean.split(' ')
    current_line = ""
    for word in words:
        if len(current_line + word) < 80:
            current_line += word + " "
        else:
            summary_lines.append(current_line.strip())
            current_line = word + " "
    if current_line:
        summary_lines.append(current_line.strip())

    for line in summary_lines:
        pdf.cell(0, 6, line, 0, 1)

    pdf.ln(5)

    # Analysis Section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Text Analysis:', 0, 1)
    pdf.set_font('Arial', size=10)

    analysis_data = [
        f"Word Count: {analysis['word_count']}",
        f"Character Count: {analysis['char_count']}",
        f"Sentence Count: {analysis['sentence_count']}",
        f"Sentiment: {analysis['sentiment_label']}",
        f"Polarity: {analysis['sentiment_polarity']}",
        f"Subjectivity: {analysis['sentiment_subjectivity']}"
    ]

    for item in analysis_data:
        pdf.cell(0, 6, item, 0, 1)

    pdf.ln(5)

    # Keywords Section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Top Keywords:', 0, 1)
    pdf.set_font('Arial', size=10)

    for word, count in analysis['keywords'][:10]:
        keyword_line = f"{word}: {count}"
        pdf.cell(0, 6, keyword_line, 0, 1)

    # Save to bytes - FIXED VERSION
    buffer = io.BytesIO()

    try:
        # Try the newer fpdf2 method first
        pdf_bytes = pdf.output()
        if isinstance(pdf_bytes, str):
            # If it returns a string, encode it
            buffer.write(pdf_bytes.encode('latin-1'))
        else:
            # If it returns bytes directly, use as is
            buffer.write(pdf_bytes)
    except TypeError:
        # Fallback for older FPDF versions
        try:
            pdf_string = pdf.output(dest='S')
            if isinstance(pdf_string, str):
                buffer.write(pdf_string.encode('latin-1'))
            else:
                buffer.write(pdf_string)
        except Exception as e:
            # Last resort: save to temporary file and read
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                pdf.output(tmp.name)
                with open(tmp.name, 'rb') as f:
                    buffer.write(f.read())
                os.unlink(tmp.name)

    buffer.seek(0)
    return buffer


def main():
    st.markdown("<h1 class='main-header'>🎤 Speech-to-Text Analyzer</h1>", unsafe_allow_html=True)

    # Initialize session state
    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = ""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    # Create tabs
    tab1, tab2 = st.tabs(["🎙️ Record Audio", "📁 Upload Audio File"])

    with tab1:
        st.markdown("### Live Audio Recording")

        # Audio recorder
        wav_audio_data = st_audiorec()

        if wav_audio_data is not None:
            st.audio(wav_audio_data, format='audio/wav')

            if st.button("🔄 Transcribe Recording", type="primary"):
                with st.spinner("Transcribing audio..."):
                    # Save audio data to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(wav_audio_data)
                        tmp_file_path = tmp_file.name

                    try:
                        # Transcribe audio
                        text, error = transcribe_audio(tmp_file_path)

                        if text:
                            st.session_state.transcribed_text = text
                            st.success("Audio transcribed successfully!")
                        else:
                            st.error(f"Transcription failed: {error}")
                    finally:
                        # Clean up temporary file
                        os.unlink(tmp_file_path)

    with tab2:
        st.markdown("### Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'mp4', 'm4a', 'flac'],
            help="Supported formats: WAV, MP3, MP4, M4A, FLAC"
        )

        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')

            if st.button("🔄 Transcribe Uploaded File", type="primary"):
                with st.spinner("Transcribing audio..."):
                    # Save uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name

                    try:
                        text, error = transcribe_audio(tmp_file_path)

                        if text:
                            st.session_state.transcribed_text = text
                            st.success("Audio transcribed successfully!")
                        else:
                            st.error(f"Transcription failed: {error}")
                    finally:
                        os.unlink(tmp_file_path)

    # Display transcribed text
    if st.session_state.transcribed_text:
        st.markdown("<h3 class='section-header'>📝 Transcribed Text</h3>", unsafe_allow_html=True)

        # Editable text area
        edited_text = st.text_area(
            "Edit transcribed text if needed:",
            value=st.session_state.transcribed_text,
            height=150,
            key="text_editor"
        )

        # Update session state if text is edited
        if edited_text != st.session_state.transcribed_text:
            st.session_state.transcribed_text = edited_text

        # Analyze button
        if st.button("📊 Analyze Text", type="primary"):
            with st.spinner("Analyzing text..."):
                st.session_state.analysis_results = analyze_text(st.session_state.transcribed_text)

    # Display analysis results
    if st.session_state.analysis_results:
        analysis = st.session_state.analysis_results

        st.markdown("<h3 class='section-header'>📊 Text Analysis Results</h3>", unsafe_allow_html=True)

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Words", analysis['word_count'])

        with col2:
            st.metric("Characters", analysis['char_count'])

        with col3:
            st.metric("Sentences", analysis['sentence_count'])

        with col4:
            st.metric("Sentiment", analysis['sentiment_label'])

        # Detailed analysis
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎭 Sentiment Analysis")
            st.write(f"**Label:** {analysis['sentiment_label']}")
            st.write(f"**Polarity:** {analysis['sentiment_polarity']} (-1 to 1)")
            st.write(f"**Subjectivity:** {analysis['sentiment_subjectivity']} (0 to 1)")

            # Sentiment visualization
            fig, ax = plt.subplots(figsize=(8, 4))
            sentiments = ['Negative', 'Neutral', 'Positive']
            colors = ['#ff4444', '#ffaa44', '#44ff44']
            values = [1 if analysis['sentiment_label'] == s else 0.3 for s in sentiments]
            bars = ax.bar(sentiments, values, color=colors, alpha=0.7)

            # Highlight current sentiment
            current_idx = sentiments.index(analysis['sentiment_label'])
            bars[current_idx].set_alpha(1.0)
            bars[current_idx].set_edgecolor('black')
            bars[current_idx].set_linewidth(2)

            ax.set_ylabel('Intensity')
            ax.set_title('Sentiment Analysis')
            ax.set_ylim(0, 1.2)
            st.pyplot(fig)

        with col2:
            st.subheader("🔑 Top Keywords")
            if analysis['keywords']:
                keywords_df = pd.DataFrame(analysis['keywords'], columns=['Keyword', 'Frequency'])
                st.dataframe(keywords_df, use_container_width=True)

                # Word cloud
                if len(analysis['keywords']) > 0:
                    word_freq_dict = dict(analysis['keywords'])
                    wordcloud = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(
                        word_freq_dict)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title('Word Cloud')
                    st.pyplot(fig)
            else:
                st.write("No keywords found.")

        # Summary
        st.subheader("📋 Summary")
        st.write(analysis['summary'])

        # PDF Export
        st.markdown("<div class='section-header'>📄 Export to PDF</div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📥 Generate PDF Report", type="secondary", use_container_width=True):
                with st.spinner("📄 Generating PDF..."):
                    pdf_buffer = create_pdf_report(st.session_state.transcribed_text, analysis)

                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_buffer.getvalue(),
                        file_name="speech_analysis_report.pdf",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )


if __name__ == "__main__":
    main()