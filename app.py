import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from googleapiclient.discovery import build
import pandas as pd
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Disney Sentiment Dashboard", page_icon="🏰", layout="wide")

# Download NLTK data
@st.cache_resource
def download_nltk():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

download_nltk()

# API Key for YouTube
API_KEY = st.secrets["YOUTUBE_API_KEY"]

# Load the model and tokenizer
@st.cache_resource
def load_assets():
    try:
        model = tf.keras.models.load_model('sentiment_bilstm_model.keras')
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None

model, tokenizer = load_assets()
if model is None or tokenizer is None:
    st.stop()

def clean_predict_text(text):
    def expand_contractions(t):
        contractions = {
            "can't": "cannot", "won't": "will not", "n't": " not",
            "i'm": "i am", "it's": "it is", "i've": "i have",
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "wasn't": "was not", "weren't": "were not", "haven't": "have not",
            "hasn't": "has not", "hadn't": "had not", "shouldn't": "should not",
            "couldn't": "could not", "wouldn't": "would not"
        }
        for contraction, expanded in contractions.items():
            t = t.replace(contraction, expanded)
        return t

    if not isinstance(text, str): return ""
    text = text.lower().strip()
    text = expand_contractions(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    stop_words = set(stopwords.words('english'))
    negation_words = {'not', 'no', 'nor', 'neither', 'never', 'none', "cannot"}
    stop_words = stop_words - negation_words
    lemmatizer = WordNetLemmatizer()
    
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def get_video_details(video_id):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    try:
        request = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        )
        response = request.execute()
        if response['items']:
            item = response['items'][0]
            return {
                'title': item['snippet']['title'],
                'channel': item['snippet']['channelTitle'],
                'views': int(item['statistics'].get('viewCount', 0)),
                'likes': int(item['statistics'].get('likeCount', 0)),
                'thumbnail': item['snippet']['thumbnails']['high']['url']
            }
    except Exception as e:
        st.error(f"Could not fetch video details: {e}")
    return None

def get_video_comments(video_id, max_comments=100):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    comments_data = []
    next_page_token = None
    
    try:
        while len(comments_data) < max_comments:
            try:
                # Calculate how many more comments we need
                remaining = max_comments - len(comments_data)
                # API limit is 100 per request
                batch_size = min(remaining, 100)
                
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    textFormat="plainText",
                    maxResults=batch_size,
                    order="relevance",
                    pageToken=next_page_token
                )
                response = request.execute()
                
                for item in response.get('items', []):
                    snippet = item['snippet']['topLevelComment']['snippet']
                    comments_data.append({
                        'text': snippet['textDisplay'],
                        'likes': snippet['likeCount'],
                        'published_at': snippet['publishedAt']
                    })
                    if len(comments_data) >= max_comments:
                        break
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
            except Exception as e:
                error_str = str(e).lower()
                if "processingfailure" in error_str or "400" in error_str:
                    st.warning("⚠️ Some comments could not be fetched due to a YouTube API processing error. Proceeding with collected data.")
                    break
                else:
                    raise e
                    
    except Exception as e:
        st.error(f"Could not fetch comments: {e}")
        
    return comments_data

# Sidebar
st.sidebar.title("Disney Marketing Insights")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/a/a4/Disney_wordmark.svg", width=150)
st.sidebar.markdown("""
**Role:** Marketing Analytics Lead
**Objective:** Optimize trailer engagement, audience perception, and campaign performance.
""")

# Main App
st.title("🏰 Disney Trailer Sentiment Intelligence Dashboard")
st.write("Analyze real-time YouTube audience sentiment to benchmark trailer performance and uncover actionable marketing insights.")

tab1, tab2 = st.tabs(["Real Time Trailer Analysis", "Individual Comment Check"])

with tab1:
    st.subheader("YouTube Trailer Benchmark")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        video_url = st.text_input("Enter Disney Trailer URL:", placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    with col2:
        max_c = st.slider("Number of comments to analyze", 50, 500, 100)

    if st.button("Run Sentiment Analysis"):
        if not video_url:
            st.warning("Please enter a YouTube URL.")
        else:
            video_id = ""
            if "v=" in video_url: video_id = video_url.split("v=")[1].split("&")[0]
            elif "be/" in video_url: video_id = video_url.split("be/")[1].split("?")[0]
            
            if video_id:
                with st.spinner('🏰 Processing Disney Intelligence Report...'):
                    # Fetch Video Metadata
                    video_info = get_video_details(video_id)
                    raw_comments_data = get_video_comments(video_id, max_comments=max_c)
                    
                    if raw_comments_data:
                        # Display Video Info
                        if video_info:
                            with st.container(border=True):
                                vcol1, vcol2 = st.columns([1, 3])
                                with vcol1:
                                    st.image(video_info['thumbnail'], width='stretch')
                                with vcol2:
                                    st.markdown(f"### {video_info['title']}")
                                    st.markdown(f"**Channel:** {video_info['channel']} | **Views:** {video_info['views']:,} | **Likes:** {video_info['likes']:,}")
                        
                        # Convert to DataFrame
                        df_comments = pd.DataFrame(raw_comments_data)
                        df_comments['published_at'] = pd.to_datetime(df_comments['published_at'])
                        df_comments = df_comments.sort_values('published_at', ascending=False)
                        
                        # Preprocess and Predict
                        processed_texts = [clean_predict_text(c) for c in df_comments['text']]
                        seqs = tokenizer.texts_to_sequences(processed_texts)
                        padded = pad_sequences(seqs, maxlen=50, padding='post', truncating='post')
                        preds = model.predict(padded)
                        df_comments['pred_class'] = np.argmax(preds, axis=1)
                        df_comments['conf'] = np.max(preds, axis=1)
                        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
                        df_comments["sentiment"] = df_comments["pred_class"].map(label_map)
                        
                        # RECENCY WEIGHTING LOGIC
                        # We split into 3 buckets: Latest 40%, Middle 40%, Old 20%
                        n = len(df_comments)
                        latest_idx = int(n * 0.4)
                        middle_idx = int(n * 0.8)
                        
                        weights = np.ones(n)
                        weights[:latest_idx] = 1.5 # Boost latest
                        weights[latest_idx:middle_idx] = 1.0 # Standard middle
                        weights[middle_idx:] = 0.5 # De-weight old
                        df_comments['weight'] = weights
                        
                        # --- CALCULATE 10 MARKETING METRICS ---
                        
                        # 1. Weighted Sentiment Score (-100 to 100)
                        # Map 0->-1, 1->0, 2->1
                        sent_map = {0: -1, 1: 0, 2: 1}
                        df_comments['sent_val'] = df_comments['pred_class'].map(sent_map)
                        weighted_score = (df_comments['sent_val'] * df_comments['weight']).sum() / df_comments['weight'].sum()
                        final_sentiment_index = weighted_score * 100
                        
                        # 2. Sentiment Momentum (Latest vs Old)
                        latest_avg = df_comments['sent_val'][:latest_idx].mean()
                        old_avg = df_comments['sent_val'][middle_idx:].mean()
                        momentum = (latest_avg - old_avg) * 100
                        
                        # 3. Brand Advocacy Rate (Strongly Positive %)
                        advocacy = (df_comments['pred_class'] == 2).mean() * 100
                        
                        # 4. Critical Friction Index (Strongly Negative %)
                        friction = (df_comments['pred_class'] == 0).mean() * 100
                        
                        # 5. Engagement Virality (Likes per comment avg)
                        avg_likes = df_comments['likes'].mean()
                        
                        # 6. Audience Polarisation (Std Dev of Sentiment)
                        polarisation = df_comments['sent_val'].std() * 100
                        
                        # 7. Production Quality Index (Keywords check)
                        prod_keywords = ['cgi', 'visual', 'animation', 'look', 'effect', 'music', 'sound', 'graphics']
                        prod_mentions = df_comments[df_comments['text'].str.lower().str.contains('|'.join(prod_keywords))]
                        prod_sentiment = prod_mentions['sent_val'].mean() * 100 if not prod_mentions.empty else 0
                        
                        # 8. Purchase/Watch Intent
                        intent_keywords = ['buy', 'watch', 'ticket', 'theater', 'cinema', 'cant wait', 'hyped', 'going']
                        intent_mentions = df_comments[df_comments['text'].str.lower().str.contains('|'.join(intent_keywords))]
                        watch_intent = (len(intent_mentions) / n) * 100
                        
                        # 9. Hype Velocity (Recent comments per day - simplified)
                        hype_velocity = latest_idx / max(1, (df_comments['published_at'].max() - df_comments['published_at'].min()).days)
                        
                        # 10. Content Fatigue Risk (Repetitive neutral feedback)
                        neutral_rate = (df_comments['pred_class'] == 1).mean() * 100
                        
                        st.success(f"Successfully analyzed {n} comments with Recency Weighting.")
                        
                        # --- DASHBOARD DISPLAY ---
                        st.subheader("📊 Executive Summary")
                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("Sentiment Index", f"{final_sentiment_index:.1f}", delta=f"{momentum:.1f}% Momentum")
                        k2.metric("Watch Intent", f"{watch_intent:.1f}%")
                        k3.metric("Advocacy Rate", f"{advocacy:.1f}%")
                        k4.metric("Friction Index", f"{friction:.1f}%", delta_color="inverse")
                        
                        st.write("---")
                        
                        col_a, col_b, col_c = st.columns([1, 1, 1])
                        with col_a:
                            st.write("**Detailed Marketing Metrics**")
                            metrics_data = pd.DataFrame({
                                "Metric": ["Polarisation", "Production Quality", "Neutral Fatigue", "Engagement Virality"],
                                "Value": [f"{polarisation:.1f}%", f"{prod_sentiment:.1f}", f"{neutral_rate:.1f}%", f"{avg_likes:.2f} likes/avg"]
                            })
                            st.table(metrics_data)
                        
                        with col_b:
                            st.write("**Strategic Action Points**")
                            if final_sentiment_index > 20:
                                st.write("🟢 **Greenlight Campaign:** Audience reception is strong. Increase global ad-spend.")
                            elif final_sentiment_index < -10:
                                st.write("🔴 **Crisis Management:** Significant backlash detected. Review creative direction or CGI quality.")
                            else:
                                st.write("🟡 **Optimization Required:** Engagement is tepid. Re-cut trailer with higher 'hype' moments.")
                                
                            if watch_intent < 5:
                                st.write("👉 **Action:** Add clearer Call-to-Action (CTA) for ticket bookings in social media posts.")
                            if prod_sentiment < 0:
                                st.write("👉 **Action:** Technical audit of visual effects required based on audience feedback.")
                            if momentum < -10:
                                st.write("👉 **Action:** Sentiment is dropping. Investigate if recent news or the trailer itself caused the dip.")

                        with col_c:
                            st.write("**Buzzword Visualization**")
                            all_text = " ".join(processed_texts)
                            if all_text.strip():
                                wc = WordCloud(width=400, height=300, background_color=None, mode="RGBA", colormap='viridis', scale=3).generate(all_text)
                                fig, ax = plt.subplots(figsize=(4, 3), dpi=600)
                                ax.imshow(wc, interpolation='bilinear')
                                ax.axis("off")
                                fig.patch.set_alpha(0)
                                ax.patch.set_alpha(0)
                                plt.tight_layout(pad=0)
                                st.pyplot(fig, clear_figure=True, width='stretch')
                            else:
                                st.write("Not enough data for word cloud.")
                        st.write("---")
                        
                        col1, col2 = st.columns(2, gap="small")
                        
                        with col1:
                            with st.container(border=True):
                                st.write("**Recency Weighted Sentiment Trend**")
                                # Simple trend plot
                                df_trend = df_comments.copy()
                                df_trend['rolling_sent'] = df_trend['sent_val'].rolling(window=10).mean()
                                line_chart = alt.Chart(df_trend).mark_line(color="#4CAF50").encode(
                                     x=alt.X('published_at:T', title='Published At'),
                                     y=alt.Y('rolling_sent:Q', title='Rolling Sentiment')
                                     ).properties(height=300).interactive()
                                st.altair_chart(line_chart, width='stretch')

                        with col2:
                            with st.container(border=True):
                                # Sentiment distribution
                                st.write("**Sentiment Distribution**")
                                sent_counts = df_comments['sentiment'].value_counts().reset_index()
                                sent_counts.columns = ['sentiment', 'count']

                                bar_chart = alt.Chart(sent_counts).mark_bar().encode(
                                     x=alt.X('sentiment:N', title='Sentiment', axis=alt.Axis(labelAngle=0)),
                                     y=alt.Y('count:Q', title='Count'),
                                     color=alt.Color('sentiment:N',
                                                     scale=alt.Scale(domain=['Positive','Neutral','Negative'],
                                                                     range=['#2E7D32', '#FFC107', '#D32F2F']),
                                                                     legend=None)
                                                                     ).properties(height=300).interactive()
                                st.altair_chart(bar_chart, width='stretch')

                        st.write("**Weighted Comment Analysis (Top 10 Latest)**")
                        st.dataframe(df_comments[['published_at', 'text', 'likes', 'sentiment', 'weight']].head(10))
                    else:
                        st.error("No comments found for this video.")
            else:
                st.error("Invalid YouTube URL.")

with tab2:
    st.subheader("Quick Sentiment Check")
    user_input = st.text_area("Enter a specific comment to test:", placeholder="e.g., 'The animation looks dated.'")
    
    if st.button("Analyze"):
        if user_input:
            processed = clean_predict_text(user_input)
            seq = tokenizer.texts_to_sequences([processed])
            pad = pad_sequences(seq, maxlen=50, padding='post', truncating='post')
            pred = model.predict(pad)[0]
            idx = np.argmax(pred)
            conf = pred[idx] * 100
            
            labels = ["Negative 😡", "Neutral 😐", "Positive 😊"]
            
            # Display Result
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.markdown(f"### Result: **{labels[idx]}**")
                st.write(f"Confidence Score: **{conf:.2f}%**")
                
                if idx == 2: st.success(" Audience seems happy with this reveal! 🏰")
                elif idx == 0: st.error(" Potential friction detected here. ⚠️")
                else: st.info(" Audience perception is balanced/neutral. ⚖️")
            
            with col_b:
                # Probability Chart
                labels_clean = ['Negative', 'Neutral', 'Positive']
                chart_data = pd.DataFrame({
                    'Sentiment': labels_clean,
                    'Probability': pred
                })
                bar_chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X('Sentiment:N', axis=alt.Axis(labelAngle=0)),
                    y='Probability:Q',
                    color=alt.Color('Sentiment:N',
                                    scale=alt.Scale(domain=labels_clean,
                                                    range=['#D32F2F', '#FFC107', '#2E7D32']),
                                    legend=None),
                    tooltip=['Sentiment', alt.Tooltip('Probability:Q', format='.0%')]
                ).properties(height=300).interactive()

                st.altair_chart(bar_chart, width='stretch')
        else:
            st.warning("Please enter a comment to analyze.")

# Footer
st.write("---")
st.markdown(
    "<p style='font-size: 14px;'>"
    "Disney Marketing Intelligence Tool Developed by <b>Nitai Satapathy & Rakshith Kumar</b>"
    "</p>",
    unsafe_allow_html=True
)