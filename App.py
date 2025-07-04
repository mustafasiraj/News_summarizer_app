import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

st.set_page_config(page_title="📰 News Summarizer", page_icon="📝")

st.title("📰 News Article Summarizer")
st.markdown("Paste a **news article URL** below and get a summary of its content.")

# User input for URL
url = st.text_input("🔗 Enter News Article URL:")

if url:
    try:
        # Step 1: Get the article content
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            # Step 2: Extract the text using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join([para.get_text() for para in paragraphs])

            if len(text.strip()) == 0:
                st.error("❌ No article text found. Try a different URL.")
            else:
                # Step 3: Load summarizer (runs only once)
                @st.cache_resource
                def load_summarizer():
                    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", framework="pt")
                
                summarizer = load_summarizer()

                # Step 4: Truncate if needed
                text = text[:1024]

                # Step 5: Generate summary
                with st.spinner("Summarizing..."):
                    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)

                st.subheader("📝 Summary:")
                st.success(summary[0]['summary_text'])

        else:
            st.error(f"⚠️ Failed to fetch article. HTTP Status Code: {response.status_code}")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
