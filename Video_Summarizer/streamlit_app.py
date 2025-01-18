import streamlit as st


import time
import tempfile
import os
from pathlib import Path
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
import google.generativeai as genai
from google.generativeai import upload_file, get_file
from dotenv import load_dotenv

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="AI Video Summarizer",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load environment variables
load_dotenv()

# Configure Google API
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

hide_github_icon = """
#GithubIcon {
  visibility: hidden;
}
"""

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    .stTextArea textarea {
        height: 100px;
        font-size: 16px;
        border-radius: 10px;
    }
    .stButton button {
        width: 100%;
        padding: 0.75rem;
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #FF2E2E;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stFileUploader"] {
        width: 100%;
        padding: 2rem;
        border: 2px dashed #FF4B4B;
        border-radius: 10px;
        background-color: rgba(255, 75, 75, 0.05);
    }
    .title-container {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8080 100%);
        border-radius: 15px;
        color: white;
    }
    .result-container {
        padding: 2rem;
        background-color: white;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

# Title section with gradient background
st.markdown("""
    <div class="title-container">
        <h1>🎥 AI Video Summarizer</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Upload your video and get AI-powered insights instantly</p>
    </div>
""", unsafe_allow_html=True)

st.markdown(hide_github_icon, unsafe_allow_html=True)

# Initialize AI agent
@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

summarizer_agent = initialize_agent()

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    # File uploader with instructions
    st.markdown("### 📤 Upload Your Video")
    st.markdown("Supported formats: MP4, AVI, MOV")
    video_file = st.file_uploader("Upload your video file", type=["mp4", "avi", "mov"])

    if video_file:
        # Save uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name

        # Display the uploaded video
        st.video(video_path, format="video/mp4", start_time=0)

with col2:
    # Query section
    st.markdown("### 🔍 Ask About Your Video")
    user_query = st.text_area(
        "Enter your question",
        placeholder="What would you like to know about the video? Ask any question...",
        help="Be specific with your questions for better results"
    )

    # Analysis button with loading state
    analyze_button = st.button("🚀 Analyze Video", use_container_width=True)

# Process and display results
if video_file and analyze_button:
    if not user_query:
        st.error("⚠️ Please enter a question about the video before analyzing.")
    else:
        try:
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Update progress
            status_text.text("📥 Uploading video...")
            progress_bar.progress(25)
            
            # Upload and process video file
            processed_video = upload_file(video_path)
            status_text.text("🔄 Processing video...")
            progress_bar.progress(50)
            
            while processed_video.state.name == "PROCESSING":
                time.sleep(1)
                processed_video = get_file(processed_video.name)
            
            status_text.text("🤖 Analyzing content...")
            progress_bar.progress(75)

            # Generate analysis prompt
            analysis_prompt = f"""
            Analyze this video and provide insights based on this query: "{user_query}"

            Please provide:
            1. Summary of key points
            2. Direct answer to the query
            3. Additional insights
            4. Related context from web research (if needed)

            Format the response in a clear, organized way using markdown.
            """

            # AI agent processing
            response = summarizer_agent.run(analysis_prompt, videos=[processed_video])
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Display results in a styled container
            st.markdown("""
                <div class="result-container">
                    <h2>🎯 Analysis Results</h2>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(response.content)

        except Exception as error:
            st.error(f"❌ An error occurred: {str(error)}")
        finally:
            # Clean up
            Path(video_path).unlink(missing_ok=True)
            # Clear progress indicators
            if 'progress_bar' in locals():
                progress_bar.empty()
            if 'status_text' in locals():
                status_text.empty()

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding: 1rem; color: #666;">
        <p>Made by Priyankesh, <a href="https://github.com/priyankeshh" target="_blank">Github</a></p>
    </div>
""", unsafe_allow_html=True)
