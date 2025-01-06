import time
import tempfile
import os
from pathlib import Path
import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
import google.generativeai as genai
from google.generativeai import upload_file, get_file
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google API
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# Streamlit app configuration
st.set_page_config(
    page_title="Video Summarizer",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Video Summarizer")

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

# File uploader for video
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if video_file is not None:
    st.write("File uploaded successfully!")

if video_file:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    # Display the uploaded video
    st.video(video_path, format="video/mp4", start_time=0)

    # User query input
    user_query = st.text_area(
        "What insights are you seeking from the video?",
        placeholder="Ask anything about the video content. The AI agent will analyze and gather additional context if needed.",
        help="Provide specific questions or insights you want from the video.",
    )

    # Analyze video button
    if st.button("🔍 Analyze Video", key="analyze_video_button"):
        if not user_query:
            st.warning("Please enter a question or insight to analyze the video.")
        else:
            try:
                with st.spinner("Processing video and gathering insights..."):
                    # Upload and process video file
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    # Generate analysis prompt
                    analysis_prompt = f"""
                    You are an advanced AI assistant tasked with analyzing the content and context of the uploaded video.
                    Your goal is to provide a comprehensive summary and insights based on the video content.

                    Here is the user's query:
                    "{user_query}"

                    Please perform the following tasks:
                    1. Summarize the main points and key takeaways from the video.
                    2. Identify any important themes, topics, or concepts discussed.
                    3. Provide answers or insights related to the user's query using information from the video.
                    4. Supplement your response with relevant web research if necessary.

                    Ensure your response is detailed, user-friendly, and actionable.
                    """

                    # AI agent processing
                    response = summarizer_agent.run(analysis_prompt, videos=[processed_video])

                # Display the result
                st.subheader("Analysis Result")
                st.markdown(response.content)

            except Exception as error:
                st.error(f"An error occurred during analysis: {error}")
            finally:
                # Clean up temporary video file
                Path(video_path).unlink(missing_ok=True)
else:
    st.info("Upload a video file to begin analysis.")

# Customize text area height
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)