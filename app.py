import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from model.predict import inference

st.set_page_config(page_title="Facial Emotion Detection", layout="centered")
st.title("Facial Emotion Detection(Real-Time)")
st.write("Allow camera access below. The app will detect your face live and predict your emotion in real time.")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class EmotionTransformer:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_img, _ = inference(img)
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

webrtc_streamer(
    key="emotion-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=EmotionTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.info("If the camera doesn't start, make sure you're using HTTPS (or localhost) and that camera permissions are allowed in your browser.")