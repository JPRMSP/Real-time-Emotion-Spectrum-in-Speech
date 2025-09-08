import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import io

st.set_page_config(page_title="Speech Emotion Spectrum", page_icon="🎤", layout="wide")
st.title("🎤 Real-time Emotion Spectrum in Speech (No Datasets, No Models)")

uploaded_file = st.file_uploader("Upload a speech file (wav/mp3)", type=["wav", "mp3"])

def analyze_speech(file):
    y, sr = librosa.load(file, sr=22050)

    # Extract acoustic features
    pitch, mag = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitch[pitch > 0]) if np.any(pitch > 0) else 0
    energy = np.mean(y ** 2)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # Rule-based emotion classification
    if pitch_mean > 180 and energy > 0.02:
        emotion = "Excited / Angry"
    elif pitch_mean < 120 and energy < 0.01:
        emotion = "Sad"
    elif zcr > 0.1 and centroid > 2500:
        emotion = "Fear / Anxiety"
    else:
        emotion = "Neutral / Calm"

    return pitch_mean, energy, zcr, centroid, emotion, y, sr

if uploaded_file is not None:
    pitch, energy, zcr, centroid, emotion, y, sr = analyze_speech(uploaded_file)

    st.subheader("📊 Extracted Features")
    st.write(f"Pitch (Hz): {pitch:.2f}")
    st.write(f"Energy: {energy:.4f}")
    st.write(f"Zero Crossing Rate: {zcr:.4f}")
    st.write(f"Spectral Centroid: {centroid:.2f}")

    st.success(f"🧠 Detected Emotion: **{emotion}**")

    # Emotion spectrum chart
    st.subheader("Emotion Spectrum")
    emotions = ["Excited/Angry", "Sad", "Fear/Anxiety", "Neutral/Calm"]
    values = [
        pitch / 300,
        1 - (energy * 50),
        zcr * 5,
        centroid / 4000
    ]
    fig, ax = plt.subplots()
    ax.bar(emotions, values, color=['red','blue','orange','green'])
    ax.set_ylim(0, 1.2)
    st.pyplot(fig)

    # Waveform
    st.subheader("Waveform")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    st.pyplot(fig)

    # Spectrogram
    st.subheader("Spectrogram")
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    fig, ax = plt.subplots()
    img = librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set_title("Spectrogram")
    st.pyplot(fig)
else:
    st.info("👆 Upload a speech file to analyze emotions.")
