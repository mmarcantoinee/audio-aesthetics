import streamlit as st
import soundfile as sf
import torch
import pandas as pd
import numpy as np
from audiobox_aesthetics.infer import initialize_predictor
import statsmodels.api as sm

# Load model once
predictor = initialize_predictor()

st.title("FCP x PoliMi | Valutazione Audio Spot Radio")
st.markdown("Carica un file audio (.wav o .mp3, max 60 secondi) per ottenere:")
st.markdown("1. Il punteggio del primo componente principale (PC1) che integra le dimensioni di (1) content-enjoyment, (2) content-usefulness, (3) production-complexity, (4) production-quality\n2. Una stima del ricordo pubblicitario assistito in condizioni di guida rilassata vs. impegnativa")

uploaded_file = st.file_uploader("Upload file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Read audio
    data, sr = sf.read(uploaded_file)

    # Check duration
    duration = len(data) / sr
    if duration > 60:
        st.error("Il file supera i 60 secondi. Carica un file più corto.")
    else:
        # Format for Audiobox model
        audio_tensor = torch.tensor(data, dtype=torch.float32).T
        batch_data = [{"path": audio_tensor, "sample_rate": sr, "filename": uploaded_file.name}]

        # Predict
        predictions = predictor.forward(batch_data)
        pred = predictions[0]

        ce = pred.get("CE", None)
        cu = pred.get("CU", None)
        pc = pred.get("PC", None)
        pq = pred.get("PQ", None)

        # Construct DataFrame
        df_scores = pd.DataFrame([{
            "file": uploaded_file.name,
            "content-enjoyment": ce,
            "content-usefulness": cu,
            "production-complexity": pc,
            "production-quality": pq
        }])

        # Manual PCA projection using training parameters
        means = np.array([6.27641759, 5.91250274, 6.00337571, 7.74871851])
        stds = np.array([0.51343856, 0.85501661, 1.24639062, 0.30531713])
        pc1_loadings = np.array([0.555112, 0.560898, 0.225606, 0.571267])

        X = df_scores[['content-enjoyment', 'content-usefulness', 'production-complexity', 'production-quality']].values
        X_scaled = (X - means) / stds
        pc1_val = float(X_scaled.dot(pc1_loadings.T))

        # Use regression model with fixed coefficients from training
        intercept = 0.1762
        beta_pc1 = 0.0238
        beta_demanding = -0.0694

        aided_relaxed = intercept + beta_pc1 * pc1_val + beta_demanding * 0
        aided_demanding = intercept + beta_pc1 * pc1_val + beta_demanding * 1

        # Output
        st.subheader("Risultati")
        st.markdown(f"**Indicatore integrativo (Sintesi dei 4 indicatori):** {pc1_val:.3f}")

        st.markdown("**Stima del ricordo pubblicitario assistito:**")
        st.write(f"Guida rilassata: {aided_relaxed * 100:.1f}%")
        st.write(f"Guida impegnativa: {aided_demanding * 100:.1f}%")
