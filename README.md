# Titanic Interactive ML Web App

A beginner-friendly Streamlit app that uses a Decision Tree Classifier to estimate Titanic passenger survival.

## Features
- Interactive passenger inputs (sliders and dropdowns)
- Model training and saving with `joblib`
- Cached model loading with Streamlit
- Live prediction and estimated survival probability
- Preprocessing pipeline for missing values and categorical features

## Run locally
1. Install Python 3.10 or newer.
2. Open this folder in VS Code or a terminal.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

The first run downloads the public Titanic CSV dataset, trains the model, and saves `titanic_model.joblib`. Later runs load the saved model.

## Deploy free with Streamlit Community Cloud
1. Upload `app.py`, `train_model.py`, `requirements.txt`, and `README.md` to a public GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
4. Deploy. The first startup may take a little longer because the model is trained and saved.
5. Copy the public app URL.

## GitHub repository contents
- `app.py` — Streamlit user interface and prediction
- `train_model.py` — data loading, preprocessing, training, and joblib saving
- `requirements.txt` — pinned package versions
- `README.md` — project documentation

## Important
- The app uses the same feature names and order during training and prediction.
- No personal information is collected.
- This is a learning demonstration; the model is not intended for real-world decisions.
