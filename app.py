import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Crop Yield Prediction System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling: larger, clearer title and an attractive, responsive layout.
st.markdown("""
<style>
    /* Use more of the available browser width */
    .block-container {
        max-width: 1600px;
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* Keep the complete project title visible */
    .hero {
        background: linear-gradient(120deg, #123d2a 0%, #167347 58%, #70a957 100%);
        padding: 1.65rem 2rem;
        border-radius: 18px;
        color: #ffffff;
        margin-bottom: 1.25rem;
        box-shadow: 0 8px 24px rgba(18, 61, 42, 0.16);
    }
    .hero-title {
        font-size: clamp(2rem, 3vw, 3.15rem);
        line-height: 1.2;
        font-weight: 850;
        letter-spacing: -0.8px;
        color: #ffffff;
        margin: 0;
        overflow-wrap: anywhere;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        line-height: 1.6;
        margin-top: 0.6rem;
        color: #e8f5e9;
    }

    h2, h3 {font-weight: 750 !important;}
    [data-testid="stSidebar"] {
        background: #f1f6f2;
    }
    [data-testid="stSidebar"] h2 {
        color: #174b32;
        font-weight: 800;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    div.stButton > button {
        width: 100%;
        min-height: 3.2rem;
        background: linear-gradient(90deg, #16834b, #20a05d);
        color: white;
        border: 0;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        font-size: 1.05rem;
        font-weight: 750;
        box-shadow: 0 4px 12px rgba(22, 131, 75, 0.18);
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #11683b, #16834b);
        color: white;
        border: 0;
    }

    .section-heading {
        font-size: 1.45rem;
        font-weight: 800;
        color: #193b2b;
        margin: 0.3rem 0 0.35rem 0;
    }
    .section-caption {
        color: #64748b;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    .result-card {
        background: linear-gradient(135deg, #e8f8ed 0%, #f8fff9 100%);
        border: 2px solid #9bd8ad;
        border-radius: 18px;
        padding: 1.7rem;
        min-height: 190px;
        box-shadow: 0 8px 22px rgba(21, 128, 61, 0.08);
    }
    .result-label {
        color: #22643d;
        font-size: 1.05rem;
        font-weight: 750;
    }
    .result-value {
        color: #14532d;
        font-size: clamp(2.3rem, 3vw, 3.3rem);
        line-height: 1.2;
        font-weight: 850;
        margin: 0.55rem 0;
        overflow-wrap: anywhere;
    }
    .result-unit {
        color: #276749;
        font-size: 1.05rem;
        font-weight: 750;
    }

    div[data-testid="stMetric"] {
        background: #f7faf8;
        border: 1px solid #e0ebe3;
        padding: 1rem;
        border-radius: 12px;
        min-height: 105px;
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 650;
    }
    .note {
        color: #64748b;
        font-size: 0.9rem;
        line-height: 1.55;
    }
    hr {margin: 1.7rem 0;}
    @media (max-width: 800px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .hero {padding: 1.25rem;}
    }
</style>
""", unsafe_allow_html=True)

DATA_FILE = "yield_df.csv"
MODEL_FILE = "best_model.pkl"

RENAME_COLUMNS = {
    "Area": "Country",
    "Item": "Crop",
    "hg/ha_yield": "Yield",
    "average_rain_fall_mm_per_year": "Rainfall",
    "pesticides_tonnes": "Pesticides",
    "avg_temp": "Temperature"
}

NUMERIC_FEATURES = ["Year", "Rainfall", "Pesticides", "Temperature"]
CATEGORICAL_FEATURES = ["Country", "Crop"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "Yield"


@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path).rename(columns=RENAME_COLUMNS)
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Dataset columns missing: " + ", ".join(missing) +
            ". Use the original crop yield dataset with Area, Item, Year, "
            "hg/ha_yield, average_rain_fall_mm_per_year, pesticides_tonnes and avg_temp."
        )

    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURES + [TARGET]).drop_duplicates()
    df["Country"] = df["Country"].astype(str)
    df["Crop"] = df["Crop"].astype(str)
    if df.empty:
        raise ValueError("No valid records were found in the dataset.")
    return df


@st.cache_resource
def train_best_model(data):
    X, y = data[FEATURES], data[TARGET]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01, max_iter=10000),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=20),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, random_state=42, n_jobs=-1
        )
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rows, fitted = [], {}

    for name, regressor in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", regressor)
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        rows.append({
            "Model": name,
            "MAE": float(mean_absolute_error(y_test, predictions)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "R² Score": float(r2_score(y_test, predictions))
        })
        fitted[name] = pipeline

    comparison = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best_name = comparison.loc[0, "Model"]
    best_model = fitted[best_name]
    joblib.dump(best_model, MODEL_FILE)
    return best_model, best_name, comparison


# -------------------- HEADER --------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">🌾 Crop Yield Prediction System</div>
    <div class="hero-subtitle">
        Smart agricultural yield estimation using historical crop data,
        rainfall, pesticide usage, temperature and location.
    </div>
</div>
""", unsafe_allow_html=True)

try:
    data = load_data(DATA_FILE)
except FileNotFoundError:
    st.error("Dataset not found. Place **yield_df.csv** in the same folder as **app.py**.")
    st.stop()
except Exception as exc:
    st.error(f"Unable to read the dataset: {exc}")
    st.stop()

with st.spinner("Preparing the crop yield prediction model..."):
    try:
        best_model, best_model_name, comparison = train_best_model(data)
    except Exception as exc:
        st.error(f"Model training failed: {exc}")
        st.stop()

countries = sorted(data["Country"].unique().tolist())
crops = sorted(data["Crop"].unique().tolist())

# -------------------- SIDEBAR INPUTS --------------------
with st.sidebar:
    st.markdown("## 🌱 Crop Conditions")
    st.caption("Enter your growing conditions below.")

    country = st.selectbox(
        "Country / Area", countries,
        index=countries.index("India") if "India" in countries else 0
    )
    crop = st.selectbox(
        "Crop Type", crops,
        index=crops.index("Maize") if "Maize" in crops else 0
    )

    year_min, year_max = int(data["Year"].min()), int(data["Year"].max())
    year_default = min(max(2013, year_min), year_max)
    year = st.slider("Year", year_min, year_max, year_default)

    rainfall_median = float(data["Rainfall"].median())
    rainfall_max = max(float(data["Rainfall"].max()) * 2, 1.0)
    rainfall = st.number_input(
        "Average Rainfall (mm/year)",
        min_value=0.0, max_value=rainfall_max,
        value=min(max(rainfall_median, 0.0), rainfall_max),
        step=10.0
    )

    pesticide_median = max(float(data["Pesticides"].median()), 0.0)
    pesticide_max = max(float(data["Pesticides"].max()) * 2, 1.0)
    pesticides = st.number_input(
        "Pesticides Used (tonnes)",
        min_value=0.0, max_value=pesticide_max,
        value=min(pesticide_median, pesticide_max),
        step=10.0
    )

    temp_min = min(-20.0, float(data["Temperature"].min()) - 5)
    temp_max = max(60.0, float(data["Temperature"].max()) + 5)
    temperature = st.number_input(
        "Average Temperature (°C)",
        min_value=temp_min, max_value=temp_max,
        value=float(data["Temperature"].median()),
        step=0.5
    )

    st.markdown("---")
    predict_clicked = st.button("🌾 Predict Crop Yield", use_container_width=True)

# -------------------- MAIN PREDICTION AREA --------------------
st.markdown('<div class="section-heading">Crop Yield Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Adjust the crop conditions in the left panel, then predict '
    'the estimated yield. Your result will appear clearly below.</div>',
    unsafe_allow_html=True
)

condition_col, result_col = st.columns([1.1, 1], gap="large")

with condition_col:
    st.markdown("### 📋 Selected Conditions")
    a, b = st.columns(2)
    a.metric("Country / Area", country)
    b.metric("Crop Type", crop)
    a.metric("Year", str(year))
    b.metric("Rainfall", f"{rainfall:,.1f} mm/year")
    a.metric("Pesticide Usage", f"{pesticides:,.1f} tonnes")
    b.metric("Temperature", f"{temperature:,.1f} °C")

with result_col:
    st.markdown("### 🎯 Prediction Result")
    if predict_clicked:
        input_df = pd.DataFrame([{
            "Country": country,
            "Crop": crop,
            "Year": year,
            "Rainfall": rainfall,
            "Pesticides": pesticides,
            "Temperature": temperature
        }], columns=FEATURES)

        try:
            predicted_yield = float(best_model.predict(input_df)[0])
            if not np.isfinite(predicted_yield):
                raise ValueError("The model returned an invalid result.")
            predicted_yield = max(0.0, predicted_yield)
            st.markdown(
                f'<div class="result-card">'
                f'<div class="result-label">ESTIMATED CROP YIELD</div>'
                f'<div class="result-value">{predicted_yield:,.2f}</div>'
                f'<div class="result-unit">hg/ha · hectograms per hectare</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.success(f"Prediction generated with {best_model_name}.")
            st.caption("This is an estimate from historical data, not a guaranteed actual yield.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    else:
        st.info("Enter your crop conditions and click **Predict Crop Yield**.")
        st.markdown(
            f'<div class="result-card">'
            f'<div class="result-label">YOUR RESULT WILL APPEAR HERE</div>'
            f'<div class="result-value">🌱 Ready</div>'
            f'<div class="result-unit">Selected model: {best_model_name}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

st.markdown("---")
st.markdown(
    f"**Model selected:** {best_model_name}  ·  "
    "Automatically chosen by comparing regression models using test-set RMSE."
)
st.markdown(
    '<div class="note">The prediction target uses the dataset unit hg/ha. '
    'The trained model includes its preprocessing steps, so no separate scaler or encoder file is needed.</div>',
    unsafe_allow_html=True
)
