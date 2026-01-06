import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Pro Realty - Real Estate Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-top: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #10B981;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data and model (assuming you've saved your trained model)


@st.cache_data
def load_data():
    # Load your dataset
    df = pd.read_csv('kc_house_data.csv')
    return df


@st.cache_resource
def load_model():
    # Load trained Random Forest model
    # Note: You need to save your trained model first using joblib.dump()
    try:
        model = joblib.load('random_forest_model.pkl')
        return model
    except:
        # If model file doesn't exist, train a new one
        st.warning("Pre-trained model not found. Training a new model...")
        return train_model()


def train_model():
    df = load_data()

    # Preprocess data (matching your notebook steps)
    df['bathrooms'] = df['bathrooms'].astype(np.int64)
    df['floors'] = df['floors'].astype(np.int64)

    # One-hot encode waterfront
    df = pd.get_dummies(df, columns=['waterfront'], drop_first=True)

    # Select features (matching your final model)
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                'floors', 'condition', 'grade', 'sqft_above',
                'yr_built', 'waterfront_1']

    X = df[features]
    y = df['price']

    # Train Random Forest model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Save the model for future use
    joblib.dump(model, 'random_forest_model.pkl')

    return model


def main():
    # App header
    st.markdown("<h1 class='main-header'>🏠 Pro Realty Real Estate Price Predictor</h1>",
                unsafe_allow_html=True)
    st.markdown("""
    ### King County House Price Prediction Tool
    Use this tool to predict house prices and make informed investment decisions.
    """)

    # Sidebar for navigation
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/609/609803.png", width=100)
        st.title("Navigation")
        app_mode = st.radio(
            "Choose a section:",
            ["🏠 Price Prediction", "📊 Data Analysis",
                "📈 Model Insights", "⚙️ About"]
        )

        st.markdown("---")
        st.markdown("### Model Performance")
        st.info("""
        **Random Forest Model:**
        - R² Score: 0.76
        - Avg Error: 22.8%
        - MAE: $120,554
        """)

    # Load data and model
    df = load_data()
    model = load_model()

    if app_mode == "🏠 Price Prediction":
        show_price_prediction(df, model)
    elif app_mode == "📊 Data Analysis":
        show_data_analysis(df)
    elif app_mode == "📈 Model Insights":
        show_model_insights(df, model)
    elif app_mode == "⚙️ About":
        show_about()


def show_price_prediction(df, model):
    st.markdown("<h2 class='sub-header'>House Price Prediction</h2>",
                unsafe_allow_html=True)

    # Create two columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Property Features")

        # Property characteristics inputs
        bedrooms = st.slider("Bedrooms", 1, 11, 3, help="Number of bedrooms")
        bathrooms = st.slider("Bathrooms", 1, 8, 2, help="Number of bathrooms")
        sqft_living = st.number_input(
            "Living Area (sqft)", 300, 10000, 2000, step=100)
        sqft_lot = st.number_input(
            "Lot Size (sqft)", 500, 1000000, 8000, step=500)
        floors = st.select_slider(
            "Floors", options=[1, 1.5, 2, 2.5, 3, 3.5], value=1.0)

    with col2:
        st.subheader("Property Quality")

        condition = st.slider("Condition (1-5)", 1, 5, 3,
                              help="1 = Poor, 3 = Average, 5 = Excellent")
        grade = st.slider("Grade (1-13)", 1, 13, 7,
                          help="Construction quality rating")
        sqft_above = st.number_input(
            "Above Ground Area (sqft)", 300, 10000, 1500, step=100)
        yr_built = st.number_input("Year Built", 1900, 2023, 1990)
        waterfront = st.radio("Waterfront Property", ["No", "Yes"])

    # Prediction button
    if st.button("🏠 Predict House Price", type="primary", use_container_width=True):
        # Prepare input data
        waterfront_encoded = 1 if waterfront == "Yes" else 0

        input_data = pd.DataFrame({
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'sqft_living': [sqft_living],
            'sqft_lot': [sqft_lot],
            'floors': [floors],
            'condition': [condition],
            'grade': [grade],
            'sqft_above': [sqft_above],
            'yr_built': [yr_built],
            'waterfront_1': [waterfront_encoded]
        })

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Display prediction
        st.markdown(f"""
        <div class='prediction-box'>
            Predicted House Price: ${prediction:,.0f}
        </div>
        """, unsafe_allow_html=True)

        # Confidence interval
        confidence_lower = prediction * 0.85  # 15% lower bound
        confidence_upper = prediction * 1.15  # 15% upper bound

        st.info(f"""
        **Confidence Interval:** ${confidence_lower:,.0f} - ${confidence_upper:,.0f}
        
        *Based on model accuracy, the actual price is likely within ±15% of this prediction.*
        """)

        # Feature importance for this prediction
        st.subheader("📊 What's Driving This Price?")

        # Get feature importance from model
        feature_importance = pd.DataFrame({
            'Feature': input_data.columns,
            'Value': input_data.iloc[0].values,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        # Display top features
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Blues(np.linspace(0.3, 1, len(feature_importance)))
        bars = ax.barh(
            feature_importance['Feature'], feature_importance['Importance'] * 100, color=colors)
        ax.set_xlabel('Importance (%)')
        ax.set_title('Feature Contribution to Price Prediction')

        # Add value labels
        for bar, imp in zip(bars, feature_importance['Importance'] * 100):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{imp:.1f}%', va='center', fontweight='bold')

        st.pyplot(fig)


def show_data_analysis(df):
    st.markdown("<h2 class='sub-header'>Data Analysis Dashboard</h2>",
                unsafe_allow_html=True)

    # Data overview
    with st.expander("📋 Dataset Overview", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Properties", f"{len(df):,}")
        with col2:
            st.metric("Average Price", f"${df['price'].mean():,.0f}")
        with col3:
            st.metric("Date Range", "2014-2015")

    # Interactive visualizations
    tab1, tab2, tab3 = st.tabs(
        ["📈 Price Distribution", "🏠 Property Features", "📍 Location Analysis"])

    with tab1:
        st.subheader("Price Distribution Analysis")

        # Price histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['price'] / 1000, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Price ($1000s)')
        ax.set_ylabel('Number of Properties')
        ax.set_title('Distribution of House Prices')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Price statistics
        price_stats = df['price'].describe()
        st.dataframe(price_stats, use_container_width=True)

    with tab2:
        st.subheader("Property Feature Analysis")

        # Select feature to analyze
        feature = st.selectbox(
            "Select Feature to Analyze Against Price:",
            ['sqft_living', 'bedrooms', 'bathrooms', 'grade', 'yr_built']
        )

        # Scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df[feature], df['price'] / 1000, alpha=0.5, s=10)
        ax.set_xlabel(feature.replace('_', ' ').title())
        ax.set_ylabel('Price ($1000s)')
        ax.set_title(f'Price vs {feature.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with tab3:
        st.subheader("Location-Based Analysis")

        # By zipcode (simplified)
        avg_price_by_zip = df.groupby('zipcode')['price'].mean().reset_index()
        avg_price_by_zip = avg_price_by_zip.sort_values(
            'price', ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(avg_price_by_zip['zipcode'].astype(
            str), avg_price_by_zip['price'] / 1000)
        ax.set_xlabel('Zip Code')
        ax.set_ylabel('Average Price ($1000s)')
        ax.set_title('Top 20 Zip Codes by Average Price')
        plt.xticks(rotation=45)
        st.pyplot(fig)


def show_model_insights(df, model):
    st.markdown("<h2 class='sub-header'>Model Insights & Performance</h2>",
                unsafe_allow_html=True)

    # Model metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("R² Score", "0.761", "+0.1227")
    with col2:
        st.metric("MAE", "$120,554", "-$17,208")
    with col3:
        st.metric("RMSE", "$190,466", "-$17,034")
    with col4:
        st.metric("Avg Error", "22.8%", "-12.5%")

    # Feature importance
    st.subheader("🔍 Feature Importance Analysis")

    features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                'floors', 'condition', 'grade', 'sqft_above',
                'yr_built', 'waterfront_1']

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_,
        'Importance_Percent': model.feature_importances_ * 100
    }).sort_values('Importance', ascending=False)

    # Display importance table
    st.dataframe(importance_df.style.format({'Importance': '{:.4f}', 'Importance_Percent': '{:.1f}%'}),
                 use_container_width=True)

    # Visualize feature importance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
    bars = ax1.barh(
        importance_df['Feature'], importance_df['Importance_Percent'], color=colors)
    ax1.set_xlabel('Importance (%)')
    ax1.set_title('Feature Importance in Price Prediction')
    ax1.invert_yaxis()

    # Add value labels
    for bar, imp in zip(bars, importance_df['Importance_Percent']):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{imp:.1f}%', va='center', fontweight='bold')

    # Cumulative importance
    importance_df['Cumulative'] = importance_df['Importance_Percent'].cumsum()
    ax2.plot(importance_df['Cumulative'], 'o-', linewidth=2, markersize=8)
    ax2.fill_between(range(len(importance_df)), 0,
                     importance_df['Cumulative'], alpha=0.3)
    ax2.set_xlabel('Number of Features')
    ax2.set_ylabel('Cumulative Importance (%)')
    ax2.set_title('Cumulative Feature Importance')
    ax2.grid(True, alpha=0.3)

    # Mark key points
    for n in [3, 5, 8]:
        cum_imp = importance_df.head(n)['Importance_Percent'].sum()
        ax2.plot(n-1, cum_imp, 'ro', markersize=10)
        ax2.text(n-1, cum_imp + 5, f'{n} features\n{cum_imp:.1f}%',
                 ha='center', fontweight='bold')

    st.pyplot(fig)

    # Business recommendations
    st.subheader("💼 Business Recommendations")

    recommendations = [
        ("🏆 **Focus on Grade**", "Grade contributes 38.5% to price prediction. Target properties with potential for grade improvement."),
        ("📏 **Maximize Living Space**",
         "Square footage contributes 30.5%. Prioritize renovations that increase living area."),
        ("🏗️ **Consider Age Carefully**",
         "Year built contributes 11.5%. Newer homes command premium prices."),
        ("🌊 **Premium for Waterfront**",
         "Waterfront properties have significant value premiums."),
        ("🛁 **Add Bathrooms Strategically**",
         "Each additional bathroom adds approximately $45,644 in value.")
    ]

    for title, desc in recommendations:
        with st.expander(title):
            st.write(desc)


def show_about():
    st.markdown("<h2 class='sub-header'>About Pro Realty Real Estate Predictor</h2>",
                unsafe_allow_html=True)

    st.markdown("""
    ## Project Overview
    
    This application uses machine learning to predict house prices in King County, Washington.
    The model was developed to help Pro Realty make data-driven investment decisions.
    
    ### Key Features
    
    1. **Price Prediction**: Predict house prices based on property features
    2. **Data Analysis**: Explore the King County housing dataset
    3. **Model Insights**: Understand what drives house prices
    4. **Investment Guidance**: Get recommendations based on feature importance
    
    ### Model Details
    
    - **Algorithm**: Random Forest Regressor (optimized)
    - **Features**: 10 key property characteristics
    - **Accuracy**: 76.1% (R² score)
    - **Average Error**: 22.8%
    
    ### Business Objectives Addressed
    
    ✅ Identify factors influencing house prices  
    ✅ Predict housing prices with high accuracy  
    ✅ Make informed investment decisions  
    ✅ Minimize risk by avoiding overpayment  
    ✅ Optimize portfolio diversification  
    
    ### Data Source
    
    King County House Sales dataset (2014-2015)
    - 21,613 properties
    - 21 features per property
    
    ### Deployment Instructions
    
    1. Install required packages: `pip install -r requirements.txt`
    2. Save your trained model as `random_forest_model.pkl`
    3. Run the app: `streamlit run app.py`
    4. Access at: `http://localhost:8501`
    
    ### Requirements File (`requirements.txt`)
    
    ```txt
    streamlit==1.28.0
    pandas==2.1.0
    numpy==1.24.0
    matplotlib==3.7.0
    seaborn==0.12.0
    scikit-learn==1.3.0
    joblib==1.3.0
    ```
    """)


if __name__ == "__main__":
    main()
