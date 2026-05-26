import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.title("🏠 Indian House Price Prediction")

st.write("Enter House Details")

# Inputs
area = st.number_input(
    "Area in Square Feet",
    min_value=500,
    max_value=10000,
    value=1200
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=2
)

# NEW INPUTS
hall = st.number_input(
    "Number of Halls",
    min_value=1,
    max_value=5,
    value=1
)

kitchen = st.number_input(
    "Number of Kitchens",
    min_value=1,
    max_value=3,
    value=1
)

car_parking = st.selectbox(
    "Car Parking Available?",
    [0, 1]
)

swimming_pool = st.selectbox(
    "Swimming Pool Available?",
    [0, 1]
)

# Predict Button
if st.button("Predict Price"):

    # Model uses only trained features
    features = pd.DataFrame(
        [[area, bedrooms, car_parking, swimming_pool]],
        columns=[
            'Area',
            'No. of Bedrooms',
            'CarParking',
            'SwimmingPool'
        ]
    )

    # Prediction
    prediction = model.predict(features)

    # Convert to Crores
    price_crore = prediction[0] / 10000000

    # Show Result
    st.success(
        f"Predicted House Price: ₹{price_crore:.2f} Crore"
    )

    # Extra Details
    st.info(
        f"""
        🛋 Halls: {hall}

        🍽 Kitchens: {kitchen}

        🚗 Car Parking: {'Yes' if car_parking == 1 else 'No'}

        🏊 Swimming Pool: {'Yes' if swimming_pool == 1 else 'No'}
        """
    )