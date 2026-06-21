# Import Libraries
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("best_model.pkl")

# Load dataset
df = pd.read_csv("beer_servings.csv")

# Drop unwanted column
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

st.title("🍺 Beer Servings Alcohol Prediction App")

st.write("Predict Total Litres of Pure Alcohol Consumption")

# Dataset overview
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Countries", df["country"].nunique())
col2.metric("Continents", df["continent"].nunique())
col3.metric("Records", len(df))

# Infographics


st.subheader("Data Visualizations")


# Chart 1
st.write("Average Beer Servings by Continent")

beer_by_continent = df.groupby("continent")["beer_servings"].mean()

fig1, ax1 = plt.subplots()

beer_by_continent.plot(
    kind="bar",
    ax=ax1,
    color="gold"
)
ax1.set_xlabel("Continent")
ax1.set_ylabel("Average Beer Servings")

st.pyplot(fig1)





# Chart 2
st.write("Average Spirit Servings by Continent")

spirit_by_continent = df.groupby("continent")["spirit_servings"].mean()

fig2, ax2 = plt.subplots()
spirit_by_continent.plot(
    kind="bar", 
    ax=ax2,
    color="orange"
)
ax2.set_xlabel("Continent")
ax2.set_ylabel("Average Spirit Servings")

st.pyplot(fig2)





# Chart 3
st.write("Distribution of Total Pure Alcohol Consumption")

fig3, ax3 = plt.subplots()
df["total_litres_of_pure_alcohol"].hist(
    ax=ax3,
    color="limegreen"
)
ax3.set_xlabel("Total Litres of Pure Alcohol")
ax3.set_ylabel("Frequency")

st.pyplot(fig3)





#Chart 4
st.write("Average Wine Servings by Continent")

wine_by_continent = df.groupby("continent")["wine_servings"].mean()

fig4, ax4 = plt.subplots()

wine_by_continent.plot(
    kind="bar",
    ax=ax4,
    color="purple"
)

ax4.set_title("Average Wine Servings by Continent")
ax4.set_xlabel("Continent")
ax4.set_ylabel("Average Wine Servings")

st.pyplot(fig4)



st.subheader("Enter Details")

country = st.selectbox(
    "Country",
    sorted(df["country"].unique())
)

continent = st.selectbox(
    "Continent",
    sorted(df["continent"].unique())
)

beer_servings = st.number_input(
    "Beer Servings",
    min_value=0.0,
    value=50.0
)

spirit_servings = st.number_input(
    "Spirit Servings",
    min_value=0.0,
    value=50.0
)

wine_servings = st.number_input(
    "Wine Servings",
    min_value=0.0,
    value=50.0
)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "country": [country],
        "beer_servings": [beer_servings],
        "spirit_servings": [spirit_servings],
        "wine_servings": [wine_servings],
        "continent": [continent]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Total Litres of Pure Alcohol: {prediction[0]:.2f}"
    )