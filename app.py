# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import os

# Set your Mapbox API key as an environment variable
os.environ["MAPBOX_API_KEY"] = "Use_your_own_API"

# Function to load and preprocess data
def load_data():
    # Load the data
    df = pd.read_csv('energy_data.csv')

    # Handle missing values (you may want to handle these differently depending on your data)
    df.fillna(0, inplace=True)

    # Convert 'year' column to integer type
    df['year'] = df['year'].astype(int)

    # Normalize data (here we use MinMaxScaler as an example)
    scaler = MinMaxScaler()
    numerical_cols = ['energy_demand', 'energy_supply', 'solar_potential']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

# Coordinates for each region
region_coordinates = {
    'North': [41.6955, -8.8345],
    'Center': [40.2033, -8.4109],
    'Lisbon': [38.7223, -9.1393],
    'Alentejo': [38.6453, -7.9143],
    'Algarve': [37.0179, -7.9304],
    'Azores': [37.7412, -25.6756],
    'Madeira': [32.6669, -16.9241],
}

# Opportunity identification function
def identify_opportunities(df):
    # Identify regions where the solar potential is high but the energy supply is low
    opportunities = df[(df['solar_potential'] > df['solar_potential'].median()) &
                       (df['energy_supply'] < df['energy_supply'].median())]

    return opportunities

def main():
    st.title("Energy Communities in Portugal")
    st.subheader("A tool for visualizing energy demand, energy supply, and solar potential")

    # Load and preprocess the data
    data = load_data()

    # Sidebar for user inputs and filters
    st.sidebar.title("Filters and User Inputs")

    # Filter by year
    year = st.sidebar.slider("Select year", min(data['year']), max(data['year']), min(data['year']))
    data = data[data['year'] == year]

    # Filter by region
    regions = st.sidebar.multiselect("Select regions", data['region'].unique())
    if regions:
        data = data[data['region'].isin(regions)]

    # Display the preprocessed dataframe
    st.write(data)

    # Create scatter plot for energy demand vs supply
    # Create scatter plot for energy demand vs supply
    fig = px.scatter(data, x="energy_demand", y="energy_supply", color="region",
                     size='solar_potential', hover_data=['year'], title="Energy Demand vs Supply")
    st.plotly_chart(fig)

    # Plotting energy demand, energy supply, or solar potential by region
    energy_type = st.sidebar.selectbox("Select energy type", ('energy_demand', 'energy_supply', 'solar_potential'))
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='region', y=energy_type, data=data)
    ax.set_title(f'{energy_type.replace("_", " ").title()} by Region')
    st.pyplot(fig)

    # Map showing energy demand, energy supply, or solar potential across different regions
    st.subheader(f"Map showing {energy_type.replace('_', ' ')} across different regions")

    # Prepare the data for the map
    data['coordinates'] = data['region'].map(region_coordinates)
    data['latitude'] = data['coordinates'].apply(lambda x: x[0])
    data['longitude'] = data['coordinates'].apply(lambda x: x[1])
    map_data = data[['latitude', 'longitude', energy_type]]

    # Define a layer to display on a map
    layer = pdk.Layer(
        "HeatmapLayer",
        map_data,
        opacity=0.8,
        get_position=['longitude', 'latitude'],
        get_weight=energy_type,
        threshold=0.3,
        radiusPixels=50,
    )
    # Set the map's initial viewport
    view_state = pdk.ViewState(
        latitude=39.6395,
        longitude=-7.8492,
        zoom=6,
        max_zoom=15,
        pitch=40.5,
        bearing=-27.36
    )

    # Use PyDeck to create the map
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v10" # You can change the map style
    )

    # Display the map in Streamlit
    st.pydeck_chart(r)

    # Identify and visualize opportunities
    st.subheader("Opportunities for Energy Communities")
    opportunities = identify_opportunities(data)
    st.write(opportunities)

if __name__ == "__main__":
    mapbox_api_key = os.getenv("MAPBOX_API_KEY")
    if mapbox_api_key is None:
        raise ValueError("Please set the MAPBOX_API_KEY environment variable")

    main()
