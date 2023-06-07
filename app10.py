# common imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# specific imports for code1
import pydeck as pdk
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# specific imports for code2
import matplotlib

# Set your Mapbox API key as an environment variable
os.environ["MAPBOX_API_KEY"] = "Use_your_own_API"

# Coordinates for each region (code 1)
region_coordinates = {
    'North': [41.6955, -8.8345],
    'Center': [40.2033, -8.4109],
    'Lisbon': [38.7223, -9.1393],
    'Alentejo': [38.6453, -7.9143],
    'Algarve': [37.0179, -7.9304],
    'Azores': [37.7412, -25.6756],
    'Madeira': [32.6669, -16.9241],
}

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

def visualize_energy_portugal():
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


def visualize_energy_demand():
    st.title("Energy Communities in Portugal")
    st.subheader("A tool for visualizing energy demand, energy supply, and solar potential")

    # Loading data from task 2 (Energy Demand)
    df = pd.read_excel("energy_demand_306 (1).xlsx")

    # Sidebar
    st.sidebar.title("Filters and User Inputs")

    # Filter by district
    district_choice = st.sidebar.multiselect("Select district", df['District_name'].unique())
    if district_choice:
        df = df[df['District_name'].isin(district_choice)]

    # Filter by Energy Demand
    energy_demand_choice = st.sidebar.multiselect("Select Energy Demand", df['energy_demand'].unique())
    if energy_demand_choice:
        df = df[df['energy_demand'].isin(energy_demand_choice)]

    # Dataframe
    st.text('Dataframe')
    st.dataframe(df)

    # Histogram of Consumption
    st.text('Histogram of Consumption')
    st.bar_chart(df['Consumo'])

    # Boxplot of Consumption
    st.text('Boxplot of Consumption')

    fig1, ax1 = plt.subplots()
    plt.boxplot(df['Consumo'])
    st.pyplot(fig=fig1)


def night_light():
    st.title("Nightlight Data Visualization")

    df = pd.read_excel('Portugal_energy (3).xlsx')

    geo = df["Geometry"]
    geo = list(geo)

    list_final = []
    for i in range(len(geo)):
        str = geo[i]
        geo[i] = geo[i].replace("[", "").replace("]", "").replace(" ", "")
        temp_lis = geo[i].split(',')
        list_final.append(temp_lis)

    list2 = []
    for i in range(len(list_final)):
        list4 = []
        for j in range(len(list_final[i])):
            if j%2 == 0:
                if j != len(list_final[i])-1:
                    list3 = [list_final[i][j], list_final[i][j+1]]
                    list4.append(list3)
        list2.append(list4)

    for i in range(len(list2)):
        for j in range(len(list2[i])):
            if list2[i][j][1] == '' or list2[i][j][0] == '':
                list2[i].pop(j)
    for i in range(len(list2)):
        for j in range(len(list2[i])):
            list2[i][j][0] = float(list2[i][j][0])
            list2[i][j][1] = float(list2[i][j][1])

    geo_list = []
    for i in range(len(list2)):
        geo_list.append(list2[i][0])


    values = df["mean_nightlight"]
    coordinates = geo_list

    x_coordinates = [coord[0] for coord in coordinates]
    y_coordinates = [coord[1] for coord in coordinates]

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    values_scaler = MinMaxScaler()

    normalized_x = x_scaler.fit_transform([[x] for x in x_coordinates])
    normalized_y = y_scaler.fit_transform([[y] for y in y_coordinates])
    normalized_values = values_scaler.fit_transform([[v] for v in values])

    fig, ax = plt.subplots(figsize=(15, 15))
    scatter = ax.scatter(normalized_x, normalized_y, c=normalized_values, cmap='viridis')
    plt.colorbar(scatter)
    ax.set_xlabel('Normalized X Coordinate')
    ax.set_ylabel('Normalized Y Coordinate')
    ax.set_title('Normalized Scatter Plot with Normalized Values')
    st.pyplot(fig)  # streamlit display plot


    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(values, bins=200, edgecolor='black')
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)  # streamlit display plot

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.boxplot(values)
    st.pyplot(fig)  # streamlit display plot

def energy_supply():
    #Reading the dataset
    df = pd.read_csv("cleaned_data.csv")
    st.dataframe(df.head(10))

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(corr, cmap="Greens",annot=True)
    st.pyplot(fig)

    df["commissioning_year"] = df['commissioning_year'].replace(np.nan, df["commissioning_year"].median())
    df["district"] = df["district"].replace(np.nan,df["district"].mode()[0])
    df["municipality"] = df["municipality"].replace(np.nan,df["municipality"].mode()[0])

    #Districts Count
    fig1  = px.bar(df["district"].value_counts(),
                x = "district",
                color = "district",
                title = "Total number of districts in Portugal")

    st.plotly_chart(fig1)

    #grouping each districts and sum of capacity in each district
    Capacity_df = df.groupby('district')['capacity_mw'].sum().reset_index()

    #barchart between capacity_mw vs district
    fig2 = px.bar(Capacity_df, x='district', y='capacity_mw',color = "district",
                labels = {"capacity_mw":"Total Energy Capacity in MegaWatt"},
                height= 600,
                title = "Total Energy Capacity In Mega Watt on Each District")

    st.plotly_chart(fig2)

    #count of Energy sources as primary fuel
    fig3 = px.histogram(df, x ="primary_fuel",
                    height = 600,
                    color = "primary_fuel",
                    title = "Total Count of Energy sources as primary fuel")

    st.plotly_chart(fig3)

    #Grouping each  primary fuels with Energy capacity
    Capacity_energy =  df.groupby('primary_fuel')['capacity_mw'].sum().sort_values(ascending = False).reset_index()

    #Barplot
    fig4 = px.bar(Capacity_energy,
                x ="primary_fuel",
                y = "capacity_mw",
                color="primary_fuel",
                title = "Energy Capacity (MW) vs primary Fuel (Energy Sources)")
    st.plotly_chart(fig4)

    #Grouping each  Municipality with Energy capacity
    Capacity_munici = df.groupby('municipality')['capacity_mw'].sum().sort_values(ascending = False).reset_index()

    # Top 10 Municipalities having Higher Energy Capacity :
    fig5 = px.bar(Capacity_munici[:10],
                y ="municipality",
                x = "capacity_mw",
                color = "municipality",
                title = "Top 10 Municipalities having higher Energy Capacity(MW)")
    st.plotly_chart(fig5)

    #  Municipalities having lower Energy Capacity :
    fig6 = px.bar(Capacity_munici[-20:],
                y ="municipality",
                x = "capacity_mw",
                color = "municipality",
                title = "Municipalities having Low Energy Capcity(MW)")
    st.plotly_chart(fig6)

    # Bar graph of Municipalities of Energy Capacity :
    fig7 = px.bar(Capacity_munici,
                x ="municipality",
                y = "capacity_mw",
                color = "municipality",
                height = 600,width = 1300,
                title = "Energy Capacity(MW) of All Municipalite")
    st.plotly_chart(fig7)

    #Grouping capcity on Commissioning_year
    Capacity_year = df.groupby('commissioning_year')['capacity_mw'].mean().sort_values(ascending = False).reset_index()

    #creating the parameter
    values = Capacity_year["capacity_mw"]
    names = Capacity_year["commissioning_year"]

    fig8 = px.pie(df, values=values[:7],
                names=names[:7],
                height=600,
                title="Top 7 commissioning_year having Higher Energy Capacity (MW)")
    fig8.update_traces(textposition="inside", textinfo="percent+label")

    st.plotly_chart(fig8)

    fig9 = px.pie(df, values=values[:15],
                names=names[:15],
                height=600,
                title="Top15_commissioning_year Vs  Energy Capacity (MW)")
    fig9.update_traces(textposition="inside", textinfo="percent+label")

    st.plotly_chart(fig9)

# driver function
def main():
    st.sidebar.title('Choose your option')
    option = st.sidebar.selectbox(
        'Which visualization do you want to see?',
        ('Energy Communities in Portugal', 'Energy Demand','Energy Supply','Nightlight visualization')
    )

    mapbox_api_key = os.getenv("MAPBOX_API_KEY")
    if mapbox_api_key is None:
        raise ValueError("Please set the MAPBOX_API_KEY environment variable")

    if option == 'Energy Communities in Portugal':
        visualize_energy_portugal()
    elif option == 'Energy Demand':
        visualize_energy_demand()
    elif option == "Energy Supply":
        energy_supply()
    else:
        night_light()

if __name__ == "__main__":
    main()
