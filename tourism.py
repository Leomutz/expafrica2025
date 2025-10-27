import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset: https://www.kaggle.com/datasets/ziya07/tourism-resource-management-dataset
# credit on coding to: https://www.youtube.com/watch?v=GMHHD4autv8
pd.set_option('display.max_columns', None)
# sns.set(style='whitegrid')

st.logo("logo.jpeg", link="https://expafrica2025-puqhfztukav7pu4gvwpkcq.streamlit.app/")
#Define page functions
def intro():
    # st.title("Welcome to the Africa Tourism Dashboard")
    st.markdown("# Discover the Future of African Tourism.\n## Insights, Innovation, and Impact in One Place")
    # st.title("Discover the Future of African Tourism.\nInsights, Innovation, and Impact in One Place")
    st.write("""
        Welcome! Use the dropdown to explore other pages.
    """)

df = pd.read_csv('tourism_resource_dataset.csv')
#change the time stamp to 'datetime' data type
df['timestamp'] = pd.to_datetime(df['timestamp'])
# drop any row with missing value
df_cleaned = df.dropna()
#environment attributes
env_fetures = df_cleaned[['temperature','air_quality_index','noise_level',]]
# environment and number of visitors
temperature_bins = [0,15,20,25,30,35,np.inf]
temperature_labels = ['<15','15-20','20-25','25-30','30-35','>35']
df_cleaned['temperature_range'] = pd.cut(df_cleaned['temperature'],bins=temperature_bins,labels=temperature_labels)

air_quality_bins = [0,50,100,150,np.inf]
air_quality_labels = ['0-50','50-100','100-150','>150']
df_cleaned['air_quality_range'] = pd.cut(df_cleaned['air_quality_index'],bins=air_quality_bins,labels=air_quality_labels)

noise_level_bins = [0,40,60,80,100,np.inf]
noise_level_labels = ['0-40','40-60','60-80','80-100','>100']
df_cleaned['noise_level_range'] = pd.cut(df_cleaned['noise_level'],bins=noise_level_bins,labels=noise_level_labels)

temperature_avg = df_cleaned.groupby('temperature_range')['visitor_count'].mean()
air_quality_avg = df_cleaned.groupby('air_quality_range')['visitor_count'].mean()
noise_level_avg = df_cleaned.groupby('noise_level_range')['visitor_count'].mean()


def environmental_page():
    st.title("Plotting ")
    st.write("The Environmental Features Visualization")
    # plot the environment features distributions

    fig, ax = plt.subplots(1, 3, figsize=(15,5))
  
    sns.histplot(df_cleaned['temperature'],kde=True,color='red',ax=ax[0])
    ax[0].set_title('Temperature Distribution')

    sns.histplot(df_cleaned['air_quality_index'],kde=True,color='salmon',ax=ax[1])
    ax[1].set_title('Air Quality Index Distribution')

    sns.histplot(df_cleaned['noise_level'],kde=True,color='green',ax=ax[2])
    ax[2].set_title('Noise Level Distribution')

    st.pyplot(fig)

def weather_visitors():
    st.title("Weather and Visitors")
    st.write("Inforation about Visitors and Weather")    
    fig, ax = plt.subplots(1, 3, figsize=(15,5))

    temperature_avg.plot(kind='bar',color='red',ax=ax[0])
    ax[0].set_title('Average Visitor Count by Temperature Range')
    ax[0].set_ylabel('Average Visitor Count')

    air_quality_avg.plot(kind='bar',color='salmon',ax=ax[1])
    ax[1].set_title('Average Visitor Count by Air Quality Range')
    ax[1].set_ylabel('Average Visitor Count')

    noise_level_avg.plot(kind='bar',color='skyblue',ax=ax[2])
    ax[2].set_title('Average Visitor Count by Noise Level Range')
    ax[2].set_ylabel('Average Visitor Count')

    plt.tight_layout()
    st.pyplot(fig)

def plotting_page():
    st.title("Plotting Page")
    st.write("Tourism Visualization Information.")

    fig, ax = plt.subplots(figsize=(15,5))
    sns.boxplot(x='season',y='visitor_count',data=df, ax=ax)
    ax.set_title('Visitor Count by Season')
    ax.set_xlabel('Season')
    ax.set_ylabel('Visitor Count')
    st.pyplot(fig)

    # # Example from random data
    # chart_data = pd.DataFrame(np.random.randn(50,3),columns=['a','b','c'])
    # st.line_chart(chart_data)

def mapping_page():
    st.title("Mapping Page")
    st.write("Tourism Map Information.")

    #Generate some random geospatial data 
    # map_data = pd.DataFrame(
    #     np.random.randn(1000,2)/[50,50]+[37.76,-122.4],
    #     columns=['lat','lon']
    # )
    df = pd.read_csv("SA-Tourist_attractions.csv")
    map_data = df[['lat','lon']]
    st.map(map_data)

def attraction_page():
    st.title("Attraction Page")
    st.write("Tourism attraction information.")

    #Display a simple data frame 
    #Dataset: https://rentechdigital.com/smartscraper/business-report-details/list-of-tourist-attractions-in-south-africa
    df = pd.read_csv("SA-Tourist_attractions.csv")
    # df_cleaned = df.dropna()
    st.dataframe(df)

#Dictionary to map page names to functions
page_names_to_funcs = {
    "_": intro,
    "Plotting Page": plotting_page,
    "Mapping Page": mapping_page,
    "Attraction Page": attraction_page,
    "Environmental Page": environmental_page,
    "Weather Visitor Page": weather_visitors,
}

#Sidebar for navigating pages
selected_page = st.sidebar.selectbox("Choose a page",
                                     options=page_names_to_funcs.keys())

#Run the function associated with the selected page
page_names_to_funcs[selected_page]()
