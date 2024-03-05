

import time
import pandas as pd
import streamlit as st
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
#from streamlit_option_menu import option_menu
#from streamlit_card import card
# import plotly.express as px
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.preprocessing import LabelEncoder
st.set_option('deprecation.showPyplotGlobalUse', False)
# Set page title and favicon
st.set_page_config(
    page_title="Road accident",
    page_icon="🚗",
    layout="wide"
)

# dashboard title
st.markdown("<h2 style='text-align: center'>ROAD ACCIDENT</h2><hr style='height:2px;border-width:0;color:gray;background-color:gray'>", unsafe_allow_html=True)
df=pd.read_csv("RTA _Dataset.csv")

df1=df.drop(['Accident_severity'],axis=1)
selected = option_menu(
    menu_title=None,
    options=["Dataset Description","Assumptions","Exploratory data analysis","Model"],
    icons=["file-earmark-excel-fill","asterisk","pie-chart-fill","modem"],
    default_index=0,
    orientation="horizontal"
)


if selected == "Dataset Description":
    
        st.header('Content')
        st.markdown("""
This data set is collected from Addis Ababa Sub city police departments for Masters research work.
The data set has been prepared from manual records of road traffic accident of the year 2017-20. All the sensitive information have been excluded during data encoding and finally it has 32 features and 12316 instances of the accident.""")
        table_data = [
        ["Time:","The time of day when the accident occurred"],
       ["Day_of_week:"," The day of the week when the accident occurred."]
       ,["Age_band_of_driver:"," The age group or band of the driver involved in the accident."]
,["Sex_of_driver:"," The gender of the driver involved in the accident."]
,["Educational_level:"," The educational level of the driver involved in the accident."]
,["Vehicle_driver_relation:"," The relationship of the driver to the vehicle (e.g., owner, renter)."]
,["Driving_experience:"," The driver's level of driving experience."]
,["Type_of_vehicle:"," The type of vehicle involved in the accident."]
,["Owner_of_vehicle: ","The ownership status of the vehicle involved in the accident."]
,["Service_year_of_vehicle:"," The number of years the vehicle has been in service."]
,["Defect_of_vehicle:"," Any defects reported in the vehicle involved in the accident."]
,["Area_accident_occured:"," The area or location where the accident occurred."]
,["Lanes_or_Medians: ","The number of lanes or medians on the road where the accident occurred."]
,["Road_allignment:"," The alignment of the road where the accident occurred."]
,["Types_of_Junction: ","The type of junction or intersection where the accident occurred."]
,["Road_surface_type:"," The type of road surface where the accident occurred."]
,["Road_surface_conditions:"," The condition of the road surface where the accident occurred."]
,["Light_conditions:"," The lighting conditions at the time of the accident."]
,["Weather_conditions:"," The weather conditions at the time of the accident."]
,["Type_of_collision:"," The type of collision that occurred."]
,["Number_of_vehicles_involved:"," The number of vehicles involved in the accident."]
,["Number_of_casualties:"," The number of casualties (injuries or deaths) in the accident."]
,["Vehicle_movement:"," The movement of the vehicle at the time of the accident."]
,["Casualty_class:"," The classification of the casualty (e.g., driver, passenger, pedestrian)."]
,["Sex_of_casualty:"," The gender of the casualty."]
,["Age_band_of_casualty:"," The age group or band of the casualty."]
,["Casualty_severity: ","The severity of the casualty (e.g., fatal, serious, slight)."]
,["Work_of_casuality:"," The work status of the casualty (e.g., employed, unemployed)."]
,["Fitness_of_casuality:"," The fitness status of the casualty."]
,["Pedestrian_movement: ","The movement of any pedestrians involved in the accident."]
,["Cause_of_accident:","The primary cause of the accident."],
["Accident_severity:"," The severity of the accident."]
    ]
        st.header('Dataset')
        st.table(table_data)

if selected == "Assumptions":
    st.markdown("<h5> #Assumption 1</h5>", unsafe_allow_html=True)
    st.write("Age band of driver plays an important role is determining the severity as above 50 can have more recovering time.")
    st.divider()
    st.markdown("<h5> #Assumption 2</h5>", unsafe_allow_html=True)
    st.write("Sex of driver can contribute on determining which gender are mostly have serverity of accident.")
    st.divider()
    st.markdown("<h5> #Assumption 3</h5>", unsafe_allow_html=True)
    st.write("Driving experience plays major role where experienced drivers will be carefully driving than newly experiencing drivers.")
    st.divider()
    st.markdown("<h5> #Assumption 4</h5>", unsafe_allow_html=True)
    st.write("Type of vehicle based on the type of vehicle like large sized or small sized contributes to severity of accidency.")
    st.divider()
    st.markdown("<h5> #Assumption 5</h5>", unsafe_allow_html=True)
    st.write("service_year_of_vehiclecand be consider where the condition of the vehicle matters.")
    st.divider()
    st.markdown("<h5> #Assumption 6</h5>", unsafe_allow_html=True)
    st.write("Defect_of_vechicle specifies conditions which can also be considered.")
    st.divider()
    st.markdown("<h5> #Assumption 7</h5>", unsafe_allow_html=True)
    st.write("Road_surface_type,Road_surface_condition,Road_alignment ,Types of junction,tells about leads to determine severity of accident.")
    st.divider()
    st.markdown("<h5> #Assumption 8</h5>", unsafe_allow_html=True)
    st.write("Light_condition may tell about that more light effects leads to severe accident.")
    st.divider()
    st.markdown("<h5> #Assumption 9</h5>", unsafe_allow_html=True)
    st.write("Weather_condition like mist,fog may lead tl cause of accident.")
    st.divider()
    st.markdown("<h5> #Assumption 10</h5>", unsafe_allow_html=True)
    st.write("Type_of_collision may help to understand new severe the accidents would be.")
    st.divider()
    st.markdown("<h5> #Assumption 11</h5>", unsafe_allow_html=True)
    st.write("causality_serverity ,fitness_of_casuality and cause_of_accident helps in determining the accident severity.")
    st.divider()
if selected == "Exploratory data analysis":
    st.markdown("<h5> EDA for Road accident dataset</h5>", unsafe_allow_html=True)
    
    st.markdown("""
This app retrieves the list of the **road accident** (from kaggle)
* **Python libraries:** pandas, streamlit, numpy, matplotlib, seaborn
* **Data source:** [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/road-traffic-accidents).
""")

    
    df=pd.read_csv(r"C:\Users\Gaming3\Documents\DATA_SETS\RTA_Dataset.csv")
    
    df.head(5)
    
    

# Detect numerical and categorical columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

# Impute missing values in numerical columns with mean
    for column in numerical_columns:
        df[column].fillna(df[column].mean(), inplace=True)

# Impute missing values in categorical columns with mode
    for column in categorical_columns:
        df[column].fillna(df[column].mode()[0], inplace=True)

        df.isna().sum() 



    categorical_columns = df.select_dtypes(include=['object']).columns

    for column in categorical_columns:
        unique_values = df[column].unique()
    # print(f"Unique values in '{column}': {unique_values}")

# dropping the columns which does not influences the cause for accidents
    categorical_columns = df.select_dtypes(include=['object']).columns

# Use the correct keyword argument to drop the columns
    df.drop(labels=["Day_of_week", "Educational_level", "Vehicle_driver_relation",
                "Owner_of_vehicle", "Area_accident_occured", "Work_of_casuality",
                "Casualty_class", "Sex_of_casualty"], axis=1, inplace=True)
    
    a = df.groupby('Cause_of_accident')['Cause_of_accident'].agg(['count'])
    b = a.sort_values(by='count',ascending=False).reset_index()
    
    ab = b.loc[0:10,]
    
    A =df.groupby(["Light_conditions"])["Accident_severity"].agg(["count"])
    B = A.sort_values(by='count',ascending=False).reset_index()

    uni,bi,insight= st.tabs(["Univariate analysis", "Bivariate analysis","insight"])

    with uni:
        with st.expander("Number of vehicles and casualities involved"):
            for column in numerical_columns:
                fig = px.histogram(df, x=column, nbins=20, title=f'Histogram of {column}')
                st.plotly_chart(fig)


            st.write("""INFERENCE
                    
                Most of the vehicles involved are two number of vehicles followed by one vehicle and the vehicles that are of four involved are comparitively less. 
                most of the casualities peoples is of from 1 followed by 2 and 3""")

        with st.expander("Graphical Representation to Probability of Top 4 Cause to Happen"):
            fig = px.bar(ab, x='Cause_of_accident', y='count', 
                title="Graphical Representation to Probability of Top 4 Cause to Happen", 
                labels={'Cause_of_accident': 'Cause of Accident', 'count': 'Count'})
            fig.update_layout(xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig)


            st.write("""INFERENCE
                    
                There are four major Road Accident causes which are [NO Distance , Changing lane to the right ,Changing lane to the left, Driving carelessly ]
                Conditional Probability of Each Cause to Happen When Maximum (Fatal) Severity Occur .""")

        with st.expander("Maximum road accident severity"):
            fig = px.bar(B, x='Light_conditions', y='count', color='count',
                title='Max Road Accident Severity',
                labels={'Light_conditions': 'Light Conditions', 'count': 'Count'})
            st.plotly_chart(fig)


            st.write("""INFERENCE
                    
                Maximum road accident severity is caused mainly on the day light is over 8000 followed by darkness-light lit which has the road accident severity of above 3500
    """)

        with st.expander("Gender distribution of drivers"):
            count = df["Sex_of_driver"].value_counts()

            plt.figure(figsize=(8, 3))
            plt.subplot(1, 2, 1)
            ax = sns.countplot(df["Sex_of_driver"])
            ax.bar_label(ax.containers[0], fontweight="black", size=15)

            fig = px.bar(x=count.index, y=count.values, labels={'x': 'Sex of Driver', 'y': 'Count'})
            st.plotly_chart(fig)

            st.write("""INFERENCE
                    
                If one bar is significantly higher or lower than others, it suggests a skew in the gender distribution of drivers in the dataset.
                The male driver is higher tha the female drivers.""")

        with st.expander("Age band of driver"):
            plt.figure(figsize=(16, 5))
            sns.countplot(df["Age_band_of_driver"])
            plt.title("Age Group", fontsize=20)

            st.pyplot()


            st.write("""INFERENCE
                    
                The count of drivers in each age category is given, with the lowest count or the Under 18 category and the highest count being for 4000 for the 2000 Age Group.
                It is clear that there is a significant variation in the number of drivers in each age category, with some categories having many more drivers than others.""")

        with st.expander("Types of vehicles"):
            # fig = px.bar(df['Type_of_vehicle'].value_counts().reset_index(), x='index', y='Type_of_vehicle', 
            #      labels={'index': 'Type of Vehicle', 'Type_of_vehicle': 'Count'},
            #      title="Type of Vehicle")
            # st.plotly_chart(fig)
            vehicle_counts = df['Type_of_vehicle'].value_counts().reset_index()
            vehicle_counts.columns = ['Type_of_vehicle', 'Count']

            fig = px.bar(vehicle_counts, x='Type_of_vehicle', y='Count',
                labels={'Type_of_vehicle': 'Type of Vehicle', 'Count': 'Count'},
                title="Type of Vehicle")
            st.plotly_chart(fig)


            st.write("""INFERENCE
                    
            There are several types of lorries and public vehicles listed, with varying seat capacities.
                it is clear that there is a significant variation in the number of each type of vehicle, with some types being much more common than others.""")

        with st.expander("Accident_severity"):
            severity_counts = df['Accident_severity'].value_counts().reset_index()
            severity_counts.columns = ['Accident_severity', 'Count']

            fig = px.bar(severity_counts, x='Accident_severity', y='Count',
                labels={'Accident_severity': 'Accident Severity', 'Count': 'Count'},
                title="Accident Severity category")
            st.plotly_chart(fig)


            st.write("""Accident severity
                    
                It is possible that this table is showing data from a specific time period or location where there were no fatal accidents.
                It is also possible that there are other factors that contribute to the severity of accidents, such as the type of vehicle involved, the age or experience of the driver, and the road conditions..""")


    with bi:     
        with st.expander("Number of casualties in Severity Type"):
            fig = px.bar(df, x='Accident_severity', y='Number_of_casualties', 
                title="Number of Casualties by Severity Type", 
                labels={'Accident_severity': 'Severity Type', 'Number_of_casualties': 'Number of Casualties'})
            st.plotly_chart(fig)


            st.write("""INFERENCE
                    
            The average or mean number of casualties is relatively low for all three severity categories, with the highest average number being only 2.5 for the Slight Injury category.
                The number of casualties decreases as the severity of injury increases. This is likely because more people tend to suffer from minor injuries than severe or fatal ones in an accident..""")
    
        with st.expander(" Number of vehicles involved in the accident"):
            
            fig = px.bar(df, x='Accident_severity', y='Number_of_vehicles_involved',
                title='Number of Vehicles Involved by Accident Severity',
                labels={'Accident_severity': 'Accident Severity', 'Number_of_vehicles_involved': 'Number of Vehicles Involved'})
            st.plotly_chart(fig)


            st.write("""INFERENCE
                    
            The number of vehicles involved in the accident decreases as the severity of the accident increases, which may be due to the fact that more serious accidents are more likely to involve fewer vehicles.""")
    
        with st.expander("Age band and percentage of Accidents"):
            a = (df.loc[df['Age_band_of_driver'] != "Unknown"].groupby('Age_band_of_driver').size() / len(df))
            a = a.sort_values(ascending=False)

            fig = px.bar(x=a.index, y=a.values*100, 
                labels={'x': 'Age Band', 'y': 'Percentage Of Total Accidents'},
                title="Age Band And Percentage Of Accidents")
            st.plotly_chart(fig)

            st.write("""INFERENCE
                    
            It shows the percentage of total accidents that drivers in different age bands are involved in.
            The percentage of total accidents decreases as the age band increases, suggesting that younger drivers are more likely to be involved in accidents than older drivers.""")
    
    
    with insight:
     st.write("""INSIGHT
                 
                These data suggest that there are several factors that may contribute to the severity and likelihood of accidents, including the age and number of vehicles involved, the type of injury sustained, and the type of road where the accident occurred. Understanding these factors can help inform policies and strategies to reduce the number and severity of accidents.
           The percentage of total accidents decreases as the age band increases, suggesting that younger drivers are more likely to be involved in accidents than older drivers.
           The number of vehicles involved in the accident decreases as the severity of the accident increases, which may be due to the fact that more serious accidents are more likely to involve fewer vehicles.""")

if selected == "Model":
    st.title(f"you have selected {selected}")
    
    categorical_columns = df.select_dtypes(include=['object']).columns

# Use the correct keyword argument to drop the columns
    df1.drop(labels=["Time","Day_of_week", "Educational_level", "Vehicle_driver_relation","Number_of_casualties",
                "Owner_of_vehicle", "Area_accident_occured", "Work_of_casuality","Fitness_of_casuality","Age_band_of_casualty",
                "Casualty_class", "Sex_of_casualty",'Road_surface_conditions','Service_year_of_vehicle','Age_band_of_driver','Road_allignment','Casualty_severity'], axis=1, inplace=True)

    # Detect numerical and categorical columns
    numerical_columns = df1.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df1.select_dtypes(include=['object']).columns

# Impute missing values in numerical columns with mean
    for column in numerical_columns:
        df1[column].fillna(df1[column].mean(), inplace=True)

# Impute missing values in categorical columns with mode
    for column in categorical_columns:
        df1[column].fillna(df1[column].mode()[0], inplace=True)


    from sklearn.preprocessing import LabelEncoder


    le = LabelEncoder()
    df['encoded_attribute'] = le.fit_transform(df['Accident_severity'])

# Display the DataFrame with the encoded attribute
    st.write('DataFrame after label encoding:')
    st.write(df)

# If you want to retrieve the mapping of original labels to encoded labels:
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

# Display the label mapping
    st.write('Label mapping:')
    st.write(label_mapping)


