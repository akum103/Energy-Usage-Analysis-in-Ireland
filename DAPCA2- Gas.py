#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#############################################################
#THIS CODE IS WRITTEN BY - NIHARIKA SUBBARAYAPPA  - x23102420
#############################################################

# WELCOME TO DAP - CA2 PROGRAMMING PROJECT. THIS PROJECT IS DONE BY 

# 1) ANKIT KUMAR - X23123061
# 2) NIHARIKA SUBBARAYAPPA - x23102420
# 3) SAVAN KUMAR PANDITH - x22182934



# This is a python file in which we are using gas.json file, which is gas usage peer county per year in Ireland
# in a JSON format. This code will follow the following steps:

##################### LIBRARIES TO INSTALL #########################

# 1) JSON
# 2) PYMONGO
# 3) SQLALCHEMY
# 4) PLOTLY
# 5) PLOTLY.GRAPH_OBJECTS
# 6) PLOTLY.EXPRESS
# 7) DASH
# 8) PANDAS

####################################################################
#*******************************************************************

############### PROGRAMMING STEPS - PSEUDO CODE ####################


# 1) Upload Json to Mongo DB
# 2) Perform transformations in MongoDB and clean the data!
# 3) Call the file from mongo DB and store it in Pandas Dataframe.
# 4) Upload the file to Postgres
# 5) Call file from postgres for viz
# 6) Perform Viz

####################################################################
#*******************************************************************


# In[3]:


############################
#IMPORTING ALL THE LIBRARIES


# mMongoDB Related Libraries
import json
import pymongo
from pymongo import MongoClient

# Data handling related libraries
import pandas as pd

#Postgres related libraries
from sqlalchemy import create_engine
from sqlalchemy import Integer
from sqlalchemy import VARCHAR
from sqlalchemy import Date

#Visualizations related libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px


# In[4]:


###########################
# UPLOADING JSON TO MONGODB

# Provide the JSON file path
json_file_path = r"C:\Users\ankit\OneDrive\Desktop\wORK fiLES\NCI Files\Semester I\DAP\gas.json"

# MongoDB connection specifics
mongodb_uri = "mongodb+srv://akum103:Sushma09(@dap.c7r880a.mongodb.net/dap"
database_name = "electricity_gas_usage"
collection_name = "gas"

# providing the connection details
client = MongoClient(mongodb_uri)
db = client[database_name]
collection = db[collection_name]

# Open file with error handling
try:
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    # Insert data into MongoDB collection
    collection.insert_many(json_data)
    print("Data inserted successfully")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Close the MongoDB connection
    client.close()
    

print("Data was inserted in MONGODB successfully")


# In[5]:


###############################
#DATA TRANSFORMATION IN MONGODB

# connect Mongo db client to call back to python for processing
try:
    # Connect to MongoDB
    client = MongoClient(mongodb_uri)
    db = client[database_name]
    collection = db[collection_name]

    
    # Define the mapping of old column names to new column names
    column_change = {
        'Statistic Label': 'type_of_connection',
        'C03815V04565': 'county_code',
        'Counties & Dublin Postal Districts': 'county'
    }

    # Iterate over the mapping and update the column names
    for old_col, new_col in column_change.items():
        update_query = {"$rename": {old_col: new_col}}
        collection.update_many({}, update_query)

    print("Column names updated successfully.")
    
    
    # Delete unnecessary columns
    delete_columns = ['TLIST(Q1)', 'C03816V04566', 'STATISTIC', 'quarter', 'sector', 'tlist(q1)','unit', 'value']

    # Iterate over the documents and delete the specified columns
    for column in delete_columns:
        update_query = {"$unset": {column: 1}}
        collection.update_many({}, update_query)

    print("Columns deleted successfully.")
    
    # Define the values to be excluded
    delete_values = ['-', '9999']

    # Iterate over the values and delete rows with those values in the county_code field
    for value in delete_values:
        delete_query = {"county_code": value}
        collection.delete_many(delete_query)

    print("Rows deleted successfully.")
    
    #rename certail row values
    sector_change_values = {
        '10 Residential': 'Residential',
        'Non-residential including power plants': 'Commercial & Power Plants'
    }

    
    # Iterate over the mapping and update the values in the Sector field
    for old_value, new_value in sector_change_values.items():
        update_query = {"Sector": old_value}
        update_operation = {"$set": {"Sector": new_value}}
        collection.update_many(update_query, update_operation)

    print("Values updated successfully.")
    
    # Deleting zeroes
    delete_zero_from_value = {"VALUE": 0}
    collection.delete_many(delete_zero_from_value)    
    print("0s from value column was deleted")
    
    # Deleting rows with empty values
    delete_empty_value_query = {"$or": [{"VALUE": 'No Field'}, {"VALUE": ''}, {"VALUE": None}]}
    collection.delete_many(delete_empty_value_query)    
    print("Rows with empty or null VALUE deleted successfully.")

    #deleting rows with county value not coded
    delete_county_with_9999 = {"county": "Not coded"}
    collection.delete_many(delete_county_with_9999)    
    print("9999 coded counties were deleted")    
    
    documents = list(collection.find())

# Lowercase all field names in each document
    for doc in documents:
        updated_doc = {}
        for key, value in doc.items():
            new_key = key.lower()
            updated_doc[new_key] = value

    # Update the document with lowercase field names
    update_query = {"$set": updated_doc}
    collection.update_one({"_id": doc["_id"]}, update_query)
    
    print("DONE.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")

finally:
    # Close the MongoDB client in the 'finally' block to ensure it's closed even if an error occurs
    if client:
        client.close()
    
        


# In[19]:


############################################################
#FETCHING  DATA FROM MONGO DB AND STORING IN DF USING PANDAS

# connecting to database
mongodb_uri = "mongodb+srv://akum103:Sushma09(@dap.c7r880a.mongodb.net/dap"
database_name = "electricity_gas_usage"
collection_name = "gas"

try:
    # Connect to MongoDB
    client = MongoClient(mongodb_uri)
    db = client[database_name]
    collection = db[collection_name]

    # Fetch data from MongoDB and store it in a DataFrame
    mongo_data = list(collection.find())
    df_gas_fetch = pd.DataFrame(mongo_data)

    print("Data fetched successfully from MongoDB.")

except Exception as e:
    print(f"An error occurred while fetching data from MongoDB: {e}")

finally:
    # Close the MongoDB connection in the 'finally' block to ensure it's closed even if an error occurs
    if client:
        client.close()


delete_columns = ['quarter', 'sector','unit', 'value']

df_gas_fetch = df_gas_fetch.drop(columns = delete_columns, errors = 'ignore')

null_values = df_gas_fetch.isnull().sum()
print('\n--------NULL VALUES--------')
print(null_values)

print('\n\n---------------------Gas Dataset---------------------')
print(df_gas_fetch)


# In[20]:


#####################################################################
# SUM UP QUARTERS TO YEARS AND GROUP BY YEAR, COUNTY CODE, AND SECTOR

# Extract the year from the 'Quarter' column
df_gas_fetch['Year'] = df_gas_fetch['Quarter'].str.extract('(\d{4})').astype(int)

# Define aggregation types to maintain all columns during aggregation
aggregation_gas = {
    'Quarter': 'first',
    'Sector': 'first',
    'VALUE': 'sum',
    'county_code': 'first',
    'UNIT': 'first',
    'type_of_connection': 'first',
    'county': 'first',
}

# Group by 'Year', 'county_code', and 'Sector', and apply the aggregation
result_gas = df_gas_fetch.groupby(['Year', 'county_code', 'Sector'], as_index=False).agg(aggregation_gas)

# Display the result
print(result_gas)

# Define the path for saving the CSV file
csv_file_path = 'gas.csv'

# Save the DataFrame to a CSV file
result_gas.to_csv(csv_file_path, index=False) 

# Alternative method: Group by 'Year', 'county_code', and 'Sector', and sum the 'VALUE'
# result_gas = df_gas.groupby(['Year', 'county_code', 'Sector'], as_index=False)['VALUE'].sum()

# Print the alternative result
# print(result_gas)


# In[22]:


################################
# UPLOAD DATAFRAME TO POSTGRESQL


# PostgreSQL connection parameters
db_params = {
    'user': 'postgres',
    'password': 'Sushma09(',
    'host': 'localhost',
    'port': '5432',
    'database': 'electricity_gas_database'
}

# Create a SQLAlchemy engine
engine = create_engine(f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}")

try:
    # Specify data types for each column if needed
    dtype_dict = {
        'Year': Integer,
        'Quarter': VARCHAR(50),
        'Sector': VARCHAR(50),
        'VALUE': Integer, 
        'county_code': VARCHAR(50),        
        'UNIT': VARCHAR(50),
        'type_of_connection': VARCHAR(50),
        'county': VARCHAR(50)   
    }
    
    # Store the DataFrame in PostgreSQL, replacing the existing data
    result_gas.to_sql('gas', engine, if_exists='replace', index=False, dtype=dtype_dict)

    print("Data uploaded successfully to PostgreSQL.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")

finally:
    # Close the database connection in the 'finally' block to ensure it's closed even if an error occurs
    engine.dispose()


# In[29]:


################################
# CALL FILE FROM POSTGRE FOR VIZ

# PostgreSQL connection parameters
DB_PARAMS = {
    'user': 'postgres',
    'password': 'Sushma09(',
    'host': 'localhost',
    'port': '5432',
    'database': 'electricity_gas_database'
}

# Create a SQLAlchemy engine
engine = create_engine(f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['database']}")

# SQL queries to retrieve data from PostgreSQL
SQL_GAS = "SELECT * FROM gas"
SQL_PRICE_LIST = "SELECT * FROM price_list"

try:
    # Read data from PostgreSQL into DataFrames
    gas_data = pd.read_sql_query(SQL_GAS, engine)
    price_list = pd.read_sql_query(SQL_PRICE_LIST, engine)

except Exception as e:
    print(f"An unexpected error occurred: {e}")

finally:
    # Close the database connection in the 'finally' block to ensure it's closed even if an error occurs
    engine.dispose()

# Select rows where the 'Year' column is greater than or equal to 2016
MASK = gas_data['Year'] >= 2016

# Apply the mask to filter rows
df_filtered = gas_data[MASK]

# Assign the filtered DataFrame to df_gas
df_gas = df_filtered
df_price_list = price_list

print('\n\033[94m-------------------------GAS DATASET FROM POSTGRES-------------------------\n')
print(df_gas)


# In[30]:


##################################################
# CREATE A NEW DATAFRAME FOR HOUSE PRICES PER YEAR


# Selecting prices for 
df_price_list_gas = df_price_list[(df_price_list['type_of_connection'] == 'Gas')
                                  & (df_price_list['user_type'] == 'House')]

# Rename the 'date' column to 'Year'
df_price_list_gas = df_price_list_gas.rename(columns={'date': 'Year'})

# Filter the gas DataFrame by the 'Residential' sector
df_gas_residential = df_gas[df_gas['Sector'] == 'Residential']

# Display the gas residential DataFrame
print(df_gas_residential)

# Merge the gas price and gas residential DataFrames on the 'Year' column
merged_df_gas_house = pd.merge(df_price_list_gas, df_gas_residential, on='Year', how='left')
print(merged_df_gas_house)

# Calculate the total gas price by multiplying relevant columns
merged_df_gas_house['Total Gas Price'] = merged_df_gas_house['ireland'] * 1 / 100 * merged_df_gas_house['VALUE'] * 1000

# Save the merged DataFrame to a CSV file
merged_df_gas_house.to_csv('price_gas.csv', index=True)
print('File Created')


# In[35]:


#######################################################
# CREATE A NEW DATAFRAME FOR COMMERCIAL PRICES PER YEAR


# Select Gas and commercial filter for the price list dataset
df_price_list_commercial = df_price_list[(df_price_list['type_of_connection'] == 'Gas')
                                           & (df_price_list['user_type'] == 'Commercial')]

# Rename column date to year
df_price_list_commercial = df_price_list_commercial.rename(columns ={'date':'Year'})

# In the gas datraset, filter sector as Commercial and Power plants.
df_gas_commercial = df_gas[df_gas['Sector'] == 'Commercial & Power Plants']

# merge the data on year left
merged_df_gas_commercial = pd.merge(df_price_list_commercial, df_gas_commercial, on = 'Year', how = 'left')
print(merged_df_gas_commercial)

# calculate the price of gas. Gas consumption value is GWh. 1 GWg is 1,000,000 Kwh. Prices are in cents per Kwh
merged_df_gas_commercial ['Total Gas Price'] = merged_df_gas_commercial['ireland'] * 1/100 * merged_df_gas_commercial['VALUE'] * 1000000
print(merged_df_gas_commercial)

#
null_values = merged_df_gas_commercial.isnull().sum()
merged_df_gas_commercial = merged_df_gas_commercial.dropna()


# In[39]:


#####################
# GAS USAGE BY COUNTY

# Starting dash command
app = dash.Dash(__name__)

# add the gas dataframe to a new dataset to maintain integrity
df_sorted = df_gas

# Plot bar graph with my df_sorted data frame with x-axis as year and y-axis as value of electricity unit
fig = px.bar(df_sorted, x='Year', y='VALUE', color='VALUE', labels={'VALUE': 'Gas Value'},
             title="County wise Gas usage", color_continuous_scale='blues')

# Define layout
app.layout = html.Div([
    html.H1(children='GAS USAGE BY COUNTY', style={'textAlign': 'center', 'color': 'red'}),
    
    # New dropdown for selecting 'Residential' or 'Commercial'
    dcc.Dropdown(
        id='sector-dropdown',
        options=[{'label': sector, 'value': sector} for sector in df_gas['Sector'].unique()],
        value=df_gas['Sector'].unique()[0],  # Set default value
        multi=False
    ),
    
    dcc.Graph(figure=fig, id='gas-bar-chart'),
    
    # Dropdown for selecting counties
    dcc.Dropdown(
        id='county-dropdown',
        options=[{'label': county, 'value': county} for county in df_gas['county'].unique()],
        value=df_gas['county'].unique()[0],  # Set default value
        multi=False
    )
])

# Define callback to update graph based on dropdown selections
@app.callback(
    Output('gas-bar-chart', 'figure'),
    [Input('county-dropdown', 'value'),
     Input('sector-dropdown', 'value')]
)
def update_graph(selected_county, selected_sector):
    filtered_df = df_sorted[(df_sorted['county'] == selected_county) & (df_sorted['Sector'] == selected_sector)]

    # Create a new DataFrame with the sum of values for each year
    sum_df = filtered_df.groupby('Year')['VALUE'].sum().reset_index()

    updated_fig = px.bar(
        sum_df, x='Year', y='VALUE',
        labels={'VALUE': 'Gas Value'},
        color='VALUE',  # Add color parameter to create a heatmap effect
        color_continuous_scale='blues'
    )

    # Add text annotations to each bar with the total value
    for i, row in sum_df.iterrows():
        updated_fig.add_annotation(
            x=row['Year'],
            y=row['VALUE'],
            text=str(row['VALUE']),
            showarrow=True,
            arrowhead=2,
            arrowcolor='black',
            ax=0,
            ay=-30
        )

    return updated_fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port = 9051)


# In[41]:


###########################
# GAS USAGE FOR RESIDENTIAL

df_top_counties_gas = merged_df_gas_commercial

# Function for creating top 10
def top_ten_per_year(group):
    return group.nlargest(10, 'Total Gas Price')

# group by top 10 year wise by value
top_ten_df_gas = df_top_counties_gas.groupby('Year', group_keys=False).apply(top_ten_per_year)


# remove 2023 year as we dont have all the data for the same
top_ten_df_gas = top_ten_df_gas[top_ten_df_gas['Year'] != 2023]


# adding it to df in order to maintain integrity
df = top_ten_df_gas

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("Residential - Top 10 Counties for Each Year for Gas"),

    # Dropdown for selecting the year
    dcc.Dropdown(
        id='year-dropdown',
        options=[
            {'label': str(year), 'value': year} for year in df['Year'].unique()
        ],
        value=df['Year'].max(),  # Set the initial selected year
        multi=False,
    ),

    # Bar plot
    dcc.Graph(id='bar-plot'),
])

# Callback to update the bar plot based on selected year
@app.callback(
    Output('bar-plot', 'figure'),
    [Input('year-dropdown', 'value')]
)
def update_bar_plot(selected_year):
    selected_data = df[df['Year'] == selected_year]

    fig = px.bar(selected_data, x='county', y='Total Gas Price', title=f'Top 10 Counties in {selected_year}')
    fig.update_layout(barmode='group', xaxis_tickangle=-45)

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port = 9052)


# In[43]:


##########################
# GAS USAGE FOR COMMERCIAL


df_top_counties_gas = merged_df_gas_house

def top_ten_per_year(group):
    return group.nlargest(10, 'Total Gas Price')

top_ten_df_gas = df_top_counties_gas.groupby('Year', group_keys=False).apply(top_ten_per_year)

top_ten_df_gas = top_ten_df_gas[top_ten_df_gas['Year'] != 2023]
# print(top_ten_df_gas)


# adding dataframe to df to maintain integrity
df = top_ten_df_gas

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("Commercial - Top 10 Counties for Each Year for Gas"),

    # Dropdown for selecting the year
    dcc.Dropdown(
        id='year-dropdown',
        options=[
            {'label': str(year), 'value': year} for year in df['Year'].unique()
        ],
        value=df['Year'].max(),  # Set the initial selected year
        multi=False,
    ),

    # Bar plot
    dcc.Graph(id='bar-plot'),
])

# Callback to update the bar plot based on selected year
@app.callback(
    Output('bar-plot', 'figure'),
    [Input('year-dropdown', 'value')]
)
def update_bar_plot(selected_year):
    selected_data = df[df['Year'] == selected_year]

    fig = px.bar(selected_data, x='county', y='Total Gas Price', title=f'Top 10 Counties in {selected_year}')
    fig.update_layout(barmode='group', xaxis_tickangle=-45)

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port = 9053)

