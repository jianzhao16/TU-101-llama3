import streamlit as st
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
from datetime import datetime
import pgeocode
import pandas as pd
import openai
from branca.element import IFrame

import os
from dotenv import load_dotenv


# Function to query OpenAI API for extracting the service and zipcode
def ask_openai_for_service_extraction(question, api_key, conversation_history):
    openai.api_key = api_key
    extraction_instruction = "Extract the type of service and zipcode from the following user query:"
    #extraction_instruction =""
    #combined_query = f"{extraction_instruction}\n{question}"
    combined_query = f"{question}"
    full_conversation = conversation_history + [{"role": "user", "content": combined_query}]
    response = openai.ChatCompletion.create(model="gpt-4", messages=full_conversation)
    
    if response.choices:
        conversation_history.append({"role": "user", "content": combined_query})
        conversation_history.append({"role": "assistant", "content": response.choices[0].message['content']})
    return response

# Function to classify service type
def classify_service_type(service_type, api_key):
    openai.api_key = api_key
    prompt = f"""
    Below are examples of service types with their categories:
    "Meal services": Food
    "Temporary housing": Shelter
    "Counseling services": Mental Health
    "Emergency food support": Food
    "Mental health counseling": Mental Health 

    Based on the examples above, classify the following service type into the correct category (Food, Shelter, Mental Health):
    "{service_type}": """
    response = openai.Completion.create(engine="davinci-002", prompt=prompt, max_tokens=60)
    
    raw_category = response.choices[0].text.strip() if response.choices else "Other"
    st.write("Raw Model Response for Classification:", raw_category)

    raw_category = raw_category.split('\n')[0]
    raw_category = raw_category.replace('1-', '').replace('2-', '').replace('3-', '').strip()

    for standard_category in ['Food', 'Shelter', 'Mental Health']:
        if standard_category.lower() in raw_category.lower():
            return standard_category

    return "Other"

# Function to safely convert time string to 24-hour format
def safe_convert_time(time_str):
    """Safely converts a time string to 24-hour format."""
    try:
        return datetime.strptime(time_str, '%I:%M %p').strftime('%H:%M')
    except ValueError:
        # Handle specific known issues
        if time_str == '0:00 AM':
            return '00:00'  # Convert '0:00 AM' to '00:00'
        elif time_str == '12:00 AM':  # Add other known issues as needed
            return '00:00'
        # Return original string or some default if no specific case matched
        return '00:00'  # Adjust this if necessary

@st.cache_data
def read_data(df):
    df = df.dropna(subset=['Latitude', 'Longitude'])  # Ensure we have the necessary location data
    # Get the current day and time
    now = datetime.now()
    current_day = now.strftime('%A')  # e.g., 'Monday'
    
    data = []
    for index, row in df.iterrows():
        # Merge and clean services information
        main_services = eval(row['Main_Services']) if pd.notna(row['Main_Services']) else []
        other_services = eval(row['Other_Services']) if pd.notna(row['Other_Services']) else []
        services = main_services + other_services
        services = [service for service in services if service != 'None']
        
        # Get the opening hours for the current day
        opening_hours_today = row[current_day] if pd.notna(row[current_day]) else 'Unavailable'

        # Format popup information to include today's hours
        info = f"""
            <strong>{row['Service_Name']}</strong><br>
            <strong>Today's Hours:</strong> {opening_hours_today}<br>
            <strong>Services:</strong> {', '.join(services)}<br>
            <strong>Serving:</strong> {row['Serving']}<br>
            <strong>Phone Number:</strong> {row['Phone_Number']}<br>
            <strong>Eligibility:</strong> {row['Eligibility']}<br>
            <strong>Languages:</strong> {row['Languages']}<br>
            <strong>Cost:</strong> {row['Cost']}
        """

        # Append this information along with latitude and longitude
        data.append({
            'latitude': row['Latitude'],
            'longitude': row['Longitude'],
            'info': info
        })

    return data

# Streamlit UI
st.markdown("# PPD Chatbot")
st.markdown("### Ask me about available services:")
#user_query = st.text_input("Enter your query (e.g., 'What is PPD')", key="user_query")
user_query = st.text_area("Enter your query (e.g., 'What is PPD')", key="user_query")

# Submit button
submit_button = st.button("Submit")

# Initialize global variables
conversation_history = []


# Load .env file
load_dotenv()
# Retrieve API key
api_key = os.getenv("API_KEY")
#api_key = ''  # Replace this with your actual OpenAI API key

if submit_button:
    response = ask_openai_for_service_extraction(user_query, api_key, conversation_history)
    if response.choices:
        extracted_info = response.choices[0].message['content'].strip()
        
        # Debugging: Display extracted information
        st.write("Answer:", extracted_info)

