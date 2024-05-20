#main code

import streamlit as st
from PIL import Image
import numpy as np
from bs4 import BeautifulSoup
import sqlite3
import os
#from googlesearch import search
from pathlib import Path
import torch
import torchvision.models as models
import requests
import cv2
import subprocess
import pandas as pd
from streamlit import columns
import time

import base64

#SMS
import twilio
import twilio.rest
from twilio.rest import Client

from IPython.display import Video

# Set the environment variable
os.environ['TWILIO_ACCOUNT_SID'] = ''
os.environ['TWILIO_AUTH_TOKEN'] = ''



def sendsms():

    # Read the list of essential items from essentials.txt
    with open('essentials.txt', 'r') as file:
        essential_items = file.read().splitlines()
    
    # Connect to the SQLite database
    conn = sqlite3.connect('grocery.db')
    cursor = conn.cursor()
    
    # Check each essential item in the list
    for item in essential_items:
        # Check if the item exists in the database
        cursor.execute("SELECT * FROM grocery WHERE class=?", (item,))
        result = cursor.fetchone()
        
        # If the item is not found, add it to needtobuy.txt
        if result is None:
            with open('needtobuy.txt', 'a') as needtobuy_file:
                needtobuy_file.write(f"{item}\n")
    
    # Close the database connection
    conn.close()
    
    # Read the contents of needtobuy.txt
    with open('needtobuy.txt', 'r') as needtobuy_file:
        needtobuy_items = needtobuy_file.read()

    # Your Twilio account SID and auth token
    # Retrieve Twilio account SID and auth token from environment variables
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    #st.write(account_sid,auth_token)
    client = Client(account_sid, auth_token)
    
    message = client.messages \
                    .create(
                         body=f"Items to buy: {needtobuy_items}",
                         from_='',
                         to=''
                     )

   
    st.write("SMS sent successfully!")
    

# Send an email or SMS with the contents of needtobuy.txt
# Add your code to send email or SMS here

# Connect to the SQLite database
conn = sqlite3.connect('grocery.db')
c = conn.cursor()

#Essentials file
def create_or_edit_essentials():
    st.write("Create or Edit Essentials List:")
    st.write("Enter the essential items (one per line) and click 'Save Essentials':")
    essentials = st.text_area("Essential Items", height=200)
    if st.button("Save Essentials"):
        with open('essentials.txt', 'w') as file:
            file.write(essentials)
        with open('needtobuy.txt', 'w') as needtobuy_file:
            needtobuy_file.truncate(0)
        st.success("Essentials saved successfully!")

def fetch_image_url(item_name):
    try:
        url = 'https://www.google.com/search?&q=' + item_name + ' vegetable'
        req = requests.get(url).text
        soup = BeautifulSoup(req, 'html.parser')
        image_url = soup.find("img").get("src")
        return image_url
    except Exception as e:
        print(f"Error fetching image: {e}")
        return None


def display_database():
    c.execute('SELECT class, count FROM grocery')
    items = c.fetchall()
    st.title("Database Contents")

    st.text("")
    for name, quantity in items:
       
        # Display the image, name, and quantity
        col1, col2, col3= st.columns(3)  # Create three columns
        with col1:
            st.text("")
            st.text("")
            st.write(name)  # Display the name
        with col2:
            st.text("")
            st.text("")
            st.text(quantity)   # Display the quantity
        with col3:
            
            quantity_input = st.number_input(f"Reduce {name} quantity", min_value=0, max_value=quantity, value=0)
            if quantity_input > 0:
                new_quantity = quantity - quantity_input
                c.execute('UPDATE grocery SET count=? WHERE class=?', (new_quantity, name))
                conn.commit()
                st.experimental_rerun()  # Rerun the app to refresh the page
      
    # Add a reset button to empty the database
    unique_key = "reset_button"
    
    if st.button("Reset Database", key=unique_key):
        c.execute('DELETE FROM grocery')
        conn.commit()
        st.write("Database has been reset.")
        st.experimental_rerun()  # Rerun the app to refresh the page

    

    

    


def essen():
    create_or_edit_essentials()
    st.title("Send Notification")
    if st.button("Send Notification"):
        sendsms()





# Classify fresh/rotten fn
def ret_fresh(res):
    threshold_fresh = 0.90  # set according to standards
    threshold_medium = 0.30  # set according to standards
    if res > threshold_fresh:
        return "The item is VERY FRESH!"
    elif threshold_fresh > res > threshold_medium:
        return "The item is FRESH"
    else:
        return "The item is NOT FRESH"
    

def pre_proc_img(image_data):
    # Convert the JpegImageFile object to bytes
    byte_stream = io.BytesIO()
    image_data = image_data.convert('RGB')  # Convert to RGB
    image_data.save(byte_stream, format='JPEG')
    image_bytes = byte_stream.getvalue()

    # img data to a np arr and read using cv2
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Cnvrt BGR to RGB & resize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100))

    # Preprocess the image
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def evaluate_rotten_vs_fresh(image_path):
    # Load and predict using the model
    model = load_model('Model/trained-freshness-model.h5')
    prediction = model.predict(pre_proc_img(image_path))

    return prediction[0][0]





def evaluate_freshness(image_path, api_key, project_id, model_version):
    url = f"https://source.roboflow.com/{api_key}/{project_id}/versions/{model_version}/infer"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": [
            {
                "type": "image_url",
                "image_url": image_path
            }
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        predictions = result["predictions"]
        # Assuming the freshness score is returned in the predictions
        freshness_score = predictions[0]["freshness"]
        return freshness_score
    else:
        return None  # Return None or handle the error as needed


def run_inference(image_path, api_key, project_id, model_version):
    # Command to install the Inference CLI
    install_command = 'pip install inference-cli'

    # Command to start the inference server
    start_command = 'inference server start'

    # Install the Inference CLI
    subprocess.run(install_command, shell=True)

    # Start the inference server
    subprocess.run(start_command, shell=True)

    # Command to run inference on an image
    infer_command = f'inference infer {image_path} --api-key {api_key} --project-id {project_id} --model-version {model_version}'
    
    # Run inference on an image
    try:
        subprocess.run(infer_command, shell=True)
    except Exception as e:
        st.error(f"Failed to get freshness prediction: {e}")



#RECIPE RECOMMEND

def display_ingredientss():
    
    c.execute('SELECT class, count FROM grocery')
    ingredients = c.fetchall()

    col1, col2, col3, col4 = st.columns(4)

    for i, (ingredient, quantity) in enumerate(ingredients):
        if i % 4 == 0:
            col = col1
        elif i % 4 == 1:
            col = col2
        elif i % 4 == 2:
            col = col3
        else:
            col = col4
        col.write(f"{ingredient} ({quantity})")



        
def get_recipes(ingredient, mode):
    url = f'https://api.spoonacular.com/recipes/findByIngredients'
    params = {
        'ingredients': ','.join(ingredient),
        'number': 15,  # Number of recipes to fetch
        'ranking': mode, #Whether to maximize used ingredients (1) or minimize missing ingredients (2) first.
        'apiKey': ''
    response = requests.get(url, params=params)
    if response.status_code == 200:
        recipes = response.json()
        return recipes
    else:
        st.error('Failed to fetch recipes')
        return None

def get_recipe_details(recipe_id):
    url = f'https://api.spoonacular.com/recipes/{recipe_id}/information'
    params = {
        'apiKey': ''  
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        recipe_details = response.json()
        return recipe_details
    else:
        st.error('Failed to fetch recipe details')
        return None


def display_recipe_recommendation():
    st.title("Recipe Recommendation")
    st.write("Select the mode for fetching recipes:")
    mode = st.radio("Mode:", ["Maximize used ingredients", "Minimize missing ingredients"])
    instruct = st.radio("Instructions:", ["With Instructions", "Without Instructions"])
    number_of_recipes = st.number_input("Number of recipes to fetch:", min_value=1, max_value=10, value=6)
    ingredients = st.text_area("Enter ingredients (one per line):", height=200)
       
    if st.button("Get Recipes"):
        ingredient_list = ingredients.split('\n')
        mode_value = 1 if mode == "Maximize used ingredients" else 2
        instruct_value = instruct == "With Instructions"
        recipes = get_recipes(ingredient_list, mode_value)
        
        if recipes:
            st.write("Recipes:")
            recipe_count = 0
            row = st.columns(3)
            for recipe in recipes:
                missing_ingredients = ', '.join(ingredient['name'] for ingredient in recipe.get('missedIngredients', []))
                if instruct_value:
                    recipe_details = get_recipe_details(recipe['id'])                                   
                    if recipe_details['instructions'] is not None:
                        with row[recipe_count % 3]:
                            st.image(recipe['image'], width=200, caption=recipe['title'])
                            st.write(f"Missing Ingredients: {missing_ingredients}")
                            st.write("Instructions:")
                            st.write(recipe_details['instructions'])
                            st.markdown("---")
                        recipe_count += 1
                        if recipe_count % 3 == 0:
                            row = st.columns(3)
                        if recipe_count >= number_of_recipes:
                            break  # Stop displaying recipes if the desired number has been reached
                else:
                    with row[recipe_count % 3]:
                        st.image(recipe['image'], width=200, caption=recipe['title'])
                        st.write(f"Missing Ingredients: {missing_ingredients}")
                        st.markdown("---")
                    recipe_count += 1
                    if recipe_count % 3 == 0:
                        row = st.columns(3)
                    if recipe_count >= number_of_recipes:
                        break  # Stop displaying recipes if the desired number has been reached
                    
        else:
            st.error("No recipes found")


def run_inference_detect(image_path):
    # Your existing code to run inference
    cmd = f"python detect.py --weights runs/train/yolov5s_results3/weights/best.pt --img 416 --conf 0.4 --source {image_path}"
    subprocess.run(cmd, shell=True)

def detect_vegetables():
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        save_image_path = 'C:/project/content/test/image' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())
        
        if img_file is not None:
            run_inference_detect(save_image_path)
            with open('output.txt', 'r') as f:
                for line in f.readlines():
                    st.info(line.strip())
                    
def front_page():
    st.write("Step into the future of grocery management with our innovative Grocery Management App. Say hello to efficiency and organization in the kitchen – welcome to the Home Grocery Management App.")
    with open("front page.mp4", "rb") as file:
        video_data = file.read()
        video_base64 = base64.b64encode(video_data).decode()

    video_html = f"""
    <video autoplay loop muted style="position: relative; right: 1; bottom: 0;top:0; min-width: 100%; min-height: 100%;max-width: 70%">
        <source type="video/mp4" src="data:video/mp4;base64,{video_base64}" />
        Your browser does not support the video tag.
    </video>
    """
    
    st.markdown(video_html, unsafe_allow_html=True)


def run():
    st.set_page_config(page_title="Grocery Management", layout="wide")    
    st.title("Home Grocery Management🍍-🍅 ")
    
    
    page = st.sidebar.selectbox("Menu", ["Home", "Database", "Recipe Recommendation", "Vegetable Detection", "Generate List"])
    # Display the selected page
    if page == "Home":
        front_page()
    elif page == "Database":
        display_database()
    elif page == "Recipe Recommendation":
        display_recipe_recommendation()
    elif page == "Vegetable Detection":
        detect_vegetables()
    elif page == "Generate List":
        essen()
    

    #RECIPE
    
           
     # Display the database contents

run()




# Close the connection when done
conn.close()

