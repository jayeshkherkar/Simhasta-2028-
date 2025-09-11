# Simhasta-2028-
Simhasta Prototype submission
# Overview
This project is flask based application which simulate the Crowd management system. 
The following features this application are:
- Zone-wise crowd monitoring  
- Folium-based maps  
- Live/recorded video footages  
- ML-based prediction & summary points
# Project Structure
 - static/ CSS, JS, videos
 - templates/ HTML templates (Jinja2)
 - app.py # Main Flask app
# Installation setup
       bash
       git clone https://github.com/<your-username>/<your-repo>.git
       cd <your-repo>
      1) Create Virtual Environment 
         Run this command in vs code powershell terminal:- python -m venv venv
      2) Activate Environment:
         Run this command in vs code powershell terminal:- venv\Scripts\activate
      3) Install dependencies: 
         Run this command in vs code powershell terminal:- pip install -r requirements.txt
         Here in this repository all libraries are mentioned in requirements.txt file.
      4) Run the flask application:
         Run this command in vs code powershell terminal: python app.py
# Important instruction is that please ensure that all the libraries mentioned in the "requirements.txt" should be installed efficiently in virtual environment and create the virtual environment folder inside the   folder named as "Dasboard".
# The provided Google Drive link contains two folders: one with 10 video footages and another with two ML models (a Joblib model for crowd density prediction and a YOLOv8 model for real-time people counting).
  Video link, ML models: - https://drive.google.com/drive/folders/1HexlVxr469xTbZbSHOHnqUzpa4iIzuI7?usp=sharing 
