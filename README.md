# Disaster Response Pipeline Project

### Introduction and motivation:
I build this project in order to apply my Data Engineer knowledge in a real life challenge: Message Classifier For Disaster Events. This work can be used by disaster relief agency. It includes a web app which allows people to input new messages and get classification result in a short time, it also has visualization part

### Project structure:
app    

| - template    
| |- master.html : Home page
| |- go.html : Model result displays in this page
|- run.py : Python source code to initialize Flask webapp 


data    

|- disaster_categories.csv : Raw dataset
|- disaster_messages.csv : Raw dataset
|- process_data.py : Data cleaning pipeline    
|- DisasterResponse.db : Database to save clean data     


models   

|- train_classifier.py # Machine learning pipeline     
|- classifier.pkl # Result model     


README.md    

### How to run project:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
