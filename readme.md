# Face Recognition

In this data science and machine learning project, we classify sports personalities. We restrict classification to only 5 people,
1) Maria Sharapova
2) Serena Williams
3) Virat Kohli
4) Roger Federer
5) Lionel Messi

Here is the folder structure,
* UI : This contains UI website code 
* server: Python flask server
* model: Contains python notebook for model building
* google_image_scrapping: code to scrap google for images
* images_dataset: Dataset used for our model training

Technologies used in this project,
1. Python 3.8.10
2. Numpy and OpenCV for data cleaning
3. Matplotlib & Seaborn for data visualization
4. Sklearn for model building
5. Jupyter notebook, visual studio code and pycharm as IDE
6. Python flask for http server
7. HTML/CSS/JavaScript for UI

# Installation Instruction

1. Clone repository
   ```bash
   git clone 
   cd 
   ```

> **NOTE :** You can create a virtualenv before running this command. Please don't forget to activate it if do.

2. Install dependencies
   ```bash
   pip install -r requirements.txt # for server

   pip install -r requirements.dev.txt # for dev pupose
   ```
3. Run server
   ```
   cd server
   python app.py
   ```

