import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
data = pd.read_csv('/Users/rishi/BTECH/Programming/My_Projects/Music_Recommendation_System/Dataset/data.csv')
application = Flask(__name__) #Initialize the flask App
from Training import Spotify_Recommendation
@application.route('/')
def home():
    return render_template('index.html')

@application.route('/predict',methods=['POST'])
def predict():
    recommendations = Spotify_Recommendation(data)
    output = recommendations.recommend("Gati Bali", 10)
    return output['name']


if __name__ == "__main__":
    application.run(debug=True)