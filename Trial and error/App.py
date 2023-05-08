
from flask import Flask, render_template, request
import pickle
import sklearn as sklearn

app = Flask(__name__)

# Load the model from the saved file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the index route
# @app.route('/')
# def index():
#     return render_template('crop.html')

import numpy as np

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pred', methods=['GET','POST'])
def pred():
    if request.method == "POST":
        playground = float(request.form['playground'])
        auditorium = float(request.form['auditorium'])
        theatre = float(request.form['theatre'])
        library = float(request.form['library'])
        laboratory = float(request.form['laboratory'])
        health_center = float(request.form['health_center'])
        conference_hall = float(request.form['conference_hall'])
        gymnasium_fitness_center = float(request.form['gymnasium_fitness_center'])
        indoor_stadium = float(request.form['indoor_stadium'])
        common_room = float(request.form['common_room'])
        computer_center = float(request.form['computer_center'])
        cafeteria = float(request.form['cafeteria'])

        # print(nitrogen+"----------------")
        data = {
            "playground" : playground,
            "auditorium" :auditorium,
            "theatre" :theatre,
            "library" :library,
            "laboratory" :laboratory,
            "health_center" :health_center,
            "conference_hall" :conference_hall,
            "gymnasium_fitness_center" :gymnasium_fitness_center,
            "indoor_stadium" :indoor_stadium,
            "common_room" :common_room,
            "computer_center" :computer_center,
            "cafeteria" :cafeteria,
        }
        sample = np.array([[playground,auditorium,theatre,library,laboratory,health_center,conference_hall,gymnasium_fitness_center,indoor_stadium,common_room,computer_center,cafeteria]])
        prediction = model.predict(sample)

    else:
        pass
    return render_template("out.html",data=data, prediction= prediction)

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the input values from the form
#     soil_ph = float(request.form['input2'])
    
#     phosphorous = 54
#     nitrogen = 26
#     potash = 13
#     temperature = 23
#     rainfall = 53

#     # Create a 2D array of the input values
#     sample = np.array([[playground,auditorium,theatre,library,laboratory,health_center,conference_hall,gymnasium_fitness_center,indoor_stadium,common_room,computer_center,cafeteria]])

#     # Make a prediction using the loaded model
#     # Make a prediction using the loaded model
#     #sample = [soil_ph, phosphorous, nitrogen, potash, temperature, rainfall, location]
#     print(sample)
#     prediction = model.predict(sample)


#     # Return the prediction as a response
#     print(prediction)
#     print(f"The predicted crop is: {prediction[0]}")
#     return render_template('result.html', prediction= prediction)
#print("Open")
    # return f'The predicted crop for the given inputs is: {prediction[0]}'

#print("OPen")

if __name__ == '__main__':
    app.run(debug=True)

