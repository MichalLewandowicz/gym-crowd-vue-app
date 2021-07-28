# -*- coding: utf-8 -*-
"""
Software Design Masters (Artificial Intelligence), Full-Time 
Athlone Institute of Technology

author: Lee O' Connor
Student ID: A00239789
Email: a00239789@student.ait.ie

Program description
<---------------------------------------------------------->

<---------------------------------------------------------->

"""
import pickle
from flask import Flask,request, jsonify
from model_files.ml_model import predict_crowdedness



app = Flask(__name__)

@app.route("/", methods=["POST"])

def predict():
    gym_details = request.get_json()
       
     #load the model
    with open("./model_files/GymModel.bin", "rb") as f_in:
         gym_model = pickle.load(f_in)
         f_in.close()
         
    prediction = predict_crowdedness(gym_details,gym_model)

    response = {
        "Gym crowd prediction": prediction
        }
    print(f"Type:\n{type(response)}\n Response:\n{response}")
    
    return jsonify(response)



if __name__ =="__main__":
    app.run(host='0.0.0.0')