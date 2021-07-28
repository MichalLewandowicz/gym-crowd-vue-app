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

import pandas as pd 

def predict_crowdedness(gym_details, gym_model):
        
    gym_details2 = {"day_of_week":[2],"is_weekend":[0], "temperature_celcius":[19], "is_during_semester":[1], "month":[2], "hour":[15], "day_of_month":[9] }
    
    gym_details_df = pd.DataFrame(gym_details2)
    prediction = gym_model.predict(gym_details_df)
    return prediction[0]



