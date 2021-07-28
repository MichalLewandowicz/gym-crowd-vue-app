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
import requests
#import json  

url = "https://gym-crowdedness-predictions.herokuapp.com/"
gym_details2 = {"day_of_week":[2],"is_weekend":[0], "temperature_celcius":[19], "is_during_semester":[1], "month":[2], "hour":[15], "day_of_month":[9] }

#json_object = json.dumps(gym_details2, indent = 4)  
#print(json_object) 

r = requests.post(url, json = gym_details2)
r.text.strip()

print(r.text.strip())