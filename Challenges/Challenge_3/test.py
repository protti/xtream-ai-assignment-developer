import requests

BASE = "http://127.0.0.1:5000/"

response = requests.put(BASE + '/predict_value/', 
                        json={"carat": 1.0, "cut": "Ideal", 
                              "color": "G", "clarity": "VS1", 
                              "depth": 61.5, "table": 55.0, 
                              "x": 6.5, "y": 6.5, "z": 4.0, 
                              "path": "C:\\Users\\jeson\\PycharmProjects\\xtream-ai-assignment-developer\\Challenges\\Challenge_3\\results\\models\\XGBoost_Diamonds_XGBRegressor_1719517235.pkl"
                              
                              })
# Send data as JSON
print(response.json())

response = requests.put(BASE + '/closer/', 
                        json={"carat": 1.0, "cut": "Ideal", 
                              "color": "G", "clarity": "VS1", 
                              "depth": 61.5, "table": 55.0, 
                              "x": 6.5, "y": 6.5, "z": 4.0, 
                              "path": "C:\\Users\\jeson\\PycharmProjects\\xtream-ai-assignment-developer\\Challenges\\Challenge_3\\results\\models\\XGBoost_Diamonds_XGBRegressor_1719517235.pkl",
                              "n_neighbors": 5
                              })


# Send data as JSON
print(response.json())