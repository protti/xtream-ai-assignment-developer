import unittest
from rest_api_function import app

class TestPredictPriceAPI(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True


    def test_predict_missing_fields(self):
        response = self.app.post('/predict-price/', json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn('Missing fields', response.json['message'])

    def test_predict_success(self):
        data = {
            "carat": 0.23,
            "cut": "Ideal",
            "color": "E",
            "clarity": "VS1",
            "depth": 61.5,
            "table": 55,
            "x": 3.95,
            "y": 3.98,
            "z": 2.43,
            "path": "results/models/XGBoost_Diamonds_XGBRegressor_1719607409.pkl"
        }
        response = self.app.post('/predict-price/', json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('price', response.json)

if __name__ == '__main__':
    unittest.main()
