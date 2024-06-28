import unittest
from resources.Observability import db, Observability
from rest_api_function import app

class TestPredictPriceAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        
        # Ensure the database is empty before each test
        with app.app_context():
            db.create_all()

    def tearDown(self):
        # Clean up / reset the database after each test
        with app.app_context():
            db.session.remove()
            db.drop_all()

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
            "path": "C:\\Users\\jeson\\PycharmProjects\\xtream-ai-assignment-developer\\Challenges\\Challenge_4\\results\\models\\XGBoost_Diamonds_XGBRegressor_optimized_1719517237.pkl"
        }
        response = self.app.post('/predict-price/', json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('price', response.json)

if __name__ == '__main__':
    unittest.main()
