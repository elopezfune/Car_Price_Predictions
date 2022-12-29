from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent


# Data Path
# =========
DATA = PROJECT / "data/car_data.csv"
TRAINING = PROJECT / "data/training.json"
TESTINGS = PROJECT / "data/testings.json"


# Constants
# ========================

# Seed for reproducibility
SEED = 42

# Present year
YEAR = 2022

# Conversion rate from Indian Rupee to Euros
CONVERSION_RATE = 0.011




# Variables
# =========
ID_VAR = "Name"


# Predictors
CAT_VARS = ['Brand','Fuel', 'Seller_Type', 'Transmission', 'Owner', 'Seats']
FLOAT_VARS = ['Year', 'Kms_Driven', 'Mileage[kmpl]', 'Engine', 'Max_Power']


# Target variable
TARGET = 'Selling_Price'


# Hyperparameters boundary
# ========================
RF_BOUNDS = [(10, 300), (1, 10), (1, 10), (2, 10), (2, 10), (0, 100)]
CB_BOUNDS = [(10, 300), (1.0e-6, 1.0), (1, 10), (0, 100)]


# Model Paths
# ===========
# Regresors
RF_MODEL = PROJECT / 'models/random_forest_regresor.pkl'
CB_MODEL = PROJECT / 'models/catboost_regresor.pkl'