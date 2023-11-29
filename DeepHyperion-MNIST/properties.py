# Make sure that any of this properties can be overridden using env.properties
import os

class MissingEnvironmentVariable(Exception):
    pass

# GA Setup
POPSIZE          = int(os.getenv('DH_POPSIZE', '800'))
NGEN             = int(os.getenv('DH_NGEN', '500000'))

RUNTIME          = int(os.getenv('DH_RUNTIME', '180'))
INTERVAL         = int(os.getenv('DH_INTERVAL', '60'))

# Mutation Hyperparameters
# range of the mutation
MUTLOWERBOUND    = float(os.getenv('DH_MUTLOWERBOUND', '0.01'))
MUTUPPERBOUND    = float(os.getenv('DH_MUTUPPERBOUND', '0.6'))

# Dataset
EXPECTED_LABEL   = int(os.getenv('DH_EXPECTED_LABEL', '5'))

#------- NOT TUNING ----------

# mutation operator probability
MUTOPPROB        = float(os.getenv('DH_MUTOPPROB', '0.5'))
MUTOFPROB        = float(os.getenv('DH_MUTOFPROB', '0.5'))


IMG_SIZE         = int(os.getenv('DH_IMG_SIZE', '28'))
num_classes      = int(os.getenv('DH_NUM_CLASSES', '10'))

INITIALPOP       = os.getenv('DH_INITIALPOP', 'seeded')

MODEL1            = os.getenv('DH_MODEL', 'models/model_mnist.h5')
MODEL2            = os.getenv('DH_MODEL', 'models/cnnClassifier.h5')

ORIGINAL_SEEDS   = os.getenv('DH_ORIGINAL_SEEDS', 'bootstraps_five')

BITMAP_THRESHOLD = float(os.getenv('DH_BITMAP_THRESHOLD', '0.5'))

DISTANCE_SEED         = float(os.getenv('DH_DISTANCE_SEED', '5.0'))
DISTANCE         = float(os.getenv('DH_DISTANCE', '2.0'))

# FEATURES             = os.getenv('FEATURES', ["Moves","Bitmaps"])
FEATURES             = os.getenv('FEATURES', ["Orientation","Bitmaps"])
# FEATURES             = os.getenv('FEATURES', ["Orientation","Moves"])
# TSHD_TYPE             = os.getenv('TSHD_TYPE', '0') # 0: no threshold
TSHD_TYPE             = os.getenv('TSHD_TYPE', '1') # 1: threshold on vectorized-rasterized seed, use DISTANCE = 2
# TSHD_TYPE             = os.getenv('TSHD_TYPE', '2') # 2: threshold on original seed, use DISTANCE = 5

RUN             = os.getenv('RUN_ID', '1') 
# try:
#     RUN = int(os.environ['RUN_ID'])
# except KeyError:
#     raise MissingEnvironmentVariable("RUN_ID does not exist. Please specify a value for this ENV variable")
# except Exception:
#     raise MissingEnvironmentVariable("Some other error?")
