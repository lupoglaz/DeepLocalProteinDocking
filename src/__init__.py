import os
import logging

logging.getLogger().setLevel(logging.INFO) # Pass down the tree
h = logging.StreamHandler()
h.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
h.setLevel(level=logging.INFO)
# No default handler (some modules won't see logger otherwise)
logging.getLogger().addHandler(h)

REPOSITORY_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
logging.info("Autodiscovery repository dir: " + REPOSITORY_DIR)


#RUTGERS ELF CLUSTER
if 'PROJECT' in os.environ:
	print("Rutgers cluster detected")
	if os.path.exists( os.path.join(os.environ['PROJECT'], "gl445-001/DockingBenchmarkV4") ):
		logging.info("RUTGERS ELF detected")
		storage_dir = os.path.join(os.environ['PROJECT'], "gl445-001")
		
		DATA_DIR = os.path.join(os.environ['PROJECT'], "gl445-001")
		
		MODELS_DIR = os.path.join(storage_dir, "DPD_models")
		if not os.path.exists(MODELS_DIR):
			os.mkdir(MODELS_DIR)

		LOG_DIR = os.path.join(storage_dir, "DPD_experiments")
		if not os.path.exists(LOG_DIR):
			os.mkdir(LOG_DIR)
	else:
		print('PROJECT variable not found')

#MILA CLUSTER
elif os.path.exists("/data/lisa/data/Proteins"):
	logging.info("SLURM detected")
	storage_dir = "/data/milatmp1/derevyag/"
	
	DATA_DIR = '/data/lisa/data/Proteins/Docking'
	
	MODELS_DIR = os.path.join(storage_dir, "DPD_models")
	if not os.path.exists(MODELS_DIR):
		os.mkdir(MODELS_DIR)

	LOG_DIR = os.path.join(storage_dir, "DPD_experiments")
	if not os.path.exists(LOG_DIR):
		os.mkdir(LOG_DIR)

#LOCAL MACHINE
else:
	logging.info("Server detected")
	storage_dir = "/media/lupoglaz"
	DATA_DIR = os.path.join(storage_dir, "ProteinsDataset")
	
	MODELS_DIR = os.path.join(storage_dir, "DPD_models")
	if not os.path.exists(MODELS_DIR):
		os.mkdir(MODELS_DIR)

	LOG_DIR = os.path.join(storage_dir, "DPD_experiments")
	if not os.path.exists(LOG_DIR):
		os.mkdir(LOG_DIR)

RESULTS_DIR = os.path.join(REPOSITORY_DIR, "results")
if not os.path.exists(RESULTS_DIR):
	os.mkdir(RESULTS_DIR)

assert os.path.exists(DATA_DIR), "Please set up correctly paths, {} doesn't exist".format(DATA_DIR)