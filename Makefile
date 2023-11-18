# Makefile for envrionment generation and training

ENV_FILE = environment.yaml
ENV_NAME = EGTF_env

# Create conda environemnt with specfied packages
create-env:
	@echo "Creating conda environment..."
	conda env create -f $(ENV_FILE)

# Delete conda environment
delete-env:
	@echo "Deleting conda environment..."
	conda env remove -n $(ENV_NAME)

# Train
make train: main
	python3 main.py