.PHONY: install data train serve dashboard test docker-build docker-up all

install:
	pip install -r requirements.txt

data:
	python data/generate_data.py

train:
	python src/train_model.py

serve:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboard/app.py

test:
	pytest tests/ -v

docker-build:
	docker-compose build

docker-up:
	docker-compose up

all: install data train serve
