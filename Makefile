env:
	uv pip compile --index-url http://repository.bog.ge/artifactory/api/pypi/python/simple requirements.txt > requirements.lock.txt
	uv pip sync --system --index-url http://repository.bog.ge/artifactory/api/pypi/python/simple requirements.lock.txt

apply:
	cd feature_repo && feast apply && feast materialize 2000-01-01 2100-01-01

build-triton:
	cd triton-mlflow && rm -rf models && python import_models.py
	cd triton-mlflow && docker build -t bla .

run-triton:
	docker run -p 8000:8000 -p 8001:8001 -p 8002:8002 bla

run-mlserver:
	mlserver start public_api