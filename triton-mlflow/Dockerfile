FROM nvcr.io/nvidia/tritonserver:23.07-py3
# COPY import_models.py import_models.py
COPY models /models
CMD ["tritonserver", "--model-repository=/models"]
# CMD ["python3", "import_models.py", "&&", "tritonserver", "--model-repository=/models"]