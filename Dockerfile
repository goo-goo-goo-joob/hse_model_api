FROM python:3.10-slim

WORKDIR /api
RUN apt-get update && \
	apt-get install -y libpq-dev gcc && \
	pip install -U pip poetry --no-cache-dir

COPY poetry.lock pyproject.toml README.md /api/
RUN poetry config virtualenvs.create false && \
	poetry install --without dev --no-root

COPY model_api/ /api/model_api/
RUN poetry install --only-root

ENV PYTHONPATH="."
CMD ["python", "/api/model_api/__main__.py", "--port", "8866", "--host", "0.0.0.0"]