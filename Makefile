pylint:
	poetry run pylint --rcfile=pyproject.toml -j 0 model_api/

flake8:
	poetry run pflake8 model_api/

bandit:
	poetry run bandit -c pyproject.toml -r model_api/

linters: pylint flake8 bandit

build:
	docker build -t googoogoojoob/model_api .

pytest:
	poetry run pytest