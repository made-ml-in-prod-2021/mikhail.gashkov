CODE = maincode
ALL = $(CODE) tests
VENV ?= .venv

venv:
	python3.9 -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -r requirements.txt

lint:
	$(VENV)/bin/flake8 --jobs 4 --statistics --show-source $(ALL)
	$(VENV)/bin/pylint --jobs 4 $(CODE)
	$(VENV)/bin/black --skip-string-normalization --check $(ALL)

format:
	$(VENV)/bin/isort --apply --recursive $(ALL)
	$(VENV)/bin/black --skip-string-normalization $(ALL)
	$(VENV)/bin/autoflake --recursive --in-place --remove-all-unused-imports $(ALL)

build:
	docker image build .

test:
	$(VENV)/bin/pytest -v tests/