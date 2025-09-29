install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint:
	flake8 refactored_analysis.py --ignore=E501,W503

test:
	pytest -v test_analysis.py
	
clean:
	rm -rf __pycache__ .pytest_cache .coverage

all: install format lint test