# BIA601 – GA Feature Selection (Django)

A Django-based web app to perform Genetic Algorithm (GA) feature selection with baseline comparisons (Filter/Wrapper/Embedded). Includes CSV upload, basic metadata extraction, and a simple UI scaffold.

## Quick start (Windows)

```bash
python -m venv venv
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python manage.py migrate
venv\Scripts\python manage.py runserver
```

Open http://127.0.0.1:8000 and try the Upload page.

## Roadmap
- CSV Upload and schema inference
- GA runner (population, selection, crossover, mutation, elitism)
- Baselines (Filter/Wrapper/Embedded)
- Async execution (Celery + Redis)
- Plotly visualizations
- Deployment (Django + worker + Redis)
