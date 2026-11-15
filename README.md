# Heart Disease Prediction API

FastAPI service that serves predictions from a Random Forest classifier trained on the
[Heart Disease dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).
Packaged with Docker and deployed on Render.

**Live URL:** `https://<your-service>.onrender.com`  (Swagger UI: `/docs`)

> Educational project, not a medical tool. The focus is Docker + deployment, not model accuracy.

## Project structure

```
app/            FastAPI app (main.py, schemas.py)
model/          Trained model (heart_model.joblib)
data/           Put heart.csv here (git-ignored)
tests/          Basic API test
train.py        Training script
Dockerfile
docker-compose.yml
requirements.txt
```

## 1. Train the model

Download `heart.csv` from Kaggle into `data/`, then:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py
```

This writes `model/heart_model.joblib`. Commit this file, since the Docker image copies it.

## 2. Run with Docker

```bash
docker-compose build
docker-compose up
```

Open http://localhost:8000/docs

## 3. Endpoints

| Method | Path       | Description                          |
|--------|------------|--------------------------------------|
| GET    | `/health`  | Liveness check                       |
| GET    | `/info`    | Model type, feature list, accuracy   |
| POST   | `/predict` | Returns `heart_disease: true/false`  |

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

Response:

```json
{"heart_disease": true, "probability": 0.87}
```

## 4. Deploy on Render

1. Push this repo to GitHub.
2. Render dashboard → **New +** → **Web Service** → connect the repo.
3. **Language**: Docker. Leave Root Directory blank (Dockerfile is at the root).
4. Choose the Free instance type and click **Create Web Service**.
5. Set **Health Check Path** to `/health` (Settings).
6. When the build finishes, test: `curl https://<your-service>.onrender.com/health`

Render sets the `PORT` environment variable automatically, and the Dockerfile reads it.
Free instances spin down after inactivity, so the first request may take about 50 seconds.

## Run tests

```bash
pip install -r requirements-dev.txt
pytest
```
