# Heart Disease Prediction API

FastAPI service that serves predictions from a Random Forest classifier trained on the
[Heart Disease dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).
Packaged with Docker and deployed on Render.

- **GitHub repo:** https://github.com/tcoder14/heart-disease-api
- **Live deployment:** https://heart-disease-api-dczh.onrender.com
- **Swagger UI:** https://heart-disease-api-dczh.onrender.com/docs

> Educational project, not a medical tool. The focus is Docker + deployment, not model accuracy.
> The service runs on Render's free tier, so the first request after a period of inactivity can take about 50 seconds.

## Project structure

```
app/                  FastAPI app (main.py, schemas.py)
model/                Trained model (heart_model.joblib)
data/                 Put heart.csv here (git-ignored, not in the repo)
tests/                Basic API test
train.py              Training script
Dockerfile
docker-compose.yml
requirements.txt      Runtime dependencies (pinned)
requirements-dev.txt  Test dependencies
pytest.ini
```

## Endpoints

| Method | Path       | Description                                     |
|--------|------------|-------------------------------------------------|
| GET    | `/health`  | Liveness check                                  |
| GET    | `/info`    | Model type, feature list, test accuracy         |
| POST   | `/predict` | Returns `heart_disease: true/false` and a probability |

Input features for `/predict` (defined in `app/schemas.py`): `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`.
Out-of-range values return a `422` validation error.

## Try the live API

**Windows PowerShell:**
```powershell
Invoke-RestMethod https://heart-disease-api-dczh.onrender.com/health

$body = '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
Invoke-RestMethod -Uri https://heart-disease-api-dczh.onrender.com/predict -Method Post -ContentType "application/json" -Body $body
```

**Mac/Linux (curl):**
```bash
curl https://heart-disease-api-dczh.onrender.com/health

curl -X POST https://heart-disease-api-dczh.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

Example response (values will vary):
```json
{"heart_disease": true, "probability": 0.87}
```

## Run locally with Docker

```bash
docker-compose build
docker-compose up
```

Then open http://localhost:8000/docs. Stop with `Ctrl+C` and `docker-compose down`.
(If `docker-compose` is not found, use `docker compose`.)

## Train the model (optional, a trained model is already included)

Download `heart.csv` from Kaggle into `data/`, then:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py                  # writes model/heart_model.joblib
```

Use the pinned versions in `requirements.txt` when training, since a model saved with one
scikit-learn version may not load under another.

## Deploy on Render

1. Push this repo to GitHub.
2. Render dashboard: **New +** → **Web Service** → connect the repo.
3. **Language:** Docker. Leave Root Directory blank (the Dockerfile is at the project root).
4. Choose the Free instance type and click **Create Web Service**.
5. In Settings, set **Health Check Path** to `/health`.

Render sets the `PORT` environment variable automatically, and the Dockerfile's start command reads it.

## Run tests

```bash
pip install -r requirements-dev.txt
pytest
```
