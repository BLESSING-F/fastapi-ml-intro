# 🌍 Language Detection API

A machine learning web service that detects the language of any text.
Type a sentence — get back the language it's written in.

**🔗 Live demo:** https://tongue-twinkle-text.lovable.app
**⚙️ API docs:** https://YOUR-RAILWAY-ADDRESS.up.railway.app/docs

---

## How it works

```
Frontend (Lovable)  →  FastAPI (/predict)  →  ML model  →  {"language": "Italian"}
                       └──── Docker container on Railway ────┘
```

1. The user types text on the website.
2. The website sends it to the API with a `POST` request.
3. The API passes the text to a trained model and returns the predicted language.

## Tech stack

| Layer | Tool |
|---|---|
| Model | scikit-learn (CountVectorizer + Multinomial Naive Bayes) |
| API | FastAPI + Uvicorn |
| Container | Docker |
| Hosting | Railway |
| Frontend | Lovable |

## Model

- **Dataset:** [Language Detection](https://www.kaggle.com/datasets/basilb2s/language-detection) (Kaggle)
- **Approach:** bag-of-words (`CountVectorizer`) + `MultinomialNB`, wrapped in a scikit-learn `Pipeline`
- **Accuracy:** ~97.8% on the test split
- **Languages (17):** Arabic, Danish, Dutch, English, French, German, Greek, Hindi, Italian, Kannada, Malayalam, Portuguese, Russian, Spanish, Swedish, Tamil, Turkish

The trained pipeline is saved with `pickle` as `trained_pipeline-0.1.0.pkl`.

## API

### `GET /`
Health check.
```json
{"status": "OK", "version": 2}
```

### `POST /predict`
Request:
```json
{"text": "Ciao, come stai?"}
```
Response:
```json
{"language": "Italian"}
```

Example with `curl`:
```bash
curl -X POST https://YOUR-RAILWAY-ADDRESS.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Ciao, come stai?"}'
```

## Run locally

### With Python
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn main:app --reload
```
Open http://127.0.0.1:8000/docs

### With Docker
```bash
docker build -t language-api .
docker run -p 8000:8000 language-api
```
Open http://127.0.0.1:8000/docs

## Project structure

```
fastapi-ml-intro/
├── main.py                       # FastAPI app: loads the model, defines endpoints
├── trained_pipeline-0.1.0.pkl    # trained scikit-learn pipeline
├── requirements.txt              # Python dependencies
├── Dockerfile                    # instructions to build the container
├── .dockerignore
└── .gitignore
```

## Deployment

The repo is connected to **Railway**. Every push to `main` automatically rebuilds the Docker image and redeploys the API. The container reads the port from Railway's `PORT` environment variable.

## Credits

Model training adapted from [basil-b2s/Language-Detector](https://github.com/basil-b2s/Language-Detector).
