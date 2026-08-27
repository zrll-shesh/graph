import os
import re
import json

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = os.environ.get('MODEL_DIR', os.path.join('output', '15_saved_models'))

app = FastAPI(
    title='INFOKOM UNESA Research Profiling API',
    description='Inference service for assigning new lecturer/paper text to existing research topics and thematic clusters, using models trained in analisis_jaringan_dosen_infokom.ipynb.',
    version='1.0.0',
)

_state = {}


def _load_json(name):
    with open(os.path.join(MODEL_DIR, name)) as f:
        return json.load(f)


@app.on_event('startup')
def load_artifacts():
    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(
            f'Model directory not found at {MODEL_DIR}. Run the notebook first so it can '
            'populate output/15_saved_models, or set the MODEL_DIR environment variable.'
        )

    config = _load_json('pipeline_config.json')
    _state['config'] = config
    _state['stopwords'] = set(_load_json('stopwords.json'))
    _state['token_pattern'] = re.compile(r'[a-zA-Z]+')

    _state['count_vec'] = joblib.load(os.path.join(MODEL_DIR, 'count_vectorizer.joblib'))
    _state['tfidf_vec'] = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib'))

    topic_model_name = config['final_topic_model_name']
    _state['topic_model_name'] = topic_model_name

    if topic_model_name == 'BERTopic':
        from bertopic import BERTopic
        _state['bertopic_model'] = BERTopic.load(os.path.join(MODEL_DIR, 'bertopic_model'))
        backend_info = _load_json('embedding_backend.json')
        _state['embedding_backend'] = backend_info['backend']
        if 'sbert_model' in backend_info:
            from sentence_transformers import SentenceTransformer
            _state['sbert_model'] = SentenceTransformer(backend_info['sbert_model'])
        else:
            _state['ppmi_svd'] = joblib.load(os.path.join(MODEL_DIR, 'ppmi_svd.joblib'))
            _state['ppmi_word_vectors'] = np.load(os.path.join(MODEL_DIR, 'ppmi_word_vectors.npy'))
        remap_raw = _load_json('bertopic_topic_remap.json')
        _state['bertopic_remap'] = {int(k): v for k, v in remap_raw.items()}
    else:
        fname = f'topic_model_{topic_model_name.lower()}.joblib'
        _state['topic_model'] = joblib.load(os.path.join(MODEL_DIR, fname))

    _state['cluster_model'] = joblib.load(os.path.join(MODEL_DIR, 'cluster_assignment_knn.joblib'))

    topic_words_raw = _load_json('topic_words.json')
    _state['topic_words'] = {int(k): v for k, v in topic_words_raw.items()}
    topic_names_raw = _load_json('topic_names.json')
    _state['topic_names'] = {int(k): v for k, v in topic_names_raw.items()}
    cluster_names_raw = _load_json('cluster_names.json')
    _state['cluster_names'] = {int(k): v for k, v in cluster_names_raw.items()}

    print(f'Loaded artifacts from {MODEL_DIR}. Topic model: {topic_model_name}, k = {config["best_k"]}')


def clean_text(text):
    text = str(text).lower()
    tokens = _state['token_pattern'].findall(text)
    tokens = [t for t in tokens if len(t) > 2 and t not in _state['stopwords']]
    return ' '.join(tokens)


def predict_topic_distribution(cleaned_text):
    topic_model_name = _state['topic_model_name']
    best_k = _state['config']['best_k']

    if topic_model_name == 'BERTopic':
        if 'sbert_model' in _state:
            new_emb = _state['sbert_model'].encode([cleaned_text], show_progress_bar=False)
        else:
            tfidf_vec = _state['tfidf_vec']
            new_tfidf = tfidf_vec.transform([cleaned_text]).toarray()
            new_emb = new_tfidf @ _state['ppmi_word_vectors']
            new_emb = new_emb / (np.linalg.norm(new_emb, axis=1, keepdims=True) + 1e-9)
        topics, _ = _state['bertopic_model'].transform([cleaned_text], embeddings=new_emb)
        mapped = _state['bertopic_remap'].get(int(topics[0]), -1)
        vec = np.zeros(best_k)
        if mapped >= 0:
            vec[mapped] = 1.0
        else:
            vec[:] = 1.0 / best_k
        return vec

    if topic_model_name == 'NMF':
        new_input = _state['tfidf_vec'].transform([cleaned_text])
        raw = _state['topic_model'].transform(new_input)
    else:
        new_input = _state['count_vec'].transform([cleaned_text])
        raw = _state['topic_model'].transform(new_input)

    raw = raw[0]
    total = raw.sum()
    return raw / total if total > 0 else np.ones(best_k) / best_k


class PredictRequest(BaseModel):
    text: str = Field(..., description='Raw title + abstract + keywords of the new document.', min_length=3)


class PredictResponse(BaseModel):
    cleaned_text: str
    topic_model_used: str
    topic_distribution: dict
    dominant_topic_id: int
    dominant_topic_name: str
    dominant_topic_words: list
    predicted_cluster_id: int
    predicted_cluster_name: str


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model_dir': MODEL_DIR,
        'topic_model': _state.get('topic_model_name'),
        'embedding_backend': _state.get('embedding_backend', 'n/a (LDA/NMF do not use a sentence embedding backbone)'),
    }


@app.get('/topics')
def list_topics():
    return {
        t: {
            'name': _state['topic_names'].get(t, ''),
            'top_words': _state['topic_words'].get(t, []),
        }
        for t in sorted(_state['topic_words'].keys())
    }


@app.get('/clusters')
def list_clusters():
    return {c: {'name': name} for c, name in sorted(_state['cluster_names'].items())}


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    cleaned = clean_text(req.text)
    if len(cleaned.split()) == 0:
        raise HTTPException(status_code=400, detail='Text became empty after cleaning; provide more descriptive text.')

    topic_vec = predict_topic_distribution(cleaned)
    dominant_topic = int(np.argmax(topic_vec))
    predicted_cluster = int(_state['cluster_model'].predict(topic_vec.reshape(1, -1))[0])

    return PredictResponse(
        cleaned_text=cleaned,
        topic_model_used=_state['topic_model_name'],
        topic_distribution={str(i): round(float(v), 4) for i, v in enumerate(topic_vec)},
        dominant_topic_id=dominant_topic,
        dominant_topic_name=_state['topic_names'].get(dominant_topic, ''),
        dominant_topic_words=_state['topic_words'].get(dominant_topic, []),
        predicted_cluster_id=predicted_cluster,
        predicted_cluster_name=_state['cluster_names'].get(predicted_cluster, ''),
    )
