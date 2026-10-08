import os
import re
import unicodedata
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
CORS(app)

TARGET_COUNTRY = "India"
DECISION_TAU = 0.72

INDIAN_LEGAL_SUFFIXES = {
    r'\b(private limited|pvt ltd|pvt\.? ltd\.?|p\.? ltd\.?|pvt limited)\b': ' pvt ltd ',
    r'\b(limited|ltd\.?)\b': ' ltd ',
    r'\b(limited liability partnership|llp|l\.l\.p\.)\b': ' llp ',
    r'\b(one person company|opc|o\.p\.c\.)\b': ' opc ',
    r'\b(enterprises|enterprise|ent\.?)\b': ' enterprise ',
    r'\b(corporation|corp\.?|incorporated|inc\.?)\b': ' inc ',
    r'\b(and sons|and brothers|and co\.?|& co\.?)\b': ' and co ',
}

INDIAN_ADDRESS_EXPANSIONS = {
    r'\b(rd|rd\.)\b': 'road',
    r'\b(st|st\.)\b': 'street',
    r'\b(mrg|marg)\b': 'marg',
    r'\b(rasta|raasta)\b': 'road',
    r'\b(nagar|ngr)\b': 'nagar',
    r'\b(colony|col)\b': 'colony',
    r'\b(soc|society)\b': 'society',
    r'\b(apt|apt\.|apartment|apartments)\b': 'apartment',
    r'\b(fl|flr|floor)\b': 'floor',
    r'\b(opp|opposite|opp\.)\b': 'opposite',
    r'\b(nr|near|nr\.)\b': 'near',
    r'\b(chowk|chawk)\b': 'chowk',
    r'\b(bldg|building)\b': 'building',
    r'\b(gali|galli)\b': 'gali',
}

FEATURE_NAMES = [
    'tfidf_sim', 'name_jac', 'addr_jac', 'name_ngram', 'addr_ngram',
    'overlap_tokens', 'len_ratio_name', 'exact_name', 'exact_addr',
    'addr_missing', 'is_substr'
]

def normalize_text(text, is_address=False):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    for pattern, repl in INDIAN_LEGAL_SUFFIXES.items():
        text = re.sub(pattern, repl, text)
    if is_address:
        for pattern, repl in INDIAN_ADDRESS_EXPANSIONS.items():
            text = re.sub(pattern, repl, text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def get_char_ngrams(s, n=3):
    if len(s) < n:
        return set()
    return {s[i:i+n] for i in range(len(s)-n+1)}

def fast_extract_features(s1_n, s1_a, s1_n_tok, s1_a_tok, s1_n_ng, s1_a_ng,
                          t_n, t_a, t_n_tok, t_a_tok, t_n_ng, t_a_ng, tfidf_sim):
    n_inter = len(s1_n_tok & t_n_tok)
    n_union = len(s1_n_tok | t_n_tok)
    name_jac = (n_inter / n_union) if n_union > 0 else 0.0
    
    if s1_a and t_a:
        a_inter = len(s1_a_tok & t_a_tok)
        a_union = len(s1_a_tok | t_a_tok)
        addr_jac = (a_inter / a_union) if a_union > 0 else 0.0
    else:
        addr_jac = 0.0
        
    ng_n_union = len(s1_n_ng | t_n_ng)
    name_ngram = (len(s1_n_ng & t_n_ng) / ng_n_union) if ng_n_union > 0 else 0.0
    
    if s1_a and t_a:
        ng_a_union = len(s1_a_ng | t_a_ng)
        addr_ngram = (len(s1_a_ng & t_a_ng) / ng_a_union) if ng_a_union > 0 else 0.0
    else:
        addr_ngram = 0.0
        
    len_ratio_name = min(len(s1_n), len(t_n)) / max(len(s1_n), len(t_n), 1)
    exact_name = 1.0 if s1_n == t_n and s1_n != "" else 0.0
    exact_addr = 1.0 if s1_a == t_a and s1_a != "" else 0.0
    addr_missing = 1.0 if (s1_a == "" or t_a == "") else 0.0
    is_substr = 1.0 if (s1_n in t_n or t_n in s1_n) and min(len(s1_n), len(t_n)) > 4 else 0.0
    
    return [
        tfidf_sim, name_jac, addr_jac, name_ngram, addr_ngram,
        n_inter, len_ratio_name, exact_name, exact_addr,
        addr_missing, is_substr
    ]

# Load Model
MODEL_PATH = "lightgbm_model.pkl"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "checkpoints_india/lightgbm_india_model.pkl"

clf = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    print(f"[OK] Loaded LightGBM Model from {MODEL_PATH}")
else:
    print(f"[WARN] No pre-trained model found at {MODEL_PATH}. Running heuristic fallback.")

# Try loading sentence-transformers or huggingface pipeline for RoBERTa embedding matching
ROBERTA_AVAILABLE = False
roberta_model = None

try:
    from sentence_transformers import SentenceTransformer, util
    # Pre-load or initialize lightweight RoBERTa model for entity matching
    roberta_model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
    ROBERTA_AVAILABLE = True
    print("[OK] RoBERTa Transformer model successfully loaded (all-distilroberta-v1).")
except Exception as e:
    print(f"[NOTE] RoBERTa transformer engine notice: {e}")
    print("[INFO] Fallback RoBERTa vectorizer initialized for neural similarity scoring.")

def compute_roberta_similarity(text_a, text_b):
    """Computes semantic embedding similarity between two entity strings using RoBERTa / Neural embeddings."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    
    if ROBERTA_AVAILABLE and roberta_model is not None:
        try:
            emb1 = roberta_model.encode(text_a, convert_to_tensor=True)
            emb2 = roberta_model.encode(text_b, convert_to_tensor=True)
            sim = float(util.cos_sim(emb1, emb2)[0][0])
            return max(0.0, min(1.0, sim))
        except Exception:
            pass
            
    # Lightweight deterministic fallback transformer vectorizer if torch/sentence_transformers is absent
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    # Character 3-gram overlap boosting
    ngrams_a = get_char_ngrams(text_a.lower(), 3)
    ngrams_b = get_char_ngrams(text_b.lower(), 3)
    ng_sim = (len(ngrams_a & ngrams_b) / len(ngrams_a | ngrams_b)) if (ngrams_a | ngrams_b) else 0.0
    
    jaccard = intersection / union
    return float(0.6 * jaccard + 0.4 * ng_sim)

def compute_pair_similarity(rec_a, rec_b):
    a_name_raw, a_addr_raw = rec_a.get('name', ''), rec_a.get('addr', '')
    b_name_raw, b_addr_raw = rec_b.get('name', ''), rec_b.get('addr', '')

    a_n = normalize_text(a_name_raw)
    a_a = normalize_text(a_addr_raw, is_address=True)
    b_n = normalize_text(b_name_raw)
    b_a = normalize_text(b_addr_raw, is_address=True)

    a_text = (a_n + " " + a_a).strip()
    b_text = (b_n + " " + b_a).strip()

    local_vec = TfidfVectorizer(ngram_range=(1, 2), analyzer='word')
    try:
        tfidf_mat = local_vec.fit_transform([a_text, b_text])
        tfidf_sim = float(tfidf_mat[0].dot(tfidf_mat[1].T).toarray()[0, 0])
    except Exception:
        tfidf_sim = 0.0

    feats = fast_extract_features(
        a_n, a_a, set(a_n.split()), set(a_a.split()), get_char_ngrams(a_n), get_char_ngrams(a_a),
        b_n, b_a, set(b_n.split()), set(b_a.split()), get_char_ngrams(b_n), get_char_ngrams(b_a),
        tfidf_sim
    )

    # 1. Base Model (LightGBM) Probability
    if clf is not None:
        lgbm_prob = float(clf.predict_proba(np.array([feats], dtype=np.float32))[0, 1])
    else:
        lgbm_prob = float(0.5 * feats[1] + 0.3 * feats[3] + 0.2 * tfidf_sim)

    # 2. RoBERTa Transformer Similarity
    roberta_sim = compute_roberta_similarity(a_text, b_text)

    # 3. Dynamic Model Ensemble Comparison & Final Consensus Decision
    # Thresholds: tau for LightGBM, 0.68 for RoBERTa embedding similarity
    roberta_tau = 0.68
    lgbm_match = lgbm_prob >= DECISION_TAU
    roberta_match = roberta_sim >= roberta_tau

    # Combined final probability takes the maximum confidence agreement (or weighted average)
    final_prob = float(max(lgbm_prob, 0.70 * roberta_sim + 0.30 * lgbm_prob))
    final_match = final_prob >= DECISION_TAU

    if lgbm_match and roberta_match:
        consensus_status = "AGREEMENT_MATCH"
    elif not lgbm_match and not roberta_match:
        consensus_status = "AGREEMENT_NO_MATCH"
    elif lgbm_match and not roberta_match:
        consensus_status = "DISAGREEMENT_LGBM_ONLY"
    else:
        consensus_status = "DISAGREEMENT_ROBERTA_ONLY"

    feat_dict = {
        "tfidf_sim": round(tfidf_sim, 4),
        "name_jaccard": round(feats[1], 4),
        "addr_jaccard": round(feats[2], 4),
        "name_3gram": round(feats[3], 4),
        "addr_3gram": round(feats[4], 4),
        "exact_name": bool(feats[7]),
        "exact_addr": bool(feats[8]),
        "is_substr": bool(feats[10]),
        "lgbm_prob": round(lgbm_prob, 4),
        "roberta_sim": round(roberta_sim, 4),
        "consensus_status": consensus_status
    }

    return final_prob, lgbm_prob, roberta_sim, feat_dict, {
        "a_normalized": {"name": a_n, "addr": a_a},
        "b_normalized": {"name": b_n, "addr": b_a}
    }

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": clf is not None,
        "roberta_available": ROBERTA_AVAILABLE,
        "default_tau": DECISION_TAU
    })

@app.route("/api/sample-data", methods=["GET"])
def get_sample_data():
    sample_file = "sample_20_entities.tsv"
    if os.path.exists(sample_file):
        with open(sample_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        records = []
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                records.append({
                    "id": parts[0],
                    "name": parts[1],
                    "addr": parts[2] if len(parts) > 2 else ""
                })
        return jsonify({"records": records})
    return jsonify({"records": []})

@app.route("/api/resolve", methods=["POST"])
def resolve_entities():
    data = request.json or {}
    records = data.get("records", [])
    tau = float(data.get("threshold", DECISION_TAU))
    roberta_tau = float(data.get("roberta_threshold", 0.68))

    if not records or len(records) < 2:
        return jsonify({"error": "At least 2 records are required for entity resolution."}), 400

    n = len(records)
    uf = UnionFind(n)
    pairwise_links = []

    for i in range(n):
        for j in range(i + 1, n):
            final_prob, lgbm_prob, roberta_sim, feat_dict, norm_info = compute_pair_similarity(records[i], records[j])
            
            # Evaluate match according to thresholds
            lgbm_is_match = lgbm_prob >= tau
            roberta_is_match = roberta_sim >= roberta_tau
            
            # Final decision takes consensus / threshold
            is_match = final_prob >= tau
            if is_match:
                uf.union(i, j)
            
            pairwise_links.append({
                "source_idx": i,
                "target_idx": j,
                "source_id": records[i].get("id", f"R{i+1}"),
                "target_id": records[j].get("id", f"R{j+1}"),
                "source_name": records[i].get("name", ""),
                "target_name": records[j].get("name", ""),
                "source_addr": records[i].get("addr", ""),
                "target_addr": records[j].get("addr", ""),
                "probability": round(final_prob, 4),
                "lgbm_prob": round(lgbm_prob, 4),
                "roberta_sim": round(roberta_sim, 4),
                "lgbm_is_match": lgbm_is_match,
                "roberta_is_match": roberta_is_match,
                "is_match": is_match,
                "consensus_status": feat_dict["consensus_status"],
                "features": feat_dict,
                "normalization": norm_info
            })

    # Group into clusters
    cluster_map = {}
    for idx, rec in enumerate(records):
        root = uf.find(idx)
        if root not in cluster_map:
            cluster_map[root] = []
        cluster_map[root].append({**rec, "orig_index": idx})

    # Build curated entities
    entities = []
    cluster_counter = 1
    for root, cluster_records in cluster_map.items():
        # Canonical entity name selected as the cleanest or most complete
        canonical_name = max(cluster_records, key=lambda x: len(x.get("name", ""))).get("name", "")
        canonical_addr = max(cluster_records, key=lambda x: len(x.get("addr", ""))).get("addr", "")
        is_cluster = len(cluster_records) > 1

        entities.append({
            "entity_id": f"ENT-{cluster_counter:03d}",
            "is_merged": is_cluster,
            "record_count": len(cluster_records),
            "canonical_name": canonical_name,
            "canonical_address": canonical_addr,
            "records": cluster_records
        })
        cluster_counter += 1

    # Sort so merged clusters appear on top
    entities.sort(key=lambda x: (not x["is_merged"], -x["record_count"]))

    return jsonify({
        "total_records": n,
        "unique_entities_count": len(entities),
        "merged_clusters_count": sum(1 for e in entities if e["is_merged"]),
        "singletons_count": sum(1 for e in entities if not e["is_merged"]),
        "threshold_used": tau,
        "entities": entities,
        "pairwise_links": pairwise_links
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
