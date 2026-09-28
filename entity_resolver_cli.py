#!/usr/bin/env python3
"""
==============================================================================
NLP MINI-PROJECT: MULTI-SOURCE BUSINESS ENTITY RESOLUTION (INDIA CORPUS)
==============================================================================
Modes Available:
  1. Interactive CLI Mode (DEFAULT): Test custom/live queries against Source 2 & Source 3 targets interactively.
  2. Batch Test Dataset Mode: Run full high-speed inference on test_source*.tsv.

Features:
- NLP Normalization (Indian Legal Suffixes + Address Expansions + Unicode NFKD)
- Sublinear TF-IDF + Character 3-gram Candidate Blocking
- Pre-Tokenized Fast Feature Extractor (No string splitting in inner loops)
- Calibrated LightGBM Decision Engine (Macro F_0.5 & Singleton Optimization)
- Step-by-Step Disk Checkpointing (Model, Validation Metrics, Predictions)
==============================================================================
"""

import os
import re
import gc
import sys
import json
import pickle
import warnings
import unicodedata
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# Suppress sklearn & lightgbm verbose warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

# ------------------------------------------------------------------------------
# SECTION 1: PATHS & DIRECTORY SETUP
# ------------------------------------------------------------------------------
KAGGLE_INPUT_DIR = "/kaggle/input/datasets/satwiksps/amazon-ml-challenge-2026/dataset"
KAGGLE_ALT_INPUT = "/kaggle/input/amazon-ml-challenge-2026/dataset"
COLAB_DIR = "/content/dataset"

if os.path.exists(os.path.join(COLAB_DIR, "train")):
    BASE_DIR = COLAB_DIR
    OUTPUT_DIR = "/content/output_india"
    CHECKPOINT_DIR = "/content/checkpoints_india"
elif os.path.exists(os.path.join(KAGGLE_INPUT_DIR, "train")):
    BASE_DIR = KAGGLE_INPUT_DIR
    OUTPUT_DIR = "/kaggle/working/output_india"
    CHECKPOINT_DIR = "/kaggle/working/checkpoints_india"
elif os.path.exists(os.path.join(KAGGLE_ALT_INPUT, "train")):
    BASE_DIR = KAGGLE_ALT_INPUT
    OUTPUT_DIR = "/kaggle/working/output_india"
    CHECKPOINT_DIR = "/kaggle/working/checkpoints_india"
else:
    BASE_DIR = "student_resource/dataset"
    OUTPUT_DIR = "output_india"
    CHECKPOINT_DIR = "checkpoints_india"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
TARGET_COUNTRY = "India"
DECISION_TAU = 0.72

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODEL_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "lightgbm_india_model.pkl")
METRICS_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "validation_metrics_india.json")
VECTORIZER_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "tfidf_vectorizer_india.pkl")
TARGET_POOL_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "interactive_target_pool_india.pkl")

# ------------------------------------------------------------------------------
# SECTION 2: NLP DOMAIN NORMALIZATION (INDIAN COMMERCIAL DATA)
# ------------------------------------------------------------------------------
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
    """
    Zero-redundancy feature extraction using pre-tokenized and pre-hashed set caches.
    """
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

def compute_macro_f05(preds_dict, gt_dict, all_s1_ids):
    """
    Computes official Macro F_0.5 evaluation metrics including singleton handling.
    """
    f05_scores = []
    precisions = []
    recalls = []
    singleton_correct = 0
    total_singletons = 0
    
    for s1_id in all_s1_ids:
        pred_set = set(preds_dict.get(s1_id, []))
        true_set = set(gt_dict.get(s1_id, []))
        
        if len(true_set) == 0:
            total_singletons += 1
            if len(pred_set) == 0:
                f05_scores.append(1.0)
                singleton_correct += 1
            else:
                f05_scores.append(0.0)
            continue
            
        if len(pred_set) == 0:
            f05_scores.append(0.0)
            recalls.append(0.0)
            continue
            
        tp = len(pred_set & true_set)
        p = tp / len(pred_set)
        r = tp / len(true_set)
        precisions.append(p)
        recalls.append(r)
        
        if (0.25 * p + r) > 0:
            f05 = (1.25 * p * r) / (0.25 * p + r)
        else:
            f05 = 0.0
        f05_scores.append(f05)
        
    return {
        "macro_f05": float(np.mean(f05_scores)),
        "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
        "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
        "singleton_accuracy": float(singleton_correct / total_singletons) if total_singletons > 0 else 1.0,
        "total_eval_entities": len(all_s1_ids),
        "total_singletons": total_singletons
    }

# ------------------------------------------------------------------------------
# SECTION 3: MODEL TRAINING / CHECKPOINT MANAGER
# ------------------------------------------------------------------------------
def load_or_train_model():
    """
    Loads saved model & TF-IDF artifacts or trains a new LightGBM classifier with offline validation.
    """
    if os.path.exists(MODEL_CHECKPOINT) and os.path.exists(METRICS_CHECKPOINT) and os.path.exists(VECTORIZER_CHECKPOINT):
        print(f"🔄 Loaded existing model checkpoint from: {MODEL_CHECKPOINT}")
        with open(MODEL_CHECKPOINT, 'rb') as f:
            clf = pickle.load(f)
        with open(VECTORIZER_CHECKPOINT, 'rb') as f:
            vec = pickle.load(f)
        with open(METRICS_CHECKPOINT, 'r') as f:
            metrics = json.load(f)
        print(f"📊 Validated Performance (India Split): Macro F_0.5 = {metrics['macro_f05']:.4f} | Precision = {metrics['mean_precision']:.4f} | Recall = {metrics['mean_recall']:.4f}")
        return clf, vec

    print("\n" + "="*70)
    print(" STEP 1: TRAINING LIGHTGBM ENTITY RESOLVER (INDIA DATASET)")
    print("="*70)
    
    print("Loading Indian training reference records...")
    s1_list = []
    for chunk in pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t", chunksize=100000):
        sub = chunk[chunk['country'] == TARGET_COUNTRY]
        if len(sub) > 0:
            s1_list.append(sub)
        if sum(len(x) for x in s1_list) >= 60000:
            break
    df_s1_tr = pd.concat(s1_list, ignore_index=True).iloc[:60000]
    del s1_list
    
    tgt_list = []
    for fname in ["train_source2.tsv", "train_source3.tsv"]:
        for chunk in pd.read_csv(os.path.join(TRAIN_DIR, fname), sep="\t", chunksize=100000):
            sub = chunk[chunk['country'] == TARGET_COUNTRY]
            if len(sub) > 0:
                tgt_list.append(sub)
            if sum(len(x) for x in tgt_list) >= 120000:
                break
    df_tgt_tr = pd.concat(tgt_list, ignore_index=True).iloc[:120000]
    del tgt_list
    
    df_gt = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep="\t")
    gt_map = {}
    for _, row in df_gt.iterrows():
        s1_id = row['source1_entity_id']
        matches = set(str(row['matched_entity_ids']).split(',')) if pd.notna(row['matched_entity_ids']) and str(row['matched_entity_ids']).strip() else set()
        gt_map[s1_id] = matches
    del df_gt
    gc.collect()

    s1_names_tr = df_s1_tr['business_name'].apply(normalize_text).tolist()
    s1_addrs_tr = df_s1_tr['business_address'].apply(lambda x: normalize_text(x, is_address=True)).tolist()
    s1_ids_tr = df_s1_tr['entity_id'].tolist()
    
    tgt_names_tr = df_tgt_tr['business_name'].apply(normalize_text).tolist()
    tgt_addrs_tr = df_tgt_tr['business_address'].apply(lambda x: normalize_text(x, is_address=True)).tolist()
    tgt_ids_tr = df_tgt_tr['entity_id'].tolist()
    del df_s1_tr, df_tgt_tr
    gc.collect()

    # Pre-tokenize & N-gram caching
    s1_n_tok_tr = [set(x.split()) for x in s1_names_tr]
    s1_a_tok_tr = [set(x.split()) for x in s1_addrs_tr]
    s1_n_ng_tr = [get_char_ngrams(x) for x in s1_names_tr]
    s1_a_ng_tr = [get_char_ngrams(x) for x in s1_addrs_tr]
    
    tgt_n_tok_tr = [set(x.split()) for x in tgt_names_tr]
    tgt_a_tok_tr = [set(x.split()) for x in tgt_addrs_tr]
    tgt_n_ng_tr = [get_char_ngrams(x) for x in tgt_names_tr]
    tgt_a_ng_tr = [get_char_ngrams(x) for x in tgt_addrs_tr]

    tgt_texts = [tgt_names_tr[i] + " " + tgt_addrs_tr[i] for i in range(len(tgt_ids_tr))]
    s1_texts = [s1_names_tr[i] + " " + s1_addrs_tr[i] for i in range(len(s1_ids_tr))]

    print("Fitting sublinear TF-IDF blocking matrix...")
    vec = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', max_features=70000, sublinear_tf=True)
    tgt_mat = vec.fit_transform(tgt_texts)
    s1_mat = vec.transform(s1_texts)
    del tgt_texts, s1_texts
    gc.collect()

    # 85% Train / 15% Validation Split
    train_idx, val_idx = train_test_split(np.arange(len(s1_ids_tr)), test_size=0.15, random_state=42)
    val_s1_ids = [s1_ids_tr[idx] for idx in val_idx]
    
    X_train, y_train = [], []
    batch_size = 1000
    
    print("Extracting feature matrices for training set...")
    for start in range(0, len(train_idx), batch_size):
        curr_batch_idx = train_idx[start:start+batch_size]
        sub_batch_s1 = s1_mat[curr_batch_idx]
        batch_sims = sub_batch_s1.dot(tgt_mat.T).toarray()
        
        for r_idx, s1_idx in enumerate(curr_batch_idx):
            cur_s1 = s1_ids_tr[s1_idx]
            sim_row = batch_sims[r_idx]
            top_k = min(30, len(tgt_ids_tr))
            top_indices = np.argpartition(sim_row, -top_k)[-top_k:]
            true_set = gt_map.get(cur_s1, set())
            
            for idx in top_indices:
                sim_val = float(sim_row[idx])
                if sim_val >= 0.05:
                    tid = tgt_ids_tr[idx]
                    feats = fast_extract_features(
                        s1_names_tr[s1_idx], s1_addrs_tr[s1_idx],
                        s1_n_tok_tr[s1_idx], s1_a_tok_tr[s1_idx],
                        s1_n_ng_tr[s1_idx], s1_a_ng_tr[s1_idx],
                        tgt_names_tr[idx], tgt_addrs_tr[idx],
                        tgt_n_tok_tr[idx], tgt_a_tok_tr[idx],
                        tgt_n_ng_tr[idx], tgt_a_ng_tr[idx],
                        sim_val
                    )
                    X_train.append(feats)
                    y_train.append(1 if tid in true_set else 0)
        del batch_sims

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"📊 Training Matrix: {X_train.shape[0]:,} candidate pairs (Positives: {np.mean(y_train)*100:.2f}%)")

    clf = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.07,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)

    # Save model and vectorizer
    with open(MODEL_CHECKPOINT, 'wb') as f:
        pickle.dump(clf, f)
    with open(VECTORIZER_CHECKPOINT, 'wb') as f:
        pickle.dump(vec, f)

    # Validation Evaluation
    print("Evaluating model performance on held-out validation split...")
    val_preds_dict = defaultdict(list)
    for start in range(0, len(val_idx), batch_size):
        curr_batch_idx = val_idx[start:start+batch_size]
        sub_batch_s1 = s1_mat[curr_batch_idx]
        batch_sims = sub_batch_s1.dot(tgt_mat.T).toarray()
        
        pair_feats, pair_recs = [], []
        for r_idx, s1_idx in enumerate(curr_batch_idx):
            cur_s1 = s1_ids_tr[s1_idx]
            sim_row = batch_sims[r_idx]
            top_k = min(30, len(tgt_ids_tr))
            top_indices = np.argpartition(sim_row, -top_k)[-top_k:]
            
            for idx in top_indices:
                sim_val = float(sim_row[idx])
                if sim_val >= 0.05:
                    tid = tgt_ids_tr[idx]
                    feats = fast_extract_features(
                        s1_names_tr[s1_idx], s1_addrs_tr[s1_idx],
                        s1_n_tok_tr[s1_idx], s1_a_tok_tr[s1_idx],
                        s1_n_ng_tr[s1_idx], s1_a_ng_tr[s1_idx],
                        tgt_names_tr[idx], tgt_addrs_tr[idx],
                        tgt_n_tok_tr[idx], tgt_a_tok_tr[idx],
                        tgt_n_ng_tr[idx], tgt_a_ng_tr[idx],
                        sim_val
                    )
                    pair_feats.append(feats)
                    pair_recs.append((cur_s1, tid))
                    
        if len(pair_feats) > 0:
            arr_val_feats = np.array(pair_feats, dtype=np.float32)
            probs = clf.predict_proba(arr_val_feats)[:, 1]
            for (s1_elem, tid), prob in zip(pair_recs, probs):
                if prob >= DECISION_TAU:
                    val_preds_dict[s1_elem].append(tid)
        del batch_sims

    val_metrics = compute_macro_f05(val_preds_dict, gt_map, val_s1_ids)
    with open(METRICS_CHECKPOINT, 'w', encoding='utf-8') as f:
        json.dump(val_metrics, f, indent=4)
        
    print(f"✅ Saved trained model to: {MODEL_CHECKPOINT}")
    print(f"💾 Saved metrics report to: {METRICS_CHECKPOINT}")
    print(f"📊 Validation Summary:")
    print(f"   - Macro F_0.5 Score   : {val_metrics['macro_f05']:.4f}")
    print(f"   - Precision (Matches) : {val_metrics['mean_precision']:.4f}")
    print(f"   - Recall (Matches)    : {val_metrics['mean_recall']:.4f}")
    print(f"   - Singleton Accuracy  : {val_metrics['singleton_accuracy']:.4f}")

    del tgt_mat, s1_mat, s1_names_tr, s1_addrs_tr, s1_ids_tr, tgt_names_tr, tgt_addrs_tr, tgt_ids_tr
    del s1_n_tok_tr, s1_a_tok_tr, s1_n_ng_tr, s1_a_ng_tr, tgt_n_tok_tr, tgt_a_tok_tr, tgt_n_ng_tr, tgt_a_ng_tr
    del X_train, y_train
    gc.collect()

    return clf, vec

# ------------------------------------------------------------------------------
# SECTION 4: TARGET CORPOUS BUILDER (FOR INTERACTIVE SEARCH)
# ------------------------------------------------------------------------------
def load_or_build_interactive_target_pool(vec):
    """
    Loads target pool for fast interactive queries against Source 2 and Source 3 entities.
    """
    if os.path.exists(TARGET_POOL_CHECKPOINT):
        with open(TARGET_POOL_CHECKPOINT, 'rb') as f:
            target_data = pickle.load(f)
        return target_data

    print("\n⏳ Building Indexed Target Corpus (Source 2 & Source 3) for Interactive Testing...")
    tgt_rows = []
    # Load from test set if available, else from train set
    data_source_dir = TEST_DIR if os.path.exists(TEST_DIR) else TRAIN_DIR
    prefix = "test_" if data_source_dir == TEST_DIR else "train_"
    
    for fname in [f"{prefix}source2.tsv", f"{prefix}source3.tsv"]:
        fpath = os.path.join(data_source_dir, fname)
        if os.path.exists(fpath):
            for chunk in pd.read_csv(fpath, sep="\t", chunksize=100000):
                sub = chunk[chunk['country'] == TARGET_COUNTRY]
                if len(sub) > 0:
                    tgt_rows.append(sub)
    
    df_tgt = pd.concat(tgt_rows, ignore_index=True) if tgt_rows else pd.DataFrame(columns=['entity_id', 'business_name', 'business_address'])
    print(f"📦 Total Indexed Target Records: {len(df_tgt):,}")

    raw_names = df_tgt['business_name'].fillna("").tolist()
    raw_addrs = df_tgt['business_address'].fillna("").tolist()
    tgt_ids = df_tgt['entity_id'].tolist()
    del df_tgt, tgt_rows
    gc.collect()

    norm_names = [normalize_text(x) for x in raw_names]
    norm_addrs = [normalize_text(x, is_address=True) for x in raw_addrs]
    
    tgt_n_tok = [set(x.split()) for x in norm_names]
    tgt_a_tok = [set(x.split()) for x in norm_addrs]
    tgt_n_ng = [get_char_ngrams(x) for x in norm_names]
    tgt_a_ng = [get_char_ngrams(x) for x in norm_addrs]

    tgt_texts = [norm_names[i] + " " + norm_addrs[i] for i in range(len(tgt_ids))]
    tgt_mat = vec.transform(tgt_texts)
    del tgt_texts
    gc.collect()

    target_data = {
        'tgt_ids': tgt_ids,
        'raw_names': raw_names,
        'raw_addrs': raw_addrs,
        'norm_names': norm_names,
        'norm_addrs': norm_addrs,
        'tgt_n_tok': tgt_n_tok,
        'tgt_a_tok': tgt_a_tok,
        'tgt_n_ng': tgt_n_ng,
        'tgt_a_ng': tgt_a_ng,
        'tgt_mat': tgt_mat
    }

    with open(TARGET_POOL_CHECKPOINT, 'wb') as f:
        pickle.dump(target_data, f)
    print(f"💾 Cached Interactive Target Pool to: {TARGET_POOL_CHECKPOINT}")
    return target_data

# ------------------------------------------------------------------------------
# SECTION 5: INTERACTIVE CLI MODE (DEFAULT)
# ------------------------------------------------------------------------------
def run_interactive_mode(clf, vec):
    """
    Provides an interactive CLI interface for querying single records or 3-source comparisons.
    """
    target_data = load_or_build_interactive_target_pool(vec)
    tgt_ids = target_data['tgt_ids']
    raw_names = target_data['raw_names']
    raw_addrs = target_data['raw_addrs']
    norm_names = target_data['norm_names']
    norm_addrs = target_data['norm_addrs']
    tgt_n_tok = target_data['tgt_n_tok']
    tgt_a_tok = target_data['tgt_a_tok']
    tgt_n_ng = target_data['tgt_n_ng']
    tgt_a_ng = target_data['tgt_a_ng']
    tgt_mat = target_data['tgt_mat']

    print("\n" + "="*70)
    print(" 🔍 NLP MINI-PROJECT: INTERACTIVE ENTITY RESOLUTION CLI (INDIA)")
    print("="*70)
    print("Options:")
    print("  [1] Query a Reference Entity (Source 1) against the full Target Pool (Source 2 & 3)")
    print("  [2] Pairwise Comparison (Source 1 vs Manual Source 2/Source 3 record)")
    print("  [3] 3-Source Triplet Verification (Source 1 vs Source 2 vs Source 3)")
    print("  [4] Switch to Batch Test Dataset Resolution")
    print("  [q] Quit")
    print("="*70)

    while True:
        choice = input("\n👉 Select an Option (1/2/3/4/q) [Default: 1]: ").strip().lower()
        if choice == "" or choice == "1":
            print("\n--- [Option 1] Query Reference Entity against Target Database ---")
            s1_name = input("Enter Source 1 Business Name (e.g., 'Infosys Technologies Pvt Ltd'): ").strip()
            if not s1_name:
                print("❌ Business Name cannot be empty.")
                continue
            s1_addr = input("Enter Source 1 Address (optional, e.g., 'Electronics City, Bangalore'): ").strip()

            s1_n = normalize_text(s1_name)
            s1_a = normalize_text(s1_addr, is_address=True)
            s1_nt = set(s1_n.split())
            s1_at = set(s1_a.split())
            s1_nng = get_char_ngrams(s1_n)
            s1_ang = get_char_ngrams(s1_a)

            s1_text = s1_n + " " + s1_a
            s1_vec = vec.transform([s1_text])
            sim_row = s1_vec.dot(tgt_mat.T).toarray()[0]

            top_k = min(30, len(tgt_ids))
            top_indices = np.argpartition(sim_row, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(-sim_row[top_indices])]

            candidates = []
            matches = []

            for idx in top_indices:
                sim_val = float(sim_row[idx])
                if sim_val >= 0.05:
                    tid = tgt_ids[idx]
                    feats = fast_extract_features(
                        s1_n, s1_a, s1_nt, s1_at, s1_nng, s1_ang,
                        norm_names[idx], norm_addrs[idx],
                        tgt_n_tok[idx], tgt_a_tok[idx],
                        tgt_n_ng[idx], tgt_a_ng[idx],
                        sim_val
                    )
                    prob = float(clf.predict_proba(np.array([feats], dtype=np.float32))[0, 1])
                    cand_info = {
                        'id': tid,
                        'name': raw_names[idx],
                        'addr': raw_addrs[idx],
                        'tfidf_sim': sim_val,
                        'prob': prob,
                        'is_match': prob >= DECISION_TAU
                    }
                    candidates.append(cand_info)
                    if prob >= DECISION_TAU:
                        matches.append(cand_info)

            print(f"\n📊 RESULTS FOR: '{s1_name}' | '{s1_addr}'")
            print(f"🎯 Total Candidates Retrieved: {len(candidates)} | Confirmed Matches (τ ≥ {DECISION_TAU}): {len(matches)}")
            
            if len(matches) > 0:
                print("\n✅ MATCHED BUSINESS ENTITIES:")
                for m in matches:
                    print(f"   🟢 [{m['id']}] Prob: {m['prob']*100:.1f}% (TF-IDF: {m['tfidf_sim']:.3f})")
                    print(f"      Name: {m['name']}")
                    print(f"      Addr: {m['addr']}")
            else:
                print("\n⚪ (No matches above decision threshold. Classified as SINGLETON entity).")

            if len(candidates) > 0:
                print(f"\n🔍 Top 5 Candidates in Blocking Pool:")
                for c in candidates[:5]:
                    status = "✅ MATCH" if c['is_match'] else "❌ REJECT"
                    print(f"   • [{c['id']}] {status} | Score: {c['prob']:.4f} | Name: {c['name']} | Addr: {c['addr']}")

        elif choice == "2":
            print("\n--- [Option 2] Direct Pairwise Entity Comparison ---")
            s1_name = input("Record 1 (Source 1) Name: ").strip()
            s1_addr = input("Record 1 Address: ").strip()
            t_name = input("Record 2 (Source 2/3) Name: ").strip()
            t_addr = input("Record 2 Address: ").strip()

            s1_n, s1_a = normalize_text(s1_name), normalize_text(s1_addr, is_address=True)
            t_n, t_a = normalize_text(t_name), normalize_text(t_addr, is_address=True)
            
            v1 = vec.transform([s1_n + " " + s1_a])
            v2 = vec.transform([t_n + " " + t_a])
            tfidf_sim = float(v1.dot(v2.T).toarray()[0, 0])

            feats = fast_extract_features(
                s1_n, s1_a, set(s1_n.split()), set(s1_a.split()), get_char_ngrams(s1_n), get_char_ngrams(s1_a),
                t_n, t_a, set(t_n.split()), set(t_a.split()), get_char_ngrams(t_n), get_char_ngrams(t_a),
                tfidf_sim
            )
            prob = float(clf.predict_proba(np.array([feats], dtype=np.float32))[0, 1])
            verdict = "MATCH (Same Entity)" if prob >= DECISION_TAU else "NON-MATCH (Different Entities)"

            print("\n" + "-"*50)
            print(f"📊 Match Probability   : {prob*100:.2f}% (Threshold = {DECISION_TAU*100}%)")
            print(f"🎯 Verdict             : {verdict}")
            print(f"📈 Feature Breakdown   :")
            print(f"   - TF-IDF Similarity : {tfidf_sim:.4f}")
            print(f"   - Name Jaccard      : {feats[1]:.4f}")
            print(f"   - Address Jaccard   : {feats[2]:.4f}")
            print(f"   - Name 3-Gram Overlap: {feats[3]:.4f}")
            print(f"   - Exact Name Match  : {bool(feats[7])}")
            print("-"*50)

        elif choice == "3":
            print("\n--- [Option 3] 3-Source Triplet Verification ---")
            s1_name = input("Source 1 (Reference) Name   : ").strip()
            s1_addr = input("Source 1 Address            : ").strip()
            s2_name = input("Source 2 Business Name      : ").strip()
            s2_addr = input("Source 2 Address            : ").strip()
            s3_name = input("Source 3 Business Name      : ").strip()
            s3_addr = input("Source 3 Address            : ").strip()

            s1_n, s1_a = normalize_text(s1_name), normalize_text(s1_addr, is_address=True)
            s1_tok, s1_atok = set(s1_n.split()), set(s1_a.split())
            s1_ng, s1_ang = get_char_ngrams(s1_n), get_char_ngrams(s1_a)
            s1_v = vec.transform([s1_n + " " + s1_a])

            # Check S1 vs S2
            s2_n, s2_a = normalize_text(s2_name), normalize_text(s2_addr, is_address=True)
            s2_v = vec.transform([s2_n + " " + s2_a])
            sim_s1_s2 = float(s1_v.dot(s2_v.T).toarray()[0, 0])
            feats_s2 = fast_extract_features(
                s1_n, s1_a, s1_tok, s1_atok, s1_ng, s1_ang,
                s2_n, s2_a, set(s2_n.split()), set(s2_a.split()), get_char_ngrams(s2_n), get_char_ngrams(s2_a),
                sim_s1_s2
            )
            prob_s2 = float(clf.predict_proba(np.array([feats_s2], dtype=np.float32))[0, 1])

            # Check S1 vs S3
            s3_n, s3_a = normalize_text(s3_name), normalize_text(s3_addr, is_address=True)
            s3_v = vec.transform([s3_n + " " + s3_a])
            sim_s1_s3 = float(s1_v.dot(s3_v.T).toarray()[0, 0])
            feats_s3 = fast_extract_features(
                s1_n, s1_a, s1_tok, s1_atok, s1_ng, s1_ang,
                s3_n, s3_a, set(s3_n.split()), set(s3_a.split()), get_char_ngrams(s3_n), get_char_ngrams(s3_a),
                sim_s1_s3
            )
            prob_s3 = float(clf.predict_proba(np.array([feats_s3], dtype=np.float32))[0, 1])

            print("\n" + "="*60)
            print(" 3-SOURCE LINKAGE EVALUATION SUMMARY")
            print("="*60)
            print(f"1. Source 1 ↔ Source 2 : Prob = {prob_s2*100:.1f}% -> {'✅ MATCH' if prob_s2 >= DECISION_TAU else '❌ NON-MATCH'}")
            print(f"2. Source 1 ↔ Source 3 : Prob = {prob_s3*100:.1f}% -> {'✅ MATCH' if prob_s3 >= DECISION_TAU else '❌ NON-MATCH'}")
            
            resolved_cluster = [s1_name]
            if prob_s2 >= DECISION_TAU:
                resolved_cluster.append(f"Source 2 ({s2_name})")
            if prob_s3 >= DECISION_TAU:
                resolved_cluster.append(f"Source 3 ({s3_name})")

            print(f"\n🏷️ Resolved Entity Cluster: {' + '.join(resolved_cluster)}")
            print("="*60)

        elif choice == "4":
            print("\nSwitching to Full Batch Test Dataset Mode...")
            run_batch_test_mode(clf, vec)
            break

        elif choice == "q":
            print("\nExiting Interactive CLI. Goodbye!")
            sys.exit(0)
        else:
            print("❌ Invalid selection. Please enter 1, 2, 3, 4, or q.")

# ------------------------------------------------------------------------------
# SECTION 6: BATCH TEST DATASET INFERENCE MODE
# ------------------------------------------------------------------------------
def run_batch_test_mode(clf, vec):
    """
    High-speed resolution across all records in test_source*.tsv for India.
    """
    print("\n" + "="*70)
    print(" STEP 2: HIGH-SPEED BATCH INFERENCE (INDIA TEST DATASET)")
    print("="*70)
    INDIA_CKPT = os.path.join(CHECKPOINT_DIR, "preds_India.pkl")

    if os.path.exists(INDIA_CKPT):
        print(f"🔄 Loaded existing test predictions checkpoint from {INDIA_CKPT}")
        with open(INDIA_CKPT, 'rb') as f:
            matching_dict, candidate_dict = pickle.load(f)
    else:
        print("Loading Indian test records...")
        s1_rows, tgt_rows = [], []
        
        for chunk in pd.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), sep="\t", chunksize=100000):
            sub = chunk[chunk['country'] == TARGET_COUNTRY]
            if len(sub) > 0:
                s1_rows.append(sub)
        df_s1_te = pd.concat(s1_rows, ignore_index=True) if s1_rows else pd.DataFrame()
        del s1_rows
        gc.collect()
        
        for fname in ["test_source2.tsv", "test_source3.tsv"]:
            for chunk in pd.read_csv(os.path.join(TEST_DIR, fname), sep="\t", chunksize=100000):
                sub = chunk[chunk['country'] == TARGET_COUNTRY]
                if len(sub) > 0:
                    tgt_rows.append(sub)
        df_tgt_te = pd.concat(tgt_rows, ignore_index=True) if tgt_rows else pd.DataFrame()
        del tgt_rows
        gc.collect()
        
        print(f"🇮🇳 India Test Dataset: {len(df_s1_te):,} Reference S1 | {len(df_tgt_te):,} Targets")
        
        matching_dict = defaultdict(list)
        candidate_dict = defaultdict(list)
        
        s1_names = df_s1_te['business_name'].apply(normalize_text).tolist()
        s1_addrs = df_s1_te['business_address'].apply(lambda x: normalize_text(x, is_address=True)).tolist()
        s1_ids = df_s1_te['entity_id'].tolist()
        del df_s1_te
        gc.collect()
        
        tgt_names = df_tgt_te['business_name'].apply(normalize_text).tolist()
        tgt_addrs = df_tgt_te['business_address'].apply(lambda x: normalize_text(x, is_address=True)).tolist()
        tgt_ids = df_tgt_te['entity_id'].tolist()
        del df_tgt_te
        gc.collect()
        
        print("Pre-tokenizing target corpora for accelerated similarity lookups...")
        tgt_n_tok = [set(x.split()) for x in tgt_names]
        tgt_a_tok = [set(x.split()) for x in tgt_addrs]
        tgt_n_ng = [get_char_ngrams(x) for x in tgt_names]
        tgt_a_ng = [get_char_ngrams(x) for x in tgt_addrs]
        
        s1_n_tok = [set(x.split()) for x in s1_names]
        s1_a_tok = [set(x.split()) for x in s1_addrs]
        s1_n_ng = [get_char_ngrams(x) for x in s1_names]
        s1_a_ng = [get_char_ngrams(x) for x in s1_addrs]
        
        tgt_texts = [tgt_names[i] + " " + tgt_addrs[i] for i in range(len(tgt_ids))]
        s1_texts = [s1_names[i] + " " + s1_addrs[i] for i in range(len(s1_ids))]
        
        print("Building sublinear TF-IDF inverted indices...")
        tgt_mat = vec.transform(tgt_texts)
        del tgt_texts
        gc.collect()
        
        s1_mat = vec.transform(s1_texts)
        del s1_texts
        gc.collect()
        
        print("🚀 Starting parallel candidate blocking & LightGBM scoring...")
        batch_size = 2000
        
        for i in range(0, len(s1_ids), batch_size):
            if i % 25000 == 0:
                print(f"  ⚡ [India] Speed Progress: {i:,}/{len(s1_ids):,} entities completed ({i/len(s1_ids)*100:.1f}%)...")
                
            sub_batch_s1 = s1_mat[i:i+batch_size]
            batch_sims = sub_batch_s1.dot(tgt_mat.T).toarray()
            
            pair_feats_batch = []
            pair_records = []
            
            for r_idx in range(len(batch_sims)):
                s1_idx = i + r_idx
                cur_s1 = s1_ids[s1_idx]
                s1_n = s1_names[s1_idx]
                s1_a = s1_addrs[s1_idx]
                s1_nt = s1_n_tok[s1_idx]
                s1_at = s1_a_tok[s1_idx]
                s1_nng = s1_n_ng[s1_idx]
                s1_ang = s1_a_ng[s1_idx]
                
                sim_row = batch_sims[r_idx]
                top_k = min(30, len(tgt_ids))
                top_indices = np.argpartition(sim_row, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(-sim_row[top_indices])]
                
                for idx in top_indices:
                    sim_val = float(sim_row[idx])
                    if sim_val >= 0.05:
                        tid = tgt_ids[idx]
                        candidate_dict[cur_s1].append(tid)
                        feats = fast_extract_features(
                            s1_n, s1_a, s1_nt, s1_at, s1_nng, s1_ang,
                            tgt_names[idx], tgt_addrs[idx],
                            tgt_n_tok[idx], tgt_a_tok[idx],
                            tgt_n_ng[idx], tgt_a_ng[idx],
                            sim_val
                        )
                        pair_feats_batch.append(feats)
                        pair_records.append((cur_s1, tid))
                        
            if len(pair_feats_batch) > 0:
                arr_feats = np.array(pair_feats_batch, dtype=np.float32)
                probs = clf.predict_proba(arr_feats)[:, 1]
                for (s1_elem, tid), prob in zip(pair_records, probs):
                    if prob >= DECISION_TAU:
                        matching_dict[s1_elem].append(tid)
                        
            del batch_sims
            
        del tgt_mat, s1_mat, s1_names, s1_addrs, tgt_names, tgt_addrs, tgt_ids
        del s1_n_tok, s1_a_tok, s1_n_ng, s1_a_ng, tgt_n_tok, tgt_a_tok, tgt_n_ng, tgt_a_ng
        gc.collect()
        
        with open(INDIA_CKPT, 'wb') as f:
            pickle.dump((dict(matching_dict), dict(candidate_dict)), f)
        print(f"💾 Step 2 Saved: Saved India predictions checkpoint to {INDIA_CKPT}")

    print("\n" + "="*70)
    print(" STEP 3: EXPORTING FINAL TSV SUBMISSION FILES")
    print("="*70)
    s1_india_full = []
    for chunk in pd.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), sep="\t", chunksize=100000):
        sub = chunk[chunk['country'] == TARGET_COUNTRY]
        if len(sub) > 0:
            s1_india_full.extend(sub['entity_id'].tolist())

    matching_tsv_path = os.path.join(OUTPUT_DIR, "matching_results_india.tsv")
    candidate_tsv_path = os.path.join(OUTPUT_DIR, "candidate_pairs_india.tsv")

    with open(matching_tsv_path, 'w', encoding='utf-8') as f_match, \
         open(candidate_tsv_path, 'w', encoding='utf-8') as f_cand:
        
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for s1_id in s1_india_full:
            cands = list(dict.fromkeys(candidate_dict.get(s1_id, [])))
            matches = list(dict.fromkeys(matching_dict.get(s1_id, [])))
            
            cand_set = set(cands)
            matches = [m for m in matches if m in cand_set]
            
            f_match.write(f"{s1_id}\t{','.join(matches)}\n")
            f_cand.write(f"{s1_id}\t{','.join(cands)}\n")

    print(f"\n🎉 SUCCESS! Generated outputs for India ({len(s1_india_full):,} entities):")
    print(f"   1. Matches TSV   : {matching_tsv_path}")
    print(f"   2. Candidate TSV : {candidate_tsv_path}")

# ------------------------------------------------------------------------------
# MAIN EXECUTION DISPATCHER
# ------------------------------------------------------------------------------
def main():
    print("🚀 Initializing NLP Business Entity Resolution Engine (India Partition)...")
    clf, vec = load_or_train_model()
    
    # Check CLI arguments: default is interactive mode
    if len(sys.argv) > 1 and sys.argv[1].lower() in ["--batch", "-b", "--test"]:
        run_batch_test_mode(clf, vec)
    else:
        run_interactive_mode(clf, vec)

if __name__ == "__main__":
    main()
