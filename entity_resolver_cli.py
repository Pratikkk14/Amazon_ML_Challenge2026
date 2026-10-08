#!/usr/bin/env python3
"""
==============================================================================
NLP MINI-PROJECT: MULTI-SOURCE BUSINESS ENTITY RESOLUTION (STANDALONE CLI)
==============================================================================
Modes Available:
  1. Interactive Multi-Record Cluster Resolver (DEFAULT):
     - Paste/type a list of custom entities (e.g. 5–50 records from different sources).
     - Instantly clusters and resolves which entities refer to the SAME business vs distinct singletons.
     - Zero dataset dependencies (doesn't require loading 4.7M rows from disk).
  2. Pairwise Entity Comparison (Record A vs Record B feature inspection).
  3. 3-Source Triplet Verification (Source 1 vs Source 2 vs Source 3).
  4. Full Batch Test Dataset Mode (test_source*.tsv).
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
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

TARGET_COUNTRY = "India"
DECISION_TAU = 0.72

# ------------------------------------------------------------------------------
# SECTION 1: NLP DOMAIN NORMALIZATION
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

# ------------------------------------------------------------------------------
# SECTION 2: MODEL DISCOVERY & ON-THE-FLY TF-IDF
# ------------------------------------------------------------------------------
def find_existing_model():
    candidate_paths = [
        "lightgbm_model.pkl",
        "lightgbm_india_model.pkl",
        "checkpoints_india/lightgbm_india_model.pkl",
        "checkpoints/lightgbm_model.pkl",
        "/kaggle/working/checkpoints/lightgbm_model.pkl",
        "/kaggle/working/lightgbm_model.pkl",
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p
    return None

def compute_pairwise_prob(rec_a, rec_b, clf):
    """
    Computes match probability between two records on the fly without heavy disk lookups.
    """
    a_name_raw, a_addr_raw = rec_a.get('name', ''), rec_a.get('addr', '')
    b_name_raw, b_addr_raw = rec_b.get('name', ''), rec_b.get('addr', '')

    a_n = normalize_text(a_name_raw)
    a_a = normalize_text(a_addr_raw, is_address=True)
    b_n = normalize_text(b_name_raw)
    b_a = normalize_text(b_addr_raw, is_address=True)

    a_text = (a_n + " " + a_a).strip()
    b_text = (b_n + " " + b_a).strip()

    # Local TF-IDF Vectorizer
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
    prob = float(clf.predict_proba(np.array([feats], dtype=np.float32))[0, 1])
    return prob, feats, tfidf_sim

# ------------------------------------------------------------------------------
# SECTION 3: CONNECTED COMPONENTS / ENTITY CLUSTERING (UNION-FIND)
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# SECTION 4: INTERACTIVE CLI
# ------------------------------------------------------------------------------
def run_cluster_resolver_mode(clf):
    """
    Allows user to paste multiple records (e.g. 5-30 records) and clusters them into real entities.
    """
    print("\n" + "="*70)
    print(" 🧩 MULTI-RECORD ENTITY CLUSTER RESOLVER (NO DATASET NEEDED)")
    print("="*70)
    print("Enter multiple business records to resolve.")
    print("Format per line: Source_Label | Business Name | Address (optional)")
    print("Example lines:")
    print("  S1 | Infosys Limited | Electronics City, Bangalore")
    print("  S2 | INFOSYS TECHNOLOGIES PVT LTD | Electronic City, Bengaluru")
    print("  S3 | Tata Consultancy Services | Whitefield, Bangalore")
    print("  S2 | TCS Ltd. | Whitefield Road, Bengaluru")
    print("  S1 | Reliance Retail | Nariman Point, Mumbai")
    print("="*70)
    print("Type 'DONE' or press ENTER on an empty line when finished:\n")

    records = []
    line_num = 1
    
    # Options to load records
    print("Input Options:")
    print("  • Press ENTER to type/paste records manually")
    print("  • Type a TSV filename (e.g., 'sample_20_entities.tsv')")
    user_input_choice = input("👉 Enter TSV filepath or press ENTER for manual input: ").strip()

    if user_input_choice and os.path.exists(user_input_choice):
        print(f"📂 Loading records from '{user_input_choice}'...")
        df_sample = pd.read_csv(user_input_choice, sep='\t')
        for _, row in df_sample.iterrows():
            rec_id = str(row.get('source_id', f"Rec_{line_num}"))
            rec_name = str(row.get('business_name', ''))
            rec_addr = str(row.get('business_address', '')) if pd.notna(row.get('business_address')) else ""
            records.append({"id": rec_id, "name": rec_name, "addr": rec_addr})
            line_num += 1
    else:
        print("\nEnter records line by line (Type 'DONE' when finished):")
        while True:
            line = input(f"Record {line_num}> ").strip()
            if not line or line.upper() == 'DONE':
                break
            parts = [p.strip() for p in line.split('|')]
            if len(parts) == 1:
                records.append({"id": f"Rec_{line_num}", "name": parts[0], "addr": ""})
            elif len(parts) == 2:
                # Could be (Name | Addr) or (Source | Name)
                records.append({"id": f"Rec_{line_num} ({parts[0]})", "name": parts[1], "addr": ""})
            else:
                records.append({"id": f"Rec_{line_num} ({parts[0]})", "name": parts[1], "addr": parts[2]})
            line_num += 1

    if len(records) < 2:
        print("❌ Need at least 2 records to perform entity resolution and merging.")
        return

    print(f"\n⚡ Resolving {len(records)} records across all pairs...")
    uf = UnionFind(len(records))
    pairwise_results = []

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            prob, feats, tfidf_sim = compute_pairwise_prob(records[i], records[j], clf)
            is_match = prob >= DECISION_TAU
            if is_match:
                uf.union(i, j)
            pairwise_results.append({
                'rec_a': records[i],
                'rec_b': records[j],
                'prob': prob,
                'is_match': is_match,
                'tfidf_sim': tfidf_sim
            })

    # Group records into clusters
    clusters = defaultdict(list)
    for idx, rec in enumerate(records):
        root = uf.find(idx)
        clusters[root].append(rec)

    print("\n" + "="*70)
    print(f" 🎯 RESOLUTION RESULTS: {len(clusters)} UNIQUE REAL-WORLD ENTITIES DETECTED")
    print("="*70)

    entity_idx = 1
    for root, cluster_records in clusters.items():
        if len(cluster_records) > 1:
            print(f"\n🟢 [ENTITY CLUSTER #{entity_idx}] -> MERGED {len(cluster_records)} RECORDS:")
            for r in cluster_records:
                addr_str = f" | Addr: '{r['addr']}'" if r['addr'] else ""
                print(f"   ✓ [{r['id']}] Name: '{r['name']}'{addr_str}")
        else:
            r = cluster_records[0]
            addr_str = f" | Addr: '{r['addr']}'" if r['addr'] else ""
            print(f"\n⚪ [ENTITY #{entity_idx}] -> SINGLETON (No Merges):")
            print(f"   • [{r['id']}] Name: '{r['name']}'{addr_str}")
        entity_idx += 1

    print("\n" + "-"*70)
    print(" 📊 DETAILED PAIRWISE MATCH CONFIDENCE SCORES:")
    print("-"*70)
    for res in pairwise_results:
        status = "✅ MERGE" if res['is_match'] else "❌ REJECT"
        print(f"   {status} | Prob: {res['prob']*100:.1f}% | {res['rec_a']['id']} ↔ {res['rec_b']['id']}")
        print(f"          Names: '{res['rec_a']['name']}' vs '{res['rec_b']['name']}'")
    print("="*70)


def run_pairwise_mode(clf):
    print("\n--- Direct Pairwise Entity Comparison ---")
    s1_name = input("Record 1 Name: ").strip()
    s1_addr = input("Record 1 Address: ").strip()
    t_name = input("Record 2 Name: ").strip()
    t_addr = input("Record 2 Address: ").strip()

    rec_a = {"name": s1_name, "addr": s1_addr}
    rec_b = {"name": t_name, "addr": t_addr}

    prob, feats, tfidf_sim = compute_pairwise_prob(rec_a, rec_b, clf)
    verdict = "MERGE (Same Real-World Business)" if prob >= DECISION_TAU else "DISTINCT (Different Entities)"

    print("\n" + "="*50)
    print(f"📊 Match Probability   : {prob*100:.2f}% (Decision Threshold = {DECISION_TAU*100}%)")
    print(f"🎯 Verdict             : {verdict}")
    print(f"📈 Feature Breakdown   :")
    print(f"   - TF-IDF Similarity : {tfidf_sim:.4f}")
    print(f"   - Name Token Jaccard: {feats[1]:.4f}")
    print(f"   - Addr Token Jaccard: {feats[2]:.4f}")
    print(f"   - Name 3-Gram Overlap: {feats[3]:.4f}")
    print(f"   - Exact Name Match  : {bool(feats[7])}")
    print("="*50)


def main():
    print("🚀 Initializing NLP Business Entity Resolution Engine...")
    model_path = find_existing_model()
    if not model_path:
        print("❌ No trained LightGBM model found (lightgbm_model.pkl).")
        print("Please place 'lightgbm_model.pkl' in this folder or train it first.")
        return

    print(f"🔄 Reusing trained model: {model_path}")
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    print("✅ Model loaded successfully! (Instant startup, zero dataset reading)\n")

    while True:
        print("="*70)
        print(" SELECT RESOLUTION MODE:")
        print("  [1] Multi-Record Cluster Resolver (Input 5–30 custom records to merge) [DEFAULT]")
        print("  [2] Pairwise Single Comparison (Record A vs Record B)")
        print("  [q] Quit")
        print("="*70)
        
        choice = input("👉 Option (1/2/q) [Default: 1]: ").strip().lower()
        if choice == "" or choice == "1":
            run_cluster_resolver_mode(clf)
        elif choice == "2":
            run_pairwise_mode(clf)
        elif choice == "q":
            print("Exiting. Goodbye!")
            break
        else:
            print("❌ Invalid option. Please enter 1, 2, or q.")

if __name__ == "__main__":
    main()
