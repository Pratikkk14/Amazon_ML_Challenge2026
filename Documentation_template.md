# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Tensor
**Team Members:** Siddharth Ravindra Metkari, Pratik Pujari, Jyotsna Kasibhotla
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We developed a large-scale business entity resolution pipeline for
linking Source 1 business records to matching records in Sources 2 and
3. The solution uses text normalization and multi-stage blocking to
reduce the comparison space, followed by an XGBoost pair classifier.

The final candidate-generation stage produced 32,365,428 candidate
pairs for 1,732,544 Source 1 test entities. The final XGBoost matcher
used a probability threshold of 0.82.

---

## 2. Methodology

### 2.1 Problem Analysis

The data contains noisy business names and addresses across independent
sources. Variations include abbreviations, legal suffixes, punctuation,
typos, word-order changes, transliteration, missing address components,
and landmark-based descriptions.

The test data also contains France in addition to the countries observed
in training, so country is treated as an open-set string feature.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Gradient-Boosted Classifier

**Core Innovation:** Combining selective blocking with an XGBoost pair
classifier using complementary name, address, numeric, exact-match,
country, and length-ratio features.

---

## 3. Candidate Generation (Blocking)

### Blocking keys used

1. Exact normalized business name + country.
2. Exact normalized business address + country.
3. Selective normalized-name token + country blocking, with common
   tokens restricted to a maximum candidate-list size.

### Candidate pairs generated

**32,365,428 candidate pairs**

for **1,732,544 Source 1 test entities**.

### How true matches were protected

Candidate sets from multiple blocking rules were combined and
deduplicated at the pair level. The resulting candidate set was the
one supplied to the final matching model.

---

## 4. Matching Model

### Features used

- Name Jaccard similarity
- Address Jaccard similarity
- Numeric-token Jaccard similarity
- Numeric overlap
- House-number match
- Exact normalized name
- Exact normalized address
- Same country
- Name length ratio
- Address length ratio

### Model type

**XGBoost binary classifier**

Configuration included:

- 500 estimators
- Maximum depth 5
- Learning rate 0.08
- Subsample 0.8
- Column sampling 0.9
- Minimum child weight 5
- L2 regularization 2.0
- Histogram tree method

### Threshold selection method

The probability threshold was selected using a held-out entity-level
validation split and F_0.5.

**Selected threshold: 0.82**

---

## 5. Results & Error Analysis

### Validation result

Held-out validation measurements for the retrained model:

- Precision: **0.978172**
- Recall: **0.837947**
- F_0.5: **0.946494**
- Threshold: **0.82**

These are validation measurements and are not claims about hidden
test-set or leaderboard performance.

### Common false positives

Hard negatives can have similar business names. Name similarity alone
is therefore insufficient; address and numeric evidence were included
to improve precision.

### Common false negatives

Missed matches can result from strong formatting differences, missing
address components, transliteration differences, or failure during
candidate generation.

---

## 6. Conclusion

The solution combines scalable blocking with an XGBoost pair classifier.
The final candidate set represents the candidates actually scored by the
matching model, and the final output contains one row for every Source 1
test entity.

---

## Appendix

### A. Code Artefacts

Source code is provided under:

`code/business_entity_resolution/src/`

The package also includes:

- `code/business_entity_resolution/README.md`
- `code/business_entity_resolution/requirements.txt`

The final outputs are:

- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

### B. Additional Results

- Source 1 entities: 1,732,544
- Source 2 + Source 3 entities: 9,969,589
- Candidate pairs: 32,365,428
- Predicted matches: 7,030,733
