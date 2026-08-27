# INFOKOM UNESA Research Collaboration Network Analysis

## About this project

This project analyzes the research landscape of lecturers in the INFOKOM
group at UNESA (Universitas Negeri Surabaya) from two angles at once: what
they research (text/topic) and who they collaborate with (network). Most
studies of academic units pick one lens or the other. The core contribution
here is putting both lenses side by side and measuring where they agree and
where they diverge, which surfaces people and patterns a single-lens study
would miss.

The pipeline covers 129 lecturers across 24 study programs, drawing on
1,151 Scopus records and 5,573 Google Scholar records, and produces:

- A 4-topic thematic model of what INFOKOM researches (NMF, chosen over
  LDA and BERTopic on measured quality).
- A 4-cluster thematic segmentation of lecturers, validated by consensus
  across four clustering algorithms.
- A 129-node, 365-edge co-authorship network, with a 93-lecturer giant
  component analyzed for community structure, brokerage, and core-
  periphery position.
- A cross-check between "who researches similar things" and "who actually
  collaborates," which turns out to agree only weakly (this is the
  project's central empirical finding, see §6).
- A broker typology and an entropy-weighted composite ranking of
  lecturers.
- A Neo4j-ready export and a FastAPI service for scoring new text against
  the trained topic/cluster models.

---

## Table of contents

1. [Data](#1-data)
2. [Full pipeline workflow](#2-full-pipeline-workflow)
3. [Text preprocessing](#3-text-preprocessing)
4. [Topic modeling: LDA vs NMF vs BERTopic](#4-topic-modeling-lda-vs-nmf-vs-bertopic)
5. [Thematic clustering](#5-thematic-clustering)
6. [Collaboration network construction](#6-collaboration-network-construction)
7. [Community detection: Louvain vs Leiden](#7-community-detection-louvain-vs-leiden)
8. [Overlapping communities (BigCLAM)](#8-overlapping-communities-bigclam)
9. [Structural holes and brokerage (Burt's constraint)](#9-structural-holes-and-brokerage-burts-constraint)
10. [Semantic-structural congruence and broker typology](#10-semantic-structural-congruence-and-broker-typology)
11. [Additional structural analyses](#11-additional-structural-analyses)
12. [Multi-criteria ranking (entropy weighting)](#12-multi-criteria-ranking-entropy-weighting)
13. [Neo4j export](#13-neo4j-export)
14. [Key results at a glance](#14-key-results-at-a-glance)
15. [Known limitations](#15-known-limitations-please-keep-these-in-the-papers-limitations-section)
16. [Environment setup](#16-environment-setup)
17. [Topic and cluster names](#17-topic-and-cluster-names)
18. [Testing the pipeline on new data (FastAPI)](#18-testing-the-pipeline-on-new-data-fastapi)
19. [Importing into Neo4j](#19-importing-into-neo4j)
20. [Files in this delivery](#20-files-in-this-delivery)

---

## 1. Data

- `dosen_infokom_final.csv`: the authoritative lecturer roster (identity,
  study program, external IDs: Scopus, Google Scholar, Sinta). 129 rows.
  The pipeline always keys off this file.
- `dosen_papers_scopus.csv`: Scopus publication records (title, abstract,
  keywords, author IDs, year). 1,151 rows. Primary source for both topic
  text and collaboration edges, since author IDs can be matched back to
  the roster precisely.
- `dosen_papers_scholar.csv`: Google Scholar records (title, year,
  venue). 5,573 rows. Used as a fallback text source for lecturers with no
  usable Scopus abstract, and folded into productivity counts. Not used
  for network edges, since Scholar does not expose reliable per-author IDs.

Lecturers are spread across 24 study programs; the three largest are S1
Bisnis Digital (16), S1 Pendidikan Teknologi Informasi (15), and S1 Sains
Data (13).

---

## 2. Full pipeline workflow

The notebook runs top to bottom as 15 numbered stages, each writing its
outputs to a matching `output/NN_stage_name/` folder:

| # | Stage | What it produces |
|---|---|---|
| 1 | Setup & data loading | Folder structure, loaded + validated CSVs |
| 2 | EDA (pre-preprocessing) | Missing-value report, publication trends, productivity stats |
| 3 | Text preprocessing | Cleaned per-lecturer corpus (Scopus, with Scholar fallback) |
| 4 | EDA (post-preprocessing) | Vocabulary stats, corpus length distribution |
| 5 | Topic modeling | LDA vs NMF vs BERTopic comparison, final NMF model |
| 6 | Thematic clustering | 4-algorithm + consensus clustering over topic distributions |
| 7 | Network construction | Co-authorship graph, degree distribution fit |
| 8 | Hard community detection | Louvain vs Leiden, significance test |
| 9 | Overlapping community detection | BigCLAM, BIC-selected k |
| 10 | Structural holes | Burt's constraint, effective size, centralities |
| 11 | Congruence analysis | ARI/NMI between thematic and structural labels, broker typology |
| 12 | Additional structural analysis | Assortativity, k-core, disparity filter backbone, temporal growth |
| 13 | Multi-criteria ranking | Entropy-weighted composite score and rank |
| 14 | Neo4j export | Node/edge CSVs + Cypher import script |
| 15 | Model persistence | All trained models, vectorizers, and graphs saved for reuse |

Two branches run **independently** by design, the thematic branch (§4-5)
never sees network data, and the network branch (§6-9) never sees topic
data, so that §10's comparison between them is a genuine cross-check
rather than a self-fulfilling one.

---

## 3. Text preprocessing

**Purpose:** turn raw, mixed Indonesian/English titles, abstracts, and
keywords into a clean per-lecturer document usable as model input.

**How it works:**
1. Lowercase, then tokenize with `[a-zA-Z]+` (strips numbers/punctuation).
2. Drop tokens ≤2 characters.
3. Drop stopwords from a merged list: Indonesian (Sastrawi), English
   (scikit-learn's `ENGLISH_STOP_WORDS`), and a custom domain list (e.g.
   "paper", "method", "ieee", "university"), 471 words total.
4. Source text is Title+Abstract+Keywords for Scopus, Title only for
   Scholar (abstracts are rarely available there).
5. Per-lecturer corpora are built by concatenating all cleaned Scopus
   papers for that lecturer; if a lecturer has no valid Scopus papers, the
   pipeline falls back to their Scholar titles.

**Results:**

| Corpus source | Lecturers |
|---|---|
| Scopus (primary) | 115 |
| Scholar (fallback) | 10 |
| No usable text (`no_data`) | 4 |

Raw vocabulary after cleaning: 11,956 unique words. Most frequent:
"learning", "students", "performance", "algorithm", "development",
"education", "digital", reflecting INFOKOM's dual emphasis on
educational technology and AI/ML.

---

## 4. Topic modeling: LDA vs NMF vs BERTopic

**Purpose:** discover latent research themes from the 125 valid
per-lecturer corpora, without deciding in advance which algorithm is
"correct" for this kind of short, domain-mixed text.

**Feature representations:** Count matrix (125×3000, `max_df=0.6`,
`min_df=3`) for LDA; TF-IDF matrix (same shape) for NMF.

**Methods compared:**

- **LDA**, generative probabilistic model; assumes each document is a
  mixture of topics and each topic a distribution over words. Fit on the
  count matrix with variational Bayes, k = 4..10.
- **NMF**, factorizes the TF-IDF matrix into non-negative document×topic
  and topic×word matrices (NNDSVDA init). Non-negativity tends to produce
  cleaner, more separable topics than LDA on short texts. k = 4..10.
- **BERTopic**, embeds each document with Sentence-BERT
  (`all-MiniLM-L6-v2`, 384-dim), reduces to 5 dimensions with UMAP
  (cosine metric), then clusters with HDBSCAN (density-based, chooses its
  own cluster count, can mark outliers as topic −1). `min_cluster_size`
  tested at 3, 4, 5, 6, 8, 10. If no internet access is available to
  download the SBERT model, this stage automatically falls back to an
  offline PPMI + truncated-SVD distributional embedding (Levy & Goldberg,
  2014) and **prints which backend was actually used**, always check
  that printed line before citing which backbone produced a given result.

**Evaluation:** NPMI-based **coherence** (do a topic's top words actually
co-occur in the same documents?) and **diversity** (how little do topics
overlap in vocabulary?), combined as `quality_score = coherence × diversity`.

**Result, final comparison:**

| Model | Best k | Coherence | Diversity | Quality score |
|---|---|---|---|---|
| LDA | 7 | 0.295 | 0.886 | 0.261 |
| **NMF** | **4** | **0.464** | **0.975** | **0.452 (winner)** |
| BERTopic | 7 | 0.333 | 0.786 | 0.261 |

NMF wins by a wide margin on both metrics, plausible given the corpus is
many short, focused per-lecturer documents rather than long generic text,
which favors TF-IDF + non-negative factorization over LDA's generative
assumptions or BERTopic's density clustering on only 125 points.

**Final model: NMF, k = 4 topics.**

| Topic | Top keywords | Interpretation | Dominant lecturers |
|---|---|---|---|
| 0 | classification, algorithm, detection, neural, machine | AI & Machine Learning | 55 |
| 1 | financial, marketing, digital, metaverse, banking | Digital Business & Fintech | 27 |
| 2 | menggunakan, berbasis, sistem, metode, aplikasi | Information Systems & Educational Applications (Indonesian-language corpus) | 10 |
| 3 | skills, thinking, vocational, school, blended | Educational Technology & Learning Development | 33 |

Every lecturer's dominant topic is the argmax of their normalized
topic-distribution row (`doc_topic_norm`).

**Artifacts saved:** `count_vectorizer.joblib`, `tfidf_vectorizer.joblib`,
`topic_model_nmf.joblib`, `topic_words.json`, `topic_names.json`,
`doc_topic_matrix.npy`, `pipeline_config.json`.

---

## 5. Thematic clustering

**Purpose:** group lecturers by their *whole* topic-distribution profile
(not just their single dominant topic), and verify the grouping is stable
across multiple clustering algorithms rather than trusting just one.

**Feature used:** `doc_topic_norm`, the 125×4 row-normalized NMF
topic-distribution matrix. Each lecturer is a point in 4-dimensional
topic-proportion space.

**Choosing k:** KMeans run for k = 2..7, scored by silhouette,
Davies-Bouldin, and Calinski-Harabasz. **k = 4** wins on silhouette
(0.632) and Davies-Bouldin (0.518) simultaneously.

**Four algorithms compared at k = 4:**

| Algorithm | How it works | Silhouette |
|---|---|---|
| KMeans | Minimizes Euclidean distance to iteratively updated centroids | 0.632 |
| Agglomerative (Ward) | Bottom-up merging that minimizes variance increase | 0.574 |
| Gaussian Mixture Model | Probabilistic; each cluster is a Gaussian, points assigned by likelihood | 0.511 |
| Spectral Clustering | Graph-based, via eigendecomposition of the nearest-neighbor Laplacian | 0.271 |

Cross-algorithm agreement (Adjusted Rand Index): KMeans and Agglomerative
agree most (ARI 0.714); Spectral diverges most from the other three (ARI
0.38-0.44), consistent with its weaker silhouette.

**Consensus clustering (final result):** a co-association matrix is built
across the four algorithms' outputs (how often two lecturers land in the
same cluster), converted to a distance matrix, and re-clustered with
average-linkage Agglomerative clustering on that precomputed distance.
This consensus labeling, not any single algorithm's raw output, is the
final `thematic_cluster` assignment. Consensus silhouette: 0.610.

**Final clusters:**

| Cluster | Interpretation | Lecturers |
|---|---|---|
| 0 | Educational Technology & Engineering | 47 |
| 1 | Computing & Artificial Intelligence | 46 |
| 2 | Digital Business & Information Systems | 22 |
| 3 | Applied Information Technology & Systems | 10 |

A 2D PCA projection (PC1 47%, PC2 33%, 80% variance explained) is used for
visualization. A KNN (k=5) model is trained to assign new lecturers to
these clusters without re-running clustering; resubstitution accuracy is
99.2%, indicating the clusters are well-separated.

---

## 6. Collaboration network construction

**Purpose:** represent actual, evidenced collaboration, co-authorship on
the same paper, as a graph.

**How it's built:** an undirected NetworkX graph where nodes are all 129
lecturers, and edges come from Scopus papers with ≥2 internal authors:
every pairwise combination of internal co-authors on such a paper gets an
edge, with weight incremented each time the same pair co-authors again.

**Basic statistics:**

| Metric | Value |
|---|---|
| Nodes | 129 |
| Edges | 365 |
| Density | 0.0442 |
| Average degree | 5.66 |
| Isolated lecturers (degree 0) | 34 |

**Degree distribution shape:** rather than assuming scale-free
structure, the pipeline fits it explicitly using the `powerlaw` package,
comparing power-law against lognormal and exponential alternatives via
log-likelihood ratio tests:

| Comparison | Log-likelihood ratio | p-value | Conclusion |
|---|---|---|---|
| Power-law vs Lognormal | −13.73 | 0.0029 | Lognormal fits significantly better |
| Power-law vs Exponential | −13.94 | 0.00002 | Exponential also fits better |

Power-law fit: α = 2.21, xmin = 4. Conclusion: **the degree distribution
is better explained by a lognormal than a pure power-law**, this network
is not an extreme scale-free hub structure, but has more moderate,
distributed variation in degree.

**Giant component:** 93 of 129 nodes and 364 of 365 edges sit in one
connected component (36 components total; the rest are tiny 1-2 node
fragments).

| Giant component metric | Value | Meaning |
|---|---|---|
| Density | 0.0851 | ~2x denser than the full graph |
| Average clustering coefficient | 0.5521 | High, many small tight-knit research cliques |
| Diameter | 8 | Longest shortest path between any two lecturers |
| Average shortest path length | 3.23 | Typical distance between two random lecturers |
| Global transitivity | 0.4714 | Probability two co-collaborators are also connected |

Both the full graph and giant component are exported as GraphML (for
Gephi/Cytoscape) and gpickle (for reloading in Python with all attributes
intact).

---

## 7. Community detection: Louvain vs Leiden

**Purpose:** find groups of lecturers who collaborate more densely with
each other than with the rest of the network, a purely structural
grouping, independent of research topic.

**Modularity** is the standard metric here: it compares actual
within-community edge density to what would be expected if edges were
randomly distributed given the same degree sequence.

**Two algorithms compared:**
- **Louvain**: fast, greedy, two-phase (local modularity-maximizing moves,
  then aggregation into super-nodes). Known theoretical weakness: can
  produce internally disconnected communities.
- **Leiden**: Louvain's successor, guarantees every resulting community
  is well-connected internally.

**Results:**

| Method | Communities | Modularity | Well-connected fraction |
|---|---|---|---|
| Louvain | 8 | 0.6147 | 100% |
| Leiden | 8 | 0.5243 | 100% |

Both are 100% well-connected here, so Louvain is selected on its higher
modularity.

**Significance test:** modularity alone doesn't prove the structure is
real rather than a side effect of the degree sequence. 200 random graphs
were generated via the configuration model (same degree sequence, randomly
rewired), Louvain was run on each, and the observed modularity was
compared to that null distribution.

| Metric | Value |
|---|---|
| Observed modularity | 0.6147 |
| Null model mean | 0.3032 |
| Null model std. dev. | 0.0107 |
| p-value | 0.00498 |

The observed modularity is roughly 29 standard deviations above the null
mean, the 8-community structure is statistically significant, not an
artifact of degree distribution.

---

## 8. Overlapping communities (BigCLAM)

**Purpose:** Louvain/Leiden force each lecturer into exactly one
community. In reality, researchers often belong to more than one active
collaboration circle. BigCLAM (Cluster Affiliation Model for Big Networks)
allows continuous, multi-community membership instead of a hard label.

**How it works:** each node u gets a non-negative affiliation strength
`F_uc` to each community c. Edge probability between u and v is modeled as
`P(u,v) = 1 − exp(−F_u · F_v^T)`, stronger shared affiliation implies
higher edge probability. Parameters are fit by maximizing graph
log-likelihood via manual gradient ascent (400-800 iterations, learning
rate 0.005).

**Choosing k:** since more communities always increase log-likelihood,
model selection uses **BIC** (`−2·log-likelihood + params·ln(n)`), which
penalizes complexity.

| k | Log-likelihood | BIC |
|---|---|---|
| 2 | −867.36 | **2577.79 (best)** |
| 3 | −716.48 | 2697.55 |
| 4 | −636.29 | 2958.70 |
| 5 | −580.96 | 3269.59 |
| 6 | −506.35 | 3541.90 |

**k = 2** is selected. Membership threshold `δ = √(−ln(1−ε))` with
ε = 10⁻⁴ (standard BigCLAM threshold); nodes below threshold everywhere
default to their single strongest community so nobody is left unassigned.

**Result:** 81 lecturers belong to exactly 1 community, **12 lecturers
have overlapping (dual) membership**, average 1.13 memberships per
lecturer. These 12 are lecturers actively bridging two collaboration
circles at once (full list in the notebook's cell output and in
`nodes_dosen.csv`).

---

## 9. Structural holes and brokerage (Burt's constraint)

**Purpose:** identify who occupies a strategically unique bridging
position between otherwise-disconnected groups, not just who has the
most connections. Per Ronald Burt's structural holes theory, a broker
whose contacts don't know each other has more informational/control value
than someone with the same degree whose contacts are all mutually
connected (redundant).

**Metrics computed** (on the 93-lecturer giant component):
- **Constraint**, how "locked in" a node's connections are by its own
  network structure; low = broker-like, high = trapped in a tight clique.
- **Effective size**, estimated count of non-redundant contacts.
- **Degree, betweenness, closeness centrality**, standard positional
  measures.

| Metric | Mean | Median | Min | Max |
|---|---|---|---|---|
| Degree | 7.83 | 7 | 1 | 25 |
| Constraint | 0.4349 | 0.3673 | 0.1673 | 1.0000 |
| Effective size | 6.01 | 4.50 | 1.00 | 22.38 |
| Betweenness | 0.0245 | 0.0069 | 0.0000 | 0.2458 |

**Top broker:** Prof. Dr. Lilik Anifah, S.T., M.T., highest degree (25)
*and* lowest constraint (0.167) simultaneously, a rare combination meaning
her many connections span many non-overlapping groups rather than one
dense clique.

---

## 10. Semantic-structural congruence and broker typology

**Purpose:** answer the project's central question, do lecturers who
collaborate closely also research similar topics? Or does collaboration
cross thematic lines?

**Method:** compare `thematic_cluster` (§5) against `community_label`
(§7, Louvain) for the 93 lecturers who have both, using Adjusted Rand
Index (ARI) and Normalized Mutual Information (NMI).

| Metric | Value |
|---|---|
| ARI (thematic vs structural) | 0.1695 |
| NMI (thematic vs structural) | 0.3360 |

A 500-iteration permutation test gives **p = 0.002** for the ARI, the
agreement is statistically significant (not zero), but its magnitude is
low.

> **Key finding:** thematic similarity explains only a small part of who
> collaborates with whom (ARI 0.17). Cross-topic, interdisciplinary
> collaboration is common at INFOKOM, collaboration is not simply
> "people with the same research theme write together."

**Neighbor topic diversity:** Shannon entropy (bits) of a lecturer's
network neighbors' thematic-cluster labels. Median constraint (0.3673)
and median neighbor diversity (0.8113 bits) are used as thresholds for a
4-way typology:

| Type | Constraint | Neighbor diversity | Meaning | Count |
|---|---|---|---|---|
| Semantic Broker | Low | High | True broker, bridges both groups and topics; most strategically valuable | 36 |
| Local Specialist | High | Low | Embedded in a tight, thematically uniform clique | 33 |
| Thematic Bridge (Embedded) | High | High | Surrounded by thematically diverse contacts, but not a structural broker itself | 13 |
| Structural Broker (Within-Topic) | Low | Low | Bridges structurally, but everyone bridged shares the same topic | 11 |

**Three-way consistency check** (thematic cluster vs Louvain community vs
BigCLAM overlap community, n=93):

| Pair | NMI |
|---|---|
| Thematic vs hard community | 0.3360 |
| Thematic vs overlapping community | 0.3041 |
| Hard vs overlapping community | 0.2706 |

All three NMI values are low, confirming these three lenses capture
genuinely different organizational dimensions rather than restating each
other, the methodological justification for analyzing all three rather
than picking just one.

---

## 11. Additional structural analyses

**Assortativity** (homophily, do similar nodes connect to each other?):

| Type | Value | Interpretation |
|---|---|---|
| By study program | 0.2835 | Moderate: lecturers somewhat prefer same-program collaborators |
| By thematic cluster | 0.4595 | Stronger: same-theme lecturers collaborate more often |
| By degree | 0.1695 | Weak: hubs mildly prefer connecting to other hubs |

**K-core decomposition:** identifies the network's layered "core." The
maximum k-core is **k = 7**, with 31 lecturers in the innermost, most
densely interconnected layer.

**Disparity filter (backbone extraction):** statistically extracts only
the edges whose weight is significant relative to each node's total edge
weight (α = 0.4), rather than an arbitrary cutoff. Of 364 giant-component
edges, **143 (39%) survive as the statistically significant backbone**,
involving 71 active nodes.

**Temporal growth:** the network is rebuilt cumulatively year by year
(2011-2026):

| Year | Cumulative edges | Density |
|---|---|---|
| 2017 | 5 | 0.0006 |
| 2019 | 50 | 0.0061 |
| 2021 | 114 | 0.0138 |
| 2023 | 224 | 0.0271 |
| 2025 | 357 | 0.0432 |
| 2026 | 365 | 0.0442 |

Growth accelerates most sharply 2021-2024, matching the publication-volume
spike seen in the raw EDA (§2).

---

## 12. Multi-criteria ranking (entropy weighting)

**Purpose:** combine every individual metric computed so far (productivity,
network centrality, brokerage, community membership) into one composite
score, without hand-picking arbitrary weights.

**Five criteria combined:** `n_papers_total`, `degree_centrality`,
`betweenness`, `inv_constraint` (1 − Burt's constraint), and
`n_community_memberships` (from BigCLAM, §8).

**Method, entropy weighting:** an objective MCDM technique from
information theory. Criteria whose values vary a lot across lecturers
(low entropy, high information content) get more weight because they
better *distinguish* lecturers; criteria that are nearly uniform across
lecturers get less weight because they carry less discriminating power.

Steps: min-max normalize each criterion → compute proportions `p_ij` →
compute entropy `e_j = −k·Σ(p_ij·ln(p_ij))`, `k = 1/ln(n)` → diversification
degree `d_j = 1 − e_j` → final weight `w_j = d_j / Σd_j`.

**Resulting weights:**

| Criterion | Weight |
|---|---|
| n_community_memberships | **0.4978 (most influential)** |
| betweenness | 0.2682 |
| n_papers_total | 0.1359 |
| degree_centrality | 0.0692 |
| inv_constraint | 0.0289 (least influential) |

`n_community_memberships` dominates the weighting because it varies the
most sharply across lecturers (mostly 1, occasionally 2), mathematically
the most discriminating signal available.

**Top 5 by composite score:**

| Rank | Lecturer | Score | Broker type |
|---|---|---|---|
| 1 | Asmunin, S.Kom., M.Kom. | 0.7578 | Semantic Broker |
| 2 | Dr. Ir. Lusia Rakhmawati, S.T., M.T. | 0.7355 | Semantic Broker |
| 3 | Rindu Puspita Wibawa, S.Kom., M.Kom. | 0.6731 | Semantic Broker |
| 4 | Rifqi Abdillah, M.Kom. | 0.6559 | Semantic Broker |
| 5 | Dr. Aries Dwi Indriyanti, S.Kom., M.Kom. | 0.6554 | Semantic Broker |

11 of the 12 BigCLAM overlap-community lecturers (§8) land in the top 10
of this ranking, consistent with the weight distribution above. Notably,
Prof. Dr. Lilik Anifah, the strongest individual broker by constraint
alone (§9), ranks only 14th here, because she belongs to just 1
community; the composite score genuinely blends dimensions rather than
just restating individual centrality.

---

## 13. Neo4j export

`output/13_neo4j_export/` contains everything needed to load the full
result set into a graph database for interactive querying:

| File | Contents |
|---|---|
| `nodes_dosen.csv` | 129 rows: all lecturer attributes (program, paper count, thematic cluster, structural community, overlap membership count, constraint, betweenness, broker type, composite rank) |
| `nodes_topic.csv` | 4 rows: NMF topic profiles (keywords, name, dominant-lecturer count) |
| `edges_collaboration.csv` | 365 rows: collaborating lecturer pairs with co-authored-paper weight |
| `edges_dosen_topic.csv` | 125 rows: lecturer → dominant topic relations with proportion weight |
| `import_script.cypher` | Ready-to-run script: creates uniqueness constraints, loads all four CSVs, builds `COLLABORATES_WITH` (undirected) and `RESEARCHES` (directed, weighted) relationships |

If no local Neo4j instance is running when the notebook executes, the
direct-connection cell will raise `ServiceUnavailable`, this is expected
in that case; the CSVs and Cypher script are still written and can be
imported manually via Neo4j Browser or `cypher-shell` at any time.

---

## 14. Key results at a glance

| Metric | Value |
|---|---|
| Lecturers | 129 |
| Scopus records | 1,151 |
| Google Scholar records | 5,573 |
| Giant component size | 93 nodes |
| Collaboration edges | 365 |
| Best topic model | NMF, k = 4 |
| Thematic clusters | 4 (consensus silhouette 0.610) |
| Best community method | Louvain, 8 communities |
| Modularity (p-value) | 0.6147 (p = 0.00498) |
| BigCLAM overlap k | 2 (12 lecturers overlap) |
| Thematic vs structural ARI (p-value) | 0.1695 (p = 0.00200) |
| Thematic vs structural NMI | 0.3360 |
| Semantic Brokers identified | 36 |
| Top-ranked lecturer (composite score) | Asmunin, S.Kom., M.Kom. (0.7578) |

---

## 15. Known limitations (please keep these in the paper's limitations section)

- A handful of lecturers (4) have no usable text in either source and are
  excluded from topic modeling and thematic clustering; they are still
  listed in the roster with a `no_data` flag.
- Lecturers outside the network's giant connected component (34 isolated,
  plus a few in small side components) do not get structural-holes,
  community, or ranking scores, because those measures are only defined
  relative to a connected neighborhood.
- The BERTopic component prefers a real sentence-transformer embedding
  backbone. If no internet access is available to download the model, it
  automatically falls back to an offline PPMI + SVD distributional
  embedding (Levy and Goldberg, 2014) and prints which backend was
  actually used, so results are never silently misrepresented. Check the
  printed output of that cell before citing which backbone was used.
- Collaboration edges are built only from Scopus data (not Google
  Scholar), since Scholar does not expose reliable per-author IDs that can
  be matched to the roster, the network may under-count collaboration
  that only appears in national/local journals not indexed by Scopus.
- All clustering and community results are unsupervised; treat them as
  hypotheses to sanity-check against domain knowledge, not as ground
  truth.
- Topic/cluster names (e.g. "AI & Machine Learning") are assigned manually
  based on top keywords, not derived automatically, domain judgment is
  still required.

---

## 16. Environment setup

```
python -m venv venv
venv\Scripts\activate        (Windows)
source venv/bin/activate     (Mac/Linux)

pip install -r requirements-notebook.txt
pip install jupyter ipykernel
```

Open `analisis_jaringan_dosen_infokom.ipynb` in VS Code, select the `venv`
interpreter as the kernel, and run all cells top to bottom. This creates an
`output/` folder next to the notebook with 15 subfolders: one per analysis
stage (charts and CSVs) plus `output/15_saved_models/` holding every
trained model and lookup table.

## 17. Topic and cluster names

The four NMF topics (section 6) and four thematic clusters (section 7)
are labeled as follows. These names are set in the notebook via the
`topic_names` and `cluster_names` dictionaries, and propagate
automatically into every chart, CSV, the Neo4j export, and the API.

Topics:

| ID | Name |
|---|---|
| 0 | AI & Machine Learning |
| 1 | Digital Business & Fintech |
| 2 | Information Systems & Educational Applications |
| 3 | Educational Technology & Learning Development |

Clusters:

| ID | Name |
|---|---|
| 0 | Educational Technology & Engineering |
| 1 | Computing & Artificial Intelligence |
| 2 | Digital Business & Information Systems |
| 3 | Applied Information Technology & Systems |

To relabel either set, edit the corresponding dictionary in the notebook
and re-run that cell and everything below it.

## 18. Testing the pipeline on new data (FastAPI)

When a new lecturer or new paper needs to be checked against the existing
topics and clusters, without re-running the whole notebook:

```
pip install -r requirements-api.txt
uvicorn app:app --reload --port 8000
```

The API loads everything from `output/15_saved_models/`, so run the
notebook at least once first. Interactive docs are available at
`http://127.0.0.1:8000/docs` once it is running.

```
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/topics
curl http://127.0.0.1:8000/clusters

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"paste the new paper's title, abstract, and keywords here\"}"
```

The response includes the predicted topic distribution, the dominant topic
(with your custom name if set), and the predicted thematic cluster.

This only covers the text-based side (topic and thematic cluster). It
cannot predict network-based measures (structural holes, community
membership, broker typology, ranking) for a brand-new lecturer, since those
depend on real co-authorship edges. To update those, add the new lecturer
and their collaboration data to the raw CSVs and re-run the notebook from
the network construction section onward.

## 19. Importing into Neo4j

After running the notebook, `output/13_neo4j_export/` contains
`nodes_dosen.csv`, `nodes_topic.csv`, `edges_collaboration.csv`,
`edges_dosen_topic.csv`, and `import_script.cypher`. Copy the CSVs into
your Neo4j import folder and run the Cypher script in Neo4j Browser or
`cypher-shell`.

## 20. Files in this delivery

- `analisis_jaringan_dosen_infokom.ipynb`: the full analysis pipeline
- `app.py`: FastAPI inference service for testing new data
- `requirements-notebook.txt`: exact package versions for the notebook
- `requirements-api.txt`: packages needed to run the API
- `README.md`: this file
