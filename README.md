# BBC News Topic Modeling — LDA & NMF  
**Internship Project @ Elevvo**

Unsupervised topic modeling on BBC news descriptions (with titles for display).  
We use **LDA (Bag-of-Words)** and **NMF (TF-IDF)** to discover hidden themes, name them in human language, and visualize how each article mixes multiple topics.

---

## ✨ What this project does
- Cleans news text (lowercasing, short-text filtering, custom news stopwords).
- Builds vectors:
  - **LDA** on Bag-of-Words (counts)
  - **NMF** on TF-IDF
- Extracts topics → **top words** per topic + **top articles** that best represent each topic.
- Assigns each document a **topic mixture score** (e.g., Doc = 70% Topic A + 20% Topic B + 10% Topic C).
- Visualizes:
  - Top words
  - Document–topic heatmap
  - Stacked bars (per-doc mixtures)
  - Word clouds
  - (Optional) pyLDAvis interactive view

---

## 🗂 Data
- File: `bbc_news_clust.csv`
- Expected columns:  
  - `description` (used for modeling)  
  - `title` (used for naming & display)
> If your column names differ, just update the notebook cells where the text column is selected.

---

## 🛠 Pipeline (step-by-step, short & clear)
1. **Load & Clean**  
   - Use `description`, convert to lowercase, drop very short rows.
2. **Custom Stopwords**  
   - Start from English stopwords and add news-generic terms (e.g., *says, bbc, year, today…*) to reduce noise.
3. **Vectorization**  
   - LDA → **CountVectorizer** with sensible `min_df` / `max_df`.  
   - NMF → **TfidfVectorizer** with similar thresholds.
4. **Train Models (K topics)**  
   - Pick `K` (e.g., 5), fit LDA & NMF.
5. **Inspect Topics**  
   - Print **top 10 words** per topic.  
   - Pull **top N titles/snippets** per topic for quick human naming.
6. **Label Topics (human names)**  
   - Map Topic IDs → labels like “Football / Premier League”, “International Conflict”…  
7. **Per-Document Scores**  
   - Compute `doc_topic = model.transform(X)` → scores per topic per doc.  
   - Dominant topic = argmax(score).
8. **Visualize & Export**  
   - Stacked bars, heatmaps, word clouds.  
   - Save CSVs: `topic_summary.csv` (counts per topic), `topic_top_examples.csv` (top titles/snippets).

---

## 🧪 Choosing K (number of topics)
- Start with **K = 5** (common for BBC-style sections), then try {7, 10, 12}.  
- Pick the K that balances:
  - **Interpretability** (you can name topics easily)
  - (Optional) **Coherence** score if you add gensim

---

## 🧾 Example outcomes (from one LDA run)
- **Topic 0**: *old, family, died, help, killed* → **Health / Society**  
- **Topic 1**: *ukraine, government, russia, israel, attack* → **International Conflict**  
- **Topic 2**: *election, minister, team, party, labour* → **Domestic Politics**  
- **Topic 3**: *england, world, cup, police, final* → **Global Sports Events**  
- **Topic 4**: *league, manchester, city, united, premier* → **Football / Premier League**

Per-document **scores** show how much each article belongs to each topic (values between 0 and 1 that sum ≈ 1). We rank the **top articles** per topic by this score to make naming fast.

---

## 📊 Visuals to include (suggested)
- *Top words per topic* (table or bar chart)  
- *Stacked bar* for first N documents (topic mixture per doc)  
- *Doc–topic heatmap* (N × K)  
- *Word clouds* per topic  
- *(Optional)* pyLDAvis HTML for interactive exploration

---

## ✅ What makes this project practical
- Small, readable cells with Arabic explanations and English code.
- Robust stopword strategy tailored for news.
- Both **LDA vs NMF** included for clarity and comparison.
- Simple exports (CSV) to plug into dashboards or reports.

---

## 💼 Context
This project was completed during my **internship at Elevvo** as a compact, educational NLP workflow that’s easy to extend (e.g., coherence metrics, n-grams, lemmatization, classifier on top of topic features).

---

## ⚠️ Troubleshooting (quick)
- *“max_df corresponds to < documents than min_df”* → decrease `min_df` (e.g., 1–2) or increase `max_df` (e.g., 0.95–0.99).  
- *Shape mismatch when making DataFrame from X* → pass a **Series** into `fit_transform` and use `DataFrame.sparse.from_spmatrix`.  
- *stop_words type error* → ensure custom stopwords are a **list** (not a set/frozenset).
