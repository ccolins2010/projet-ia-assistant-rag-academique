# 🧠 Assistant IA Académique — RAG + Ollama + Streamlit  
Assistant intelligent capable de répondre à des questions à partir de documents internes (RAG), exécuter des calculs, obtenir la météo, faire des recherches web et gérer des TODO — le tout en local grâce à **Ollama** et **LangChain**.

---

## 🚀 Fonctionnalités principales

### 🔹 1. RAG (Retrieval Augmented Generation)
- Charge automatiquement les documents du dossier `RAG_Data/`  
- Indexe avec **ChromaDB** et `sentence-transformers/all-MiniLM-L6-v2`  
- Analyse les questions et extrait les passages pertinents  
- Construit un contexte contrôlé pour éviter les hallucinations  
- Répond **uniquement** avec les données présentes dans les documents internes  

➡️ Fonctionne avec : `.txt`, `.pdf`, `.docx`

---

### 🔹 2. Outils intégrés

#### 🧮 Calculatrice intelligente
- Comprend : `2+3*4`, `23²`, `sqrt16`, `sin45`, `cos30`, etc.  
- Conversion auto degrés → radians (`sin30°` → `sin(0.52)`)  
- Nettoyage automatique des expressions  

#### 🌦️ Météo
- Récupération de la météo en temps réel  
- Exemple : `donne-moi la météo pour Nice`

#### 🔍 Recherche web
- Utilise **DuckDuckGo Search**  
- Ne s’active **que si l’utilisateur donne son accord** (réponse “oui”)  

#### 📝 Gestion TODO
- `add: faire les courses`  
- `done: 1`  
- `list`  
- Stockage dans `memory_store.json` (ignoré par Git)

#### 💬 Smalltalk
- Gère les salutations simples : bonjour, salut, etc.

---

## 🗂️ Architecture du projet

```text
Projet_IA/
│
├── app.py                 # Application Streamlit (interface principale)
├── agents.py              # Outils : calculatrice, météo, web, TODO
├── router.py              # Détection d’intention (calc / météo / web / rag / todo)
├── rag_core.py            # Moteur RAG (Chroma + embeddings + Ollama)
├── rag.py                 # API simplifiée pour utiliser le moteur RAG
├── reindex_once.py        # Script pour réindexer les documents
│
├── RAG_Data/              # Documents internes utilisés par le RAG
│
├── requirements.txt       # Dépendances Python
├── .gitignore             # Exclusions Git
└── README.md              # Documentation


git clone https://github.com/ccolins2010/projet-ia-assistant-rag-academique.git
cd projet-ia-assistant-rag-academique
pip install -r requirements.txt
streamlit run app.py

