# 🧠 Assistant IA Académique — RAG + Ollama + Streamlit  
Assistant intelligent capable de répondre à des questions à partir de documents internes (RAG), exécuter des calculs, obtenir la météo, faire des recherches web et gérer des TODO — le tout en local grâce à **Ollama** et **LangChain**.

---

## 🚀 Fonctionnalités principales

### 🔹 1. RAG (Retrieval Augmented Generation)
- Charge automatiquement les documents du dossier `RAG_Data/`  
- Indexe avec **ChromaDB** et `sentence-transformers/all-MiniLM-L6-v2`  
- Analyse les questions et extrait les passages pertinents  
- Construit un contexte contrôlé pour éviter les hallucinations  
- Répond **uniquement** avec les données des documents internes  

➡️ Fonctionne avec : `.txt`, `.pdf`, `.docx`

---

### 🔹 2. Outils intégrés

#### 🧮 Calculatrice intelligente
- Expressions mathématiques : `2+3*4`, `23²`, `sin45`, `cos30 + sqrt16`, etc.  
- Conversion automatique degrés → radians  
- Nettoyage d’expression tolérant aux fautes

#### 🌦️ Météo
- Récupération en temps réel via API  
- Exemple : `donne-moi la météo pour Paris`

#### 🔍 Recherche web
- Utilise **DuckDuckGo** via `ddgs`  
- Déclenchée uniquement si l’utilisateur donne son accord (oui/non)

#### 📝 Gestion TODO
- Ajout (`add:`), validation (`done:`) et liste des tâches  
- Stockage dans `memory_store.json` (non versionné)

#### 💬 Smalltalk
- Gestion des salutations simples : bonjour, salut, etc.

---

## 🗂️ Architecture du projet

```text
Projet_IA/
│
├── app.py                 # Application Streamlit (UI principale)
├── agents.py              # Outils : calculatrice, météo, web, TODO
├── router.py              # Détection d’intention (calc / météo / web / rag / smalltalk)
├── rag_core.py            # Moteur RAG (Chroma + embeddings + Ollama)
│
├── RAG_Data/              # Dossier contenant les documents internes (cours, PDF, etc.)
│
├── requirements.txt       # Dépendances Python
├── .gitignore             # Fichiers / dossiers ignorés par Git
└── README.md              # Documentation du projet
