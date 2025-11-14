# 🧠 Assistant IA Académique — RAG + Ollama + Streamlit  

Assistant intelligent capable de :

- répondre à des questions à partir de documents internes (RAG),
- exécuter des calculs,
- obtenir la météo,
- faire des recherches web (avec consentement),
- gérer une TODO-list persistante,
- envoyer la **dernière réponse par e-mail**,

le tout orchestré avec **Ollama**, **LangChain** et **Streamlit**.

---

## 🚀 Fonctionnalités principales

### 🔹 1. RAG (Retrieval-Augmented Generation)

- Charge automatiquement les documents du dossier `RAG_Data/`
- Supporte : `.txt`, `.pdf`, `.docx`
- Indexe les documents dans un **vector store ChromaDB** persistant (`chroma_store/`)
- Utilise les embeddings : `sentence-transformers/all-MiniLM-L6-v2`
- LLM local : **Ollama** avec le modèle `llama3.2:3b`
- Contrôle les hallucinations grâce à :
  - un filtrage lexical (_has_lexical_overlap),
  - une règle stricte : si le contexte ne parle pas de la question → réponse **exacte** : `Je ne sais pas.`

> 👉 Le RAG est toujours utilisé **en premier** pour répondre à une question.  
> Si aucune réponse fiable n’est trouvée, l’assistant propose une recherche web (oui/non).

---

### 🔹 2. Outils intégrés (Agents)

Implémentés dans `agents.py`, appelés automatiquement selon l’intention détectée dans `router.py`.

#### 🧮 Calculatrice intelligente

- Comprend des expressions comme :
  - `2+3*4`
  - `23²`
  - `sqrt16`
  - `sin45`, `cos30`, `tan60`
  - `(145 + 268) × 3 – 42`
- Normalisation automatique :
  - `,` → `.`  
  - `^` → `**`  
  - `×`, `÷`, `−`, `–` → `*`, `/`, `-`
  - conversion degrés → radians (`sin30°` → `sin(0.5235...)`)
- Évaluation sécurisée via **AST** (pas de `eval` Python).

#### 🌦️ Météo

- Comprend des phrases comme :
  - `donne-moi la météo pour Nice`
  - `quel temps fait-il à Lyon ?`
- Normalise le nom de ville à partir d’un texte libre.
- Géocodage via **Nominatim (OpenStreetMap)**.
- Météo courante via **Open-Meteo**.
- Fallback sur un petit dictionnaire local (Paris, Lyon, Marseille…) si le web ne répond pas.

#### 🔍 Recherche web

- Utilise **DuckDuckGo Search** via la librairie `ddgs`.
- L’utilisateur peut déclencher explicitement la recherche web avec des phrases comme :
  - `recherche sur le web ...`
  - `cherche sur internet ...`
- Si le RAG ne trouve rien, l’assistant demande :
  > `Je n’ai rien trouvé dans les documents internes. Veux-tu que je cherche sur le web ? (oui/non)`

#### 📝 Gestion TODO

- Commandes possibles (langage naturel) :
  - `ajoute : réviser IA`
  - `liste` / `list`
  - `termine 2` / `done: 2`
- Liste stockée dans `todo_store.json` (persistant entre les sessions).

#### 💬 Smalltalk

- Gère les salutations simples : `bonjour`, `salut`, `coucou`, `hello`…
- Utilise un LLM local (Ollama, `llama3.2:3b`) avec un prompt “assistant amical et bref”.

---

### 🔹 3. Envoi de la dernière réponse par e-mail

- Configuré via `.env` :

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=ton.email@gmail.com
SMTP_PASS=mot_de_passe_application
SMTP_FROM=ton.email@gmail.com
projet-ia-assistant-rag-academique/
│
├── app.py                 # Application Streamlit (UI + orchestration RAG / tools / web / e-mail)
├── agents.py              # Outils : calculatrice, météo, recherche web, TODO
├── router.py              # Détection d’intention (calc / météo / web / rag / todo / smalltalk)
├── rag_core.py            # Moteur RAG (Chroma + embeddings + Ollama)
├── rag.py                 # (optionnel) API simplifiée autour du moteur RAG
├── reindex_once.py        # Script pour forcer une réindexation des documents
│
├── RAG_Data/              # Documents internes utilisés par le RAG
│   ├── Cours_IA.txt
│   ├── Cours_Pytho.txt
│   └── Cours_Reseaux.txt
│
├── chroma_store/          # Index vectoriel persistant (créé automatiquement)
├── todo_store.json        # Stockage persistant des tâches TODO
├── memory_store.json      # Historique de conversation (chat) persistant
│
├── requirements.txt       # Dépendances Python
├── .env                   # Variables d’environnement (SMTP, etc.)
├── .gitignore             # Exclusions Git
└── README.md              # Documentation du projet

pip install -r requirements.txt
ollama run llama3.2:3b
streamlit run app.py
streamlit run app.py
