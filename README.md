# 🧠 Assistant IA Académique — RAG + Ollama + Streamlit

Assistant intelligent académique capable de :

- répondre à des questions à partir de **documents internes** (RAG),
- exécuter des **calculs** (calculatrice sécurisée),
- donner la **météo**,
- faire des **recherches web** (avec consentement explicite),
- gérer une **TODO-list persistante**,
- envoyer la **dernière réponse par e-mail**,

le tout orchestré avec **Ollama**, **LangChain** et **Streamlit**.

---

## ✅ Objectifs du projet (côté TP)

Ce projet répond aux exigences :

- **RAG complet** sur des fichiers locaux (cours académiques).
- **Agents / outils** : calculatrice, météo, recherche web, TODO.
- **Routage intelligent** : choix automatique entre RAG, outils, smalltalk.
- **Mémoire conversationnelle** persistante.
- **Interface conversationnelle** avec Streamlit.
- **Recherche web** intégrée (avec consentement utilisateur).
- **Envoi d’e-mails** de la dernière réponse.
- Code structuré, versionné, avec documentation d’architecture.

---

## 🚀 1. Fonctionnalités principales

### 🔹 1.1. RAG (Retrieval-Augmented Generation)

- Charge automatiquement les documents du dossier `RAG_Data/`
- Supporte les formats : `.txt`, `.pdf`, `.docx`
- Indexation dans un **vector store Chroma** persistant : `chroma_store/`
- Embeddings : `sentence-transformers/all-MiniLM-L6-v2`
- LLM local : **Ollama** (`llama3.2:3b`)
- **Contrôle des hallucinations** :
  - test de recouvrement lexical (_has_lexical_overlap),
  - si le contexte ne parle pas clairement de la question → réponse EXACTE :  
    `Je ne sais pas.`

🧠 **Logique de priorité :**

1. La question part **d’abord** dans le RAG (documents internes).
2. Si rien de pertinent n’est trouvé :
   - l’assistant répond :  
     `Je n’ai rien trouvé dans les documents internes. Veux-tu que je cherche sur le web ? Réponds par oui ou non.`
   - si l’utilisateur répond **oui** → recherche web,
   - si **non** → l’assistant reste sur les docs internes / smalltalk.

---

### 🔹 1.2. Outils intégrés (Agents)

Les outils sont implémentés dans `agents.py`, et sélectionnés automatiquement via le routeur `router.py`.

#### 🧮 Calculatrice intelligente

- Comprend des expressions comme :
  - `2 + 3 * 4`
  - `2^8`
  - `23²`, `10³`
  - `sqrt16`, `log10(100)`, `exp2`
  - `sin45`, `cos30`, `tan60`, `sin 45°`, `cos 30deg`
- Normalisations automatiques :
  - `,` → `.`  
  - `^` → `**`  
  - `×`, `÷`, `−`, `–` → `*`, `/`, `-`
  - conversion degrés → radians (`sin30°` → `sin(0.5235...)`)
- Sécurisée :
  - pas de `eval` Python,
  - parsing via **AST**,
  - seules certaines opérations / fonctions / constantes sont autorisées.

#### 🌦️ Météo

- Comprend des requêtes en langage naturel :
  - `quel temps fait-il à Lyon ?`
  - `donne-moi la météo pour Nice aujourd'hui`
  - `meteo paris`
- Étapes :
  1. Normalisation du nom de ville (`_normalize_city_free_text`).
  2. Géocodage via **Nominatim (OpenStreetMap)**.
  3. Météo actuelle via **Open-Meteo**.
  4. Fallback sur un petit dictionnaire interne (`Paris`, `Lyon`, `Marseille`, etc.) si les APIs externes échouent.

#### 🔍 Recherche web

- Utilise **DuckDuckGo Search** via la librairie `ddgs`.
- Deux manières de l’utiliser :
  - **explícite** :  
    `recherche sur le web la cuisine italienne`  
    `cherche sur internet les réseaux de neurones`
  - **après échec du RAG** (avec consentement) :
    - l’assistant demande **oui/non**
    - si **oui**, il affiche une liste de résultats formatés (titre + lien + extrait).

#### 📝 Gestion TODO

- Commandes en langage naturel :
  - `ajoute : réviser IA`
  - `ajoute réviser réseaux`
  - `liste` / `list`
  - `termine 2` / `done: 2`
- Les tâches sont stockées dans `todo_store.json` (persistance entre les sessions).
- L’interface Streamlit reformate le JSON en liste lisible avec :
  - ✅ tâches terminées  
  - 🔹 tâches en cours

#### 💬 Smalltalk

- Gère les salutations simples :
  - `bonjour`, `salut`, `coucou`, `bonsoir`, `hello`, `hey`…
- Utilise un LLM local via Ollama (`llama3.2:3b`) avec un prompt simple :
  > "Tu es un assistant amical et bref."

---

### 🔹 1.3. Envoi de la dernière réponse par e-mail

- Configuration dans `.env` :

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=ton.email@gmail.com
SMTP_PASS=mot_de_passe_application
SMTP_FROM=ton.email@gmail.com


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

