# 🎓 Assistant IA Académique — RAG + Agents + Ollama + Streamlit

Assistant académique capable de :

- répondre à des questions à partir de **documents internes** (RAG),
- effectuer des **calculs** (calculatrice sécurisée),
- donner la **météo**,
- faire des **recherches web** (avec consentement explicite ou détection d’actualité),
- gérer une **TODO-list persistante**,
- envoyer la **dernière réponse par e-mail**,
- discuter en **smalltalk** avec un modèle local Ollama.

Le tout est orchestré avec **Streamlit**, des outils maison dans `agents.py`, un **RAG léger sans LLM**, et **Ollama** uniquement pour la partie conversationnelle.

---

## ✅ Objectifs pédagogiques (TP)

Ce projet illustre :

- un **RAG simple** basé sur des fichiers texte locaux,
- des **agents / outils** (calculatrice, météo, TODO, web),
- un **routage intelligent** (smalltalk / outils / RAG / web),
- une **mémoire conversationnelle** persistante,
- une **interface conversationnelle** avec Streamlit,
- une **recherche web** avec demande de consentement,
- un **envoi d’e-mails** via SMTP,
- un code structuré et versionné (Git).

---

## 🧠 1. RAG interne (sans LLM)

**Fichier :** `rag_core.py`

### Principe

- On charge tous les fichiers **`.txt`** du dossier `RAG_Data/`.
- Chaque fichier est découpé en **sections Markdown** à partir des lignes qui commencent par `##`.
- Chaque section devient un petit “document” avec :
  - `page_content` : titre + texte de la section,
  - `metadata["source"]` : chemin du fichier,
  - `metadata["section_title"]` : titre de la section.

### Recherche d’une réponse

Pour une question :

1. La question est **normalisée** (minuscules, accents enlevés, ponctuation simplifiée).
2. On essaie d’abord de trouver une **section dont le titre correspond** à la question :
   - soit le titre est contenu dans la question,
   - soit la question est contenue dans le titre.
3. Sinon, on calcule un **score combiné** pour chaque section :
   - similarité floue entre le titre et la question,
   - nombre de mots-clés communs (normalisés).
4. On garde la meilleure section **seulement si le score est suffisant**  
   (pour éviter de raconter n’importe quoi).
5. Si aucune section n’est jugée pertinente :
   - `answer = "La réponse ne se trouve pas dans les documents internes."`
   - `source_documents = []`

📌 **Important :**  
Le RAG **ne fait appel à aucun LLM**.  
La réponse est un **extrait brut** de tes cours (`Cours_IA.txt`, `Cours_Python.txt`, `Cours_Reseaux.txt`, etc.).

---

## 🤖 2. Agents / Outils (`agents.py`)

Tous les outils renvoient du **texte prêt à afficher** dans `app.py`.

---

### 🧮 2.1. Calculatrice sécurisée

- Analyse les expressions mathématiques via l’AST Python (pas de `eval`).
- Opérations et fonctions autorisées :
  - `+`, `-`, `*`, `/`, `**`
  - `sqrt`, `sin`, `cos`, `tan`, `log`, `log10`, `exp`
  - constantes : `pi`, `e`

#### Expressions comprises

Exemples d’expressions reconnues :

- `2 + 3 * 4`
- `2^8` → `2**8`
- `2² + 3³`
- `sqrt16`, `log50`, `exp2`
- `sin45`, `cos30`, `tan60`  
  → les angles sont interprétés en **degrés** puis convertis en radians :
  - `sin45` → `sin(0,785398...)`
  - `sin 45°` ou `sin(45deg)` idem
- `e4` → `e**4`
- `5(4*5)` → `5*(4*5)` (multiplication implicite)

#### Sécurité

- Seuls certains types de nœuds AST sont autorisés.
- Les noms non autorisés lèvent une erreur (`Symbole non autorisé`).
- En cas de problème :  
  `Résultat: Erreur calcul: ...`

---

### 🌦️ 2.2. Météo mondiale (wttr.in)

Fonctions :

- `tool_weather(city: str)` (asynchrone)
- `tool_weather_sync(city: str)` (synchrone pour Streamlit)

Caractéristiques :

- Utilise `wttr.in` en mode JSON (`format=j1`).
- Supporte des requêtes en texte libre :
  - `meteo rouen`
  - `donne la météo à nantes`
  - `meteo vinci`
- La fonction `_normalize_city_free_text()` :
  - filtre les mots outils (`meteo`, `à`, `la`, etc.),
  - récupère le nom de ville probable,
  - renvoie un nom propre : `Rouen`, `Nantes`, `Vinci`, etc.

Exemple de retour :

```text
Ville: Vinci
Température: 4°C
Vent: 22 km/h

### 🌐 2.3. Recherche web (DuckDuckGo)

**Fonction :** `tool_web_search(query: str, max_results: int = 5)`

- Utilise la librairie `ddgs` (DuckDuckGo Search).
- Retourne une **liste JSON** de résultats :

```json
[
  {
    "title": "Titre du résultat",
    "href": "https://exemple.com",
    "body": "Petit extrait du contenu..."
  }
]

### 📝 2.4. TODO-list persistante

- **Fichier de stockage :** `todo_store.json`  
- **Fonction principale :** `tool_todo(cmd: str)`

#### Commandes supportées

**➕ Ajouter une tâche :**

- `ajoute faire les courses`  
- `ajoute : reviser le cours IA`  
- `add reviser le cours réseaux`  

**✅ Marquer une tâche comme terminée :**

- `termine 2`  
- `done 2`  

**📋 Lister les tâches :**

- `liste`  
- `list`  

**🗑️ Vider la liste :**

- `efface tout`  
- `reset`  
- `clear`  

Les tâches sont stockées en **JSON**, et `app.py` reformate la réponse en liste lisible dans l’interface Streamlit.

## 💬 3. Smalltalk (Ollama)

- **Fichiers concernés :** `app.py` et `router.py`  

Le smalltalk gère les messages du type :

- `bonjour`, `salut`, `coucou`  
- `ça va ?`, `comment tu vas ?`, etc.

`router.py` détecte ces formulations et retourne l’intention **`smalltalk`**.

Dans `app.py`, on utilise un modèle local via `ChatOllama` :

- **Modèle configurable** via la variable d’environnement `OLLAMA_MODEL`  
  - valeur par défaut : `llama3.2:3b`
- **Prompt système utilisé :**  
  > "Tu es un assistant amical, bref et poli."

📌 **Important :**  
Ollama **n’est pas utilisé pour le RAG**.  
Il sert uniquement pour la **discussion générale (smalltalk)**.

## 📧 4. Envoi d’e-mail (dernière réponse)

Dans `app.py` :

- L’assistant détecte des commandes du type :
  - `envoi la reponse à ccolins2010@yahoo.fr`
  - `envoie la réponse à mon mail ...`
  - `peux-tu envoyer la réponse par email à ...`
- Une adresse e-mail est extraite avec une **regex**, avec correction de petites fautes
  (par exemple : `yahoo;fr` → `yahoo.fr`).
- La fonction `send_email_smtp()` envoie **la dernière réponse de l’assistant**
  à l’adresse détectée.

Configuration SMTP dans `.env` :

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=ton.email@gmail.com
SMTP_PASS=mot_de_passe_application
SMTP_FROM=ton.email@gmail.com


```markdown
## 🔀 5. Routage des requêtes (app.py + router.py)

La fonction principale `handle_user_query()` dans `app.py` suit cet ordre logique :

1. **Réponse oui/non après échec du RAG**
   - Si on attend une réponse à  
     > « Souhaites-tu que je cherche sur le web ? (oui / non) »
   - alors :
     - si l’utilisateur répond **oui** → appel à `tool_web_search(...)`
     - si l’utilisateur répond **non** → l’assistant reste sur les documents internes.

2. **Détection d’une commande e-mail**
   - Si la phrase contient une commande du type :  
     `envoi la réponse à ...`
   - alors `send_email_smtp()` est appelé.

3. **Ajout du message utilisateur à l’historique**
   - Le message est stocké dans `memory_store.json`.

4. **Détection rapide de certains cas**
   - Si le texte contient `calcule`, `combien fait`, etc. → `intent = "calc"`
   - Si le texte parle d’**actualité** ou de  
     `qui est le président ...` → `intent = "web"`

5. **Routage général via `router.py`**
   - Si ce n’est pas un cas forcé, `router.py` décide de l’intention :
     - `smalltalk`, `weather`, `todo`, `web` ou `rag`.

6. **Si un outil est déclenché (calc, météo, todo, web)**
   - `app.py` appelle l’outil correspondant dans `agents.py`,
   - formate la réponse,
   - l’affiche,
   - et l’ajoute à l’historique.

7. **Sinon → RAG interne**
   - `answer_question()` est appelé avec la question.
   - Si une section pertinente est trouvée → on renvoie **l’extrait de cours + la source**.
   - Sinon → on propose :

     > Je n’ai rien trouvé dans les documents internes.  
     > 👉 Souhaites-tu que je cherche sur le web ? (oui / non)

## 🛠️ 7. Installation & Lancement

### 7.1. Cloner le projet

```bash
git clone <URL_DU_REPO>
cd projet-ia-assistant-rag-academique

### 7.2. Créer un environnement virtuel

```bash
python -m venv .venv

### 7.3. Installer les dépendances

```bash
pip install -r requirements.txt

### 7.4. Configurer Ollama

1. Installer **Ollama** sur ta machine (depuis le site officiel).
2. Télécharger le modèle utilisé par l’assistant, puis lancer le serveur :

```bash
ollama pull llama3.2:3b
ollama serve

### 7.5. Configurer le fichier `.env`

Créer un fichier `.env` à la racine du projet avec le contenu suivant :

```env
# Modèle utilisé pour le smalltalk (Ollama)
OLLAMA_MODEL=llama3.2:3b

# SMTP pour l'envoi d'e-mails
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=ton.email@gmail.com
SMTP_PASS=mot_de_passe_application
SMTP_FROM=ton.email@gmail.com

### 7.6. Lancer l’application Streamlit

Dans le terminal (en ayant bien activé l'environnement virtuel) :

```bash
streamlit run app.py

L’interface sera accessible sur : <http://localhost:8501/>

## ✅ 8. Exemples de requêtes à tester

Quelques exemples de requêtes à essayer dans l’interface :

- `qu’est-ce que l’IA ?`
- `Brève histoire de l'IA`
- `c’est quoi Python`
- `c’est quoi un réseau informatique`
- `calcule 2²+log50`
- `calcule sin45`
- `calcule e4`
- `meteo rouen`
- `meteo vinci`
- `ajoute reviser le cours IA`
- `liste`
- `termine 1`
- `efface tout`
- `qui est le president des USA`
- `actualité intelligence artificielle`
- `envoi la reponse à monmail@exemple.com`
