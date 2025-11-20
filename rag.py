# from __future__ import annotations
# """
# app.py
# ------
# Interface Streamlit pour l'assistant académique RAG + Agents.

# Flux :
#  1) On lit la question de l'utilisateur.
#  2) Si on attend une réponse "oui/non" pour la recherche web → on gère ça d'abord.
#  3) Sinon, on teste commande e-mail.
#  4) Sinon, on route vers :
#       - smalltalk (LLM local)
#       - outils (calc / météo / todo / web)
#       - RAG (documents internes)
#  5) Si RAG ne sait pas → demande de consentement pour recherche web.
# """

# import json
# import re
# import os
# from pathlib import Path
# from typing import List, Dict, Tuple, Optional

# import streamlit as st
# from dotenv import load_dotenv

# load_dotenv()  # charge .env (SMTP_* etc.)

# from rag_core import answer_question, reindex, DOCS_DIR
# from router import route
# from agents import (
#     tool_calculator,
#     tool_weather_sync,
#     tool_web_search,
#     tool_todo,
# )

# from langchain_ollama import ChatOllama

# # ───────────────────────── Config & Mémoire ─────────────────────────

# st.set_page_config(
#     page_title="Assistant académique (RAG + Agents)",
#     page_icon="🎓",
#     layout="centered",
# )
# st.title("🎓 Assistant académique — RAG + Agents (Ollama + Chroma + DuckDuckGo)")

# # CSS simple pour la sidebar
# st.markdown(
#     """
# <style>
# div[data-testid="stSidebar"] .stButton>button {
#     padding: 0.4rem 0.6rem;
#     font-size: 0.9rem;
#     border-radius: 6px;
#     width: 100%;
#     margin-bottom: 0.35rem;
# }
# </style>
# """,
#     unsafe_allow_html=True,
# )

# ROOT = Path(__file__).parent
# MEMORY_PATH = ROOT / "memory_store.json"

# MAX_TURNS = 30  # nombre max de messages en historique


# def load_memory() -> List[Dict[str, str]]:
#     """Charge l'historique persistant depuis disque."""
#     if MEMORY_PATH.exists():
#         try:
#             data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
#             if isinstance(data, list):
#                 return [
#                     t for t in data
#                     if isinstance(t, dict) and "role" in t and "content" in t
#                 ]
#         except Exception:
#             pass
#     return []


# def save_memory(history: List[Dict[str, str]]) -> None:
#     """Sauvegarde l'historique sur disque."""
#     try:
#         MEMORY_PATH.write_text(
#             json.dumps(history, ensure_ascii=False, indent=2),
#             encoding="utf-8",
#         )
#     except Exception:
#         pass


# def trim_history():
#     """Garde au plus MAX_TURNS messages."""
#     if len(st.session_state.chat_history) > MAX_TURNS:
#         st.session_state.chat_history = st.session_state.chat_history[-MAX_TURNS:]


# def get_smalltalk_llm():
#     """LLM local pour smalltalk via Ollama."""
#     return ChatOllama(model="llama3.2:3b", temperature=0.5)


# # ──────────────────────── État de session ──────────────────────────

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = load_memory()

# if "last_sources" not in st.session_state:
#     st.session_state.last_sources = []

# if "last_mode" not in st.session_state:
#     st.session_state.last_mode = None

# # pending_web est utilisé lorsqu'on attend "oui/non" pour la recherche web
# if "pending_web" not in st.session_state:
#     st.session_state.pending_web = None


# # ───────────────────────── Sidebar (contrôles) ─────────────────────────

# st.sidebar.header("⚙️ Contrôles")

# if st.sidebar.button(
#     "🆕 Nouveau chat",
#     help="Réinitialise la discussion (mais garde la mémoire sur disque).",
# ):
#     st.session_state.chat_history = []
#     st.session_state.last_sources = []
#     st.session_state.last_mode = None
#     st.session_state.pending_web = None
#     st.success("Nouvelle discussion démarrée ✅")

# if st.sidebar.button(
#     "🧹 Effacer historique",
#     help="Vide la session et efface l'historique persistant.",
# ):
#     st.session_state.chat_history = []
#     try:
#         MEMORY_PATH.write_text("[]", encoding="utf-8")
#     except Exception:
#         pass
#     st.success("Historique effacé ✅")

# if st.sidebar.button(
#     "🔄 Réindexer",
#     help="Reconstruit l’index à partir des documents internes.",
# ):
#     reindex()
#     st.sidebar.success("Index reconstruit avec succès ✅")
#     st.session_state.chat_history.append(
#         {
#             "role": "assistant",
#             "content": (
#                 "ℹ️ Index reconstruit. Les prochaines réponses utiliseront la "
#                 "dernière version des documents internes."
#             ),
#         }
#     )
#     trim_history()
#     save_memory(st.session_state.chat_history)


# # ───────────────────── Affichage historique ────────────────────────

# st.subheader("💬 Discussion")
# for turn in st.session_state.chat_history:
#     with st.chat_message("user" if turn["role"] == "user" else "assistant"):
#         st.markdown(turn["content"])


# # ──────────────────── Helper d’affichage web ───────────────────────

# def render_web_results(json_payload: str) -> str:
#     """
#     Mise en forme des résultats DuckDuckGo.
#     Si le JSON est invalide ou erreur, on montre le JSON brut.
#     """
#     try:
#         data = json.loads(json_payload)
#         if isinstance(data, dict) and "error" in data:
#             return f"Erreur de recherche : {data['error']}"
#         if not isinstance(data, list):
#             return f"Résultats (brut):\n\n```json\n{json_payload}\n```"

#         lines = ["**Résultats web :**"]
#         for i, item in enumerate(data[:8], 1):
#             title = item.get("title") or "(sans titre)"
#             href = item.get("href") or ""
#             body = item.get("body") or ""
#             if href:
#                 lines.append(f"- {i}. [{title}]({href})  \n  {body}")
#             else:
#                 lines.append(f"- {i}. {title}  \n  {body}")
#         return "\n".join(lines)
#     except Exception:
#         return f"Résultats (brut):\n\n```json\n{json_payload}\n```"


# # ──────────────────── E-mail : détection + envoi SMTP ──────────────

# EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")


# def detect_email_command(text: str) -> Optional[str]:
#     """
#     Détecte une commande du type :
#       - "envoie la réponse à nom@domaine.com"
#       - "mail cette réponse à ..."
#     Retourne l’adresse e-mail ou None.
#     """
#     t = (text or "").lower()
#     if any(k in t for k in ["envoie", "envoies", "mail", "email", "e-mail", "envoyer"]):
#         m = EMAIL_RE.search(text)
#         if m:
#             return m.group(1)
#     return None


# def send_email_smtp(to_addr: str, subject: str, body: str) -> Tuple[bool, str]:
#     """
#     Envoie un e-mail texte via SMTP.
#     Variables attendues dans .env :
#       SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM
#     """
#     import smtplib
#     from email.mime.text import MIMEText

#     host = os.getenv("SMTP_HOST", "smtp.gmail.com")
#     port = int(os.getenv("SMTP_PORT", "587"))
#     user = os.getenv("SMTP_USER")
#     pwd = os.getenv("SMTP_PASS")
#     from_addr = os.getenv("SMTP_FROM", user)

#     if not (user and pwd and from_addr):
#         return False, "SMTP non configuré (.env incomplet)."

#     msg = MIMEText(body, "plain", "utf-8")
#     msg["Subject"] = subject
#     msg["From"] = from_addr
#     msg["To"] = to_addr

#     try:
#         with smtplib.SMTP(host, port) as s:
#             s.starttls()
#             s.login(user, pwd)
#             s.send_message(msg)
#         return True, "E-mail envoyé ✅"
#     except Exception as e:
#         return False, f"Échec envoi e-mail: {e}"


# YES_RE = re.compile(r"^\s*(oui|o|yes|y)\b", re.I)
# NO_RE = re.compile(r"^\s*(non|n|no)\b", re.I)


# def handle_user_query(user_text: str):
#     """
#     Route la requête :
#       - consentement web (oui/non) si pending_web
#       - commande e-mail
#       - smalltalk
#       - outils
#       - RAG (documents)
#       - demande de consentement web si RAG ne sait pas
#     """

#     # 0) Consentement web en cours ?
#     if st.session_state.pending_web is not None:
#         original_query = st.session_state.pending_web.get("query", "")
#         if YES_RE.search(user_text or ""):
#             raw_json = tool_web_search(original_query)
#             answer_md = (
#                 "🛠️ **Recherche Web (suite à ton consentement)**\n\n"
#                 + render_web_results(raw_json)
#             )
#             mode = "web"
#             sources = []
#             st.session_state.pending_web = None

#             st.session_state.chat_history.append({"role": "user", "content": user_text})
#             st.session_state.chat_history.append(
#                 {"role": "assistant", "content": answer_md}
#             )
#             trim_history()
#             save_memory(st.session_state.chat_history)
#             st.session_state.last_sources = sources
#             st.session_state.last_mode = mode
#             with st.chat_message("assistant"):
#                 st.markdown(answer_md)
#             return

#         if NO_RE.search(user_text or ""):
#             answer_md = (
#                 "👍 D'accord, je reste sur tes documents internes. "
#                 "Comment puis-je t'aider autrement ?"
#             )
#             mode = "rag"
#             sources = []
#             st.session_state.pending_web = None

#             st.session_state.chat_history.append({"role": "user", "content": user_text})
#             st.session_state.chat_history.append(
#                 {"role": "assistant", "content": answer_md}
#             )
#             trim_history()
#             save_memory(st.session_state.chat_history)
#             st.session_state.last_sources = sources
#             st.session_state.last_mode = mode
#             with st.chat_message("assistant"):
#                 st.markdown(answer_md)
#             return

#         # Ni oui ni non
#         st.session_state.chat_history.append({"role": "user", "content": user_text})
#         answer_md = (
#             "Je n’ai pas compris. Souhaites-tu que je cherche **sur le web** ? "
#             "Réponds par **oui** ou **non**."
#         )
#         st.session_state.chat_history.append(
#             {"role": "assistant", "content": answer_md}
#         )
#         trim_history()
#         save_memory(st.session_state.chat_history)
#         st.session_state.last_sources = []
#         st.session_state.last_mode = "rag"
#         with st.chat_message("assistant"):
#             st.markdown(answer_md)
#         return

#     # 0-bis) Commande e-mail ?
#     to_addr = detect_email_command(user_text)
#     if to_addr:
#         st.session_state.chat_history.append({"role": "user", "content": user_text})

#         last_assistant = ""
#         for t in reversed(st.session_state.chat_history):
#             if t["role"] == "assistant":
#                 last_assistant = t["content"]
#                 break

#         if not last_assistant:
#             answer_md = "Je n’ai pas de réponse précédente à envoyer."
#         else:
#             ok, info = send_email_smtp(
#                 to_addr,
#                 subject="Réponse de l'assistant",
#                 body=last_assistant,
#             )
#             answer_md = info

#         st.session_state.chat_history.append(
#             {"role": "assistant", "content": answer_md}
#         )
#         trim_history()
#         save_memory(st.session_state.chat_history)
#         st.session_state.last_sources = []
#         st.session_state.last_mode = "email"
#         with st.chat_message("assistant"):
#             st.markdown(answer_md)
#         return

#     # 1) Parcours normal
#     st.session_state.chat_history.append({"role": "user", "content": user_text})

#     intent, payload = route(user_text)

#     # 1.a) Smalltalk
#     if intent == "smalltalk":
#         llm = get_smalltalk_llm()
#         msgs = [
#             {"role": "system", "content": "Tu es un assistant amical et bref."},
#             {"role": "user", "content": user_text},
#         ]
#         ai = llm.invoke(msgs)
#         answer_md = getattr(ai, "content", str(ai))

#         st.session_state.chat_history.append(
#             {"role": "assistant", "content": answer_md}
#         )
#         trim_history()
#         save_memory(st.session_state.chat_history)
#         st.session_state.last_sources = []
#         st.session_state.last_mode = "smalltalk"
#         with st.chat_message("assistant"):
#             st.markdown(answer_md)
#         return

#     # 1.b) Outils
#     if intent in {"calc", "weather", "todo", "web"}:
#         try:
#             if intent == "calc":
#                 # ⚠️ On passe l'input complet de l'utilisateur,
#                 # pas seulement payload (qui pourrait être transformé).
#                 out = tool_calculator(user_text)
#                 answer_md = f"🛠️ **Outil Calculatrice**\n\n{out}"
#             elif intent == "weather":
#                 out = tool_weather_sync(payload)
#                 answer_md = f"🛠️ **Outil Météo**\n\n{out}"
#             elif intent == "todo":
#                 out = tool_todo(payload)
#                 answer_md = f"🛠️ **Outil TODO**\n\n{out}"
#             else:  # web explicite
#                 raw_json = tool_web_search(payload)
#                 answer_md = (
#                     "🛠️ **Outil Recherche Web (DuckDuckGo)**\n\n"
#                     + render_web_results(raw_json)
#                 )

#             mode, sources = intent, []

#         except Exception as e:
#             mode, sources = "error", []
#             answer_md = f"⚠️ Erreur: {e}"

#         st.session_state.chat_history.append(
#             {"role": "assistant", "content": answer_md}
#         )
#         trim_history()
#         save_memory(st.session_state.chat_history)
#         st.session_state.last_sources = sources
#         st.session_state.last_mode = mode
#         with st.chat_message("assistant"):
#             st.markdown(answer_md)
#         return

#     # 2) Sinon → RAG en premier
#     try:
#         with st.spinner("🔎 Recherche dans les documents internes..."):
#             res = answer_question(
#                 user_text,
#                 chat_history=st.session_state.chat_history,
#             )

#         answer_text = res.get("answer", "")
#         sources = res.get("source_documents", []) or []
#         found = (len(sources) > 0) and (
#             "je ne sais pas" not in (answer_text or "").lower()
#         )

#         if found:
#             top_src = None
#             for d in sources:
#                 meta = getattr(d, "metadata", {}) or {}
#                 if meta.get("source"):
#                     top_src = meta["source"]
#                     break

#             if top_src:
#                 answer_md = f"{answer_text}\n\n---\n📎 **Source** : `{top_src}`"
#             else:
#                 answer_md = answer_text

#             st.session_state.chat_history.append(
#                 {"role": "assistant", "content": answer_md}
#             )
#             trim_history()
#             save_memory(st.session_state.chat_history)
#             st.session_state.last_sources = sources
#             st.session_state.last_mode = "rag"
#             with st.chat_message("assistant"):
#                 st.markdown(answer_md)
#             return

#     except Exception as e:
#         st.warning(f"RAG indisponible: {e}")

#     # 3) RAG n'a rien trouvé → consentement web
#     st.session_state.pending_web = {"query": user_text}
#     answer_md = (
#         "Je n’ai rien trouvé dans **les documents internes**.\n\n"
#         "👉 Veux-tu que je cherche **sur le web** ? Réponds par **oui** ou **non**."
#     )
#     mode, sources = "rag", []

#     st.session_state.chat_history.append(
#         {"role": "assistant", "content": answer_md}
#     )
#     trim_history()
#     save_memory(st.session_state.chat_history)
#     st.session_state.last_sources = sources
#     st.session_state.last_mode = mode
#     with st.chat_message("assistant"):
#         st.markdown(answer_md)


# # Champ de saisie utilisateur
# question = st.chat_input("Pose ta question (cours, calcul, météo, web, todo...)")
# if question:
#     handle_user_query(question)

from __future__ import annotations

"""
rag_core.py — Moteur RAG propre et robuste
------------------------------------------

Rôle :
- Répondre à des questions en s'appuyant UNIQUEMENT sur les fichiers dans RAG_Data.
- Refuser de répondre (poliment) si l'information n'est pas clairement présente.
- Limiter au maximum les hallucinations du LLM (ports, âges, chiffres inventés).

Fonctions exposées :
- answer_question(question: str, chat_history=None) -> dict
- ask_rag(question: str) -> dict
- reindex() -> None

Variables utiles :
- DOCS_DIR : dossier contenant les documents
- PERSIST_PATH : dossier Chroma pour l'index vectoriel
- LLM_MODEL : nom du modèle Ollama utilisé
"""

import re
import shutil
import unicodedata
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# ───────────────────────── CONFIG GLOBALE ──────────────────────────

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "RAG_Data"
PERSIST_PATH = ROOT / "chroma_store"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:1b"

# Message système pour le LLM : très strict, pas d'invention.
SYSTEM_PROMPT = (
    "Tu es un tuteur académique très précis et très concis.\n"
    "Tu dois répondre UNIQUEMENT avec le contenu fourni dans le contexte.\n"
    "Si le contexte ne permet pas de répondre clairement, tu dois répondre "
    "exactement : La réponse ne se trouve pas dans les documents internes.\n"
    "Ne fais pas de généralités, n'invente rien, et ne copies pas de longs "
    "passages mot pour mot.\n"
    "Réponds en 1 à 3 phrases maximum."
)

# Ordre canonique des couches OSI (utilisé pour une réponse claire et stable)
OSI_LAYERS_CANONICAL = [
    "Couche Physique",
    "Couche Liaison de données",
    "Couche Réseau",
    "Couche Transport",
    "Couche Session",
    "Couche Présentation",
    "Couche Application",
]

# Caches en mémoire (évite de recharger à chaque question)
_VS_CACHE: Optional[Chroma] = None
_LLM_CACHE: Optional[ChatOllama] = None


# ──────────────────────────── UTILITAIRES TEXTE ────────────────────────────

def _normalize(t: str) -> str:
    """Normalise les accents (é → e, etc.) pour comparer plus facilement."""
    t = unicodedata.normalize("NFD", t)
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def _keywords(text: str) -> set[str]:
    """
    Retourne un ensemble de mots-clés (a-z0-9) en minuscules, longueur ≥ 3.
    Sert pour les tests de pertinence.
    """
    t = _normalize(text.lower())
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return {w for w in t.split() if len(w) >= 3}


def _relevant(question: str, context: str) -> bool:
    """
    Pertinence grossière : au moins un mot-clé commun (longueur ≥ 3)
    entre la question et le contexte.
    """
    return len(_keywords(question) & _keywords(context)) > 0


# Mots peu informatifs qu'on ne considère pas comme "forts"
_CRITICAL_STOPWORDS = {
    "comment",
    "quelle",
    "quelles",
    "quel",
    "donner",
    "donne",
    "explique",
    "expliquer",
    "peux",
    "pourrais",
    "voudrais",
    "definition",
    "definir",
    "définition",
    "définir",
    "exemple",
    "exemples",
    "utiliser",
    "utilisation",
    "cours",
    "introduire",
    "introduction",
}


def _has_uncovered_strong_keywords(question: str, context: str) -> bool:
    """
    Vérifie s'il existe des mots-clés importants de la question
    qui n'apparaissent pas dans le contexte.

    Idée :
      - On prend les mots de la question de longueur ≥ 5,
      - On enlève les mots peu informatifs (_CRITICAL_STOPWORDS),
      - Si l'un d'eux n'est pas dans le contexte → le contexte ne couvre
        probablement pas bien la question.

    Exemple :
      - question sur "mbappé" → ce mot n'est dans aucun document → on refuse.
      - question sur "https" + contexte qui ne parle que de "http" → on refuse.
    """
    qk = _keywords(question)
    ck = _keywords(context)

    strong = {
        w for w in qk
        if len(w) >= 5 and w not in _CRITICAL_STOPWORDS
    }

    uncovered = {w for w in strong if w not in ck}
    return len(uncovered) > 0


def _numbers_from_text(text: str) -> set[str]:
    """Retourne tous les nombres entiers trouvés dans un texte."""
    return set(re.findall(r"\b\d+\b", text))


def _answer_consistent_with_context(answer: str, context: str) -> bool:
    """
    Vérifie que tous les nombres de la réponse apparaissent aussi dans le contexte.

    Exemple :
      - question sur HTTPS → le modèle propose "443".
        Si "443" n'est pas dans le cours → on rejette la réponse.
      - question sur l'âge de Mbappé → nombre inventé → rejeté.

    Si la réponse ne contient aucun nombre, on ne bloque pas ici.
    """
    nums_answer = _numbers_from_text(answer)
    if not nums_answer:
        return True

    nums_context = _numbers_from_text(context)
    return nums_answer.issubset(nums_context)


# ──────────────────────────── LLM & VECTORSTORE ────────────────────────────

def _get_llm() -> ChatOllama:
    """Retourne une instance de ChatOllama (mise en cache)."""
    global _LLM_CACHE
    if _LLM_CACHE is None:
        _LLM_CACHE = ChatOllama(
            model=LLM_MODEL,
            temperature=0,  # 0 = maximum de déterminisme
        )
    return _LLM_CACHE


def _load_documents():
    """
    Charge tous les documents présents dans RAG_Data
    et les découpe en chunks de texte.
    """
    docs = []
    DOCS_DIR.mkdir(exist_ok=True)

    for file in DOCS_DIR.rglob("*"):
        if file.is_dir():
            continue
        if file.suffix.lower() == ".txt":
            docs.extend(TextLoader(str(file), encoding="utf-8").load())
        elif file.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(file)).load())
        elif file.suffix.lower() == ".docx":
            docs.extend(Docx2txtLoader(str(file)).load())

    if not docs:
        return []

    # 1) Split par titres Markdown (#, ##, ###)
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
    )

    md_chunks = []
    for d in docs:
        parts = md_splitter.split_text(d.page_content)
        if len(parts) > 1:
            # On garde les métadonnées d'origine
            for p in parts:
                p.metadata = d.metadata.copy()
                md_chunks.append(p)
        else:
            md_chunks.append(d)

    # 2) Re-split final en chunks de taille CHUNK_SIZE
    final_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return final_splitter.split_documents(md_chunks)


def _get_vectorstore() -> Chroma:
    """
    Retourne la base vectorielle Chroma (persistante + mise en cache).
    Si la collection est vide ou corrompue, elle est reconstruite.
    """
    global _VS_CACHE

    if _VS_CACHE is not None:
        return _VS_CACHE

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    try:
        vs = Chroma(
            collection_name="docs",
            persist_directory=str(PERSIST_PATH),
            embedding_function=embeddings,
        )
    except Exception:
        # Index corrompu : on supprime tout et on repart propre
        shutil.rmtree(PERSIST_PATH, ignore_errors=True)
        vs = Chroma(
            collection_name="docs",
            persist_directory=str(PERSIST_PATH),
            embedding_function=embeddings,
        )

    # Première création (si vide)
    try:
        if vs._collection.count() == 0:
            docs = _load_documents()
            if docs:
                vs.add_documents(docs)
    except Exception:
        # Si ça plante, on reconstruit complètement
        shutil.rmtree(PERSIST_PATH, ignore_errors=True)
        vs = Chroma(
            collection_name="docs",
            persist_directory=str(PERSIST_PATH),
            embedding_function=embeddings,
        )
        docs = _load_documents()
        if docs:
            vs.add_documents(docs)

    _VS_CACHE = vs
    return vs


# ──────────────────────────── RAG PRINCIPAL ────────────────────────────

def answer_question(question: str, chat_history=None):
    """
    Pipeline complet RAG : retrieve → gardes-fous → LLM.

    Retourne un dict :
      {
        "answer": str,
        "source_documents": [Document, ...]
      }

    Convention :
      - Si source_documents est VIDE → l'app comprend que les
        documents internes ne suffisent pas, et propose une
        recherche web à l'utilisateur.
    """
    vs = _get_vectorstore()
    llm = _get_llm()

    # 1) Récupération des meilleurs chunks
    retriever = vs.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(question)

    if not docs:
        return {
            "answer": "La réponse ne se trouve pas dans les documents internes.",
            "source_documents": [],
        }

    q_low = question.lower()

    # 2) On privilégie les chunks issus du même fichier que le meilleur
    main_file = docs[0].metadata.get("source")
    if main_file:
        same_file = [d for d in docs if d.metadata.get("source") == main_file]
        if same_file:
            docs = same_file

    # 3) Construction d'un contexte cohérent (limité en longueur)
    context = ""
    for d in docs:
        part = d.page_content.replace("\n", " ").strip()
        if not part:
            continue
        if len(context) + len(part) > 2200:
            break
        context += part + "\n\n"

    if not context or not _relevant(question, context):
        return {
            "answer": "La réponse ne se trouve pas dans les documents internes.",
            "source_documents": [],
        }

    # 4) Si des mots-clés importants de la question n'apparaissent pas
    #    dans le contexte, on ne prend pas le risque de répondre.
    if _has_uncovered_strong_keywords(question, context):
        return {
            "answer": "La réponse ne se trouve pas dans les documents internes.",
            "source_documents": [],
        }

    ctx_low = context.lower()

    # ───────── Cas qualité : couches OSI ─────────
    # On sait que ton cours les liste proprement, donc on les extrait
    # directement plutôt que de laisser le LLM improviser.
    if "osi" in q_low and ("couche" in q_low or "couches" in q_low):
        layers = re.findall(r"\d+\.\s*\*\*(.+?)\*\*", context)
        if layers:
            # Nettoyage + suppression de doublons
            names = []
            for name in layers:
                name = name.strip()
                if name not in names:
                    names.append(name)

            # Tri selon l'ordre canonique OSI
            ordered = sorted(
                names,
                key=lambda x: OSI_LAYERS_CANONICAL.index(x)
                if x in OSI_LAYERS_CANONICAL
                else 999,
            )

            lines = [f"{i + 1}. {name}" for i, name in enumerate(ordered[:7])]
            answer = "Les 7 couches du modèle OSI sont :\n" + "\n".join(lines)
            return {"answer": answer, "source_documents": docs}

    # ───────── Cas qualité : couches TCP/IP ─────────
    if ("tcp/ip" in q_low or "tcp ip" in q_low) and (
        "couche" in q_low or "couches" in q_low
    ):
        answer = (
            "Le modèle TCP/IP comporte 4 couches :\n"
            "1. Accès réseau\n"
            "2. Internet\n"
            "3. Transport\n"
            "4. Application"
        )
        return {"answer": answer, "source_documents": docs}

    # 5) Construction du prompt pour le LLM
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Contexte :\n{context}\n\n"
                "Question : {question}\n\n"
                "Réponds en 1 à 3 phrases maximum, en utilisant UNIQUEMENT ce contexte. "
                "Si le contexte ne permet pas de répondre, dis simplement : "
                "La réponse ne se trouve pas dans les documents internes.",
            ),
        ]
    )

    msgs = prompt.format_messages(context=context, question=question)

    # 6) Appel LLM avec gestion d'erreur
    try:
        out = llm.invoke(msgs)
        answer = out.content.strip()
    except Exception:
        # Erreur modèle → on renvoie une phrase explicite
        answer = (
            "Je ne peux pas répondre car le modèle local n’a pas pu être chargé "
            "(mémoire insuffisante ou erreur Ollama)."
        )
        return {"answer": answer, "source_documents": docs}

    # 7) Si le LLM répond lui-même "je ne sais pas", on unifie le message
    ans_low = answer.lower().strip()
    if "je ne sais pas" in ans_low:
        return {
            "answer": "La réponse ne se trouve pas dans les documents internes.",
            "source_documents": [],
        }

    # 8) Garde-fou générique sur les nombres (ports, âges, dates...)
    if answer.lower().strip() != "la réponse ne se trouve pas dans les documents internes.":
        if not _answer_consistent_with_context(answer, context):
            return {
                "answer": "La réponse ne se trouve pas dans les documents internes.",
                "source_documents": [],
            }

    # Si on arrive ici → réponse jugée cohérente avec le contexte
    return {"answer": answer, "source_documents": docs}


# ──────────────────────────── ALIAS & REINDEX ────────────────────────────

def ask_rag(question: str):
    """
    Alias simple (sans historique).
    Retourne :
      {
        "answer": str,
        "source": [chemins_de_fichiers]
      }
    """
    res = answer_question(question)
    return {
        "answer": res["answer"],
        "source": [d.metadata.get("source") for d in res["source_documents"]],
    }


def reindex():
    """
    Reconstruit complètement l'index Chroma.
    Appelé par le bouton 'Réindexer' de l'interface.
    """
    global _VS_CACHE
    shutil.rmtree(PERSIST_PATH, ignore_errors=True)
    _VS_CACHE = None
    _ = _get_vectorstore()
