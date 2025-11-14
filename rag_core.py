from __future__ import annotations
"""
rag_core.py
-----------
RAG simple basé sur :

- Dossier DOCS_DIR = RAG_Data/ (.txt, .pdf, .docx)
- Vector store persistant : Chroma
- Embeddings : sentence-transformers/all-MiniLM-L6-v2
- LLM : Ollama (llama3.2:3b)

Logique générale :

1) On recharge (ou crée) un index Chroma sur les documents de RAG_Data/.
2) On récupère quelques chunks les plus proches (k=4).
3) On privilégie les chunks provenant du même fichier que le top-1.
4) On construit un contexte concaténé, limité en taille.
5) On vérifie la pertinence via un test de recouvrement lexical
   **tolérant aux fautes d’orthographe**.
6) Si le contexte ne semble pas parler de la question → « Je ne sais pas. »
7) Sinon, on appelle le LLM local via LangChain.

Cette logique fonctionne quelle que soit la liste de fichiers
présents dans RAG_Data/ (tu peux en ajouter/enlever librement).
"""

from pathlib import Path
from typing import List, Dict, Any

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# ───────────────────── Constantes & chemins ─────────────────────

ROOT = Path(__file__).parent

# Dossier contenant tes cours / docs.
DOCS_DIR = ROOT / "RAG_Data"

# Dossier où Chroma va stocker l’index (persistant).
PERSIST_PATH = ROOT / "chroma_store"

# Taille des chunks pour le découpage.
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Modèle d’embedding HuggingFace (téléchargé une seule fois).
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Modèle Ollama utilisé pour la génération.
LLM_MODEL = "llama3.2:3b"

SYSTEM_PROMPT = (
    "Tu es un tuteur précis et concis.\n"
    "- Tu dois répondre UNIQUEMENT à partir du contexte fourni.\n"
    "- Si le contexte ne parle pas clairement du sujet de la question,\n"
    "  réponds EXACTEMENT : « Je ne sais pas. »\n"
)


# ─────────────────── Chargement / découpage des docs ───────────────────

def _load_documents():
    """
    Charge les documents depuis DOCS_DIR (txt, pdf, docx)
    et les découpe en morceaux (chunks) prêts pour l’indexation.
    """
    if not DOCS_DIR.exists():
        raise RuntimeError(f"Dossier introuvable: {DOCS_DIR.resolve()}")

    loaders = []

    for p in DOCS_DIR.rglob("*.txt"):
        loaders.append(TextLoader(str(p), encoding="utf-8"))
    for p in DOCS_DIR.rglob("*.pdf"):
        loaders.append(PyPDFLoader(str(p)))
    for p in DOCS_DIR.rglob("*.docx"):
        loaders.append(Docx2txtLoader(str(p)))

    if not loaders:
        raise RuntimeError(
            f"Aucun .txt / .pdf / .docx trouvé dans {DOCS_DIR.resolve()}"
        )

    docs = []
    for ld in loaders:
        docs.extend(ld.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def _get_vectorstore() -> Chroma:
    """
    Retourne un vector store Chroma persistant.

    - Si le dossier existe et contient déjà une collection,
      on le réutilise.
    - Sinon, on (ré)indexe tous les documents.
    """
    PERSIST_PATH.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = Chroma(
        collection_name="docs",
        embedding_function=embeddings,
        persist_directory=str(PERSIST_PATH),
    )

    # Si la collection est vide → première indexation.
    try:
        count = vs._collection.count()
    except Exception:
        count = 0

    if count == 0:
        docs = _load_documents()
        if not docs:
            raise RuntimeError("Aucun document à indexer.")
        vs.add_documents(docs)

    return vs


def _get_llm() -> ChatOllama:
    """Retourne le LLM local (Ollama)."""
    return ChatOllama(model=LLM_MODEL, temperature=0.0)


# ─────────────── Extraction de mots-clés & recouvrement ───────────────

_STOPWORDS_FR = {
    "le", "la", "les", "un", "une", "des",
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "de", "du", "dans", "en", "et", "ou", "au", "aux",
    "c", "ce", "cet", "cette", "ces",
    "est", "suis", "es", "sommes", "êtes", "sont",
    "quoi", "que", "qui", "dont", "où",
    "à", "a", "pour", "par", "avec", "sans",
    "comment", "pourquoi", "quel", "quelle", "quels", "quelles",
}


def _extract_keywords(text: str) -> set[str]:
    """
    Transforme un texte brut en un petit ensemble de mots-clés :

    - mise en minuscules
    - suppression de la ponctuation
    - retrait des stopwords
    - on garde uniquement les tokens de longueur >= 3
    """
    import re

    if not text:
        return set()

    t = text.lower()
    t = re.sub(r"[^a-zà-öø-ÿ0-9]+", " ", t)
    tokens = t.split()

    return {
        tok for tok in tokens
        if len(tok) >= 3 and tok not in _STOPWORDS_FR
    }


def _has_lexical_overlap(question: str, context: str) -> bool:
    """
    Vérifie s'il existe un recouvrement lexical **robuste** entre
    la question et le contexte.

    Objectifs :
    - Éviter les hallucinations quand le contexte est totalement hors-sujet.
    - Accepter les fautes d'orthographe courantes, les pluriels, etc.

    Stratégie :
    - On extrait les mots-clés de la question et du contexte.
    - Si l'intersection exacte n'est pas vide → OK.
    - Sinon, on cherche un recouvrement "flou" :
      * préfixes identiques (au moins 4 lettres)
      * inclusion (ex: "reseau" vs "reseaux")
    """

    q_kw = _extract_keywords(question)
    if not q_kw:
        # pas de mots utiles dans la question → on ne bloque pas
        return True

    ctx_kw = _extract_keywords(context)
    if not ctx_kw:
        return False

    # 1) Intersection exacte
    if q_kw & ctx_kw:
        return True

    # 2) Recouvrement approximatif pour tolérer les fautes / variantes
    #    Exemple : "aplications" vs "applications"
    for q in q_kw:
        for c in ctx_kw:
            # On ne regarde que des mots un peu longs (>= 4)
            if len(q) >= 4 and len(c) >= 4:
                # l'un est contenu dans l'autre (ex: "reseau" / "reseaux")
                if q in c or c in q:
                    return True
                # même préfixe sur 4 lettres (ex: "appl" ...)
                if q[:4] == c[:4]:
                    return True

    return False


# ───────────────────────── RAG principal ─────────────────────────

def answer_question(
    question: str,
    chat_history: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """
    RAG minimal sans chain toute faite.
    Retourne {"answer": str, "source_documents": [docs...]}

    Étapes :

      1) Récupérer k=4 chunks les plus proches via Chroma.
      2) Garder ceux du même fichier que le meilleur résultat.
      3) Concaténer ces chunks dans un seul contexte (limité à 2000 caractères).
      4) Contrôler la pertinence via _has_lexical_overlap().
         - Si le contexte n'a aucun lien lexical avec la question → "Je ne sais pas."
      5) Sinon, appeler le LLM (Ollama) avec un prompt clair.
    """
    vs = _get_vectorstore()
    llm = _get_llm()

    # 1) Récupération des documents
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    source_docs = retriever.invoke(question)

    if not source_docs:
        return {"answer": "Je ne sais pas.", "source_documents": []}

    # 2) Filtrer par fichier principal (celui du premier document)
    primary_source = None
    for d in source_docs:
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source")
        if src:
            primary_source = src
            break

    if primary_source:
        same_file_docs = []
        for d in source_docs:
            meta = getattr(d, "metadata", {}) or {}
            if meta.get("source") == primary_source:
                same_file_docs.append(d)
        if same_file_docs:
            source_docs = same_file_docs

    # 3) Contexte concaténé (limité à max_chars)
    max_chars = 2000
    context_parts: List[str] = []
    total = 0
    for d in source_docs:
        chunk = (d.page_content or "").strip().replace("\n", " ")
        if not chunk:
            continue
        left = max_chars - total
        if left <= 0:
            break
        use = chunk[:left]
        context_parts.append(use)
        total += len(use)

    context = "\n\n---\n\n".join(context_parts) if context_parts else ""

    # Si aucun contexte exploitable → on ne force pas le LLM
    if not context:
        return {"answer": "Je ne sais pas.", "source_documents": []}

    # 4) Filtre de pertinence (évite de répondre à côté de la plaque)
    if not _has_lexical_overlap(question, context):
        return {"answer": "Je ne sais pas.", "source_documents": []}

    # 5) Construction d’un historique léger (optionnel)
    history_text = ""
    if chat_history:
        for turn in chat_history[-6:]:
            role = turn.get("role", "user")
            content = (turn.get("content", "") or "").replace("\n", " ")
            if role == "user":
                history_text += f"Utilisateur: {content}\n"
            else:
                history_text += f"Assistant: {content}\n"

    # Prompt clair pour limiter les hallucinations.
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Historique (récent):\n{history}\n\n"
            "Contexte (extraits de documents internes):\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Consignes:\n"
            "- Réponds UNIQUEMENT avec les informations du contexte ci-dessus.\n"
            "- Si le contexte ne permet pas de répondre précisément, dis exactement : « Je ne sais pas. »"
        ),
    ])

    messages = prompt.format_messages(
        history=history_text,
        context=context,
        question=question,
    )

    result = llm.invoke(messages)
    answer_text = getattr(result, "content", str(result))

    return {"answer": answer_text, "source_documents": source_docs}


# ─────────────────── Fonctions utilitaires (réindex) ───────────────────

def reindex():
    """
    Réindexation complète : efface le dossier Chroma et reconstruit.
    Utilisé par le bouton 'Réindexer' de la sidebar.
    """
    import shutil
    if PERSIST_PATH.exists():
        shutil.rmtree(PERSIST_PATH, ignore_errors=True)
    _ = _get_vectorstore()
