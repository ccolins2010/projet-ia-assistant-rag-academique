from __future__ import annotations

"""
rag_core.py — Moteur RAG (version améliorée et corrigée)
--------------------------------------------------------

Corrections et améliorations effectuées :
- Ajout du split intelligent Markdown (#, ##, ###)
- Chunk size optimisé : 900 + overlap 150
- Conserve MiniLM + Chroma (pas de changement de techno)
- Priorité au bon fichier source pour éviter les mauvaises réponses
- Vérification stricte de pertinence (anti-hallucination renforcée)
- Réponses concises et exactes seulement si trouvées dans le contexte
- Compatible avec llama3.2:1b (moins de RAM)

Mise à jour récente :
- Suppression des warnings de dépréciation LangChain en important :
  - Chroma depuis `langchain_chroma`
  - HuggingFaceEmbeddings depuis `langchain_huggingface`

NOUVELLE MISE À JOUR :
- Cas spécial pour les questions contenant "OSI" :
  on ne garde que les passages qui parlent explicitement du modèle OSI,
  pour éviter de répondre avec les 4 couches du modèle TCP/IP.
"""

import shutil
from pathlib import Path
from typing import Optional
import unicodedata
import re

# ⚠️ IMPORTS MIS À JOUR POUR ÉVITER LES WARNINGS
# Ancien (déprécié) :
#   from langchain_community.vectorstores import Chroma
#   from langchain_community.embeddings import HuggingFaceEmbeddings
# Nouveau (recommandé) :
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


#──────────────────────────────────────────────
# CONFIG
#──────────────────────────────────────────────

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "RAG_Data"
PERSIST_PATH = ROOT / "chroma_store"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "llama3.2:1b"

SYSTEM_PROMPT = (
    "Tu es un tuteur académique précis et concis.\n"
    "Tu NE dois répondre qu’avec le contenu strictement présent dans le contexte.\n"
    "Si l'information n’est pas dans le contexte → tu dois répondre : Je ne sais pas.\n"
    "Pas d'invention, pas d'ajout, pas de généralités."
)

# caches
_VS_CACHE: Optional[Chroma] = None
_LLM_CACHE: Optional[ChatOllama] = None


#──────────────────────────────────────────────
# LLM
#──────────────────────────────────────────────

def _get_llm() -> ChatOllama:
    global _LLM_CACHE
    if _LLM_CACHE is None:
        _LLM_CACHE = ChatOllama(model=LLM_MODEL, temperature=0)
    return _LLM_CACHE


#──────────────────────────────────────────────
# DOCUMENTS + SPLIT INTELLIGENT
#──────────────────────────────────────────────

def _load_documents():
    docs = []

    DOCS_DIR.mkdir(exist_ok=True)

    for file in DOCS_DIR.rglob("*"):

        if file.suffix.lower() == ".txt":
            docs.extend(TextLoader(str(file), encoding="utf-8").load())

        elif file.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(file)).load())

        elif file.suffix.lower() == ".docx":
            docs.extend(Docx2txtLoader(str(file)).load())

    # Split par titres Markdown
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
            for p in parts:
                p.metadata = d.metadata.copy()
                md_chunks.append(p)
        else:
            md_chunks.append(d)

    # Re-split propre (fallback)
    final_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return final_splitter.split_documents(md_chunks)


#──────────────────────────────────────────────
# VECTORSTORE (CHROMA)
#──────────────────────────────────────────────

def _get_vectorstore() -> Chroma:
    global _VS_CACHE

    if _VS_CACHE:
        return _VS_CACHE

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    try:
        vs = Chroma(
            collection_name="docs",
            persist_directory=str(PERSIST_PATH),
            embedding_function=embeddings,
        )
    except Exception:
        shutil.rmtree(PERSIST_PATH, ignore_errors=True)
        vs = Chroma(
            collection_name="docs",
            persist_directory=str(PERSIST_PATH),
            embedding_function=embeddings,
        )

    # Première création
    try:
        # NOTE : l’API Chroma garde la compatibilité, _collection.count() reste disponible
        if vs._collection.count() == 0:
            docs = _load_documents()
            vs.add_documents(docs)
    except Exception:
        shutil.rmtree(PERSIST_PATH, ignore_errors=True)
        vs = Chroma(
            collection_name="docs",
            persist_directory=str(PERSIST_PATH),
            embedding_function=embeddings,
        )
        docs = _load_documents()
        vs.add_documents(docs)

    _VS_CACHE = vs
    return vs


#──────────────────────────────────────────────
# ANTI-HALLUCINATIONS
#──────────────────────────────────────────────

def _normalize(t: str) -> str:
    t = unicodedata.normalize("NFD", t)
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def _keywords(text: str) -> set[str]:
    t = _normalize(text.lower())
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return {w for w in t.split() if len(w) >= 3}


def _relevant(question: str, context: str) -> bool:
    # On reste simple pour ne pas casser le reste :
    # au moins un mot-clé en commun
    return len(_keywords(question) & _keywords(context)) > 0


#──────────────────────────────────────────────
# RAG PRINCIPAL
#──────────────────────────────────────────────

def answer_question(question: str, chat_history=None):
    vs = _get_vectorstore()
    llm = _get_llm()

    retriever = vs.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    if not docs:
        return {"answer": "Je ne sais pas.", "source_documents": []}

    # ───────── Cas spécial : questions sur le modèle OSI ─────────
    #
    # Problème observé :
    # - Pour la question "Quelles sont les 7 couches du modèle OSI ?"
    #   le retriever ramenait parfois le bloc sur le modèle TCP/IP
    #   (4 couches) car il contient aussi les mots "modèle" et "OSI".
    #
    # Correction :
    # - Si la question contient "osi", on filtre les documents pour ne garder
    #   que ceux dont le contenu mentionne explicitement "osi" ou
    #   "open systems interconnection".
    # - Si le filtre ne trouve rien, on garde les docs d'origine (fallback).
    q_low = question.lower()
    if "osi" in q_low:
        osi_docs = []
        for d in docs:
            content_low = d.page_content.lower()
            if "osi" in content_low or "open systems interconnection" in content_low:
                osi_docs.append(d)
        if osi_docs:
            docs = osi_docs  # on spécialise la recherche sur l'OSI

    # Priorité → fichier du meilleur chunk
    main_file = docs[0].metadata.get("source")
    if main_file:
        same_file = [d for d in docs if d.metadata.get("source") == main_file]
        if same_file:
            docs = same_file

    # Construit un contexte cohérent
    context = ""
    for d in docs:
        part = d.page_content.replace("\n", " ").strip()
        if len(context) + len(part) < 2200:
            context += part + "\n\n"

    if not context or not _relevant(question, context):
        return {"answer": "Je ne sais pas.", "source_documents": []}

    # Prompt RAG strict
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Contexte :\n{context}\n\nQuestion : {question}\n\n"
                "Ta réponse doit être concise et issue EXCLUSIVEMENT du contexte.",
            ),
        ]
    )

    msgs = prompt.format_messages(context=context, question=question)
    out = llm.invoke(msgs)
    answer = out.content.strip()

    return {"answer": answer, "source_documents": docs}


#──────────────────────────────────────────────
# ALIAS SIMPLIFIÉ
#──────────────────────────────────────────────

def ask_rag(question: str):
    res = answer_question(question)
    return {
        "answer": res["answer"],
        "source": [d.metadata.get("source") for d in res["source_documents"]],
    }


#──────────────────────────────────────────────
# REINDEX
#──────────────────────────────────────────────

def reindex():
    global _VS_CACHE
    shutil.rmtree(PERSIST_PATH, ignore_errors=True)
    _VS_CACHE = None
    _ = _get_vectorstore()
