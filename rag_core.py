from __future__ import annotations

"""
rag_core.py — RAG simple sans LLM ni LangChain
---------------------------------------------

Objectif : répondre UNIQUEMENT à partir des fichiers .txt dans RAG_Data.

Stratégie :
- On charge tous les .txt de RAG_Data.
- On découpe chaque fichier en sections Markdown (chaque ligne '## ...').
- Chaque section devient un "Document" avec :
    - page_content      : titre + texte de la section
    - metadata["source"]: chemin du fichier
    - metadata["section_title"]: titre de la section

Pour répondre à une question :
1. On normalise la question (minuscules, sans accents, etc.).
2. On cherche d'abord une section dont le titre matche la question :
   - soit le titre est contenu dans la question,
   - soit la question est contenue dans le titre.
   Exemple :
     "Brève histoire de l’IA"  <->  "## 5. Brève histoire de l’IA"
     "Systèmes experts et représentation des connaissances"  <->  "## 8. ..."

3. Si aucun titre ne matche directement, on calcule un score COMBINÉ pour chaque
   section :
     score = similitude_titre + 0.3 * nb_mots_commun
   où :
     - similitude_titre = ratio difflib entre question et titre normalisés
     - nb_mots_commun  = |mots(question) ∩ mots(section)|

4. Si le meilleur score est trop faible (peu de mots en commun ET titre peu
   similaire), on considère que la réponse n’est pas dans les documents internes.

On ne fait AUCUN appel à un LLM ici. Les réponses sont des extraits de cours.
"""

import re
import unicodedata
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ───────────────────────── CONFIG GLOBALE ──────────────────────────

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "RAG_Data"
PERSIST_PATH = ROOT / "chroma_store"   # gardé pour compat avec app.py

# ⚠️ IMPORTANT : ce nom DOIT correspondre au modèle installé dans Ollama
LLM_MODEL = "llama3.2:3b"              # utilisé par app.py pour l'agent


@dataclass
class Document:
    page_content: str
    metadata: Dict[str, str]


# Cache en mémoire de toutes les sections
_SECTIONS_CACHE: Optional[List[Document]] = None


# ──────────────────────────── UTILITAIRES ────────────────────────────

def _normalize(t: str) -> str:
    """
    Normalisation agressive :
    - passe en NFD
    - enlève les accents
    - met en minuscules
    - remplace tout ce qui n'est pas [a-z0-9] par un espace
    - compresse les espaces
    """
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = t.lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


def _keywords(text: str) -> set[str]:
    """
    Découpe en mots-significatifs (>= 2 lettres), normalisés.
    On garde >=2 pour que des abréviations comme 'ia' soient prises en compte.
    """
    norm = _normalize(text)
    return {w for w in norm.split() if len(w) >= 2}


def _parse_markdown_sections(text: str, source: str) -> List[Document]:
    """
    Découpe un fichier Markdown en sections à partir des lignes qui commencent par '## '.

    Exemple :
      ## 1. Qu’est-ce que l’intelligence artificielle ?
      (texte...)
      ## 2. Le test de Turing
      (texte...)

    Chaque section devient un Document.
    """
    lines = text.splitlines()

    sections: List[Document] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):  # nouveau titre de section
            # Si on avait déjà une section en cours, on la pousse
            if current_title is not None:
                content = (current_title + "\n\n" + "\n".join(current_lines)).strip()
                sections.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": source,
                            "section_title": current_title,
                        },
                    )
                )

            # On démarre une nouvelle section
            current_title = stripped.lstrip("#").strip()
            current_lines = []
        else:
            if current_title is not None:
                current_lines.append(line)

    # Dernière section si existante
    if current_title is not None:
        content = (current_title + "\n\n" + "\n".join(current_lines)).strip()
        sections.append(
            Document(
                page_content=content,
                metadata={
                    "source": source,
                    "section_title": current_title,
                },
            )
        )

    # Si le fichier n'avait pas de "##", on met tout en une seule section
    if not sections:
        sections.append(
            Document(
                page_content=text.strip(),
                metadata={"source": source, "section_title": ""},
            )
        )

    return sections


def _load_all_sections() -> List[Document]:
    """
    Charge TOUS les fichiers .txt de RAG_Data et les découpe en sections.
    """
    sections: List[Document] = []
    DOCS_DIR.mkdir(exist_ok=True)

    for file in DOCS_DIR.rglob("*.txt"):
        try:
            raw = file.read_text(encoding="utf-8")
        except Exception:
            continue
        sections.extend(_parse_markdown_sections(raw, str(file)))

    return sections


def _get_sections() -> List[Document]:
    """
    Retourne la liste de toutes les sections ; charge si nécessaire.
    """
    global _SECTIONS_CACHE
    if _SECTIONS_CACHE is None:
        _SECTIONS_CACHE = _load_all_sections()
    return _SECTIONS_CACHE


def _best_section_for_question(question: str) -> Tuple[Optional[Document], str]:
    """
    Retourne la meilleure section pour une question, + raison ('title' ou 'combined').

    - D'abord : match direct sur les titres de sections (inclusion).
    - Sinon : score combiné titre + mots-clés.
    """
    sections = _get_sections()
    if not sections:
        return None, "none"

    q_norm = _normalize(question)

    # 1) Match direct sur les titres de sections (inclusion stricte)
    title_matches: List[Document] = []
    for doc in sections:
        title = doc.metadata.get("section_title", "") or ""
        t_norm = _normalize(title)
        if not t_norm:
            continue

        # Si le titre est contenu dans la question, ou l'inverse
        if t_norm in q_norm or q_norm in t_norm:
            title_matches.append(doc)

    if title_matches:
        # On prend le titre le plus court (souvent le plus spécifique)
        def _title_len(d: Document) -> int:
            return len(_normalize(d.metadata.get("section_title", "")))

        best = min(title_matches, key=_title_len)
        return best, "title"

    # 2) Score combiné : similarité du titre + 0.3 * nombre de mots communs
    q_words = _keywords(question)
    if not q_words:
        return None, "none"

    best_doc: Optional[Document] = None
    best_score = 0.0
    best_overlap = 0
    best_title_sim = 0.0

    for doc in sections:
        title = doc.metadata.get("section_title", "") or ""
        t_norm = _normalize(title)
        if not t_norm:
            continue

        # Similarité de titre (fuzzy)
        title_sim = difflib.SequenceMatcher(None, q_norm, t_norm).ratio()

        # Recouvrement de mots-clés avec le titre + le contenu
        text_for_keywords = title + "\n\n" + doc.page_content
        d_words = _keywords(text_for_keywords)
        overlap = len(q_words & d_words)

        score = title_sim + 0.3 * overlap

        if score > best_score:
            best_score = score
            best_doc = doc
            best_overlap = overlap
            best_title_sim = title_sim

    # Seuil minimal : on exige un minimum de signal
    # - soit au moins 2 mots en commun
    # - soit un titre assez proche (>= 0.5)
    if best_doc is None or (best_overlap < 2 and best_title_sim < 0.5):
        return None, "none"

    return best_doc, "combined"


def _shorten_answer(text: str, max_chars: int = 1200) -> str:
    """
    Coupe proprement si la section est très longue, pour ne pas saturer l'UI.
    On coupe à max_chars en essayant de s'arrêter en fin de phrase.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    # Essayer de couper à la dernière ponctuation forte
    idx = max(
        truncated.rfind("."),
        truncated.rfind("!"),
        truncated.rfind("?"),
    )
    if idx != -1 and idx > max_chars * 0.5:
        return truncated[: idx + 1].strip() + " [...]"
    return truncated.strip() + " [...]"


# ──────────────────────────── API PRINCIPALE ────────────────────────────

def answer_question(question: str, chat_history=None):
    """
    Retourne :
      {
        "answer": str,
        "source_documents": [Document, ...]
      }

    Si aucune section pertinente n'est trouvée :
      - answer = "La réponse ne se trouve pas dans les documents internes."
      - source_documents = []
    """
    question = (question or "").strip()
    if not question:
        return {
            "answer": "La réponse ne se trouve pas dans les documents internes.",
            "source_documents": [],
        }

    best, reason = _best_section_for_question(question)

    if best is None:
        return {
            "answer": "La réponse ne se trouve pas dans les documents internes.",
            "source_documents": [],
        }

    answer_text = _shorten_answer(best.page_content)
    return {
        "answer": answer_text,
        "source_documents": [best],
    }


def ask_rag(question: str):
    """
    Alias simple pour un appel RAG (sans historique).
    Retourne :
      {
        "answer": str,
        "source": [chemins_de_fichiers]
      }
    """
    res = answer_question(question)
    return {
        "answer": res["answer"],
        "source": [d.metadata.get("source") for d in res.get("source_documents", [])],
    }


def reindex():
    """
    Reconstruit complètement la "base" à partir des fichiers internes.
    (Ici, on recharge simplement les sections depuis les .txt).
    """
    global _SECTIONS_CACHE
    _SECTIONS_CACHE = _load_all_sections()
