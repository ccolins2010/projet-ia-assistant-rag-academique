from __future__ import annotations
import re

"""
router.py — Détection des intentions
-----------------------------------

Ce fichier décide si la question doit aller :
- vers smalltalk
- vers un outil (calc, weather, todo, web)
- vers le RAG

CORRECTIONS IMPORTANTES :
- _looks_like_math strict → évite les faux positifs (ex : "7 couches du modèle OSI")
- expressions math reconnues correctement (sin, cos, sqrt, log…)
- TODO : 
    - "ajoute ...", "termine X", "liste" → comme avant
    - "vide tout", "reset", "clear", "efface tout", "supprime tout"
      sont maintenant bien routés vers tool_todo().
"""

# Détecte les mots qui indiquent une recherche météo
_WEATHER_RE = re.compile(r"\b(meteo|météo|weather|temperature|température|temps)\b", re.I)

# Détecte si l'utilisateur veut explicitement une recherche web
_WEB_RE = re.compile(r"\b(cherche|recherche|search)\b", re.I)

# Détecte les commandes TODO
_TODO_ADD_RE = re.compile(r"\b(ajoute|ajouter|add)\b", re.I)
_TODO_DONE_RE = re.compile(r"\b(termine|fini|finis|done)\b", re.I)
_TODO_LIST_RE = re.compile(r"\b(liste|tasks|taches|tâches)\b", re.I)

# NOUVEAU : détection des commandes de reset TODO
_TODO_CLEAR_RE = re.compile(
    r"\b(vide tout|vide la liste|reset|clear|efface tout|supprime tout)\b",
    re.I,
)

# Détecte formulaire de calcul
_MATH_HINT_RE = re.compile(r"[0-9+\-*/^()]")

# Fonctions math reconnues → on va les permettre
_MATH_FUNC_RE = re.compile(r"\b(sin|cos|tan|sqrt|log|log10|ln|exp|pi|π)\b", re.I)


def _looks_like_math(text: str) -> bool:
    """
    Détection STRICTE des calculs :
    - au moins un chiffre
    - présence d'un opérateur (+ - * / ^) OU d'une fonction math
    - ET correspond à notre regex math
    → Empêche la phrase 'les 7 couches du modèle OSI' d'aller en calculatrice

    Supporte aussi :
    - sin45
    - log100
    - sqrt16
    - etc.
    """

    if not text:
        return False

    t = text.replace(" ", "").lower()

    # 1. Doit contenir un chiffre
    if not any(c.isdigit() for c in t):
        return False

    # 2. Doit contenir un opérateur ou une fonction math
    if not re.search(r"[+\-*/^]", t) and not _MATH_FUNC_RE.search(t):
        return False

    # 3. Structure compatible math
    if not _MATH_HINT_RE.search(text):
        return False

    return True


def route(text: str):
    """
    Retourne un tuple (intent, payload)
    intent ∈ {"smalltalk", "calc", "weather", "todo", "web", "rag"}

    PRIORITÉ :
    1. météo
    2. calculatrice
    3. todo
    4. recherche web explicite
    5. smalltalk simple
    6. rag par défaut
    """

    t = (text or "").strip().lower()

    # ────────── METEO
    if _WEATHER_RE.search(t):
        return "weather", t

    # ────────── CALCUL
    if _looks_like_math(text):
        return "calc", text

    # ────────── TODO (ADD / DONE / LIST / CLEAR)
    #
    # IMPORTANT :
    # tool_todo(cmd: str) dans agents.py attend du TEXTE BRUT.
    # Donc on renvoie toujours `text` comme payload.
    if _TODO_ADD_RE.search(t):
        return "todo", text

    if _TODO_DONE_RE.search(t):
        return "todo", text

    if _TODO_LIST_RE.search(t):
        return "todo", text

    if _TODO_CLEAR_RE.search(t):
        return "todo", text

    # ────────── Recherche web explicite
    if _WEB_RE.search(t):
        return "web", text

    # ────────── Smalltalk (phrases basiques)
    if t in {"bonjour", "salut", "hello", "hi"}:
        return "smalltalk", text

    # ────────── PAR DÉFAUT → RAG
    return "rag", text
