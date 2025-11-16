from __future__ import annotations

"""
agents.py — Outils officiels de l’assistant
------------------------------------------

• tool_calculator     : Calculatrice sécurisée (AST)
                        - sin45, sin 45°, sin(45deg)
                        - sqrt16, log100, exp2, 2^3, 5², 3³, etc.
                        - log(x) est interprété comme log10(x)

• tool_weather        : Météo mondiale via wttr.in (Rouen, Nantes, London, Brazil, etc.)
• tool_weather_sync   : Version synchrone pour Streamlit
• tool_web_search     : Recherche DuckDuckGo (ddgs)
• tool_todo           : To-do list persistante (JSON)
                        - ajoute ...
                        - termine X
                        - liste
                        - vide tout (reset)

Toutes les fonctions renvoient du TEXTE prêt à afficher dans app.py.
"""

import ast
import json
import math
import operator as op
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import httpx
from ddgs import DDGS


# ╔═══════════════════════════════════════════╗
# ║           1. CALCULATRICE (AST)           ║
# ╚═══════════════════════════════════════════╝

# On autorise seulement un sous-ensemble sûr de Python
_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}

# ⚠️ CHOIX IMPORTANT :
#   - "log" = log10 (logarithme base 10)
#   - "log10" = idem (pour être explicite)
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log10,     # on interprète "log" comme log10
    "log10": math.log10,
    "exp": math.exp,
}

_ALLOWED_CONSTS = {
    "pi": math.pi,
    "e": math.e,
}


def _eval_ast(node: ast.AST) -> float:
    """Évalue récursivement un AST mathématique limité et sécurisé."""

    # Constantes (pi, e…)
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTS:
            return float(_ALLOWED_CONSTS[node.id])
        raise ValueError(f"Symbole non autorisé : {node.id}")

    # Nombres (Python 3.8+)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Constante non numérique")

    # Opérateurs unaires (ex : -x)
    if isinstance(node, ast.UnaryOp):
        return _ALLOWED_OPS[type(node.op)](_eval_ast(node.operand))

    # Opérateurs binaires (x + y, x * y, etc.)
    if isinstance(node, ast.BinOp):
        return _ALLOWED_OPS[type(node.op)](
            _eval_ast(node.left),
            _eval_ast(node.right),
        )

    # Appels de fonctions autorisées (sqrt, sin, log10…)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fname = node.func.id
        if fname not in _ALLOWED_FUNCS:
            raise ValueError(f"Fonction non autorisée : {fname}")
        args = [_eval_ast(a) for a in node.args]
        return float(_ALLOWED_FUNCS[fname](*args))

    raise ValueError("Expression invalide (AST)")


# ─────────────────────────────
# Extraction / normalisation
# ─────────────────────────────

_MATH_EXPR_RE = re.compile(
    r"(?:sqrt|sin|cos|tan|log10|log|exp|\d|[+\-*/().,^°²³ ]+)+",
    re.I,
)


def _extract_math_expr(text: str) -> str:
    """
    Extrait et normalise une expression mathématique à partir d'une phrase.

    Gère :
    - opérateurs unicode → ASCII
    - '2^3' → '2**3'
    - '5²' → '5**2', '3³' → '3**3'
    - 'sin45' / 'sin 45' / 'sin 45°' / 'sin(45deg)' → sin(radians(45))
    - 'sqrt16' / 'sqrt 16' → 'sqrt(16)'
    - 'log100' / 'log 100' → 'log(100)' (et "log" = log10)
    - 'exp2' / 'exp 2' → 'exp(2)'
    """

    if not text:
        return ""

    raw = text.strip()

    # Normalisation des opérateurs unicode
    raw = (
        raw.replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )

    # Premier essai : zone "math" dans le texte
    m = _MATH_EXPR_RE.search(raw)
    expr = m.group(0).strip() if m else raw

    # Normalisations de base
    expr = expr.replace(",", ".")
    expr = expr.replace("^", "**")

    # Puissances ² / ³
    expr = re.sub(r"(\d+)\s*²", r"\1**2", expr)
    expr = re.sub(r"(\d+)\s*³", r"\1**3", expr)

    # sin45 / cos30 / tan60 (sans ° explicitement) → interprétation en DEGRÉS
    def _inline_deg(match: re.Match) -> str:
        func = match.group(1).lower()
        val = float(match.group(2))
        rad = val * math.pi / 180.0
        return f"{func}({rad})"

    expr = re.sub(
        r"\b(sin|cos|tan)\s*([0-9]+(?:\.[0-9]+)?)\b",
        _inline_deg,
        expr,
        flags=re.I,
    )

    # Gestion des notations avec ° ou "deg" EXPLICITES
    # sin 45° / sin(45deg)
    def _deg_token_to_rad(match: re.Match) -> str:
        func = match.group(1)
        number = float(match.group(2))
        rad = number * math.pi / 180.0
        return f"{func}({rad})"

    expr = re.sub(
        r"\b(sin|cos|tan)\s+([0-9]+(?:\.[0-9]+)?)\s*°\b",
        _deg_token_to_rad,
        expr,
        flags=re.I,
    )
    expr = re.sub(
        r"\b(sin|cos|tan)\s*\(\s*([0-9]+(?:\.[0-9]+)?)\s*deg\s*\)",
        _deg_token_to_rad,
        expr,
        flags=re.I,
    )

    # sqrt16 / log100 / exp2 → ajout de parenthèses
    # ⚠️ IMPORTANT :
    # On place "log" AVANT "log10" dans l'alternative pour éviter que
    # "log100" soit interprété comme "log10(0)".
    # Exemple sans cette précaution :
    #   - regex voit "log10" dans "log100" → groupe1="log10", groupe2="0"
    #   - devient "log10(0)" → math domain error
    #
    # Avec cet ordre ("log" d'abord) :
    #   - "log100" → groupe1="log", groupe2="100" → "log(100)" ✅
    expr = re.sub(
        r"\b(sqrt|log|log10|exp)\s*([0-9]+(?:\.[0-9]+)?)\b",
        r"\1(\2)",
        expr,
        flags=re.I,
    )

    # ⚠️ IMPORTANT :
    # On NE fait PAS de remplacement du type "log(\d+) → log10(...)". 
    # Ça évite les horreurs du style "log10(10)(0)".

    return expr


def tool_calculator(text: str) -> str:
    """Outil CALCUL — renvoie une réponse prête à afficher."""

    expr = _extract_math_expr(text)

    if not expr:
        return "Expression reconnue: (vide)\nRésultat: Erreur — expression vide"

    try:
        node = ast.parse(expr, mode="eval").body
        val = _eval_ast(node)

        if abs(val - int(val)) < 1e-12:
            result = int(val)
        else:
            # limiter les décimales pour un rendu propre
            result = float(f"{val:.10f}".rstrip("0").rstrip("."))

        return f"Expression reconnue: `{expr}`\nRésultat: **{result}**"

    except Exception as e:
        return f"Expression reconnue: `{expr}`\nRésultat: Erreur calcul: {e}"


# ╔═══════════════════════════════════════════╗
# ║     2. MÉTÉO MONDIALE (wttr.in)          ║
# ╚═══════════════════════════════════════════╝

# On garde quelques presets au cas où, mais wttr.in gère déjà très bien
_CITY_PRESET = {
    "paris": "Paris",
    "lyon": "Lyon",
    "marseille": "Marseille",
    "reims": "Reims",
}


def _normalize_city_free_text(raw: str) -> str:
    """
    Exemples :
      "meteo rouen"        → "Rouen"
      "la météo à nantes"  → "Nantes"
      "meteo brazil"       → "Brazil"
      "meteo londre"       → "Londre" (wttr.in gère assez bien)
    """
    if not raw:
        return "Paris"

    text = raw.strip()

    tokens = re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ']+", text)
    stop = {
        "meteo", "météo", "la", "le", "les", "du", "de", "des",
        "a", "à", "au", "aux", "pour", "stp", "svp", "il", "fait",
        "quelle", "quel", "donne", "donner",
    }

    filtered = [t for t in tokens if t.lower() not in stop]

    if not filtered:
        return "Paris"

    city = " ".join(filtered).strip()
    return city.title()


async def tool_weather(city: str = "Paris") -> str:
    """
    Météo via wttr.in (fonctionne pour la plupart des villes / pays du monde).
    - Gère les phrases complètes : "meteo rouen", "la météo à nantes", etc.
    - Retourne : Ville, Température, Vent.
    """
    normalized = _normalize_city_free_text(city)

    # Petit fallback sur le preset (corrige quelques variantes)
    preset = _CITY_PRESET.get(normalized.lower())
    target = preset or normalized

    url = f"https://wttr.in/{target}"
    params = {"format": "j1", "lang": "fr"}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()

        current = (data.get("current_condition") or [{}])[0]
        temp_c = current.get("temp_C", "?")
        wind_kmh = current.get("windspeedKmph", "?")

        return (
            f"Ville: **{target}**\n"
            f"Température: **{temp_c}°C**\n"
            f"Vent: **{wind_kmh} km/h**"
        )

    except Exception:
        return "Ville inconnue ou service météo indisponible."


def tool_weather_sync(city: str = "Paris") -> str:
    """Enveloppe synchrone pour Streamlit."""
    import asyncio

    try:
        return asyncio.run(tool_weather(city))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(tool_weather(city))
        finally:
            loop.close()


# ╔═══════════════════════════════════════════╗
# ║        3. RECHERCHE WEB (DuckDuckGo)     ║
# ╚═══════════════════════════════════════════╝

def tool_web_search(query: str, max_results: int = 5) -> str:
    """
    Recherche texte via DuckDuckGo (ddgs).
    Retourne un JSON (string) que app.py formate joliment.
    """
    cleaned = query.strip()

    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(
                    cleaned,
                    region="fr-fr",
                    safesearch="moderate",
                    max_results=max_results,
                )
            )

        payload = [
            {
                "title": r.get("title"),
                "href": r.get("href"),
                "body": r.get("body"),
            }
            for r in results
        ]
        return json.dumps(payload, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Recherche échouée : {e}"}, ensure_ascii=False)


# ╔═══════════════════════════════════════════╗
# ║          4. TODO LISTE PERSISTANTE        ║
# ╚═══════════════════════════════════════════╝

_TODO_PATH = Path(__file__).parent / "todo_store.json"

try:
    _TODO: List[Dict] = json.loads(_TODO_PATH.read_text(encoding="utf-8"))
    if not isinstance(_TODO, list):
        _TODO = []
except Exception:
    _TODO = []


def _save_todo():
    try:
        _TODO_PATH.write_text(
            json.dumps(_TODO, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def tool_todo(cmd: str) -> str:
    """
    To-do list persistante.

    Commandes reconnues (en texte libre) :
      - "ajoute ..." / "add ..."            → ajoute une tâche (sans doublons exacts)
      - "termine 2" / "done 2"             → marque la tâche #2 comme faite
      - "liste" / "list"                   → renvoie la liste complète (JSON)
      - "vide tout" / "reset" / "clear"    → vide complètement la liste

    Retour :
      - En cas de succès : JSON (liste de tâches)
      - En cas d'erreur : texte explicite
    """
    global _TODO

    text = (cmd or "").strip().lower()

    # ───── VIDER LA LISTE ─────
    # Exemples : "vide tout", "reset", "clear", "efface tout"
    if text.startswith(("vide tout", "vide la liste", "reset", "clear", "efface tout", "supprime tout")):
        _TODO = []
        _save_todo()
        return json.dumps(_TODO, ensure_ascii=False)

    # ───── AJOUT D'UNE TÂCHE ─────
    if text.startswith("ajoute") or text.startswith("add"):
        content = re.sub(r"^(ajoute|add)\s*:?", "", cmd, flags=re.I).strip()
        if not content:
            return "Texte vide."

        # Anti-doublon : on ne rajoute pas si une tâche avec le même texte existe déjà (insensible à la casse)
        content_norm = content.strip().lower()
        for t in _TODO:
            if t.get("text", "").strip().lower() == content_norm:
                # On ne rajoute pas, on renvoie simplement l'état actuel de la liste
                return json.dumps(_TODO, ensure_ascii=False)

        item = {"id": len(_TODO) + 1, "text": content, "done": False}
        _TODO.append(item)
        _save_todo()
        return json.dumps(_TODO, ensure_ascii=False)

    # ───── TERMINER UNE TÂCHE ─────
    if text.startswith("termine") or text.startswith("done"):
        m = re.search(r"(\d+)", text)
        if not m:
            return "ID manquant."
        idx = int(m.group(1))
        for t in _TODO:
            if t["id"] == idx:
                t["done"] = True
                _save_todo()
                return json.dumps(_TODO, ensure_ascii=False)
        return "ID inconnu."

    # ───── LISTER LES TÂCHES ─────
    if text in {"liste", "list"}:
        return json.dumps(_TODO, ensure_ascii=False)

    # ───── COMMANDE INCONNUE ─────
    return "Commande inconnue (ajoute, termine X, liste, vide tout)."


__all__ = [
    "tool_calculator",
    "tool_weather",
    "tool_weather_sync",
    "tool_web_search",
    "tool_todo",
]
