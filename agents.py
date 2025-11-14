from __future__ import annotations

"""
agents.py
---------
Outils/agents utilisés par l'app :

- Calculatrice sécurisée (via AST, sans eval)
- Météo (Nominatim + Open-Meteo, avec fallback local)
- Recherche web (DuckDuckGo via ddgs)
- TODO persistant sur disque (JSON)

Compatible Python 3.10
"""

import ast
import json
import math
import operator as op
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import httpx
from ddgs import DDGS  # pip install ddgs


# =====================================================================
#  Helpers NLU / Normalisation
# =====================================================================
# Ces helpers servent à nettoyer/normaliser le texte libre
# avant de le passer aux vrais outils.


# ---------------------------------------------------------------------
# 1) Calculatrice : extraction/normalisation d'expressions math
# ---------------------------------------------------------------------

# Fonctions autorisées dans les expressions
_MATH_FUNC = r"(sqrt|sin|cos|tan|log10|log|exp)"

# Expression math autorisée :
# - chiffres
# - espaces
# - opérateurs + - * / ( ) . , ^ et symboles ° ² ³
_MATH_EXPR_RE = re.compile(
    rf"(?:{_MATH_FUNC}|\d|\s|[+\-*/().,^°²³])+",
    re.I,
)


def _extract_math_expr(text: str) -> str:
    """
    Extrait et normalise une expression math à partir d'un texte libre.

    Normalisations :
      - virgules → points (2,5 → 2.5)
      - ^ → ** (2^3 → 2**3)
      - ×, ·, ÷, −, –, — → *, /, -
      - ² / ³ → puissances
      - sin45, cos30, tan60 → sin(radians)
      - sqrt16, log10 100, exp2 → sqrt(16), log10(100), exp(2)
      - sin 45° / sin(45deg) → conversion degrés → radians
      - équilibrage simple des parenthèses

    Si rien de mathématique n'est détecté, retourne "".
    """
    if not text:
        return ""

    raw = text.strip()

    # 1) Première tentative : on prend le premier bloc "math"
    m = _MATH_EXPR_RE.search(raw)
    expr = (m.group(0) if m else raw).strip()

    # 2) Si on a récupéré du vide, on cherche le premier vrai token math
    if not expr:
        m2 = re.search(r"(sqrt|sin|cos|tan|log10|log|exp|\d)", raw, flags=re.I)
        if m2:
            expr = raw[m2.start():].strip()
        else:
            return ""

    # 3) Normalisations basiques
    expr = expr.replace(",", ".")   # 2,5 -> 2.5
    expr = expr.replace("^", "**")  # 2^3 -> 2**3

    # 3-bis) Opérateurs Unicode → ASCII (copier/coller depuis Word, etc.)
    expr = (
        expr
        .replace("×", "*")
        .replace("∙", "*")
        .replace("·", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )

    # 3-ter) Équilibrage simple des parenthèses
    def _balance_parentheses(s: str) -> str:
        open_count = 0
        out: List[str] = []
        for ch in s:
            if ch == "(":
                open_count += 1
                out.append(ch)
            elif ch == ")":
                if open_count > 0:
                    open_count -= 1
                    out.append(ch)
                # sinon : parenthèse fermante en trop → on l'ignore
            else:
                out.append(ch)
        # on ferme les parenthèses restantes à la fin
        return "".join(out) + (")" * open_count)

    expr = _balance_parentheses(expr)

    # 4) Puissances avec ² / ³
    expr = re.sub(r"(\d+)\s*²", r"\1**2", expr)
    expr = re.sub(r"(\d+)\s*³", r"\1**3", expr)

    # 5) Cas "sin45", "cos30", "tan60" → sin(radians)
    def _inline_deg(mf: re.Match) -> str:
        func = mf.group(1).lower()
        val = float(mf.group(2))
        rad = val * math.pi / 180.0
        return f"{func}({rad})"

    expr = re.sub(
        r"\b(sin|cos|tan)\s*([0-9]+(?:\.[0-9]+)?)\b",
        _inline_deg,
        expr,
        flags=re.I,
    )

    # 6) Cas "sqrt16", "sqrt 16", "log10 100", "exp2" → on ajoute les ()
    expr = re.sub(
        r"\b(sqrt|log10|log|exp)\s*([0-9]+(?:\.[0-9]+)?)\b",
        r"\1(\2)",
        expr,
        flags=re.I,
    )

    # 7) Cas sin'45  → sin(45)
    expr = re.sub(
        r"\b(sin|cos|tan)\s*'\s*([0-9]+(?:\.[0-9]+)?)\b",
        r"\1(\2)",
        expr,
        flags=re.I,
    )

    # 8) Cas sin 45° / sin 45deg → sin(45°/deg)
    expr = re.sub(
        r"\b(sin|cos|tan)\s+([0-9]+(?:\.[0-9]+)?)\s*°",
        r"\1(\2°)",
        expr,
        flags=re.I,
    )
    expr = re.sub(
        r"\b(sin|cos|tan)\s+([0-9]+(?:\.[0-9]+)?)\s*deg\b",
        r"\1(\2deg)",
        expr,
        flags=re.I,
    )

    # 9) Conversion sin(45°) / sin(45deg) -> sin(radians)
    def _deg_to_rad(mf: re.Match) -> str:
        func = mf.group(1)
        inside = mf.group(2)
        m_deg = re.match(
            r"\s*([0-9]+(?:\.[0-9]+)?)\s*(°|deg)\s*$",
            inside,
            flags=re.I,
        )
        if not m_deg:
            return f"{func}({inside})"
        val = float(m_deg.group(1))
        rad = val * math.pi / 180.0
        return f"{func}({rad})"

    expr = re.sub(
        r"\b(sin|cos|tan)\s*\(\s*([^)]+)\s*\)",
        _deg_to_rad,
        expr,
        flags=re.I,
    )

    return expr


# ---------------------------------------------------------------------
# 2) Météo : normalisation d’une ville dans un texte libre
# ---------------------------------------------------------------------

_STOPWORDS_CITY = {
    "aujourd'hui", "auj", "demain", "stp", "svp", "merci",
    "s'il", "te", "plaît", "plait", "moi", "please", "today",
    "meteo", "météo", "quelle", "est", "la", "le", "de", "du",
    "a", "à", "pour", "il", "fait", "temps", "donne", "donner", "donnes",
    "au",  # pour éviter "Au Brazil" -> on garde "Brazil"
}


def _normalize_city_free_text(raw: str) -> str:
    """
    Exemples :
      "quelle est la météo à Paris aujourd'hui ?" → "Paris"
      "météo pour lyon stp"                        → "Lyon"
      "la meteo au Brazil"                         → "Brazil"

    Si aucune ville claire n'est trouvée, retourne "Paris".
    """
    if not raw:
        return "Paris"

    text = raw.strip()

    # On essaie de récupérer ce qui vient après "à", "a", "pour" ou "au"
    m = re.search(r"(?:\bà|\ba|\bpour|\bau)\s+([a-zA-ZÀ-ÖØ-öø-ÿ' -]{2,})", text, re.I)
    candidate = m.group(1) if m else text

    # On coupe aux ponctuations fortes
    candidate = re.split(r"[?,!.;:()\[\]\{\}\n\r]", candidate)[0]

    # On enlève les mots parasites ("meteo", "aujourd'hui", etc.)
    tokens = [
        t for t in re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ']{2,}", candidate)
        if t.lower() not in _STOPWORDS_CITY
    ]

    if not tokens:
        return "Paris"

    return " ".join(tokens).strip().title()


# ---------------------------------------------------------------------
# 3) Web : nettoyage requête DuckDuckGo
# ---------------------------------------------------------------------

def _clean_web_query(text: str) -> str:
    """
    Nettoie une requête web de type :
      - "recherche: ..."
      - "cherche sur le web ..."
      - "internet ..."
    """
    if not text:
        return ""
    cleaned = re.sub(
        r"^\s*(recherche|cherche( sur (le )?web)?|internet|google)\s*:?\s*",
        "",
        text,
        flags=re.I,
    )
    return cleaned.strip() or text.strip()


# ---------------------------------------------------------------------
# 4) TODO : passage texte NL → commande canonique
# ---------------------------------------------------------------------

def _normalize_todo_command(text: str) -> str:
    """
    Map en commandes simples :
      - "ajoute …" / "add: …"  →  "add: <texte>"
      - "termine 2" / "done: 2"→  "done: 2"
      - "liste" / "list"       →  "list"
    """
    t = (text or "").strip()

    # Ajout de tâche
    if re.search(r"\b(ajoute|ajouter|add)\b", t, re.I):
        m = re.search(r"(?:ajoute|ajouter|add)\s*:?\s*(.*)", t, re.I)
        payload = (m.group(1) if m else "").strip()
        return f"add: {payload}" if payload else "list"

    # Terminer une tâche (par numéro)
    if re.search(r"\b(termine|finis|done)\b", t, re.I):
        m = re.search(r"(?:termine|finis|done)\s*:?\s*(\d+)", t, re.I)
        return f"done: {m.group(1)}" if m else "list"

    # Lister les tâches
    if re.search(r"\b(liste|list)\b", t, re.I):
        return "list"

    # Sinon on laisse tel quel (permet le debug)
    return t


# =====================================================================
#  Calculatrice sécurisée (AST)
# =====================================================================

# Opérateurs autorisés
_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
}

# Fonctions de math autorisées
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,      # ln
    "log10": math.log10,
    "exp": math.exp,
}

# Constantes autorisées
_ALLOWED_CONSTS = {
    "pi": math.pi,
    "e": math.e,
}


def _eval_ast(node: ast.AST) -> float:
    """
    Évalue récursivement l’AST d’une expression mathématique restreinte.
    Empêche tout appel/accès non autorisé (sécurité).
    """

    # Constantes (pi, e, …)
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTS:
            return float(_ALLOWED_CONSTS[node.id])
        raise ValueError(f"symbole non autorisé: {node.id}")

    # Nombres
    if hasattr(ast, "Num") and isinstance(node, ast.Num):
        return float(node.n)  # type: ignore[attr-defined]
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("constante non numérique")

    # Opérations unaires / binaires
    if isinstance(node, ast.UnaryOp):
        return _ALLOWED_OPS[type(node.op)](_eval_ast(node.operand))
    if isinstance(node, ast.BinOp):
        return _ALLOWED_OPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))

    # Appels de fonctions autorisées
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func = node.func.id
        if func not in _ALLOWED_FUNCS:
            raise ValueError(f"fonction non autorisée: {func}")
        args = [_eval_ast(a) for a in node.args]
        return float(_ALLOWED_FUNCS[func](*args))

    raise ValueError("expression invalide")


def tool_calculator(expr: str) -> str:
    """
    Évalue une expression mathématique contenue dans un texte naturel.
    Retourne un bloc texte prêt pour l'affichage (markdown).

    Exemple :
      "calcul moi (145 + 268) × 3 – 42"
        → Expression reconnue: `(145 + 268) * 3 - 42`
        → Résultat: **949**
    """
    try:
        normalized = _extract_math_expr(expr)
        if not normalized:
            return (
                "🛠️ Calculatrice\n\n"
                "Expression reconnue: *vide*\n"
                "Résultat: Erreur calcul: expression vide"
            )

        node = ast.parse(normalized, mode="eval").body
        val = _eval_ast(node)

        # Si le résultat est très proche d'un entier, on l'affiche comme entier
        if abs(val - int(val)) < 1e-12:
            result = str(int(val))
        else:
            # On limite à 10 décimales pour éviter des trucs comme 0.7853981633974483
            result = f"{val:.10f}".rstrip("0").rstrip(".")

        return (
            "🛠️ Calculatrice\n\n"
            f"Expression reconnue: `{normalized}`\n"
            f"Résultat: **{result}**"
        )
    except Exception as e:
        return (
            "🛠️ Calculatrice\n\n"
            f"Expression reconnue: `{_extract_math_expr(expr)}`\n"
            f"Résultat: Erreur calcul: {e}"
        )


# =====================================================================
#  Météo (Open-Meteo) + Nominatim
# =====================================================================

# Quelques villes en dur, au cas où le géocodage web échoue
_CITY_PRESET: Dict[str, Tuple[float, float]] = {
    "paris": (48.8566, 2.3522),
    "lyon": (45.7640, 4.8357),
    "marseille": (43.2965, 5.3698),
    "evry": (48.6239, 2.4289),
    "rennes": (48.1173, -1.6778),
}


async def _geocode_city(city: str) -> Optional[Tuple[float, float]]:
    """
    Géocodage via Nominatim (OpenStreetMap).
    Retourne (lat, lon) ou None en cas d'erreur ou d'absence de résultat.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": city, "format": "json", "limit": 1}
    headers = {"User-Agent": "RAG-Academique/1.0 (education use)"}
    try:
        async with httpx.AsyncClient(timeout=15, headers=headers) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            if not data:
                return None
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        return None


async def tool_weather(city: str = "Paris") -> str:
    """
    Donne la météo courante via Open-Meteo.

    Étapes :
      1. Normalisation de la ville (texte libre → "Paris", "Lyon", "Brazil"...).
      2. Géocodage via Nominatim (internet).
      3. Si le géocodage échoue → fallback sur _CITY_PRESET.
    """
    normalized = _normalize_city_free_text(city)
    city_key = normalized.lower()

    # 1) Géocodage web prioritaire
    coords = await _geocode_city(normalized)

    # 2) Fallback sur les presets locaux
    if not coords:
        coords = _CITY_PRESET.get(city_key)

    if not coords:
        return (
            "Météo indisponible: ville introuvable ou service de géocodage indisponible.\n"
            "Essaie une autre orthographe ou une grande ville (Paris, Lyon, Marseille...)."
        )

    lat, lon = coords
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "current_weather": True}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            cw = r.json().get("current_weather") or {}
            if not cw:
                return "Météo indisponible pour cette position."
            t = cw.get("temperature")
            w = cw.get("windspeed")
            return (
                "🛠️ Météo\n\n"
                f"Ville: **{normalized}**\n"
                f"Température: **{t}°C**\n"
                f"Vent: **{w} km/h**"
            )
    except httpx.HTTPError as e:
        return f"Météo indisponible (problème réseau ou service). Détail: {e}"


def tool_weather_sync(city: str = "Paris") -> str:
    """
    Enveloppe synchrone pour usage simple dans Streamlit.
    Gère les environnements avec ou sans event loop existant.
    """
    import asyncio
    try:
        return asyncio.run(tool_weather(city))
    except RuntimeError:
        # Cas où une boucle asyncio existe déjà (rare avec Streamlit, mais safe)
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(tool_weather(city))
        finally:
            loop.close()


# =====================================================================
#  Recherche Web (DuckDuckGo)
# =====================================================================

def tool_web_search(query: str, max_results: int = 5) -> str:
    """
    Recherche texte via DuckDuckGo, retourne une chaîne JSON avec une liste de
    résultats : [{"title", "href", "body"}, ...].

    La mise en forme (markdown) est gérée par render_web_results() côté app.
    """
    q = _clean_web_query(query)

    def clip(s: Optional[str], n: int = 300) -> str:
        s = s or ""
        return s[:n]

    try:
        with DDGS() as ddgs:
            results_iter = ddgs.text(
                q,
                region="fr-fr",
                safesearch="moderate",
                max_results=max_results,
            )
            results = list(results_iter)

        payload = [
            {
                "title": clip(r.get("title")),
                "href": r.get("href"),
                "body": clip(r.get("body")),
            }
            for r in results
        ]
        return json.dumps(payload, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Recherche échouée: {e}"}, ensure_ascii=False)


# =====================================================================
#  TODO (persistant sur disque)
# =====================================================================

_TODO: List[Dict] = []
_TODO_PATH = Path(__file__).parent / "todo_store.json"


def _todo_load() -> None:
    """Charge la liste TODO depuis disque (si présente)."""
    global _TODO
    try:
        if _TODO_PATH.exists():
            data = json.loads(_TODO_PATH.read_text(encoding="utf-8"))
            _TODO = data if isinstance(data, list) else []
    except Exception:
        _TODO = []


def _todo_save() -> None:
    """Sauvegarde la liste TODO vers disque (silencieux si erreur)."""
    try:
        _TODO_PATH.write_text(
            json.dumps(_TODO, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


# Chargement au moment de l'import du module
_todo_load()


def tool_todo(cmd: str) -> str:
    """
    Gestion d'une TODO-list persistante.

    Commandes en langage naturel :
      - "ajoute : réviser IA"   → ajout
      - "liste" / "list"        → liste
      - "termine 2" / "done: 2" → termine tâche n°2

    Retourne une chaîne (markdown ou JSON) prête à afficher.
    """
    q = _normalize_todo_command(cmd)

    # Ajout
    if q.startswith("add:"):
        text = q[4:].strip()
        if not text:
            return "Texte vide."
        item = {"id": len(_TODO) + 1, "text": text, "done": False}
        _TODO.append(item)
        _todo_save()
        return f"Ajouté: {item}"

    # Terminer une tâche
    if q.startswith("done:"):
        try:
            idx = int(q[5:].strip())
        except ValueError:
            return "ID invalide."
        for it in _TODO:
            if it["id"] == idx:
                it["done"] = True
                _todo_save()
                return f"Terminé: {it}"
        return "ID introuvable."

    # Liste
    if q == "list":
        return json.dumps(_TODO, ensure_ascii=False)

    return "Commande inconnue (utilise 'add', 'done' ou 'list')."


__all__ = [
    "tool_calculator",
    "tool_weather",
    "tool_weather_sync",
    "tool_web_search",
    "tool_todo",
]
