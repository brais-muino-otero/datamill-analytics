"""
i18n/translator.py
==================
Sistema de internacionalizacion (i18n) ligero y sin dependencias externas.

Idea: cada idioma es un fichero JSON (i18n/locales/<codigo>.json) con la MISMA
estructura de claves anidadas. Al arrancar cargamos todos los JSON en memoria;
en tiempo de ejecucion resolvemos claves con notacion de punto, p.ej.:

    translator.get("en", "nav.supervised")  ->  "Supervised Learning"

Estrategia de respaldo (fallback) en 3 niveles para que la UI NUNCA quede vacia:
    1. Clave en el idioma pedido.
    2. Clave en el idioma por defecto (DEFAULT_LANGUAGE).
    3. La propia clave como texto (asi detectas a simple vista que falta traducir).
"""
import json
from pathlib import Path
from typing import Callable

from config import DEFAULT_LANGUAGE, LOCALES_DIR


class Translator:
    """Carga y sirve traducciones desde ficheros JSON."""

    def __init__(self, locales_dir: Path, default_lang: str = DEFAULT_LANGUAGE) -> None:
        self.default_lang = default_lang
        # {codigo_idioma: {dict anidado de traducciones}}
        self._translations: dict[str, dict] = {}
        self._load_all(locales_dir)

    # ------------------------------------------------------------------ #
    # Carga
    # ------------------------------------------------------------------ #
    def _load_all(self, locales_dir: Path) -> None:
        """Lee TODOS los *.json de la carpeta de locales (una sola vez, al arrancar)."""
        for json_file in sorted(locales_dir.glob("*.json")):
            lang_code = json_file.stem                 # "es.json" -> "es"
            with open(json_file, "r", encoding="utf-8") as f:
                self._translations[lang_code] = json.load(f)

        # Si ni siquiera existe el idioma por defecto, algo va muy mal: fallamos pronto.
        if self.default_lang not in self._translations:
            raise FileNotFoundError(
                f"No se encontro el idioma por defecto '{self.default_lang}' en {locales_dir}"
            )

    # ------------------------------------------------------------------ #
    # Resolucion de claves
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve(tree: dict, dotted_key: str):
        """Navega un dict anidado con clave de puntos ('a.b.c'). None si no existe."""
        node = tree
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return None
        return node

    def get(self, lang: str, key: str, **fmt_kwargs) -> str:
        """
        Devuelve la traduccion de `key` para el idioma `lang`.
        Acepta kwargs para interpolar plantillas ({name}) con str.format.
        """
        # 1) idioma pedido  ->  2) idioma por defecto
        value = self._resolve(self._translations.get(lang, {}), key)
        if value is None:
            value = self._resolve(self._translations[self.default_lang], key)
        # 3) ultimo recurso: la propia clave (senal visible de "falta traducir")
        if value is None:
            return key
        # Interpolacion opcional: t("msg.greeting", name="Brais")
        return value.format(**fmt_kwargs) if fmt_kwargs else value

    # ------------------------------------------------------------------ #
    # Azucar sintactico
    # ------------------------------------------------------------------ #
    def scoped(self, lang: str) -> Callable[..., str]:
        """
        Devuelve una funcion `t(key, **kwargs)` ya "fijada" a un idioma.
        Permite escribir layouts limpios:  html.H1(t("pages.supervised.title"))
        """
        return lambda key, **kwargs: self.get(lang, key, **kwargs)


# Instancia unica (singleton) compartida por toda la app.
translator = Translator(LOCALES_DIR, DEFAULT_LANGUAGE)
