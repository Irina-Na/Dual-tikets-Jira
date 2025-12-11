
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any, List

import pandas as pd
from openai import OpenAI

# --- ошибки OpenAI (с запасным вариантом, если классы изменятся) ---
try:
    from openai import APIError, RateLimitError
except ImportError:  # на всякий случай, если сигнатуры SDK поменяются
    APIError = Exception
    RateLimitError = Exception

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

# --- LangSmith: обёртка клиента и декоратор трассировки ---
try:
    from langsmith.wrappers import wrap_openai
    from langsmith import traceable
except ImportError:
    # Если langsmith не установлен — просто делаем заглушки,
    # чтобы код продолжал работать без трассировки.
    def wrap_openai(client):
        return client

    def traceable(*dargs, **dkwargs):
        def decorator(func):
            return func
        return decorator

from bug_dedup_prompts import PAIR_SYSTEM_PROMPT, PairLLMResult  # системный промпт для пар
from tikets_preraratior import format_ticket_json, format_ticket_markdown, build_pair_user_input
from dotenv import load_dotenv

# сразу после импортов
load_dotenv()  # подхватит переменные из .env в текущей папке


# ─────────────────────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────────────────────

MODEL_NAME = "gpt-5.1"
SERVICE_TIER_FLEX = "flex"

# Таймаут увеличиваем, потому что flex может отвечать медленнее
DEFAULT_TIMEOUT_SECONDS = 900.0  # 15 minutes per flex guidance to reduce timeouts
# Пути по умолчанию для CLI (используются, если аргументы не переданы)
DEFAULT_ISSUES_PATH = ""
DEFAULT_PAIRS_PATH = "C:\Users\Ironia\PycharmProjects\Dual-tikets-Jira\labler"
DEFAULT_OUTPUT_PATH = "C:\Users\Ironia\PycharmProjects\Dual-tikets-Jira\labler\docs\output"


# ─────────────────────────────────────────────────────────────
# Клиент OpenAI + LangSmith
# ─────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    """
    Создаём OpenAI-клиент с увеличенным таймаутом и оборачиваем его в LangSmith.

    Чтобы трассировка реально работала, должны быть выставлены:
      - OPENAI_API_KEY
      - LANGCHAIN_API_KEY
      - LANGCHAIN_TRACING_V2 = "true"
      - (опц.) LANGCHAIN_PROJECT
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Нужно выставить переменную окружения OPENAI_API_KEY")

    client = OpenAI(timeout=DEFAULT_TIMEOUT_SECONDS)
    # Оборачиваем клиент — все chat.completions.create будут логироваться в LangSmith
    client = wrap_openai(client)

    # По желанию можно подсказать пользователю, если LangSmith выключен
    if not os.getenv("LANGCHAIN_API_KEY"):
        print(
            "[WARN] LANGCHAIN_API_KEY не задан — трассировка в LangSmith не активна. "
            "См. комментарии в модуле для настройки."
        )
    return client


# ─────────────────────────────────────────────────────────────
# Вызов LLM с ретраями
# ─────────────────────────────────────────────────────────────

def is_retryable_error(exc: BaseException) -> bool:
    """Определяем, нужно ли ретраить исключение (rate limit / временные ошибки)."""
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIError):
        # 408, 429, 5xx можно ретраить
        status = getattr(exc, "status_code", None)
        if status in (408, 429, 500, 502, 503):
            return True
    return False


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    retry=retry_if_exception(is_retryable_error),
    reraise=True,
)
def call_llm_for_pair_chat(
    client: OpenAI,
    user_input_str: str,
    response_format: Any
) -> tuple[PairLLMResult, Dict[str, Any]]:
    """
    Вызов gpt-5.1 (Chat Completions) в режиме flex для одной пары.

    - system message = PAIR_SYSTEM_PROMPT
    - user message = markdown с двумя тикетами
    - text_format (Responses API) = PairLLMResult (Pydantic), чтобы получить структурированный ответ
    - service_tier = "flex" (дешевле, но медленнее; подходит для оффлайн-разметки)
    """
    # Flex может отвечать медленнее, поэтому выставляем увеличенный таймаут на уровень запроса.
    response = client.with_options(timeout=DEFAULT_TIMEOUT_SECONDS).responses.parse(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": PAIR_SYSTEM_PROMPT},
            {"role": "user", "content": user_input_str},
        ],
        text_format=response_format,
        service_tier=SERVICE_TIER_FLEX,
    )

    parsed: PairLLMResult = response.output_parsed
    raw_json: Dict[str, Any] = response.model_dump()
    return parsed, raw_json


# ВАЖНО:
# @traceable делает отдельный trace в LangSmith для КАЖДОЙ пары.
# Внутри него LLM-вызов будет дочерним run'ом, т.к. клиент обёрнут через wrap_openai.
@traceable(name="bug_pair_labeling", run_type="chain")
def label_pair(
    client: OpenAI,
    issue1: Dict[str, Any],
    issue2: Dict[str, Any],
) -> tuple[PairLLMResult, Dict[str, Any]]:
    """
    Формирует input для пары тикетов, вызывает LLM и возвращает результат.

    Этот шаг — корневой trace в LangSmith.
    """
    user_input = build_pair_user_input(issue1, issue2)
    data, raw_json = call_llm_for_pair_chat(client, user_input, PairLLMResult)

    label = data.label
    if label not in ("duplicate", "probable_duplicate", "regression", "unrelated"):
        print(f"Неверный label от LLM: {label!r}, raw={data}")
        label = "unknown"

    parsed_result = PairLLMResult(
        issue_key_1=issue1.get("key", ""),
        issue_key_2=issue2.get("key", ""),
        label=label,
        reason=data.reason,
    )
    raw_json = raw_json or parsed_result.model_dump()
    return parsed_result, raw_json


# ─────────────────────────────────────────────────────────────
# Обработка датафреймов
# ─────────────────────────────────────────────────────────────

def load_issues(path: str) -> pd.DataFrame:
    """Загрузка таблицы тикетов (issues) из parquet/csv/xlsx."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif ext in (".csv",):
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Неизвестный формат файла тикетов: {path}")
    if "key" not in df.columns:
        raise ValueError("В таблице тикетов нет колонки 'key'")
    return df


def load_pairs(path: str) -> pd.DataFrame:
    """Загрузка таблицы пар (issue_key_1, issue_key_2)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif ext in (".csv",):
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Неизвестный формат файла пар: {path}")
    required_cols = {"issue_key_1", "issue_key_2"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"В таблице пар не хватает колонок: {missing}")
    return df


def build_issue_lookup(df_issues: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Создаём словарь key -> dict(issue_row) для быстрого доступа."""
    df_issues = df_issues.copy()
    df_issues["key"] = df_issues["key"].astype(str)
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in df_issues.to_dict(orient="records"):
        lookup[row["key"]] = row
    return lookup


def label_all_pairs(
    df_issues: pd.DataFrame,
    df_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Основная функция:
    - на вход: df_issues (все тикеты), df_pairs (список пар)
    - на выход: df_result с размеченными парами
    """
    client = get_client()
    issue_lookup = build_issue_lookup(df_issues)

    results: List[Dict[str, Any]] = []

    for idx, row in df_pairs.iterrows():
        key1 = str(row["issue_key_1"])
        key2 = str(row["issue_key_2"])

        issue1 = issue_lookup.get(key1)
        issue2 = issue_lookup.get(key2)

        if issue1 is None or issue2 is None:
            print(f"[WARN] Не найден один из тикетов: {key1}, {key2} — пропускаю")
            continue

        try:
            pair_res, raw_json = label_pair(client, issue1, issue2)
        except Exception as e:
            print(f"[ERROR] Ошибка при обработке пары {key1} / {key2}: {e}")
            continue

        results.append(
            {
                "issue_key_1": pair_res.issue_key_1,
                "issue_key_2": pair_res.issue_key_2,
                "label": pair_res.label,
                "reason": pair_res.reason,
                "raw_json": json.dumps(raw_json, ensure_ascii=False),
            }
        )

    df_result = pd.DataFrame(results)
    return df_result


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-разметка пар баг-тикетов (duplicate/probable_duplicate/regression/unrelated) с помощью gpt-5.1 + LangSmith tracing"
    )
    parser.add_argument(
        "--issues",
        default=DEFAULT_ISSUES_PATH,
        help="Путь к файлу с тикетами (parquet/csv/xlsx) с колонкой 'key'",
    )
    parser.add_argument(
        "--pairs",
        default=DEFAULT_PAIRS_PATH,
        help="Путь к файлу с парами (parquet/csv/xlsx) с колонками 'issue_key_1', 'issue_key_2'",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Путь к выходному файлу (csv или xlsx). По расширению определяется формат.",
    )
    args = parser.parse_args()
    if not args.issues:
        parser.error("--issues is required (или установите DEFAULT_ISSUES_PATH)")
    if not args.pairs:
        parser.error("--pairs is required (или установите DEFAULT_PAIRS_PATH)")
    if not args.output:
        parser.error("--output is required (или установите DEFAULT_OUTPUT_PATH)")
    return args


def save_result(df: pd.DataFrame, path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext in (".xlsx", ".xls"):
        df.to_excel(path, index=False)
    else:
        raise ValueError(f"Неизвестный формат выходного файла: {path}")


def main() -> None:
    args = parse_args()

    print(f"Загружаю тикеты из: {args.issues}")
    df_issues = load_issues(args.issues)
    print(f"Тикетов: {len(df_issues)}")

    print(f"Загружаю пары из: {args.pairs}")
    df_pairs = load_pairs(args.pairs)
    print(f"Пар-кандидатов: {len(df_pairs)}")

    print("Запускаю LLM-разметку пар (gpt-5.1, flex, LangSmith tracing)...")
    df_result = label_all_pairs(df_issues, df_pairs)
    print(f"Получено размеченных пар: {len(df_result)}")

    print(f"Сохраняю результат в: {args.output}")
    save_result(df_result, args.output)
    print("Готово.")


if __name__ == "__main__":
    main()
