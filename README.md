# Dual-tikets-Jira






pair_llm_labeling.py - Основной модуль для LLM-разметки ПАР баг-тикетов.

Пайплайн:
- на вход: таблица тикетов (issues) и таблица пар (candidate_pairs)
- для каждой пары строим markdown-описание issue_1 и issue_2
- вызываем gpt-5.1 (Chat Completions API, service_tier="flex"),
  обёрнутый в LangSmith (wrap_openai),
- получаем одну из 4 меток: "duplicate", "probable_duplicate", "regression", "unrelated"
- сохраняем результат в CSV/Excel

❗ LangSmith:
- трассировка включается установкой env-переменных:
    LANGCHAIN_API_KEY
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_PROJECT=<project_name>
- OpenAI-клиент оборачивается через langsmith.wrappers.wrap_openai
- функция label_pair помечена @traceable -> одна трасса на каждую пару

Ожидаемые входные данные:

1) Таблица тикетов (issues) с колонками минимум:
   - key                (строка, например "CHTI-7434")
   - summary            (краткое описание)
   - description        (полное описание / тело тикета)
   - platform           (android / ios / desktop / unknown, опционально)
   - components         (строка с компонентами, опционально)
   - environment        (строка с окружением, опционально)
   - status             (строка, опционально)
   - resolution         (строка, опционально)
   - created, updated   (опционально, строки)

2) Таблица пар (candidate_pairs) с колонками:
   - issue_key_1
   - issue_key_2

Файл системных промптов:
- bug_dedup_prompts.py в том же каталоге, с переменной PAIR_SYSTEM_PROMPT.

Пример запуска из консоли:

    python pair_llm_labeling.py \\
        --issues issues.parquet \\
        --pairs candidate_pairs.csv \\
        --output pairs_labeled.xlsx
