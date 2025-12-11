import json
import pandas as pd

from typing import Any, Dict, List, Literal


# ─────────────────────────────────────────────────────────────
# Форматирование тикетов и пар для промпта
# ───────────────────────────────────────────────────────────── 


def build_pair_user_input(issue1: Dict[str, Any], issue2: Dict[str, Any], format_type: Literal["markdown", "json"] = "markdown") -> str:
    """
    Build a user prompt for a pair of issues, allowing either markdown or JSON layout.
    """
    if format_type == "json":
        payload = {
            "issue_1": _issue_to_json_dict(issue1),
            "issue_2": _issue_to_json_dict(issue2),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    if format_type != "markdown":
        raise ValueError(f"Unsupported format_type={format_type!r}. Use 'markdown' or 'json'.")

    parts: List[str] = [""]
    parts.append(format_ticket_markdown(issue1, tag="issue_1"))
    parts.append("")
    parts.append(format_ticket_markdown(issue2, tag="issue_2"))
    return "\n\n".join(parts)


def _issue_to_json_dict(issue: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize issue fields for JSON output, skipping empty values."""
    def clean(val: Any) -> Any:
        if val is None:
            return None
        if isinstance(val, str):
            val = val.strip()
            return val or None
        return val

    status_val = clean(issue.get("status"))
    resol_val = clean(issue.get("resolution"))
    if status_val and resol_val:
        status_combined = f"{status_val} {resol_val}".strip()
    else:
        status_combined = status_val or resol_val or None

    ticket = {
        "id": clean(issue.get("key")),
        "summary": clean(issue.get("summary")),
        "description": clean(issue.get("description")),
        "components": clean(issue.get("components")),
        "environment": clean(issue.get("environment")),
        "labels": clean(issue.get("labels")),
        "status": status_combined,
        "created_time": clean(issue.get("created")),
        "updated_time": clean(issue.get("updated")),
        "epic_link": clean(issue.get("epic_link")),
        "stand": clean(issue.get("stand")),
        "sprint": clean(issue.get("sprint")),
        "affects_versions": clean(issue.get("affects_versions")),
        "fix_versions": clean(issue.get("fix_versions")),
    }

    return {k: v for k, v in ticket.items() if v is not None}


def format_ticket_jsonsuffix: str) -> str:
    """
    Собирает JSON-описание одного тикета из колонок с суффиксом `_suffix`.

    Ожидаемые колонки (пример для suffix="1"):
      key_1, summary_1, description_1, components_1, environment_1,
      labels_1, status_1, resolution_1, created_1, updated_1,
      epic_link_1, stand_1, sprint_1, affects_versions_1, fix_versions_1
    """

    def get(col_base: str):
        col = f"{col_base}_{suffix}"
        if col in row and pd.notna(row[col]):
            v = str(row[col]).strip()
            return v if v else None
        return None

    key              = get("key")
    summary          = get("summary")
    description      = get("description")
    components       = get("components")
    environment      = get("environment")
    labels           = get("labels")
    status_val       = get("status")
    resol_val        = get("resolution")
    created          = get("created")
    updated          = get("updated")
    epic_link        = get("epic_link")
    stand            = get("stand")
    sprint           = get("sprint")
    affects_versions = get("affects_versions")
    fix_versions     = get("fix_versions")

    # status: status + " " + resolution (если есть)
    if status_val and resol_val:
        status_combined = f"{status_val} {resol_val}"
    else:
        status_combined = status_val or resol_val or None

    ticket = {
        "id": key,
        "summary": summary,
        "description": description,
        "components": components,
        "environment": environment,
        "labels": labels,
        "status": status_combined,
        "created_time": created,
        "updated_time": updated,
        "epic_link": epic_link,
        "stand": stand,
        "sprint": sprint,
        "affects_versions": affects_versions,
        "fix_versions": fix_versions,
    }

    # Выкидываем пустые поля, чтобы JSON был компактный
    ticket = {k: v for k, v in ticket.items() if v is not None}

    return json.dumps(ticket, ensure_ascii=False, indent=2)




def format_ticket_markdown(issue: Dict[str, Any], tag: str) -> str:
    """
    Формирует компактное markdown-описание одного тикета.

    tag — пометка "issue_1" или "issue_2", просто для читабельности.
    """
    key = issue.get("key", "")
    summary = (issue.get("summary") or "").strip()
    description = (issue.get("description") or "").strip()
    platform = issue.get("platform") or "unknown"
    components = issue.get("components") or ""
    environment = issue.get("environment") or ""
    status = issue.get("status") or ""
    resolution = issue.get("resolution") or ""
    created = issue.get("created") or ""
    updated = issue.get("updated") or ""

    lines = []
    lines.append(f"### {tag}: {key}")
    if summary:
        lines.append(f"**Кратко (summary):** {summary}")
    lines.append(f"**Платформа:** {platform}")
    if components:
        lines.append(f"**Компоненты:** {components}")
    if environment:
        lines.append(f"**Окружение:** {environment}")
    if status:
        lines.append(f"**Статус:** {status}")
    if resolution:
        lines.append(f"**Resolution:** {resolution}")
    if created or updated:
        lines.append(f"**Создан / обновлён:** {created} / {updated}")
    if description:
        lines.append("")
        lines.append("**Описание (включая шаги, ФР и ОР, если есть):**")
        lines.append(description.strip())

    return "\n".join(lines)
