import json
import pandas as pd

def format_ticket(row: pd.Series, suffix: str) -> str:
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
