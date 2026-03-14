from __future__ import annotations

from typing import Any

import numpy as np


def parse_constraints(constraints_path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(constraints_path, "r", encoding="utf-8") as f:
        lines = {
            line.split(":")[0]: line.split(":", 1)[1].strip()
            for line in f
            if ":" in line
        }

    capacities = np.array(
        list(map(int, lines["capacities"].split(","))), dtype=np.int32
    )
    conf_raw = lines.get("conflicts", "")
    conflict_pairs = (
        np.array(
            [tuple(map(int, c.split(","))) for c in conf_raw.split(";") if c],
            dtype=np.int32,
        )
        if conf_raw
        else np.zeros((0, 2), dtype=np.int32)
    )
    return capacities, conflict_pairs


def evaluate_solution(
    solution: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    capacities: np.ndarray,
    item_groups: np.ndarray,
    conflict_pairs: np.ndarray,
    group_max: int,
    bonus_val: int = 50,
) -> dict[str, Any]:
    sol = np.asarray(solution, dtype=np.int8).reshape(-1)
    selected_indices = np.where(sol == 1)[0]

    errors: list[str] = []
    conflict_violations: list[tuple[int, int]] = []

    if selected_indices.size == 0:
        errors.append("No items selected")
        total_weights = np.zeros(3, dtype=np.int64)
        unique_groups = np.zeros(0, dtype=np.int32)
        counts = np.zeros(0, dtype=np.int32)
        base_score = 0
        bonus_score = 0
    else:
        total_weights = np.sum(weights[selected_indices], axis=0)
        cap_over = total_weights - capacities
        if np.any(cap_over > 0):
            errors.append(
                "Capacity violation: "
                + ", ".join(
                    [
                        f"w{i}={int(total_weights[i])}>{int(capacities[i])}"
                        for i in range(len(capacities))
                        if cap_over[i] > 0
                    ]
                )
            )

        unique_groups, counts = np.unique(
            item_groups[selected_indices], return_counts=True
        )
        over_groups = unique_groups[counts > group_max]
        if over_groups.size > 0:
            errors.append(
                "Group max violation: "
                + ", ".join(
                    [
                        f"g{int(g)}={int(c)}>{group_max}"
                        for g, c in zip(unique_groups, counts)
                        if c > group_max
                    ]
                )
            )

        selected_group_set = set(int(g) for g in unique_groups.tolist())
        for pair in np.asarray(conflict_pairs):
            g1 = int(pair[0])
            g2 = int(pair[1])
            if g1 in selected_group_set and g2 in selected_group_set:
                conflict_violations.append((g1, g2))

        if conflict_violations:
            errors.append(f"Conflict violations: {len(conflict_violations)} pairs")

        base_score = int(np.sum(values[selected_indices]))
        bonus_count = int(np.sum((counts >= 3) & (counts <= 5)))
        bonus_score = int(bonus_count * bonus_val)

    total_score = int(base_score + bonus_score)
    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "errors": errors,
        "selected_count": int(selected_indices.size),
        "selected_indices": selected_indices.astype(np.int32),
        "selected_groups": unique_groups.astype(np.int32),
        "group_counts": counts.astype(np.int32),
        "total_weights": np.asarray(total_weights, dtype=np.int64),
        "capacities": np.asarray(capacities, dtype=np.int64),
        "base_score": int(base_score),
        "bonus_score": int(bonus_score),
        "total_score": int(total_score),
        "conflict_violation_count": int(len(conflict_violations)),
        "conflict_violations": conflict_violations,
    }


def format_solution_report(
    solver_name: str,
    elapsed_sec: float,
    status: str,
    evaluation: dict[str, Any],
    objective_value: Any | None = None,
    timestamp: str | None = None,
    full_output: bool = True,
    max_items_preview: int = 40,
    max_groups_preview: int = 20,
) -> str:
    selected_items = evaluation["selected_indices"]
    selected_groups = evaluation["selected_groups"]
    group_counts = evaluation["group_counts"]

    if full_output:
        items_text = ",".join(map(str, selected_items))
        items_label = "Selected Items (full)"
    else:
        items_text = ",".join(map(str, selected_items[:max_items_preview]))
        if selected_items.size > max_items_preview:
            items_text += ",..."
        items_label = "Selected Items (preview)"

    if full_output:
        group_parts = [
            f"g{int(g)}:{int(c)}" for g, c in zip(selected_groups, group_counts)
        ]
        groups_text = ", ".join(group_parts)
        groups_label = "Selected Groups (full)"
    else:
        group_parts = [
            f"g{int(g)}:{int(c)}"
            for g, c in zip(
                selected_groups[:max_groups_preview], group_counts[:max_groups_preview]
            )
        ]
        groups_text = ", ".join(group_parts)
        if selected_groups.size > max_groups_preview:
            groups_text += ", ..."
        groups_label = "Selected Groups (preview)"

    weights = evaluation["total_weights"]
    caps = evaluation["capacities"]
    usage = [
        f"w{i}={int(weights[i])}/{int(caps[i])} ({(float(weights[i]) / float(caps[i]) * 100.0) if caps[i] else 0.0:.1f}%)"
        for i in range(len(caps))
    ]

    objective_line = (
        f"Objective Value (Solver): {objective_value}"
        if objective_value is not None
        else "Objective Value (Solver): N/A"
    )
    validation_label = "VALID" if evaluation["is_valid"] else "INVALID"
    error_line = (
        "Validation Errors: none"
        if evaluation["is_valid"]
        else "Validation Errors: " + " | ".join(evaluation["errors"])
    )

    ts = timestamp or ""
    ts_header = f"[{ts}]\n" if ts else ""

    return (
        f"{ts_header}"
        f"Solver: {solver_name}\n"
        f"Status: {status}\n"
        f"Validation: {validation_label}\n"
        f"{objective_line}\n"
        f"Recalculated Score: {evaluation['total_score']} (base={evaluation['base_score']}, bonus={evaluation['bonus_score']})\n"
        f"Selected Item Count: {evaluation['selected_count']}\n"
        f"{items_label}: [{items_text}]\n"
        f"Selected Group Count: {selected_groups.size}\n"
        f"{groups_label}: [{groups_text}]\n"
        f"Capacity Usage: {', '.join(usage)}\n"
        f"Conflict Violations: {evaluation['conflict_violation_count']}\n"
        f"{error_line}\n"
        f"Execution Time: {elapsed_sec:.4f} seconds\n" + "-" * 50 + "\n"
    )
