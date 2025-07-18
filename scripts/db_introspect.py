#!/usr/bin/env python3
"""MySQL Schema Introspection Helper for SiloFlow
================================================

This standalone utility connects to a MySQL 8.0+ server and extracts a rich
schema description that helps developers understand table structures and
relationships before writing complex SQL queries.

Features
--------
1. Lists all tables in one or more schemas (databases).
2. Captures column details (name, type, nullability, default, extra flags).
3. Extracts primary- and foreign-key constraints from *information_schema*.
4. Optionally records approximate row counts per table (uses InnoDB stats).
5. Writes structured metadata to a JSON file **and** prints a concise Markdown
   summary to stdout for quick inspection.

Example usage
-------------
    # Introspect the two key schemas and save as schema_dump.json
    python scripts/db_introspect.py \
        --host 127.0.0.1 --port 3306 \
        --user root --password secret \
        --schemas cloud_server,cloud_lq \
        --output schema_dump.json

    # Generate Markdown only (no JSON file)
    python scripts/db_introspect.py --schemas cloud_server --no-json
"""
from __future__ import annotations

import argparse
import json
import sys
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_tables(conn, schema: str) -> List[str]:
    """Return table names for *schema* sorted alphabetically."""
    # Use explicit lowercase aliases so row objects expose predictable keys
    q = text(
        """
        SELECT TABLE_NAME   AS table_name
        FROM information_schema.tables
        WHERE table_schema = :schema
        ORDER BY TABLE_NAME
        """
    )
    return [row._mapping["table_name"] for row in conn.execute(q, {"schema": schema})]


def fetch_columns(conn, schema: str, table: str) -> List[Dict[str, Any]]:
    """Return column metadata for *schema.table* ordered by ordinal position."""
    q = text(
        """
        SELECT
            COLUMN_NAME     AS column_name,
            DATA_TYPE       AS data_type,
            COLUMN_TYPE     AS column_type,
            IS_NULLABLE     AS is_nullable,
            COLUMN_DEFAULT  AS column_default,
            COLUMN_KEY      AS column_key,
            EXTRA           AS extra
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
        ORDER BY ORDINAL_POSITION
        """
    )
    rows = conn.execute(q, {"schema": schema, "table": table})
    return [dict(row._mapping) for row in rows]


def fetch_constraints(conn, schema: str, table: str) -> Dict[str, Any]:
    """Return PK & FK constraint details for *schema.table*."""

    # Primary-key columns ----------------------------------------------------
    pk_q = text(
        """
        SELECT COLUMN_NAME AS column_name
        FROM information_schema.key_column_usage
        WHERE table_schema = :schema
          AND table_name   = :table
          AND constraint_name = 'PRIMARY'
        ORDER BY ORDINAL_POSITION
        """
    )
    pk_cols = [row._mapping["column_name"] for row in conn.execute(pk_q, {"schema": schema, "table": table})]

    # Foreign-keys -----------------------------------------------------------
    fk_q = text(
        """
        SELECT
            COLUMN_NAME           AS column_name,
            REFERENCED_TABLE_SCHEMA AS referenced_table_schema,
            REFERENCED_TABLE_NAME   AS referenced_table_name,
            REFERENCED_COLUMN_NAME  AS referenced_column_name,
            CONSTRAINT_NAME        AS constraint_name
        FROM information_schema.key_column_usage
        WHERE table_schema = :schema
          AND table_name   = :table
          AND referenced_table_name IS NOT NULL
        ORDER BY constraint_name, ORDINAL_POSITION
        """
    )
    fks = [dict(row._mapping) for row in conn.execute(fk_q, {"schema": schema, "table": table})]

    return {"primary_key": pk_cols, "foreign_keys": fks}


def fetch_row_count(conn, schema: str, table: str) -> int:
    """Return *approximate* row count from information_schema (InnoDB stats)."""
    q = text(
        """
        SELECT TABLE_ROWS AS table_rows
        FROM information_schema.tables
        WHERE table_schema = :schema AND table_name = :table
        """
    )
    row = conn.execute(q, {"schema": schema, "table": table}).first()
    return int(row.table_rows) if row and row.table_rows is not None else -1


# ---------------------------------------------------------------------------
# Main introspection routine
# ---------------------------------------------------------------------------

def introspect_schema(conn, schema: str, include_counts: bool = False) -> Dict[str, Any]:
    """Gather all metadata for *schema* and return as a nested dict."""
    schema_info: Dict[str, Any] = {}
    for tbl in fetch_tables(conn, schema):
        col_meta = fetch_columns(conn, schema, tbl)
        constraints = fetch_constraints(conn, schema, tbl)
        tbl_info: Dict[str, Any] = {
            "columns": col_meta,
            **constraints,
        }
        if include_counts:
            tbl_info["approx_rows"] = fetch_row_count(conn, schema, tbl)
        schema_info[tbl] = tbl_info
    return schema_info


def render_markdown(metadata: Dict[str, Any]) -> str:
    """Return a human-readable Markdown overview of *metadata*."""
    lines: List[str] = []
    for schema, tables in metadata.items():
        lines.append(f"# Schema `{schema}`\n")
        for tbl, info in tables.items():
            lines.append(f"## Table `{schema}.{tbl}`\n")

            # Column list ----------------------------------------------------
            lines.append("| Column | Type | Null | Key | Default | Extra |\n|--------|------|------|-----|---------|-------|")
            for col in info["columns"]:
                lines.append(
                    f"| {col['column_name']} | {col['column_type']} | {col['is_nullable']} | "
                    f"{col.get('column_key', '') or ''} | {col['column_default'] if col['column_default'] is not None else ''} | {col['extra']} |"
                )
            lines.append("")

            # Primary key --------------------------------------------------
            pk = info.get("primary_key", [])
            if pk:
                lines.append(f"**Primary key**: {', '.join(pk)}\n")

            # Foreign keys -------------------------------------------------
            fks = info.get("foreign_keys", [])
            if fks:
                lines.append("**Foreign keys**:")
                for fk in fks:
                    lines.append(
                        f"* `{fk['column_name']}` → `{fk['referenced_table_schema']}.{fk['referenced_table_name']}.{fk['referenced_column_name']}`"
                    )
                lines.append("")

            # Row count ----------------------------------------------------
            if "approx_rows" in info and info["approx_rows"] >= 0:
                lines.append(f"Approx. rows: {info['approx_rows']:,}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MySQL schema introspection utility")
    p.add_argument("--host", default="127.0.0.1", help="MySQL host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=3306, help="MySQL port (default: 3306)")
    p.add_argument("--user", required=True, help="MySQL username")
    p.add_argument("--password", help="MySQL password (will prompt if omitted)")
    p.add_argument(
        "--schemas",
        default="cloud_server,cloud_lq",
        help="Comma-separated list of schemas/databases to introspect",
    )
    p.add_argument("--output", default="schema_dump.json", help="JSON file to write (default: schema_dump.json)")
    p.add_argument("--no-json", action="store_true", help="Do not write JSON file")
    p.add_argument("--md-output", help="Markdown file to write. If omitted, prints to stdout. If set to 'auto', uses <output>.md")
    p.add_argument("--include-counts", action="store_true", help="Include approximate row counts from information_schema")
    return p.parse_args(argv)


def build_engine(user: str, pwd: str, host: str, port: int):
    from urllib.parse import quote_plus

    user_enc = quote_plus(user)
    pwd_enc = quote_plus(pwd)
    return create_engine(
        f"mysql+pymysql://{user_enc}:{pwd_enc}@{host}:{port}/information_schema",
        pool_pre_ping=True,
    )


def main(argv: List[str] | None = None):
    args = parse_args(argv)

    password = args.password or getpass(f"Password for MySQL user '{args.user}': ")

    engine = build_engine(args.user, password, args.host, args.port)

    metadata: Dict[str, Any] = {}
    schemas = [s.strip() for s in args.schemas.split(",") if s.strip()]
    with engine.connect() as conn:
        for schema in schemas:
            print(f"Introspecting schema '{schema}'…", file=sys.stderr)
            metadata[schema] = introspect_schema(conn, schema, include_counts=args.include_counts)

    # Write JSON ------------------------------------------------------------
    if not args.no_json:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        print(f"\n📄 JSON schema dump written to {out_path}\n", file=sys.stderr)

    # Handle Markdown summary ---------------------------------------------
    md = render_markdown(metadata)
    md_out_arg = args.md_output
    if md_out_arg is None:
        # Default behaviour – print to console
        print(md)
    else:
        md_path = Path(args.output).with_suffix(".md") if md_out_arg == "auto" else Path(md_out_arg)
        md_path.write_text(md, encoding="utf-8")
        print(f"\n📝 Markdown schema written to {md_path}\n", file=sys.stderr)


if __name__ == "__main__":
    main() 