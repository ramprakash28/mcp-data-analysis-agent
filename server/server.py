"""
MCP Data Analysis Server
========================
A custom Model Context Protocol (MCP) server that exposes data analysis
capabilities as tools. Built with the official `mcp` Python library.

Tools provided:
  - load_dataset          : Load a CSV file into memory
  - get_dataset_info      : Shape, columns, and data types
  - get_summary_statistics: Descriptive stats (mean, std, min, max, etc.)
  - analyze_missing_values: Count and % of missing data per column
  - compute_correlations  : Pearson correlation matrix for numeric columns
  - detect_outliers       : IQR-based outlier detection per column
  - filter_data           : Filter rows by a column condition
  - group_and_aggregate   : Group-by with aggregation (sum, mean, count, etc.)
  - generate_visualization: Save a chart to disk and return its path
  - save_report           : Write a markdown report to disk
"""

import asyncio
import json
import os
import io
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for servers
import matplotlib.pyplot as plt

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# ── in-memory dataset store ────────────────────────────────────────────────
_datasets: dict[str, pd.DataFrame] = {}

server = Server("data-analysis-server")


# ── helpers ────────────────────────────────────────────────────────────────

def _require(dataset_name: str) -> pd.DataFrame:
    if dataset_name not in _datasets:
        raise KeyError(
            f"Dataset '{dataset_name}' not found. "
            f"Available: {list(_datasets.keys()) or 'none loaded yet'}"
        )
    return _datasets[dataset_name]


def _to_json(obj) -> str:
    """Serialize numpy / pandas types to plain JSON."""
    return json.dumps(obj, default=lambda o: float(o) if hasattr(o, "item") else str(o), indent=2)


# ── tool definitions ───────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="load_dataset",
            description="Load a CSV file from disk into memory for analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path":    {"type": "string", "description": "Absolute or relative path to the CSV file."},
                    "dataset_name": {"type": "string", "description": "Alias used to reference this dataset in subsequent calls."},
                },
                "required": ["file_path", "dataset_name"],
            },
        ),
        types.Tool(
            name="get_dataset_info",
            description="Return shape, column names, data types, and first 5 rows of a loaded dataset.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string"},
                },
                "required": ["dataset_name"],
            },
        ),
        types.Tool(
            name="get_summary_statistics",
            description="Compute descriptive statistics (count, mean, std, min, 25/50/75%, max) for numeric columns.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string"},
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Subset of columns to analyse. Omit for all numeric columns.",
                    },
                },
                "required": ["dataset_name"],
            },
        ),
        types.Tool(
            name="analyze_missing_values",
            description="Return count and percentage of missing values for every column.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string"},
                },
                "required": ["dataset_name"],
            },
        ),
        types.Tool(
            name="compute_correlations",
            description="Compute the Pearson correlation matrix for numeric columns.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string"},
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to include. Omit for all numeric columns.",
                    },
                },
                "required": ["dataset_name"],
            },
        ),
        types.Tool(
            name="detect_outliers",
            description="Detect outliers in a numeric column using the IQR method.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string"},
                    "column":       {"type": "string", "description": "Numeric column to inspect."},
                },
                "required": ["dataset_name", "column"],
            },
        ),
        types.Tool(
            name="filter_data",
            description="Filter rows where column satisfies a condition (gt, lt, eq, contains).",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string"},
                    "column":       {"type": "string"},
                    "operator":     {"type": "string", "enum": ["gt", "lt", "eq", "ne", "contains"]},
                    "value":        {"description": "Threshold value (string or number)."},
                },
                "required": ["dataset_name", "column", "operator", "value"],
            },
        ),
        types.Tool(
            name="group_and_aggregate",
            description="Group the dataset by one column and aggregate another with a function.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string"},
                    "group_by":     {"type": "string", "description": "Column to group by."},
                    "agg_column":   {"type": "string", "description": "Column to aggregate."},
                    "agg_func":     {"type": "string", "enum": ["sum", "mean", "count", "min", "max", "median"]},
                },
                "required": ["dataset_name", "group_by", "agg_column", "agg_func"],
            },
        ),
        types.Tool(
            name="generate_visualization",
            description="Create and save a chart. Returns the file path to the saved PNG.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string"},
                    "chart_type":   {"type": "string", "enum": ["bar", "line", "scatter", "histogram", "box"]},
                    "x_column":     {"type": "string"},
                    "y_column":     {"type": "string", "description": "Required for bar, line, scatter."},
                    "title":        {"type": "string"},
                    "output_path":  {"type": "string", "description": "Where to save the PNG (default: charts/<title>.png)."},
                },
                "required": ["dataset_name", "chart_type", "x_column"],
            },
        ),
        types.Tool(
            name="save_report",
            description="Write a markdown analysis report to disk.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content":     {"type": "string", "description": "Full markdown content of the report."},
                    "output_path": {"type": "string", "description": "File path to save (default: report.md)."},
                },
                "required": ["content"],
            },
        ),
    ]


# ── tool implementations ───────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    # ── load_dataset ──────────────────────────────────────────────────────
    if name == "load_dataset":
        file_path    = arguments["file_path"]
        dataset_name = arguments["dataset_name"]

        df = pd.read_csv(file_path)
        _datasets[dataset_name] = df

        return [types.TextContent(
            type="text",
            text=_to_json({
                "status":   "loaded",
                "dataset":  dataset_name,
                "rows":     len(df),
                "columns":  list(df.columns),
                "dtypes":   {c: str(t) for c, t in df.dtypes.items()},
            }),
        )]

    # ── get_dataset_info ──────────────────────────────────────────────────
    elif name == "get_dataset_info":
        df = _require(arguments["dataset_name"])
        return [types.TextContent(
            type="text",
            text=_to_json({
                "shape":   list(df.shape),
                "columns": list(df.columns),
                "dtypes":  {c: str(t) for c, t in df.dtypes.items()},
                "head":    df.head().to_dict(orient="records"),
            }),
        )]

    # ── get_summary_statistics ────────────────────────────────────────────
    elif name == "get_summary_statistics":
        df   = _require(arguments["dataset_name"])
        cols = arguments.get("columns")
        sub  = df[cols] if cols else df.select_dtypes(include="number")
        return [types.TextContent(
            type="text",
            text=sub.describe().to_json(indent=2),
        )]

    # ── analyze_missing_values ────────────────────────────────────────────
    elif name == "analyze_missing_values":
        df      = _require(arguments["dataset_name"])
        missing = df.isnull().sum()
        pct     = (missing / len(df) * 100).round(2)
        return [types.TextContent(
            type="text",
            text=_to_json({
                col: {"missing_count": int(missing[col]), "missing_pct": float(pct[col])}
                for col in df.columns
            }),
        )]

    # ── compute_correlations ──────────────────────────────────────────────
    elif name == "compute_correlations":
        df   = _require(arguments["dataset_name"])
        cols = arguments.get("columns")
        sub  = df[cols] if cols else df.select_dtypes(include="number")
        corr = sub.corr().round(4)
        return [types.TextContent(type="text", text=corr.to_json(indent=2))]

    # ── detect_outliers ───────────────────────────────────────────────────
    elif name == "detect_outliers":
        df  = _require(arguments["dataset_name"])
        col = arguments["column"]
        series = df[col].dropna()
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR     = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        return [types.TextContent(
            type="text",
            text=_to_json({
                "column":         col,
                "Q1":             float(Q1),
                "Q3":             float(Q3),
                "IQR":            float(IQR),
                "lower_bound":    float(lower),
                "upper_bound":    float(upper),
                "outlier_count":  len(outliers),
                "outlier_pct":    round(len(outliers) / len(df) * 100, 2),
                "sample_outliers": outliers[col].head(10).tolist(),
            }),
        )]

    # ── filter_data ───────────────────────────────────────────────────────
    elif name == "filter_data":
        df       = _require(arguments["dataset_name"])
        col      = arguments["column"]
        operator = arguments["operator"]
        value    = arguments["value"]

        if operator == "gt":       mask = df[col] > float(value)
        elif operator == "lt":     mask = df[col] < float(value)
        elif operator == "eq":     mask = df[col] == value
        elif operator == "ne":     mask = df[col] != value
        elif operator == "contains": mask = df[col].astype(str).str.contains(str(value), case=False)
        else:                      raise ValueError(f"Unknown operator: {operator}")

        filtered = df[mask]
        return [types.TextContent(
            type="text",
            text=_to_json({"rows_matched": len(filtered), "sample": filtered.head(10).to_dict(orient="records")}),
        )]

    # ── group_and_aggregate ───────────────────────────────────────────────
    elif name == "group_and_aggregate":
        df       = _require(arguments["dataset_name"])
        group_by = arguments["group_by"]
        agg_col  = arguments["agg_column"]
        agg_func = arguments["agg_func"]

        result = df.groupby(group_by)[agg_col].agg(agg_func).reset_index()
        result.columns = [group_by, f"{agg_func}_{agg_col}"]
        return [types.TextContent(
            type="text",
            text=result.to_json(orient="records", indent=2),
        )]

    # ── generate_visualization ────────────────────────────────────────────
    elif name == "generate_visualization":
        df         = _require(arguments["dataset_name"])
        chart_type = arguments["chart_type"]
        x_col      = arguments["x_column"]
        y_col      = arguments.get("y_column")
        title      = arguments.get("title", f"{chart_type.capitalize()} of {x_col}")
        out_path   = arguments.get("output_path", f"charts/{title.replace(' ', '_')}.png")

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "bar":
            data = df.groupby(x_col)[y_col].mean()
            data.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
        elif chart_type == "line":
            ax.plot(df[x_col], df[y_col], marker="o", linewidth=2)
        elif chart_type == "scatter":
            ax.scatter(df[x_col], df[y_col], alpha=0.6)
            ax.set_xlabel(x_col); ax.set_ylabel(y_col)
        elif chart_type == "histogram":
            df[x_col].dropna().plot(kind="hist", bins=30, ax=ax, color="steelblue", edgecolor="white")
        elif chart_type == "box":
            cols = [x_col, y_col] if y_col else [x_col]
            df[cols].dropna().plot(kind="box", ax=ax)

        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        return [types.TextContent(
            type="text",
            text=_to_json({"status": "saved", "path": out_path, "title": title}),
        )]

    # ── save_report ───────────────────────────────────────────────────────
    elif name == "save_report":
        content    = arguments["content"]
        out_path   = arguments.get("output_path", "report.md")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(content, encoding="utf-8")
        return [types.TextContent(
            type="text",
            text=_to_json({"status": "saved", "path": out_path, "chars": len(content)}),
        )]

    else:
        raise ValueError(f"Unknown tool: {name}")


# ── entry point ────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
