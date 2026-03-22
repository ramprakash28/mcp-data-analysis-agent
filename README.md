# MCP Data Analysis Agent

An **autonomous data analysis agent** powered by [Claude AI](https://www.anthropic.com/claude) and a custom [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server. Point it at any CSV file, give it a goal in plain English, and it will explore the data, detect patterns, generate visualisations, and produce a full markdown report — all on its own.

---

## Architecture

```
User Prompt (CSV path + goal)
        │
        ▼
┌───────────────────┐       stdio        ┌──────────────────────────┐
│  Claude Agent     │ ◄─────────────────► │  MCP Data Analysis Server│
│  (claude-opus-4-6)│    tool calls /     │  (server/server.py)      │
│  agent/agent.py   │    tool results     │                          │
└───────────────────┘                    │  Tools:                  │
        │                                │  • load_dataset          │
        │  Final markdown report         │  • get_dataset_info      │
        ▼                                │  • get_summary_statistics│
   report.md + charts/                   │  • analyze_missing_values│
                                         │  • compute_correlations  │
                                         │  • detect_outliers       │
                                         │  • filter_data           │
                                         │  • group_and_aggregate   │
                                         │  • generate_visualization│
                                         │  • save_report           │
                                         └──────────────────────────┘
```

Claude connects to the MCP server over **stdio transport**, calls the tools autonomously in the right order, reasons over the results, and writes a final markdown report to disk.

---

## Features

- **10 MCP tools** for end-to-end data analysis
- **Autonomous agentic loop** — Claude decides which tools to call and in what order
- **Outlier detection** via IQR method
- **Correlation analysis** for numeric columns
- **Group-by aggregations** to surface business insights
- **Chart generation** (bar, line, scatter, histogram, box) saved as PNG
- **Markdown report** persisted to disk
- Works with **any CSV file** — no schema configuration needed

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/ramprakash28/mcp-data-analysis-agent.git
cd mcp-data-analysis-agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Anthropic API key
```bash
export ANTHROPIC_API_KEY=your_key_here
```

### 4. Run the agent
```bash
# Use the included sample dataset
python main.py data/sample_sales.csv

# Custom analysis goal
python main.py data/sample_sales.csv --goal "Identify the top performing regions and products"

# Your own data
python main.py /path/to/your/data.csv --goal "Find trends and anomalies"
```

---

## MCP Tools Reference

| Tool | Description |
|---|---|
| `load_dataset` | Load a CSV file into the server's memory |
| `get_dataset_info` | Return shape, columns, dtypes, and first 5 rows |
| `get_summary_statistics` | Descriptive stats: count, mean, std, min, max, quartiles |
| `analyze_missing_values` | Count and % of missing data per column |
| `compute_correlations` | Pearson correlation matrix for numeric columns |
| `detect_outliers` | IQR-based outlier detection with bounds and sample values |
| `filter_data` | Filter rows by column condition (gt, lt, eq, ne, contains) |
| `group_and_aggregate` | Group-by with sum / mean / count / min / max / median |
| `generate_visualization` | Create and save bar, line, scatter, histogram, or box chart |
| `save_report` | Write the final markdown report to disk |

---

## Project Structure

```
mcp-data-analysis-agent/
├── server/
│   └── server.py          # MCP server — exposes 10 data analysis tools
├── agent/
│   └── agent.py           # Claude agent — agentic loop with tool execution
├── data/
│   └── sample_sales.csv   # Sample dataset (50 sales records)
├── main.py                # CLI entry point
├── requirements.txt
└── .gitignore
```

---

## Example Output

After running the agent on `sample_sales.csv`, you will find:

- **`report.md`** — a comprehensive markdown analysis report including:
  - Dataset overview and schema
  - Descriptive statistics for all numeric columns
  - Missing value analysis
  - Correlation findings
  - Outlier summary
  - Regional and product performance breakdown
  - Key business insights and recommendations
- **`charts/`** — PNG visualisations generated during the analysis

---

## Why MCP?

The [Model Context Protocol](https://modelcontextprotocol.io) is an open standard that lets AI models like Claude connect to external tools and data sources in a structured, composable way. Instead of hardcoding tool logic into the agent, MCP allows you to:

- **Decouple** the AI layer from the data layer
- **Reuse** the same MCP server across multiple agents and applications
- **Extend** capabilities by adding new tools without modifying the agent
- **Standardise** how AI models interact with your infrastructure

This project demonstrates a real-world MCP use case where a data analysis server is exposed as a set of tools that Claude can discover, reason about, and use autonomously.

---

## Tech Stack

| Layer | Technology |
|---|---|
| AI Model | Claude (claude-opus-4-6) via Anthropic API |
| Protocol | Model Context Protocol (MCP) |
| Transport | stdio |
| Data processing | pandas, numpy |
| Visualisation | matplotlib |
| Language | Python 3.10+ |

---

## Roadmap

- [ ] Support for JSON, Parquet, and Excel files
- [ ] PostgreSQL / SQLite connector as an MCP tool
- [ ] HTML report with embedded charts
- [ ] Streamlit dashboard for interactive analysis
- [ ] Docker support

---

## License

MIT
