"""
Data Analysis Agent
===================
A Claude-powered autonomous agent that connects to the MCP data analysis
server and orchestrates a full analysis of a CSV dataset.

Architecture
------------
  User Prompt
      │
      ▼
  Claude (claude-opus-4-6)  ◄──── System Prompt
      │
      │  tool_use blocks
      ▼
  MCP Client  ──stdio──►  MCP Server (server/server.py)
      │                        │
      │  tool results           │  pandas / matplotlib
      ◄────────────────────────┘
      │
      ▼
  Final markdown report

Usage
-----
  from agent.agent import DataAnalysisAgent

  agent = DataAnalysisAgent()
  report = await agent.run("data/sample_sales.csv", "Analyse sales trends and top products")
  print(report)
"""

import asyncio
import json
import os
from pathlib import Path

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ── constants ──────────────────────────────────────────────────────────────

MODEL        = "claude-opus-4-6"
MAX_TOKENS   = 8096
SERVER_SCRIPT = Path(__file__).parent.parent / "server" / "server.py"

SYSTEM_PROMPT = """You are an expert data scientist and analyst. Your job is to:

1. Load and explore the dataset thoroughly
2. Compute descriptive statistics for all numeric columns
3. Identify and report missing values
4. Detect outliers in key numeric columns
5. Compute correlations between numeric variables
6. Perform meaningful group-by aggregations to surface business insights
7. Generate at least one visualization (bar chart or histogram)
8. Synthesise all findings into a clear, structured markdown report

Always follow this order:
  load_dataset → get_dataset_info → get_summary_statistics →
  analyze_missing_values → compute_correlations → detect_outliers →
  group_and_aggregate (as needed) → generate_visualization → save_report

Be thorough, professional, and data-driven in your analysis.
End with save_report to persist the markdown report to disk."""


# ── agent class ────────────────────────────────────────────────────────────

class DataAnalysisAgent:
    """Autonomous data analysis agent backed by Claude + a custom MCP server."""

    def __init__(self, model: str = MODEL):
        self.model  = model
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    async def run(self, csv_path: str, goal: str = "Perform a comprehensive data analysis") -> str:
        """
        Run the agent on a CSV file.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file to analyse.
        goal : str
            Natural-language description of the analysis goal.

        Returns
        -------
        str
            The final assistant message (usually a summary of the report).
        """
        server_params = StdioServerParameters(
            command="python",
            args=[str(SERVER_SCRIPT)],
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # ── fetch tool schemas from MCP server ──────────────────
                tools_result = await session.list_tools()
                tools = [
                    {
                        "name":         t.name,
                        "description":  t.description,
                        "input_schema": t.inputSchema,
                    }
                    for t in tools_result.tools
                ]

                print(f"[agent] Connected to MCP server — {len(tools)} tools available")
                print(f"[agent] Analysing: {csv_path}")
                print(f"[agent] Goal: {goal}\n")

                # ── initialise conversation ──────────────────────────────
                messages = [
                    {
                        "role": "user",
                        "content": (
                            f"Please analyse the dataset located at: {csv_path!r}\n\n"
                            f"Goal: {goal}\n\n"
                            "Use the available MCP tools to explore, analyse, visualise the data, "
                            "and finally save a comprehensive markdown report."
                        ),
                    }
                ]

                # ── agentic loop ─────────────────────────────────────────
                final_text = ""
                iteration  = 0

                while True:
                    iteration += 1
                    print(f"[agent] ── iteration {iteration} ──")

                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=MAX_TOKENS,
                        system=SYSTEM_PROMPT,
                        tools=tools,
                        messages=messages,
                    )

                    print(f"[agent] stop_reason={response.stop_reason}  "
                          f"content_blocks={len(response.content)}")

                    # collect text for return value
                    for block in response.content:
                        if hasattr(block, "text"):
                            final_text = block.text

                    # done
                    if response.stop_reason == "end_turn":
                        print("[agent] Analysis complete.")
                        break

                    # no tool calls → nothing more to do
                    tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
                    if not tool_use_blocks:
                        break

                    # append assistant turn
                    messages.append({"role": "assistant", "content": response.content})

                    # execute each tool call via MCP
                    tool_results = []
                    for block in tool_use_blocks:
                        print(f"[agent]   → {block.name}({json.dumps(block.input)[:120]})")
                        try:
                            result = await session.call_tool(block.name, block.input)
                            content_text = (
                                result.content[0].text
                                if result.content
                                else '{"status": "ok", "result": null}'
                            )
                            print(f"[agent]     ✓ {content_text[:120]}")
                        except Exception as exc:
                            content_text = json.dumps({"error": str(exc)})
                            print(f"[agent]     ✗ {content_text}")

                        tool_results.append({
                            "type":        "tool_result",
                            "tool_use_id": block.id,
                            "content":     content_text,
                        })

                    # append user turn with tool results
                    messages.append({"role": "user", "content": tool_results})

                return final_text
