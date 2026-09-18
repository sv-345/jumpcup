"""Jump Trading Probability Cup — autonomous floor bot.

A free-data, fault-tolerant daemon that submits honest de-vigged / calibrated
probability forecasts to SportsPredict (weighted Brier scoring). The Python
engine does all deterministic work (odds pull, de-vig, modelling, sanity gates,
state); the thin submission step is driven over MCP by the agent loop.
"""

__version__ = "1.0.0"
