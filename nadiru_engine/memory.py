"""
Memory Store — SQLite log of every interaction.
The Conductor's learning source. Nadis read from it via /query.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class MemoryStore:
    """SQLite-backed interaction log with computed user signals."""

    def __init__(self, db_path: str = "nadiru.db"):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS nadis (
                nadi_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                default_priority TEXT DEFAULT 'balanced',
                connected_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS interactions (
                request_id TEXT PRIMARY KEY,
                nadi_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                prompt TEXT NOT NULL,
                content TEXT NOT NULL,
                model TEXT NOT NULL,
                provider TEXT NOT NULL,
                tokens_in INTEGER DEFAULT 0,
                tokens_out INTEGER DEFAULT 0,
                cost_estimate REAL DEFAULT 0.0,
                latency_ms INTEGER DEFAULT 0,
                task_type TEXT DEFAULT 'unknown',
                complexity INTEGER DEFAULT 3,
                routing_reason TEXT DEFAULT '',
                outcome TEXT DEFAULT 'neutral',
                FOREIGN KEY (nadi_id) REFERENCES nadis(nadi_id)
            );

            CREATE INDEX IF NOT EXISTS idx_interactions_nadi
                ON interactions(nadi_id);
            CREATE INDEX IF NOT EXISTS idx_interactions_timestamp
                ON interactions(timestamp);
            CREATE INDEX IF NOT EXISTS idx_interactions_provider
                ON interactions(provider);
            CREATE INDEX IF NOT EXISTS idx_interactions_model
                ON interactions(model);

            CREATE TABLE IF NOT EXISTS user_signals (
                signal_key TEXT PRIMARY KEY,
                signal_value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        self._conn.commit()

    # --- Nadi Registration ---

    def register_nadi(self, name: str, description: str = "",
                      default_priority: str = "balanced") -> dict:
        nadi_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO nadis (nadi_id, name, description, default_priority, connected_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (nadi_id, name, description, default_priority, now)
        )
        self._conn.commit()
        return {"nadi_id": nadi_id, "connected_at": now}

    def get_nadi(self, nadi_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM nadis WHERE nadi_id = ?", (nadi_id,)
        ).fetchone()
        return dict(row) if row else None

    # --- Interaction Logging ---

    def log_interaction(self, nadi_id: str, prompt: str, content: str,
                        model: str, provider: str, tokens_in: int = 0,
                        tokens_out: int = 0, cost_estimate: float = 0.0,
                        latency_ms: int = 0, task_type: str = "unknown",
                        complexity: int = 3, routing_reason: str = "") -> str:
        request_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO interactions "
            "(request_id, nadi_id, timestamp, prompt, content, model, provider, "
            "tokens_in, tokens_out, cost_estimate, latency_ms, task_type, "
            "complexity, routing_reason, outcome) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (request_id, nadi_id, now, prompt, content, model, provider,
             tokens_in, tokens_out, cost_estimate, latency_ms, task_type,
             complexity, routing_reason, "neutral")
        )
        self._conn.commit()

        # Check for implicit feedback on PREVIOUS interaction
        self._evaluate_implicit_feedback(nadi_id, prompt, now)

        return request_id

    def _evaluate_implicit_feedback(self, nadi_id: str, current_prompt: str,
                                     current_time: str):
        """Check if this request implies feedback on the previous one."""
        prev = self._conn.execute(
            "SELECT request_id, prompt, timestamp FROM interactions "
            "WHERE nadi_id = ? AND timestamp < ? "
            "ORDER BY timestamp DESC LIMIT 1",
            (nadi_id, current_time)
        ).fetchone()

        if not prev:
            return

        prev_time = datetime.fromisoformat(prev["timestamp"])
        curr_time = datetime.fromisoformat(current_time)
        delta = (curr_time - prev_time).total_seconds()

        if delta > 300:
            return  # Too long ago, not feedback

        similarity = self._jaccard_similarity(current_prompt, prev["prompt"])

        if similarity > 0.8 and delta < 60:
            outcome = "rejected"
        elif similarity < 0.3:
            outcome = "accepted"
        else:
            outcome = "neutral"

        if outcome != "neutral":
            self._conn.execute(
                "UPDATE interactions SET outcome = ? WHERE request_id = ?",
                (outcome, prev["request_id"])
            )
            self._conn.commit()

    @staticmethod
    def _jaccard_similarity(text_a: str, text_b: str) -> float:
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    # --- Querying ---

    def query_interactions(self, nadi_id: Optional[str] = None,
                           since: Optional[str] = None,
                           until: Optional[str] = None,
                           model: Optional[str] = None,
                           provider: Optional[str] = None,
                           min_cost: Optional[float] = None,
                           max_cost: Optional[float] = None,
                           limit: int = 50,
                           offset: int = 0) -> dict:
        conditions = []
        params = []

        if nadi_id:
            conditions.append("nadi_id = ?")
            params.append(nadi_id)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        if until:
            conditions.append("timestamp <= ?")
            params.append(until)
        if model:
            conditions.append("model = ?")
            params.append(model)
        if provider:
            conditions.append("provider = ?")
            params.append(provider)
        if min_cost is not None:
            conditions.append("cost_estimate >= ?")
            params.append(min_cost)
        if max_cost is not None:
            conditions.append("cost_estimate <= ?")
            params.append(max_cost)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        # Get total count
        total = self._conn.execute(
            f"SELECT COUNT(*) FROM interactions {where}", params
        ).fetchone()[0]

        # Get page
        rows = self._conn.execute(
            f"SELECT * FROM interactions {where} "
            f"ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            params + [limit, offset]
        ).fetchall()

        return {
            "interactions": [dict(r) for r in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    # --- User Signals (Computed Aggregates) ---

    def compute_user_signals(self) -> dict:
        """Compute aggregate signals from interaction history."""
        now = datetime.now(timezone.utc).isoformat()

        total = self._conn.execute(
            "SELECT COUNT(*) FROM interactions"
        ).fetchone()[0]

        # Success rates by task type
        success_rates = {}
        rows = self._conn.execute(
            "SELECT task_type, outcome, COUNT(*) as cnt "
            "FROM interactions "
            "WHERE outcome IN ('accepted', 'rejected') "
            "GROUP BY task_type, outcome"
        ).fetchall()

        type_counts = {}
        for row in rows:
            t = row["task_type"]
            if t not in type_counts:
                type_counts[t] = {"accepted": 0, "rejected": 0}
            type_counts[t][row["outcome"]] = row["cnt"]

        for t, counts in type_counts.items():
            total_rated = counts["accepted"] + counts["rejected"]
            if total_rated > 0:
                success_rates[t] = round(counts["accepted"] / total_rated, 2)

        # Local success rates specifically
        local_success = {}
        rows = self._conn.execute(
            "SELECT task_type, outcome, COUNT(*) as cnt "
            "FROM interactions "
            "WHERE provider = 'ollama' AND outcome IN ('accepted', 'rejected') "
            "GROUP BY task_type, outcome"
        ).fetchall()

        local_counts = {}
        for row in rows:
            t = row["task_type"]
            if t not in local_counts:
                local_counts[t] = {"accepted": 0, "rejected": 0}
            local_counts[t][row["outcome"]] = row["cnt"]

        for t, counts in local_counts.items():
            total_rated = counts["accepted"] + counts["rejected"]
            if total_rated > 0:
                local_success[t] = round(counts["accepted"] / total_rated, 2)

        # Average cost by task type
        avg_costs = {}
        rows = self._conn.execute(
            "SELECT task_type, AVG(cost_estimate) as avg_cost "
            "FROM interactions WHERE cost_estimate > 0 "
            "GROUP BY task_type"
        ).fetchall()
        for row in rows:
            avg_costs[row["task_type"]] = round(row["avg_cost"], 6)

        # Preferred models by task type (most accepted)
        preferred = {}
        rows = self._conn.execute(
            "SELECT task_type, model, COUNT(*) as cnt "
            "FROM interactions WHERE outcome = 'accepted' "
            "GROUP BY task_type, model "
            "ORDER BY cnt DESC"
        ).fetchall()
        seen_types = set()
        for row in rows:
            if row["task_type"] not in seen_types:
                preferred[row["task_type"]] = row["model"]
                seen_types.add(row["task_type"])

        # Recent error providers (last 24h) — providers that returned errors
        error_providers = []
        rows = self._conn.execute(
            "SELECT DISTINCT provider FROM interactions "
            "WHERE content LIKE 'ERROR:%' "
            "AND timestamp >= datetime('now', '-1 day')"
        ).fetchall()
        error_providers = [row["provider"] for row in rows]

        signals = {
            "total_interactions": total,
            "local_success_rate": local_success,
            "overall_success_rate": success_rates,
            "avg_cost_per_type": avg_costs,
            "preferred_models": preferred,
            "recent_error_providers": error_providers,
        }

        # Cache in user_signals table
        self._conn.execute(
            "INSERT OR REPLACE INTO user_signals (signal_key, signal_value, updated_at) "
            "VALUES (?, ?, ?)",
            ("latest", json.dumps(signals), now)
        )
        self._conn.commit()

        return signals

    def get_cached_signals(self) -> Optional[dict]:
        """Get cached signals, returns None if never computed."""
        row = self._conn.execute(
            "SELECT signal_value FROM user_signals WHERE signal_key = 'latest'"
        ).fetchone()
        if row:
            return json.loads(row["signal_value"])
        return None

    def get_interaction_count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM interactions"
        ).fetchone()[0]

    def close(self):
        self._conn.close()
