"""Local SQLite Database Lookup Tool with parameterized queries."""
import sqlite3
from pathlib import Path
from typing import Any, Dict, List
from backend.tools.base import BaseTool
from utils.logger import logger


class DatabaseLookupTool(BaseTool):
    name = "database_lookup"
    description = "Queries a local SQLite database for customer, inventory, or order records using specific filters."

    def __init__(self, db_path: str = "./lookup.db"):
        self.db_path = Path(db_path).resolve()
        self._init_db()

    def _init_db(self):
        """Initializes tables and seed records if database does not exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS customers (
                    customer_id TEXT PRIMARY KEY,
                    name TEXT,
                    email TEXT,
                    tier TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inventory (
                    sku TEXT PRIMARY KEY,
                    product_name TEXT,
                    stock INTEGER,
                    unit_price REAL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    customer_id TEXT,
                    status TEXT,
                    total_amount REAL
                )
            """)
            # Insert seed data if empty
            cursor.execute("SELECT COUNT(*) FROM customers")
            if cursor.fetchone()[0] == 0:
                cursor.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", [
                    ("CUST-101", "Alex Smith", "alex@enterprise.com", "enterprise"),
                    ("CUST-102", "Jordan Lee", "jordan@startup.io", "pro")
                ])
                cursor.executemany("INSERT INTO inventory VALUES (?, ?, ?, ?)", [
                    ("SKU-4001", "Precision Sensor Module", 142, 89.50),
                    ("SKU-4002", "Edge Inference Gateway", 35, 499.00)
                ])
                cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", [
                    ("ORD-9821", "CUST-101", "shipped", 1890.00),
                    ("ORD-9822", "CUST-102", "processing", 499.00)
                ])
                conn.commit()

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "enum": ["customers", "inventory", "orders"],
                        "description": "Database table name to query."
                    },
                    "query_field": {"type": "string", "description": "Column name to filter against."},
                    "query_value": {"type": "string", "description": "Exact match filter value."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5}
                },
                "required": ["table", "query_field", "query_value"]
            }
        }

    def execute(self, table: str, query_field: str, query_value: str, limit: int = 5, **kwargs) -> Dict[str, Any]:
        # Whitelist defense against SQL injection
        allowed_tables = {
            "customers": ["customer_id", "name", "email", "tier"],
            "inventory": ["sku", "product_name", "stock", "unit_price"],
            "orders": ["order_id", "customer_id", "status", "total_amount"]
        }
        
        tbl = table.lower().strip()
        field = query_field.lower().strip()

        if tbl not in allowed_tables:
            raise ValueError(f"Access denied to table '{tbl}'")
        if field not in allowed_tables[tbl]:
            raise ValueError(f"Invalid query field '{field}' for table '{tbl}'")

        # Safe parameterized execution
        query = f"SELECT * FROM {tbl} WHERE {field} = ? LIMIT ?"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, (str(query_value), int(limit)))
            rows = [dict(row) for row in cursor.fetchall()]

        return {"table": tbl, "match_count": len(rows), "records": rows}

