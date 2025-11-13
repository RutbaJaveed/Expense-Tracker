import sqlite3
import datetime as dt
import calendar
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd
import matplotlib.pyplot as plt


DB_PATH = "expense_tracker.db"


def now_iso():
    return dt.datetime.now().isoformat(timespec="seconds")


def current_period_ym() -> str:
    return dt.datetime.now().strftime("%Y-%m")


def period_bounds(period_ym: str):
    year, month = map(int, period_ym.split("-"))
    start = dt.datetime(year, month, 1)
    if month == 12:
        end = dt.datetime(year + 1, 1, 1)
    else:
        end = dt.datetime(year, month + 1, 1)
    return start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")


def is_valid_ym(s: str) -> bool:
    try:
        year, month = map(int, s.split("-"))
        return 1 <= month <= 12 and 2000 <= year <= 2100
    except Exception:
        return False


class ExpenseTrackerPro:
    def __init__(self, db_path: str = DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._create_tables()

    def _create_tables(self):
        cur = self.conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                income_monthly REAL NOT NULL DEFAULT 0,
                savings_goal_monthly REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                UNIQUE(user_id, name),
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                category_id INTEGER NOT NULL,
                amount REAL NOT NULL CHECK(amount >= 0),
                currency TEXT NOT NULL DEFAULT 'INR',
                tx_time TEXT NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY(category_id) REFERENCES categories(id) ON DELETE RESTRICT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS budgets_monthly (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                category_id INTEGER NOT NULL,
                period_ym TEXT NOT NULL,
                limit_amount REAL NOT NULL CHECK(limit_amount >= 0),
                UNIQUE(user_id, category_id, period_ym),
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY(category_id) REFERENCES categories(id) ON DELETE CASCADE
            )
        """)

        cur.execute("CREATE INDEX IF NOT EXISTS ix_expenses_user_time ON expenses(user_id, tx_time)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_expenses_user_cat_time ON expenses(user_id, category_id, tx_time)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_budgets_user_period ON budgets_monthly(user_id, period_ym)")

        self.conn.commit()

    # ------------- USERS -------------
    def register_user(self, name: str, income: float, savings: float):
        name = name.strip()
        if not name:
            print("❌ Name cannot be empty.")
            return
        if any(x < 0 for x in (income, savings)):
            print("❌ Income and savings goal must be non-negative.")
            return
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO users (name, income_monthly, savings_goal_monthly, created_at) VALUES (?, ?, ?, ?)",
                    (name, income, savings, now_iso())
                )
            print("✅ User registered successfully.")
        except sqlite3.IntegrityError:
            print("⚠️ User already exists.")

    def get_user_id(self, name: str) -> Optional[int]:
        row = self.conn.execute("SELECT id FROM users WHERE name=?", (name,)).fetchone()
        return int(row["id"]) if row else None

    def get_user(self, user_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()

    # ------------- CATEGORIES -------------
    def _norm(self, s: Optional[str]) -> Optional[str]:
        return s.strip().lower() if isinstance(s, str) else None

    def ensure_category(self, user_id: int, name: str) -> int:
        name = self._norm(name)
        if not name:
            name = "uncategorized"
        row = self.conn.execute(
            "SELECT id FROM categories WHERE user_id=? AND name=?", (user_id, name)
        ).fetchone()
        if row:
            return int(row["id"])
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO categories (user_id, name) VALUES (?, ?)", (user_id, name)
            )
        return int(cur.lastrowid)

    # ------------- EXPENSES -------------
    def add_expense(self, user_id: int, category: str, amount: float,
                    tx_time: Optional[str] = None, notes: Optional[str] = None):
        if amount < 0:
            print("❌ Amount must be non-negative.")
            return
        cat_id = self.ensure_category(user_id, category)
        tx_time = tx_time or now_iso()
        with self.conn:
            self.conn.execute("""
                INSERT INTO expenses (user_id, category_id, amount, currency, tx_time, notes, created_at)
                VALUES (?, ?, ?, 'INR', ?, ?, ?)
            """, (user_id, cat_id, amount, tx_time, notes, now_iso()))
        print("✅ Expense added.")
        self._check_budget_after_transaction(user_id, cat_id, tx_time)

    def edit_expense(self, expense_id: int, category: Optional[str], amount: Optional[float],
                     tx_time: Optional[str] = None, notes: Optional[str] = None):
        row = self.conn.execute(
            "SELECT user_id, category_id, tx_time FROM expenses WHERE id=?", (expense_id,)
        ).fetchone()
        if not row:
            print("❌ Expense not found.")
            return

        user_id = int(row["user_id"])
        old_cat_id = int(row["category_id"])
        old_time = row["tx_time"]

        updates, params = [], []
        if category is not None:
            new_cat_id = self.ensure_category(user_id, category)
            updates.append("category_id=?")
            params.append(new_cat_id)
        if amount is not None:
            if amount < 0:
                print("❌ Amount must be non-negative.")
                return
            updates.append("amount=?")
            params.append(amount)
        if tx_time is not None:
            updates.append("tx_time=?")
            params.append(tx_time)
        if notes is not None:
            updates.append("notes=?")
            params.append(notes)

        if not updates:
            print("ℹ️ Nothing to update.")
            return

        params.append(expense_id)
        with self.conn:
            self.conn.execute(f"UPDATE expenses SET {', '.join(updates)} WHERE id=?", params)
        print("✏️ Expense updated.")

        new_cat_id = self.conn.execute(
            "SELECT category_id, tx_time FROM expenses WHERE id=?", (expense_id,)
        ).fetchone()
        self._check_budget_after_transaction(user_id, old_cat_id, old_time)
        self._check_budget_after_transaction(user_id, int(new_cat_id["category_id"]), new_cat_id["tx_time"])

    def delete_expense(self, expense_id: int):
        with self.conn:
            cur = self.conn.execute("DELETE FROM expenses WHERE id=?", (expense_id,))
        if cur.rowcount:
            print("🗑️ Expense deleted.")
        else:
            print("❌ Expense not found.")

    def list_expenses(self, user_id: int, limit: int = 20):
        rows = self.conn.execute("""
            SELECT e.id, e.amount, e.tx_time, e.notes, c.name AS category
            FROM expenses e
            JOIN categories c ON c.id = e.category_id
            WHERE e.user_id=?
            ORDER BY e.tx_time DESC
            LIMIT ?
        """, (user_id, limit)).fetchall()
        if not rows:
            print("ℹ️ No expenses found.")
            return
        print("\n🧾 Recent expenses:")
        for r in rows:
            print(f"#{r['id']} | {r['tx_time']} | ₹{r['amount']:.2f} | {r['category']} | {r['notes'] or ''}")

    # ------------- BUDGETS -------------
    def set_monthly_budget(self, user_id: int, category: str, limit_amount: float, period_ym: Optional[str] = None):
        if limit_amount < 0:
            print("❌ Budget must be non-negative.")
            return
        period_ym = period_ym or current_period_ym()
        if not is_valid_ym(period_ym):
            print("❌ Invalid period. Use YYYY-MM.")
            return
        cat_id = self.ensure_category(user_id, category)
        with self.conn:
            self.conn.execute("""
                INSERT INTO budgets_monthly (user_id, category_id, period_ym, limit_amount)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, category_id, period_ym)
                DO UPDATE SET limit_amount=excluded.limit_amount
            """, (user_id, cat_id, period_ym, limit_amount))
        print(f"💰 Budget for '{category}' in {period_ym} set to ₹{limit_amount:.2f}")
        self.check_budget_for_period(user_id, period_ym)

    def delete_monthly_budget(self, user_id: int, category: str, period_ym: Optional[str] = None):
        period_ym = period_ym or current_period_ym()
        cat_id = self.ensure_category(user_id, category)
        with self.conn:
            cur = self.conn.execute(
                "DELETE FROM budgets_monthly WHERE user_id=? AND category_id=? AND period_ym=?",
                (user_id, cat_id, period_ym)
            )
        if cur.rowcount:
            print(f"🗑️ Budget for '{category}' in {period_ym} deleted.")
        else:
            print("❌ No budget found for that category/month.")

    def _check_budget_after_transaction(self, user_id: int, category_id: int, tx_time: str):
        period_ym = tx_time[:7]
        start, end = period_bounds(period_ym)
        limit_row = self.conn.execute(
            "SELECT limit_amount FROM budgets_monthly WHERE user_id=? AND category_id=? AND period_ym=?",
            (user_id, category_id, period_ym)
        ).fetchone()
        if not limit_row:
            return
        limit_amt = float(limit_row["limit_amount"])
        spent = self.conn.execute("""
            SELECT COALESCE(SUM(amount), 0) AS spent
            FROM expenses
            WHERE user_id=? AND category_id=? AND tx_time >= ? AND tx_time < ?
        """, (user_id, category_id, start, end)).fetchone()["spent"] or 0.0
        if spent > limit_amt:
            cat_name = self.conn.execute("SELECT name FROM categories WHERE id=?", (category_id,)).fetchone()["name"]
            print(f"⚠️ Budget exceeded for '{cat_name}' in {period_ym} → Spent ₹{spent:.2f} / Limit ₹{limit_amt:.2f}")

    def check_budget_for_period(self, user_id: int, period_ym: Optional[str] = None):
        period_ym = period_ym or current_period_ym()
        start, end = period_bounds(period_ym)
        rows = self.conn.execute("""
            SELECT c.name AS category, b.limit_amount,
                   COALESCE((
                       SELECT SUM(e.amount) FROM expenses e
                       WHERE e.user_id=b.user_id AND e.category_id=b.category_id
                         AND e.tx_time >= ? AND e.tx_time < ?
                   ), 0) AS spent
            FROM budgets_monthly b
            JOIN categories c ON c.id = b.category_id
            WHERE b.user_id=? AND b.period_ym=?
            ORDER BY c.name ASC
        """, (start, end, user_id, period_ym)).fetchall()
        if not rows:
            print("ℹ️ No budgets set for this month.")
            return
        print(f"\n📊 Budget summary ({period_ym}):")
        for r in rows:
            remaining = r["limit_amount"] - (r["spent"] or 0)
            status = "✅" if remaining >= 0 else "⚠️"
            print(f"{r['category']:<15} → Spent ₹{(r['spent'] or 0):.2f} / Limit ₹{r['limit_amount']:.2f} / Remaining ₹{remaining:.2f} {status}")

    # ------------- SAVINGS -------------
    def savings_progress(self, user_id: int, period_ym: Optional[str] = None):
        period_ym = period_ym or current_period_ym()
        user = self.get_user(user_id)
        if not user:
            print("❌ User not found.")
            return
        income = float(user["income_monthly"])
        goal = float(user["savings_goal_monthly"])
        start, end = period_bounds(period_ym)
        spent = self.conn.execute("""
            SELECT COALESCE(SUM(amount), 0) AS spent
            FROM expenses
            WHERE user_id=? AND tx_time >= ? AND tx_time < ?
        """, (user_id, start, end)).fetchone()["spent"] or 0.0
        remaining = income - spent
        print(f"\n💰 Savings Progress ({period_ym}):")
        print(f"Income: ₹{income:.2f} | Spent: ₹{spent:.2f} | Remaining: ₹{remaining:.2f}")
        print(f"Savings Goal: ₹{goal:.2f}")
        if remaining > goal:
            print(f"✅ You’re above your goal by ₹{(remaining - goal):.2f}. Nice! 🎉")
        elif abs(remaining - goal) < 1e-9:
            print("✅ You met your goal exactly! 👏")
        else:
            print(f"⚠️ You’re ₹{(goal - remaining):.2f} below your goal. Consider trimming expenses.")

    # ------------- ANALYTICS -------------
    def expenses_df(self, user_id: int, start_iso: Optional[str] = None, end_iso: Optional[str] = None) -> pd.DataFrame:
        q = """
            SELECT e.tx_time, e.amount, c.name AS category, e.notes
            FROM expenses e
            JOIN categories c ON c.id = e.category_id
            WHERE e.user_id=?
        """
        params = [user_id]
        if start_iso:
            q += " AND e.tx_time >= ?"
            params.append(start_iso)
        if end_iso:
            q += " AND e.tx_time < ?"
            params.append(end_iso)
        q += " ORDER BY e.tx_time ASC"
        
        df = pd.read_sql_query(q, self.conn, params=params, parse_dates=["tx_time"])
        return df
    
    def monthly_report(self, user_id: int, period_ym: Optional[str] = None) -> Dict[str, float]:
        period_ym = period_ym or current_period_ym()
        start, end = period_bounds(period_ym)
        rows = self.conn.execute("""
            SELECT c.name AS category, SUM(e.amount) AS total
            FROM expenses e
            JOIN categories c ON c.id = e.category_id
            WHERE e.user_id=? AND e.tx_time >= ? AND e.tx_time < ?
            GROUP BY c.name
            ORDER BY total DESC
        """, (user_id, start, end)).fetchall()
        data = {r["category"]: (r["total"] or 0.0) for r in rows}
        if not data:
            print("ℹ️ No expenses for this period.")
        return data

    def plot_pie(self, data: Dict[str, float], title: str):
        total = sum(data.values()) if data else 0
        if total <= 0:
            print("ℹ️ Nothing to plot.")
            return
        plt.figure(figsize=(6, 6))
        plt.pie(list(data.values()), labels=list(data.keys()), autopct="%1.1f%%", startangle=140)
        plt.title(title)
        plt.axis("equal")
        plt.tight_layout()
        plt.show()
    
    def plot_monthly_trend(self, user_id: int, months: int = 12):
        df = pd.read_sql_query("""
            SELECT substr(tx_time, 1, 7) AS ym, SUM(amount) AS total
            FROM expenses
            WHERE user_id=?
            GROUP BY ym
            ORDER BY ym
        """, self.conn, params=[user_id])
        if df.empty:
            print("ℹ️ No data to plot.")
            return
        df["total"] = df["total"].fillna(0.0)
        df = df.tail(months)
        plt.figure(figsize=(8, 4))
        plt.plot(df["ym"], df["total"], marker="o")
        plt.title(f"Monthly Spend Trend (last {len(df)} months)")
        plt.xlabel("Month")
        plt.ylabel("Total Spend (₹)")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def detect_anomalies(self, user_id: int, z: float = 2.0):
        df = pd.read_sql_query("""
            SELECT substr(e.tx_time, 1, 7) AS ym, c.name AS category, SUM(e.amount) AS spent
            FROM expenses e
            JOIN categories c ON c.id = e.category_id
            WHERE e.user_id=?
            GROUP BY ym, c.name
            ORDER BY ym
        """, self.conn, params=[user_id])
        if df.empty:
            print("ℹ️ Not enough data.")
            return
        pt = df.pivot_table(index="ym", columns="category", values="spent", aggfunc="sum").fillna(0.0)
        if len(pt.index) < 3:
            print("ℹ️ Need at least 3 months of data for anomaly detection.")
            return

        now = dt.datetime.now()
        last_complete = (now.replace(day=1) - dt.timedelta(days=1)).strftime("%Y-%m")
        if last_complete not in pt.index:
            last_complete = pt.index[-1]

        s = pt.loc[last_complete]
        means = pt.drop(index=[last_complete], errors="ignore").mean()
        stds = pt.drop(index=[last_complete], errors="ignore").std(ddof=0).replace(0, pd.NA)
        zscores = (s - means) / stds
        anomalies = zscores[zscores.abs() >= z].dropna()

        if anomalies.empty:
            print(f"✅ No category anomalies in {last_complete} (|z| < {z}).")
            return

        print(f"🚨 Anomalies in {last_complete} (|z| ≥ {z}):")
        for cat, zval in anomalies.sort_values(key=lambda x: -x.abs()).items():
            print(f"- {cat}: z={zval:.2f} (spent ₹{s[cat]:.2f}, mean ₹{means[cat]:.2f})")

    def forecast_month_end(self, user_id: int, period_ym: Optional[str] = None):
        period_ym = period_ym or current_period_ym()
        start_dt = dt.datetime.strptime(period_ym + "-01", "%Y-%m-%d")
        days_in_month = calendar.monthrange(start_dt.year, start_dt.month)[1]
        start_iso, end_iso = period_bounds(period_ym)
        today = dt.datetime.now()
        # if forecasting a past month, assume it's the end of that month
        day_of_month = min(today.day, days_in_month) if today.strftime("%Y-%m") == period_ym else days_in_month

        spent_mtd = self.conn.execute("""
            SELECT COALESCE(SUM(amount), 0) AS spent
            FROM expenses
            WHERE user_id=? AND tx_time >= ? AND tx_time < ?
        """, (user_id, start_iso, min(end_iso, now_iso()))).fetchone()["spent"] or 0.0

        avg_per_day = spent_mtd / max(day_of_month, 1)
        forecast_total = avg_per_day * days_in_month

        # Sum of budgets for this month (if any)
        budgets_row = self.conn.execute("""
            SELECT COALESCE(SUM(limit_amount), 0) AS total_budget
            FROM budgets_monthly
            WHERE user_id=? AND period_ym=?
        """, (user_id, period_ym)).fetchone()
        total_budget = float(budgets_row["total_budget"] or 0.0)

        user = self.get_user(user_id)
        income = float(user["income_monthly"]) if user else 0.0
        goal = float(user["savings_goal_monthly"]) if user else 0.0

        print(f"\n📈 Forecast for {period_ym}:")
        print(f"MTD spent: ₹{spent_mtd:.2f} over {day_of_month} day(s) → Avg/day ₹{avg_per_day:.2f}")
        print(f"Forecast month-end spend: ₹{forecast_total:.2f}")
        if total_budget > 0:
            delta = total_budget - forecast_total
            status = "✅ under" if delta >= 0 else "⚠️ over"
            print(f"Budgets total: ₹{total_budget:.2f} → {status} by ₹{abs(delta):.2f}")
        if income > 0:
            remaining = income - forecast_total
            goal_gap = remaining - goal
            print(f"Income: ₹{income:.2f} → Remaining after forecast: ₹{remaining:.2f}")
            if goal > 0:
                if goal_gap >= 0:
                    print(f"✅ On track to meet savings goal by ₹{goal_gap:.2f}")
                else:
                    print(f"⚠️ Off track by ₹{abs(goal_gap):.2f}")

    def import_csv(self, user_id: int, filepath: str):
        p = Path(filepath)
        if not p.exists():
            print("❌ File not found.")
            return

        df = pd.read_csv(p)
        cols = {c.lower().strip(): c for c in df.columns}

        def find_col(cands: List[str]) -> Optional[str]:
            for c in cands:
                for k, orig in cols.items():
                    if k == c or c in k:
                        return orig
            return None

        date_col = find_col(["date", "transaction_date", "posted_date", "time"])
        amount_col = find_col(["amount", "debit"])
        category_col = find_col(["category", "cat"])
        notes_col = find_col(["description", "memo", "note"])

        if not date_col or not amount_col:
            print("❌ CSV must have at least date and amount columns.")
            return

        df["__date"] = pd.to_datetime(df[date_col], errors="coerce")
        df["__amount"] = pd.to_numeric(df[amount_col], errors="coerce")
        df = df.dropna(subset=["__date", "__amount"])
        df = df[df["__amount"] > 0]

        df["__category"] = (df[category_col].astype(str) if category_col else "uncategorized")
        df["__notes"] = (df[notes_col].astype(str) if notes_col else None)

        cat_cache: Dict[str, int] = {}
        to_insert = []
        created_at = now_iso()
        for _, row in df.iterrows():
            cat_name = str(row["__category"]).strip().lower() if row["__category"] is not None else "uncategorized"
            if cat_name not in cat_cache:
                cat_cache[cat_name] = self.ensure_category(user_id, cat_name)
            cat_id = cat_cache[cat_name]

            tx_time = row["__date"].to_pydatetime().isoformat(timespec="seconds")
            amount = float(row["__amount"])
            notes = None if pd.isna(row["__notes"]) else str(row["__notes"])

            to_insert.append((user_id, cat_id, amount, "INR", tx_time, notes, created_at))

        if not to_insert:
            print("ℹ️ Nothing to import.")
            return

        with self.conn:
            self.conn.executemany("""
                INSERT INTO expenses (user_id, category_id, amount, currency, tx_time, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, to_insert)
        print(f"✅ Imported {len(to_insert)} transactions.")
        self.check_budget_for_period(user_id, current_period_ym())

    # ------------- EXPORT -------------
    def export_expenses_csv(self, user_id: int, filepath: str):
        df = self.expenses_df(user_id)
        if df.empty:
            print("ℹ️ No expenses to export.")
            return
        try:
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"📤 Exported {len(df)} rows to '{filepath}' successfully.")
        except Exception as e:
            print(f"❌ Failed to export CSV: {e}")

    def close(self):
        self.conn.close()
        print("🔒 Database connection closed.")


# --------------------- CLI HELPERS ---------------------
def get_float_input(prompt: str, allow_zero=True) -> float:
    while True:
        s = input(prompt).strip()
        try:
            v = float(s)
            if v < 0 or (not allow_zero and v == 0):
                print("❌ Enter a positive number.")
                continue
            return v
        except ValueError:
            print("❌ Invalid number. Try again.")


def get_valid_user(tracker: ExpenseTrackerPro):
    name = input("Enter your name: ").strip()
    user_id = tracker.get_user_id(name)
    if not user_id:
        print("❌ User not found.")
        return None, None
    return name, user_id


def pick_expense_interactive(tracker: ExpenseTrackerPro, user_id: int, period_ym: Optional[str] = None, limit: int = 20) -> Optional[int]:
    params = [user_id]
    date_filter = ""
    if period_ym:
        if not is_valid_ym(period_ym):
            print("❌ Invalid period. Use YYYY-MM.")
            return None
        start, end = period_bounds(period_ym)
        date_filter = " AND e.tx_time >= ? AND e.tx_time < ?"
        params.extend([start, end])
    params.append(limit)

    rows = tracker.conn.execute(f"""
        SELECT e.id, e.tx_time, e.amount,
               c.name AS category,
               COALESCE(e.notes, '') AS notes
        FROM expenses e
        JOIN categories c ON c.id = e.category_id
        WHERE e.user_id=? {date_filter}
        ORDER BY e.tx_time DESC
        LIMIT ?
    """, params).fetchall()

    if not rows:
        print("ℹ️ No expenses found. Add some first or import CSV.")
        return None

    print("\n🧾 Pick an expense:")
    for i, r in enumerate(rows, 1):
        print(f"{i:>2}. #{r['id']} | {r['tx_time']} | ₹{r['amount']:.2f} | {r['category']} | {r['notes'] or ''}")

    while True:
        sel = input("Enter number (or 'q' to cancel): ").strip().lower()
        if sel == 'q':
            return None
        if sel.isdigit():
            idx = int(sel)
            if 1 <= idx <= len(rows):
                return int(rows[idx - 1]["id"])
        print("❌ Invalid choice. Try again.")


# --------------------- MAIN MENU ---------------------
def main():
    tracker = ExpenseTrackerPro(DB_PATH)

    while True:
        print("\n=== 💼 Expense Tracker ===")
        print("1. Register User")
        print("2. Add Expense")
        print("3. Edit Expense")
        print("4. Delete Expense")
        print("5. View/Filter Expenses")
        print("6. Set Monthly Budget")
        print("7. View Budget Summary (This/Chosen Month)")
        print("8. View Savings Progress (This/Chosen Month)")
        print("9. Monthly Report + Pie (This/Chosen Month)")
        print("10. Import CSV")
        print("11. Plot Monthly Trend (12m)")
        print("12. Detect Anomalies")
        print("13. Forecast Month-End")
        print("14. Export Expenses CSV")
        print("15. Exit")

        choice = input("Enter your choice: ").strip()

        if choice == '1':
            name = input("Enter your name: ").strip()
            income = get_float_input("Monthly income (₹): ")
            savings = get_float_input("Monthly savings goal (₹): ")
            tracker.register_user(name, income, savings)

        elif choice == '2':
            name, user_id = get_valid_user(tracker)
            if not user_id:
                continue
            category = input("Category: ").strip()
            amount = get_float_input("Amount (₹): ")
            time_in = input("Time (YYYY-MM-DD HH:MM:SS) or blank for now: ").strip()
            tx_time = time_in if time_in else None
            notes = input("Notes (optional): ").strip() or None
            tracker.add_expense(user_id, category, amount, tx_time, notes)

        elif choice == '3':  # Edit Expense
            name, user_id = get_valid_user(tracker)
            if not user_id:
                continue
            filt = input("Filter picker by month (YYYY-MM) (optional): ").strip() or None
            exp_id = pick_expense_interactive(tracker, user_id, filt, limit=20)
            if not exp_id:
                continue

            category = input("New Category (blank to skip): ").strip()
            amount_str = input("New Amount (blank to skip): ").strip()
            time_in = input("New Time (YYYY-MM-DD HH:MM:SS) (blank to skip): ").strip()
            notes = input("New Notes (blank to skip): ").strip()

            amount = None
            if amount_str:
                try:
                    amount = float(amount_str)
                    if amount < 0:
                        print("❌ Amount must be non-negative.")
                        continue
                except ValueError:
                    print("❌ Invalid amount.")
                    continue

            tracker.edit_expense(
                exp_id,
                category if category else None,
                amount,
                time_in if time_in else None,
                notes if notes else None
            )

        elif choice == '4':  # Delete Expense
            name, user_id = get_valid_user(tracker)
            if not user_id:
                continue
            filt = input("Filter picker by month (YYYY-MM) (optional): ").strip() or None
            exp_id = pick_expense_interactive(tracker, user_id, filt, limit=20)
            if not exp_id:
                continue
            confirm = input(f"Delete expense #{exp_id}? (y/N): ").strip().lower()
            if confirm == 'y':
                tracker.delete_expense(exp_id)
            else:
                print("❌ Cancelled.")

        elif choice == '5':  # View/Filter Expenses
            name, user_id = get_valid_user(tracker)
            if not user_id:
                continue
            ym = input("Filter by month (YYYY-MM) (blank for all): ").strip() or None
            cat = input("Filter by category (blank for all): ").strip()
            search = input("Search in notes (blank for none): ").strip()
            try:
                limit = int(input("How many rows? (default 50): ").strip() or "50")
            except ValueError:
                limit = 50

            params = [user_id]
            where = "WHERE e.user_id=?"
            if ym:
                if not is_valid_ym(ym):
                    print("❌ Invalid period. Use YYYY-MM.")
                    continue
                start, end = period_bounds(ym)
                where += " AND e.tx_time >= ? AND e.tx_time < ?"
                params.extend([start, end])
            if cat:
                where += " AND LOWER(c.name)=LOWER(?)"
                params.append(cat)
            if search:
                where += " AND LOWER(COALESCE(e.notes,'')) LIKE LOWER(?)"
                params.append(f"%{search}%")

            params.append(limit)
            rows = tracker.conn.execute(f"""
                SELECT e.id, e.tx_time, e.amount, c.name AS category, COALESCE(e.notes,'') AS notes
                FROM expenses e
                JOIN categories c ON c.id = e.category_id
                {where}
                ORDER BY e.tx_time DESC
                LIMIT ?
            """, params).fetchall()

            if not rows:
                print("ℹ️ No expenses found for the chosen filters.")
            else:
                print("\n🧾 Expenses:")
                for r in rows:
                    print(f"#{r['id']} | {r['tx_time']} | ₹{r['amount']:.2f} | {r['category']} | {r['notes'] or ''}")

        elif choice == '6':
            name, user_id = get_valid_user(tracker)
            if not user_id:
                continue
            cat = input("Category: ").strip()
            limit_amt = get_float_input("Monthly budget (₹): ")
            ym = input("Period (YYYY-MM) (blank for this month): ").strip()
            tracker.set_monthly_budget(user_id, cat, limit_amt, ym or None)

        elif choice == '7':
            name, user_id = get_valid_user(tracker)
            if not user_id:
                continue
            ym = input("Period (YYYY-MM) (blank for this month): ").strip()
            tracker.check_budget_for_period(user_id, ym or None)

        elif choice == '8':
            name, user_id = get_valid_user(tracker)
            if not user_id:
                continue
            ym = input("Period (YYYY-MM) (blank for this month): ").strip()
            tracker.savings_progress(user_id, ym or None)

        elif choice == '9':
            name, user_id = get_valid_user(tracker)
            if not user_id:
                continue
            ym = input("Period (YYYY-MM) (blank for this month): ").strip()
            data = tracker.monthly_report(user_id, ym or None)
            if data:
                print("\n📊 Expense Report:")
                for k, v in data.items():
                    print(f"{k.capitalize()}: ₹{v:.2f}")
                tracker.plot_pie(data, f"Expense Distribution - {name} ({(ym or current_period_ym())})")

        elif choice == '10':
            name, user_id = get_valid_user(tracker)
            if not user_id:
                continue
            path = input("Path to CSV: ").strip()
            tracker.import_csv(user_id, path)

        elif choice == '11':
            name, user_id = get_valid_user(tracker)
            if not user_id:
                continue
            tracker.plot_monthly_trend(user_id, months=12)

        elif choice == '12':
            name, user_id = get_valid_user(tracker)
            if not user_id:
                continue
            try:
                z = float(input("Anomaly z-threshold (default 2.0): ").strip() or "2.0")
            except ValueError:
                z = 2.0
            tracker.detect_anomalies(user_id, z=z)

        elif choice == '13':
            name, user_id = get_valid_user(tracker)
            if not user_id:
                continue
            ym = input("Forecast period (YYYY-MM) (blank for this month): ").strip()
            tracker.forecast_month_end(user_id, ym or None)

        elif choice == '14':
            name, user_id = get_valid_user(tracker)
            if not user_id:
                continue
            path = input("Output CSV path (e.g., expenses.csv): ").strip()
            tracker.export_expenses_csv(user_id, path)

        elif choice == '15':
            tracker.close()
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice. Try again.")


if __name__ == "__main__":
    main()
