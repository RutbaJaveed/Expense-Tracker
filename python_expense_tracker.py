import sqlite3
import matplotlib.pyplot as plt
import datetime

class ExpenseTracker:
    def __init__(self):
        self.conn = sqlite3.connect('py_expense_tracker.db',detect_types=sqlite3.PARSE_DECLTYPES)
        self.cur = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        self.cur.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY,
                            name TEXT UNIQUE,
                            income REAL,
                            savings REAL)''')
        self.cur.execute('''CREATE TABLE IF NOT EXISTS expenses (
                            id INTEGER PRIMARY KEY,
                            user_id INTEGER,
                            category TEXT,
                            amount REAL,
                            time TIMESTAMP,
                            FOREIGN KEY(user_id) REFERENCES users(id))''')
        self.cur.execute('''CREATE TABLE IF NOT EXISTS budget_limits (
                            id INTEGER PRIMARY KEY,
                            user_id INTEGER,
                            category TEXT,
                            bug_limit REAL,
                            FOREIGN KEY(user_id) REFERENCES users(id))''')
        self.conn.commit()

    def register_user(self, name, income, savings):
        try:
            self.cur.execute("INSERT INTO users (name, income, savings) VALUES (?, ?, ?)", (name, income, savings))
            self.conn.commit()
            print("User registered successfully.")
        except sqlite3.IntegrityError:
            print("User already exists.")

    def add_expense(self, user_id, category, amount):
        current_time = datetime.datetime.now().isoformat()
        self.cur.execute("INSERT INTO expenses (user_id, category, amount, time) VALUES (?, ?, ?, ?)", (user_id, category, amount, current_time))
        self.conn.commit()
        self.check_budget_limit(user_id, category, amount)

    def delete_expense(self, expense_id):
        self.cur.execute("DELETE FROM expenses WHERE id=?", (expense_id,))
        self.conn.commit()

    def edit_expense(self, expense_id, category, amount):
        current_time = datetime.datetime.now().isoformat()
        self.cur.execute("UPDATE expenses SET category=?, amount=?, time=? WHERE id=?", (category, amount, current_time, expense_id))
        self.conn.commit()

    def generate_pie_chart(self, data, title):
        # Extract labels (categories) and values (amounts) from the data dictionary
        labels = data.keys()
        values = data.values()
        # Plot the pie chart using Matplotlib
        plt.pie(values, labels=labels, autopct='%1.1f%%') # Create the pie chart with percentages as labels
        plt.title(title)
        plt.legend(title="Categories:",loc="upper right")# Add a legend with a title to the upper right corner
        plt.show()

    def get_user_id(self, name):
        self.cur.execute("SELECT id FROM users WHERE name=?", (name,))
        user = self.cur.fetchone()
        return user[0] if user else None

    def get_expenses_by_user(self, user_id):
        self.cur.execute("SELECT id, category, amount FROM expenses WHERE user_id=?", (user_id,))
        return self.cur.fetchall()

    def get_total_income_and_savings(self, user_id):
        self.cur.execute("SELECT income, savings FROM users WHERE id=?", (user_id,))
        return self.cur.fetchall()

    def set_budget_limit(self, user_id, category, bug_limit):
        try:
            self.cur.execute("INSERT OR REPLACE INTO budget_limits (user_id, category, bug_limit) VALUES (?, ?, ?)",
                             (user_id, category, bug_limit))
            self.conn.commit()
            print("Budget limit set successfully.")
        except sqlite3.Error as e:
            print("An error occurred:", e)

    def check_budget_limit(self, user_id, category, amount):
        self.cur.execute("SELECT bug_limit FROM budget_limits WHERE user_id=? AND category=?", (user_id, category))
        bug_limit = self.cur.fetchone()
        if bug_limit and amount > bug_limit[0]:
            print("Warning: Expense exceeds the budget limit!")

    def generate_report(self, user_id, period='weekly'):
        # if period == 'weekly':
        #     start_date = datetime.datetime.now() - datetime.timedelta(days=7)
        # elif period == 'monthly':
        #     start_date = datetime.datetime.now() - datetime.timedelta(days=30)
        # else:
        #     print("Invalid period specified. Please choose 'weekly' or 'monthly'.")
        #     return

        self.cur.execute("SELECT category, SUM(amount) FROM expenses WHERE user_id=? GROUP BY category", (user_id,))
        expenses = self.cur.fetchall()
        report = {category: amount for category, amount in expenses}
        return report

    def close_connection(self):
        self.conn.close()

def get_float_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    tracker = ExpenseTracker()

    while True:
        print("\nExpense Tracker Menu:")
        print("1. Register User")
        print("2. Add expense")
        print("3. Delete expense")
        print("4. Edit expense")
        print("5. View expenses")
        # print("6. View all expenses")
        print("6. Set budget limit")
        print("7. Generate report")
        print("8. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            name = input("Enter your name: ")
            income = get_float_input("Enter your income: ")
            savings = get_float_input("Enter the amount you want to save: ")
            tracker.register_user(name, income, savings)
        elif choice == '2':
            name = input("Enter your name: ")
            user_id = tracker.get_user_id(name)
            if user_id:
                category = input("Enter expense category: ")
                amount = get_float_input("Enter expense amount: ")
                tracker.add_expense(user_id, category, amount)
                print("Expense added successfully.")
            else:
                print("User not found.")
        elif choice == '3':
            name = input("Enter your name: ")
            user_id = tracker.get_user_id(name)
            if user_id:
                expenses = tracker.get_expenses_by_user(user_id)
                if expenses:
                    print("Select the expense to delete:")
                    for expense in expenses:
                        print(f"ID: {expense[0]}, Category: {expense[1]}, Amount: {expense[2]}")
                    expense_id = input("Enter the ID of the expense to delete: ")
                    # Check if the entered expense ID is valid
                    valid_expense_ids = [str(expense[0]) for expense in expenses]
                    if expense_id in valid_expense_ids:
                        tracker.delete_expense(expense_id)
                        print("Expense deleted successfully.")
                    else:
                        print("Invalid expense ID.")
                else:
                    print("No expenses found for this user.")
            else:
                print("User not found.")
        elif choice == '4':
            name = input("Enter your name: ")
            user_id = tracker.get_user_id(name)
            if user_id:
                expenses = tracker.get_expenses_by_user(user_id)
                if expenses:
                    print("Select the expense to edit:")
                    for expense in expenses:
                        print(f"ID: {expense[0]}, Category: {expense[1]}, Amount: {expense[2]}")
                    expense_id = input("Enter the ID of the expense to edit: ")
                    valid_expense_ids = [str(expense[0]) for expense in expenses]
                    if expense_id in valid_expense_ids:
                        category = input("Enter new category: ")
                        amount = get_float_input("Enter new amount: ")
                        tracker.edit_expense(expense_id, category, amount)
                        print("Expense edited successfully.")
                    else:
                        print("Invalid expense ID.")
                else:
                    print("No expenses found for this user.")
            else:
                print("User not found.")
        elif choice == '5':
            name = input("Enter your name: ")
            user_id = tracker.get_user_id(name)
            if user_id:
                expenses = tracker.get_expenses_by_user(user_id)
                tracker.generate_pie_chart({expense[1]: expense[2] for expense in expenses}, f'Expenses - {name}')
                #total_income, total_savings = tracker.get_total_income_and_savings(user_id)
                #print(f"Total Income: {total_income}, Total Savings: {total_savings}")
            else:
                print("User not found.")
        # elif choice == '6':
        #     all_expenses = tracker.cur.execute("SELECT * FROM expenses")
        #     for expense in all_expenses:
        #         print(f"Category: {expense[2]}, Amount: {expense[3]}")
        elif choice == '6':
            name = input("Enter your name: ")
            user_id = tracker.get_user_id(name)
            if user_id:
                category = input("Enter expense category: ")
                bug_limit = get_float_input("Enter budget limit for the category: ")
                tracker.set_budget_limit(user_id, category, bug_limit)
            else:
                print("User not found.")
        elif choice == '7':
            name = input("Enter your name: ")
            user_id = tracker.get_user_id(name)
            if user_id:
                # period = input("Enter the period for the report (weekly/monthly): ")
                period = input("Enter the period for the report (weekly/monthly): ")
                report = tracker.generate_report(user_id, period)
                if report:
                    print("Expense report:")
                    for category, amount in report.items():
                        print(f"Category: {category}, Total Amount: {amount}")
                else:
                    print("No expenses found for this period.")
            else:
                print("User not found.")
        elif choice == '8':
            print("Exiting...")
            tracker.close_connection()
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
