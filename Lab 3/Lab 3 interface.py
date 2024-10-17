import tkinter as tk
from tkinter import ttk
import json
import sympy as sp


class EquationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Формування умови")

        # Введення для обмежень x1, x2
        tk.Label(root, text="Обмеження для x1:").grid(row=0, column=0)
        self.x1_constraints_entry = tk.Entry(root)
        self.x1_constraints_entry.grid(row=0, column=1)

        tk.Label(root, text="Обмеження для x2:").grid(row=1, column=0)
        self.x2_constraints_entry = tk.Entry(root)
        self.x2_constraints_entry.grid(row=1, column=1)

        tk.Label(root, text="T:").grid(row=2, column=0)
        self.t_entry = tk.Entry(root)
        self.t_entry.grid(row=2, column=1)

        tk.Label(root, text="y(s):").grid(row=3, column=0)
        self.y_entry = tk.Entry(root)
        self.y_entry.grid(row=3, column=1)

        # Кнопка обчислення виразу для u(s)
        self.calculate_button = tk.Button(root, text="Обчислити u(s)", command=self.calculate_u)
        self.calculate_button.grid(row=3, column=2)

        tk.Label(root, text="u(s):").grid(row=4, column=0)
        self.u_entry = tk.Entry(root)
        self.u_entry.grid(row=4, column=1)

        # Введення кількості рівнянь ПУ
        tk.Label(root, text="Кількість рівнянь ПУ:").grid(row=5, column=0)
        self.pu_count_entry = tk.Entry(root)
        self.pu_count_entry.grid(row=5, column=1)
        self.pu_button = tk.Button(root, text="Згенерувати ПУ", command=self.generate_pu_equations)
        self.pu_button.grid(row=5, column=2)

        # Введення кількості рівнянь КУ
        tk.Label(root, text="Кількість рівнянь КУ:").grid(row=6, column=0)
        self.ku_count_entry = tk.Entry(root)
        self.ku_count_entry.grid(row=6, column=1)
        self.ku_button = tk.Button(root, text="Згенерувати КУ", command=self.generate_ku_equations)
        self.ku_button.grid(row=6, column=2)

        self.pu_frame = tk.Frame(root)
        self.pu_frame.grid(row=7, columnspan=3)

        self.ku_frame = tk.Frame(root)
        self.ku_frame.grid(row=8, columnspan=3)

        # Кнопка збереження
        self.save_button = tk.Button(root, text="Зберегти в JSON", command=self.save_to_json)
        self.save_button.grid(row=9, column=1)

        # Завантаження попередніх даних
        self.load_from_json()

    def generate_pu_equations(self):
        for widget in self.pu_frame.winfo_children():
            widget.destroy()

        try:
            pu_count = int(self.pu_count_entry.get())
        except ValueError:
            return

        self.pu_equations = []
        for i in range(pu_count):
            tk.Label(self.pu_frame, text=f"ПУ{i + 1}:").grid(row=i, column=0)
            pu_type = ttk.Combobox(self.pu_frame, values=["1", "похідна по t"])
            pu_type.grid(row=i, column=1)
            pu_type.set("1")  # Значення за замовчуванням
            pu_type.state(['readonly'])  # Заборонити введення
            pu_expression = tk.Entry(self.pu_frame)
            pu_expression.grid(row=i, column=2)
            self.pu_equations.append((pu_type, pu_expression))

    def generate_ku_equations(self):
        for widget in self.ku_frame.winfo_children():
            widget.destroy()

        try:
            ku_count = int(self.ku_count_entry.get())
        except ValueError:
            return

        self.ku_equations = []
        for i in range(ku_count):
            tk.Label(self.ku_frame, text=f"КУ{i + 1}:").grid(row=i, column=0)
            ku_type = ttk.Combobox(self.ku_frame, values=["1", "похідна по x1"])
            ku_type.grid(row=i, column=1)
            ku_type.set("1")  # Значення за замовчуванням
            ku_type.state(['readonly'])  # Заборонити введення
            ku_expression = tk.Entry(self.ku_frame)
            ku_expression.grid(row=i, column=2)
            self.ku_equations.append((ku_type, ku_expression))

    def calculate_u(self):
        # Отримуємо вираз для y(s)
        y_expr = self.y_entry.get()

        # Символьні змінні для обчислень
        t, x1, x2 = sp.symbols('t x1 x2')

        # Перетворюємо рядковий вираз у символьний
        y = sp.sympify(y_expr)

        # Обчислюємо другі похідні для u(s)
        d2y_dt2 = sp.diff(y, t, 2)
        d2y_dx1_2 = sp.diff(y, x1, 2)
        d2y_dx2_2 = sp.diff(y, x2, 2)

        # Вираз для u(s)
        u_expr = d2y_dt2 - d2y_dx1_2 - d2y_dx2_2

        # Виводимо результат у поле u(s)
        self.u_entry.delete(0, tk.END)
        self.u_entry.insert(0, str(u_expr))

        # Обробка ПУ
        for pu_type, pu_expr in self.pu_equations:
            if pu_type.get() == "1":
                # Підставляємо t = 0 у y(s)
                pu_value = y.subs(t, 0)
            elif pu_type.get() == "похідна по t":
                # Обчислюємо похідну по t та підставляємо t = 0
                dy_dt = sp.diff(y, t)
                pu_value = dy_dt.subs(t, 0)

            # Виводимо результат у відповідне поле ПУ
            pu_expr.delete(0, tk.END)
            pu_expr.insert(0, str(pu_value))

        # Обробка КУ
        for ku_type, ku_expr in self.ku_equations:
            if ku_type.get() == "1":
                # Просто вираз y(s)
                ku_value = y
            elif ku_type.get() == "похідна по x1":
                # Обчислюємо похідну по x1
                dy_dx1 = sp.diff(y, x1)
                ku_value = dy_dx1

            # Виводимо результат у відповідне поле КУ
            ku_expr.delete(0, tk.END)
            ku_expr.insert(0, str(ku_value))

    def save_to_json(self):
        data = {
            'x1_constraints': self.x1_constraints_entry.get(),
            'x2_constraints': self.x2_constraints_entry.get(),
            'T': self.t_entry.get(),
            'y(s)': self.y_entry.get(),
            'u(s)': self.u_entry.get(),
            'pu_equations': [(pu_type.get(), pu_expr.get()) for pu_type, pu_expr in self.pu_equations],
            'ku_equations': [(ku_type.get(), ku_expr.get()) for ku_type, ku_expr in self.ku_equations],
        }

        with open('equations.json', 'w') as f:
            json.dump(data, f)

    def load_from_json(self):
        try:
            with open('equations.json', 'r') as f:
                data = json.load(f)

            self.x1_constraints_entry.insert(0, data.get('x1_constraints', ''))
            self.x2_constraints_entry.insert(0, data.get('x2_constraints', ''))
            self.t_entry.insert(0, data.get('T', ''))
            self.y_entry.insert(0, data.get('y(s)', ''))
            self.u_entry.insert(0, data.get('u(s)', ''))

            # Завантаження ПУ
            pu_equations = data.get('pu_equations', [])
            if pu_equations:
                self.pu_count_entry.insert(0, str(len(pu_equations)))
                self.generate_pu_equations()
                for i, (pu_type, pu_expr) in enumerate(pu_equations):
                    self.pu_equations[i][0].set(pu_type)
                    self.pu_equations[i][1].insert(0, pu_expr)
            else:
                # Якщо ПУ не задані, за замовчуванням одне рівняння
                self.pu_count_entry.insert(0, "1")
                self.generate_pu_equations()

            # Завантаження КУ
            ku_equations = data.get('ku_equations', [])
            if ku_equations:
                self.ku_count_entry.insert(0, str(len(ku_equations)))
                self.generate_ku_equations()
                for i, (ku_type, ku_expr) in enumerate(ku_equations):
                    self.ku_equations[i][0].set(ku_type)
                    self.ku_equations[i][1].insert(0, ku_expr)
            else:
                # Якщо КУ не задані, за замовчуванням одне рівняння
                self.ku_count_entry.insert(0, "1")
                self.generate_ku_equations()

        except FileNotFoundError:
            # Якщо файл не знайдено, генеруємо одне рівняння для ПУ та КУ
            self.pu_count_entry.insert(0, "1")
            self.generate_pu_equations()
            self.ku_count_entry.insert(0, "1")
            self.generate_ku_equations()

if __name__ == '__main__':
    root = tk.Tk()
    app = EquationApp(root)
    root.mainloop()
