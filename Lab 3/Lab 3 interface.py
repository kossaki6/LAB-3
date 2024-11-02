import tkinter as tk
from tkinter import ttk
import json
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class EquationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Формування умови")

        # Додаємо обробник для закриття вікна
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Встановлення шрифту за замовчуванням
        default_font = ("Arial", 14)  # Шрифт Arial, розмір 14
        self.root.option_add("*Font", default_font)

        # Параметри
        self.T = 10  # Кінцеве значення для t
        self.t = 0  # Початкове значення для t
        self.x1_constraints = [0,1]  # Обмеження для x1
        self.x2_constraints = [0,1]  # Обмеження для x2
        self.points = [] # Точки дискретизацій

        tk.Label(root, text="Режими роботи:").grid(row=0, column=0)
        self.solve_mode_button = tk.Button(root, text="Розв'язування", command=self.solve_mode)
        self.solve_mode_button.grid(row=0, column=1)
        self.solve_mode_button = tk.Button(root, text="Формування умови", command=self.problem_generation_mode)
        self.solve_mode_button.grid(row=0, column=2)

        # Введення для обмежень x1, x2, T
        tk.Label(root, text="Обмеження для x1:").grid(row=1, column=0)
        self.x1_constraints_entry = tk.Entry(root)
        self.x1_constraints_entry.grid(row=1, column=1)
        self.x1_constraints_entry.bind("<Return>", self.update_constraints)  # Оновлення обмеження

        tk.Label(root, text="Обмеження для x2:").grid(row=2, column=0)
        self.x2_constraints_entry = tk.Entry(root)
        self.x2_constraints_entry.grid(row=2, column=1)
        self.x2_constraints_entry.bind("<Return>", self.update_constraints)  # Оновлення обмеження

        tk.Label(root, text="T:").grid(row=3, column=0)
        self.t_entry = tk.Entry(root)
        self.t_entry.grid(row=3, column=1)
        self.t_entry.bind("<Return>", self.update_T)  # Оновлення T при натисканні Enter

        self.y_label = tk.Label(root, text="y(s):")
        self.y_label.grid(row=4, column=0)
        self.y_entry = tk.Entry(root)
        self.y_entry.grid(row=4, column=1)

        # Кнопка обчислення виразу для u(s)
        self.calculate_button = tk.Button(root, text="Обчислити u(s)", command=self.calculate_u)
        self.calculate_button.grid(row=4, column=2)

        tk.Label(root, text="u(s):").grid(row=5, column=0)
        self.u_entry = tk.Entry(root)
        self.u_entry.grid(row=5, column=1)

        # Введення кількості рівнянь ПУ
        tk.Label(root, text="Кількість рівнянь ПУ:").grid(row=6, column=0)
        self.pu_count_entry = tk.Entry(root)
        self.pu_count_entry.grid(row=6, column=1)
        self.pu_button = tk.Button(root, text="Згенерувати ПУ", command=self.generate_pu_equations)
        self.pu_button.grid(row=6, column=2)

        # Введення кількості рівнянь КУ
        tk.Label(root, text="Кількість рівнянь КУ:").grid(row=7, column=0)
        self.ku_count_entry = tk.Entry(root)
        self.ku_count_entry.grid(row=7, column=1)
        self.ku_button = tk.Button(root, text="Згенерувати КУ", command=self.generate_ku_equations)
        self.ku_button.grid(row=7, column=2)

        self.pu_frame = tk.Frame(root)
        self.pu_frame.grid(row=8, columnspan=3)

        self.ku_frame = tk.Frame(root)
        self.ku_frame.grid(row=9, columnspan=3)

        # Кнопка збереження
        self.save_button = tk.Button(root, text="Зберегти в JSON", command=self.save_to_json)
        self.save_button.grid(row=10, column=1)
        self.solve_button = tk.Button(root, text="Розв'язати", command=self.solve)

        # Графік y(s)
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=3, rowspan=10)

        # повзунок для зміни t
        self.slider = ttk.Scale(root, from_=0, to=self.T, orient="horizontal", command=self.update_graph)
        self.slider.grid(row=11, column=3, sticky="we")
        self.t_slider = tk.Label(root, text="Значення t:")
        self.t_slider.grid(row=11, column=2)
        self.T_slider = tk.Label(root, text=f"T: {self.T}")
        self.T_slider.grid(row=11, column=4)

        # Графік для точок дискретизації
        self.figure_points = plt.Figure(figsize=(5, 4))
        self.ax_points = self.figure_points.add_subplot(111)
        self.ax_points.set_title("Графік для створення точок")
        self.ax_points.set_xlim(self.x1_constraints)
        self.ax_points.set_ylim(self.x2_constraints)
        self.ax_points.set_xlabel("x1")
        self.ax_points.set_ylabel("x2")

        # Полотно для графіка точок
        self.canvas_points = FigureCanvasTkAgg(self.figure_points, master=root)
        self.canvas_points.mpl_connect("button_press_event", self.on_click)

        # Додаємо обробник натискань миші на графік
        self.canvas.mpl_connect("button_press_event", self.on_click)

        self.clean_button = tk.Button(root, text="Очистити", command=self.clean_points)

        # Завантаження попередніх даних
        self.load_from_json()

        # Початковий графік
        self.update_graph()
        self.update_graph_from_points()


    def solve_mode(self):
        self.save_button.grid(row=10, column=0)
        self.solve_button.grid(row=10, column=1)
        self.clean_button.grid(row=10, column=3)
        self.canvas_points.get_tk_widget().grid(row=0, column=3, rowspan=10)

        self.y_label.grid_forget()
        self.y_entry.grid_forget()
        self.calculate_button.grid_forget()
        self.save_button.grid_forget()
        self.slider.grid_forget()
        self.canvas.get_tk_widget().grid_forget()
        self.T_slider.grid_forget()
        self.t_slider.grid_forget()


    def problem_generation_mode(self):
        self.y_label.grid(row=4, column=0)
        self.y_entry.grid(row=4, column=1)
        self.calculate_button.grid(row=4, column=2)
        self.slider.grid(row=11, column=3, sticky="we")
        self.canvas.get_tk_widget().grid(row=0, column=3, rowspan=10)
        self.T_slider.grid(row=11, column=4)
        self.t_slider.grid(row=11, column=2)
        self.save_button.grid(row=10, column=1)

        self.solve_button.grid_forget()
        self.clean_button.grid_forget()
        self.canvas_points.get_tk_widget().grid_forget()

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

        try:
            # Перетворюємо рядковий вираз у символьний
            y = sp.sympify(y_expr)
        except ValueError:
            return

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

    def update_T(self, event=None):
        # Оновлення значення T
        try:
            self.T = float(self.t_entry.get())
        except ValueError:
            self.T = 10  # Значення за замовчуванням, якщо введено некоректно

        self.T_slider.config(text=f"T: {self.T}")
        # Оновлення меж повзунка
        self.slider.config(to=self.T)
        self.update_graph()

    def update_constraints(self,event=None):
        try:
            self.x1_constraints[0], self.x1_constraints[1] = map(float, self.x1_constraints_entry.get().split(' '))
            self.x2_constraints[0], self.x2_constraints[1] = map(float, self.x2_constraints_entry.get().split(' '))
        except ValueError:
            # Якщо не вдалося перетворити, використати стандартні значення
            self.x1_constraints = [0, 1]
            self.x2_constraints = [0, 1]

        self.update_graph_from_points()

    def update_graph(self, event=None):
        # Оновлення значення t
        self.t = self.slider.get()

        # Символьні змінні для обчислень
        t, x1, x2 = sp.symbols('t x1 x2')
        # Отримуємо вираз для y(s)
        y_expr = self.y_entry.get()

        try:
            # Перетворюємо рядковий вираз у символьний
            y = sp.sympify(y_expr)
        except ValueError:
            return

        # Отримання меж для x1 і x2
        try:
            self.x1_constraints[0], self.x1_constraints[1] = map(float, self.x1_constraints_entry.get().split(' '))
            self.x2_constraints[0], self.x2_constraints[1] = map(float, self.x2_constraints_entry.get().split(' '))
        except ValueError:
            # Якщо не вдалося перетворити, використати стандартні значення
            self.x1_constraints = [0, 1]
            self.x2_constraints = [0, 1]

        # Генерація даних для графіку
        x1_values = np.linspace(self.x1_constraints[0], self.x1_constraints[1], 15)
        x2_values = np.linspace(self.x2_constraints[0], self.x2_constraints[1], 15)
        x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)

        # Перетворення символьного виразу в числову функцію
        y_func = sp.lambdify((x1, x2, t), y_expr, 'numpy')

        y_values = y_func(x1_grid, x2_grid, self.t)

        # Оновлення графіку
        self.ax.clear()
        self.ax.plot_surface(x1_grid, x2_grid, y_values, cmap='viridis')
        self.ax.set_title(f"Графік функції y(x1, x2), t={self.t:.2f}")
        self.ax.set_xlabel("x1")
        self.ax.set_ylabel("x2")
        self.ax.set_zlabel("y")

        # Оновлення відображення
        self.canvas.draw()

    def on_click(self, event=None):
        """Обробка події натиску на графіку для створення точок."""
        if event.inaxes != self.ax_points:  # Перевірка, чи натискання графіку
            return

        x, y = event.xdata, event.ydata
        self.points.append((x, y))

        # Відображення точки на графіку
        self.ax_points.plot(x, y, 'ro')  # 'ro' - червона точка
        self.canvas_points.draw()

    def update_graph_from_points(self, event=None):
        # Очищення графіку та точок
        self.ax_points.clear()
        for x, y in self.points:
            self.ax_points.plot(x, y, 'ro')  # 'ro' - червона точка

        self.ax_points.set_xlim(self.x1_constraints)
        self.ax_points.set_ylim(self.x2_constraints)

        # Оновлення відображення
        self.canvas_points.draw()

    def save_to_json(self):
        data = {
            'x1_constraints': self.x1_constraints_entry.get(),
            'x2_constraints': self.x2_constraints_entry.get(),
            'T': self.t_entry.get(),
            'y(s)': self.y_entry.get(),
            'u(s)': self.u_entry.get(),
            'pu_equations': [(pu_type.get(), pu_expr.get()) for pu_type, pu_expr in self.pu_equations],
            'ku_equations': [(ku_type.get(), ku_expr.get()) for ku_type, ku_expr in self.ku_equations],
            'points': self.points  # Додаємо координати точок у JSON
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
            self.points = data.get('points', [])  # Завантаження точок у JSON

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

    def clean_points(self):
        self.points = []
        self.ax_points.clear()
        self.ax_points.set_xlim(self.x1_constraints)
        self.ax_points.set_ylim(self.x2_constraints)

        self.canvas_points.draw()

    def solve(self):
        pass

    def on_closing(self):
        # Завершення роботи програми
        plt.close('all')  # Закрити всі фігури matplotlib
        self.root.destroy()  # Знищити головне вікно

if __name__ == '__main__':
    root = tk.Tk()
    app = EquationApp(root)
    root.mainloop()
