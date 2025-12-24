import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import math
import pandas as pd

# -----------------------------------
# Safe dictionary for allowed functions
# -----------------------------------
safe_dict = {
    "x": None,
    "np": np,
    "math": math,
    "sp": sp,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "pi": np.pi
}


# -----------------------------------
# 1️⃣ Bisection Method
# -----------------------------------
def bisection_method_gui(eq, a, b, tol=1e-6, parent=None):
    a = float(a); b = float(b)
    def f(x):
        try: return eval(eq, safe_dict | {"x": x})
        except: return None
    if f(a) is None or f(b) is None: return
    root=None; root_val=None; root_at_endpoint=False
    if f(a)==0: root=a; root_val=0; root_at_endpoint=True
    elif f(b)==0: root=b; root_val=0; root_at_endpoint=True
    elif f(a)*f(b)>0: messagebox.showerror("Error","f(a) and f(b) must have opposite signs!"); return
    table=[]; max_iter=50
    if not root_at_endpoint:
        for _ in range(max_iter):
            c=(a+b)/2; fc=f(c)
            table.append([a,b,c,fc,abs(a-b)])
            if abs(a-b)<tol: root=c; root_val=fc; break
            if f(a)*fc<0: b=c
            else: a=c
    else: table.append([a,b,root,root_val,0])
    frame=tk.Frame(parent); frame.pack(pady=5,fill='both',expand=True)
    output_text=tk.Text(frame,height=30,width=35); output_text.pack(side='left',padx=5,fill='both',expand=True)
    plot_frame=tk.Frame(frame); plot_frame.pack(side='left',padx=5,fill='both',expand=True)
    output_text.insert(tk.END,"Iter   a            b            c            f(c)        interval width\n")
    output_text.insert(tk.END,"-"*65+"\n")
    for i,row in enumerate(table,1):
        output_text.insert(tk.END,f"{i:<5}{row[0]:<12.6f}{row[1]:<12.6f}{row[2]:<12.6f}{row[3]:<12.6f}{row[4]:<12.6f}\n")
    output_text.insert(tk.END,f"\nRoot approximation: x={root:.6f}\n")
    x_vals=np.linspace(min([r[0] for r in table])-1,max([r[1] for r in table])+1,500)
    y_vals=[f(x) for x in x_vals]
    fig,ax=plt.subplots(figsize=(5,4))
    ax.plot(x_vals,y_vals,label='f(x)',color='blue')
    ax.scatter([r[2] for r in table],[r[3] for r in table],color='red',label='Midpoints')
    ax.scatter(root, root_val, color='green', s=50, label='Root')
    ax.axhline(0,color='black',linestyle='--')
    ax.set_title('Bisection Method'); ax.grid(True); ax.legend()
    canvas=FigureCanvasTkAgg(fig,master=plot_frame); canvas.draw()
    canvas.get_tk_widget().pack(fill='both',expand=True)

# -----------------------------------
# 2️⃣ Taylor Method
# -----------------------------------
def taylor_method_gui(f_input, degree, x0, point, x_eval, parent=None):
    x=sp.symbols('x'); f=sp.sympify(f_input); f_num=sp.lambdify(x,f,"numpy")
    degree=int(degree); x0=float(x0)
    derivs=[sp.diff(f,x,i).subs(x,x0) for i in range(degree+1)]
    taylor_poly=sum(derivs[i]/math.factorial(i)*(x-x0)**i for i in range(degree+1))
    p_func=sp.lambdify(x,taylor_poly,"numpy")
    frame=tk.Frame(parent); frame.pack(pady=5,fill='both',expand=True)
    output_text=tk.Text(frame,height=10,width=35); output_text.pack(side='left',padx=5,fill='both',expand=True)
    plot_frame=tk.Frame(frame); plot_frame.pack(side='left',padx=5,fill='both',expand=True)
    if point!='': point_val=float(point); output_text.insert(tk.END,f"\nAbsolute error at x={point_val}: {abs(f_num(point_val)-p_func(point_val))}\n")
    if x_eval!='': x_eval_val=float(x_eval); output_text.insert(tk.END,f"Taylor approximation at x={x_eval_val}: {p_func(x_eval_val)}\n")
    output_text.insert(tk.END,f"Taylor polynomial:\n{taylor_poly}\n")
    xs_plot=np.linspace(x0-6,x0+6,500)
    ys_exact=[f_num(xi) for xi in xs_plot]; ys_taylor=[p_func(xi) for xi in xs_plot]
    fig,ax=plt.subplots(figsize=(5,4))
    ax.plot(xs_plot,ys_exact,label='Exact f(x)',color='blue')
    ax.plot(xs_plot,ys_taylor,'--',label=f'Taylor degree {degree}',color='orange')
    if x_eval!='': ax.scatter([x_eval_val],[p_func(x_eval_val)],color='red',label=f'x_eval={x_eval_val}')
    ax.set_title('Taylor Series Approximation'); ax.grid(True); ax.legend()
    canvas=FigureCanvasTkAgg(fig,master=plot_frame); canvas.draw()
    canvas.get_tk_widget().pack(fill='both',expand=True)
# -----------------------------------
# 3️⃣ Neville Interpolation
# -----------------------------------
def neville_gui(xx,yy,x_val,parent=None):
    xx=list(map(float,xx.split())); yy=list(map(float,yy.split())); x=float(x_val); n=len(xx)
    if len(xx)!=len(yy):messagebox.showerror("Error","num of xs and num of ys must be equal"); return
    frame=tk.Frame(parent); frame.pack(pady=5,fill='both',expand=True)
    output_text=tk.Text(frame,height=10,width=35); output_text.pack(side='left',padx=5,fill='both',expand=True)
    plot_frame=tk.Frame(frame); plot_frame.pack(side='left',padx=5,fill='both',expand=True)
    q=[[0]*n for _ in range(n)]
    for i in range(n): q[i][0]=yy[i]
    for j in range(1,n):
        for i in range(j,n):
            q[i][j]=((x-xx[i-j])*q[i][j-1]-(x-xx[i])*q[i-1][j-1])/(xx[i]-xx[i-j])
    f_point=q[n-1][n-1]
    # Neville Table formatted manually
    output_text.insert(tk.END, "Neville Table:\n")

    header = " | ".join([f"Q[i,{j}]" for j in range(n)])
    output_text.insert(tk.END, header + "\n")

    for i in range(n):
        row = " | ".join([f"{q[i][j]:.5f}" for j in range(n)])
        output_text.insert(tk.END, row + "\n")

    output_text.insert(tk.END, f"\nValue at x={x}: {f_point:.5f}\n")
    x_vals=np.linspace(min(xx),max(xx),500); y_vals=[]
    for xs in x_vals:
        q_temp=[[0]*n for _ in range(n)]
        for i in range(n): q_temp[i][0]=yy[i]
        for j in range(1,n):
            for i in range(j,n):
                q_temp[i][j]=((xs-xx[i-j])*q_temp[i][j-1]-(xs-xx[i])*q_temp[i-1][j-1])/(xx[i]-xx[i-j])
        y_vals.append(q_temp[n-1][n-1])
    fig,ax=plt.subplots(figsize=(5,4))
    ax.plot(x_vals,y_vals,'-',color='blue',label='Neville Curve')
    ax.scatter(xx,yy,color='red',label='Data Points')
    ax.scatter(x,f_point,color='green',label=f'x={x}')
    ax.set_title('Neville Interpolation'); ax.grid(True); ax.legend()
    canvas=FigureCanvasTkAgg(fig,master=plot_frame); canvas.draw()
    canvas.get_tk_widget().pack(fill='both',expand=True)

# -----------------------------------
# 4️⃣ Lagrange Interpolation
# -----------------------------------
def lagrange_gui(x_str, y_str, x_val, parent=None):
    x = list(map(float, x_str.split()))
    y = list(map(float, y_str.split()))
    if len(x)!=len(y):messagebox.showerror("Error","num of xs and num of ys must be equal"); return
    xp = float(x_val)
    n = len(x)

    # GUI output
    frame = tk.Frame(parent); frame.pack(pady=5, fill='both', expand=True)
    output_text = tk.Text(frame, height=10, width=35); output_text.pack(side='left', padx=5, fill='both', expand=True)
    plot_frame = tk.Frame(frame); plot_frame.pack(side='left', padx=5, fill='both', expand=True)

    # Compute Lagrange interpolation
    total = 0
    for i in range(n):
        L = np.prod([(xp - x[j])/(x[i] - x[j]) for j in range(n) if i != j])
        total += L * y[i]
            # عرض جدول معاملات لاجرانج
    output_text.insert(tk.END, "\nLagrange Basis Table:\n")
    for i in range(n):
        L_i = np.prod([(xp - x[j])/(x[i] - x[j]) for j in range(n) if i != j])
        output_text.insert(tk.END, f"L[{i}] at x={xp}: {L_i:.5f}, contributes: {L_i*y[i]:.5f}\n")
            # عرض المعادلة بالشكل الأكاديمي
    output_text.insert(tk.END, "\nThe Lagrange Interpolating Polynomial\n")
    output_text.insert(tk.END, "General form:\n")
    output_text.insert(tk.END, "P(x) = Σ f(xi) · Li(x)\n")
    output_text.insert(tk.END, "Li(x) = Π (x - xj)/(xi - xj),  j ≠ i\n")

    # معادلة خاصة بالقيم المدخلة
    poly_equation = "P(x) = "
    terms = []
    for i in range(n):
        terms.append(f"f(x{i})·L{i}(x)")
    poly_equation += " + ".join(terms)
    output_text.insert(tk.END, f"\nConstructed Polynomial:\n{poly_equation}\n")

    # معادلة عند النقطة xp
    poly_at_xp = f"P({xp:.2f}) = "
    terms_at_xp = []
    for i in range(n):
        terms_at_xp.append(f"f(x{i})·L{i}({xp:.2f})")
    poly_at_xp += " + ".join(terms_at_xp)
    output_text.insert(tk.END, f"\n{poly_at_xp}\n")

    output_text.insert(tk.END, f"Value at x={xp}: {total:.5f}\n")

    # Plot interpolation curve
    x_vals = np.linspace(min(x), max(x), 500)
    y_vals = np.sum([y[i]*np.prod([(x_vals - x[j])/(x[i] - x[j]) for j in range(n) if i != j], axis=0) for i in range(n)], axis=0)

    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(x_vals, y_vals, '-', color='blue', label='Lagrange Curve')
    ax.scatter(x, y, color='red', label='Data Points')
    ax.scatter(xp, total, color='green', label=f'x={xp}')
    ax.set_title('Lagrange Interpolation'); ax.grid(True); ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame); canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)



# -----------------------------------
# 5️⃣ Newton Divided Differences
# -----------------------------------
def newton_gui(xs_str, ys_str, xq, parent=None):
    xs = list(map(float, xs_str.split()))
    ys = list(map(float, ys_str.split()))
    if len(xs)!=len(ys):messagebox.showerror("Error","num of xs and num of ys must be equal"); return
    xq = float(xq)
    n = len(xs)

    frame = tk.Frame(parent); frame.pack(pady=5, fill='both', expand=True)
    output_text = tk.Text(frame, height=14, width=35); output_text.pack(side='left', padx=5, fill='both', expand=True)
    plot_frame = tk.Frame(frame); plot_frame.pack(side='left', padx=5, fill='both', expand=True)

    table = [ys.copy()]
    for j in range(1, n):
        table.append([(table[j-1][i+1] - table[j-1][i])/(xs[i+j] - xs[i]) for i in range(n-j)])

    column_names = ['x', 'f(x)'] + [f'{i}st Diff' if i==1 else f'{i}th Diff' for i in range(1, n)]
    col_width = 12
    
    header = "".join(f"{name:<{col_width}}" for name in column_names)
    separator = "-" * (col_width * n + 2 * col_width)
    
    output_text.insert(tk.END, "Newton Divided Difference Table:\n")
    output_text.insert(tk.END, header + "\n")
    output_text.insert(tk.END, separator + "\n")
    
    for i in range(n):
        row_str = f"{xs[i]:<{col_width}.4f}{table[0][i]:<{col_width}.4f}"
        
        for j in range(1, n):
            if i < n - j: 
                value = table[j][i]
                row_str += f"{value:<{col_width}.4f}"
            else:
                row_str += " " * col_width
        
        output_text.insert(tk.END, row_str.strip() + "\n")
        
    output_text.insert(tk.END, "\n")
    x_symbol = sp.symbols('x')
    P_x = table[0][0]
    term_symbol = 1
    
    for j in range(1, n):
        term_symbol *= (x_symbol - xs[j-1])
        P_x += table[j][0] * term_symbol
        
    simplified_poly = sp.simplify(P_x)
    
    output_text.insert(tk.END, f"Interpolating Polynomial P(x):\n")
    output_text.insert(tk.END, f"{str(simplified_poly)}\n")
    output_text.insert(tk.END, "-" * 30 + "\n")

    result = table[0][0]; term = 1
    for j in range(1, n):
        term *= (xq - xs[j-1])
        result += table[j][0] * term

    output_text.insert(tk.END, f"Approximation at x={xq}: {result}\n")

    xs_plot = np.linspace(min(xs), max(xs), 500)
    ys_plot = []
    for xp in xs_plot:
        val = table[0][0]; term = 1
        for j in range(1, n):
            term *= (xp - xs[j-1])
            val += table[j][0] * term
        ys_plot.append(val)

    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(xs_plot, ys_plot, '-', label='Newton Approx')
    ax.scatter(xs, ys, color='red', label='Data Points')
    ax.scatter(xq, result, color='green', label=f'x={xq}')
    ax.set_title('Newton Divided Differences'); ax.grid(True); ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame); canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
# -----------------------------------
# 6️⃣ Forward Difference (without spacing check)
# -----------------------------------
def forward_difference_gui(x_str, y_str, x_eval='', parent=None):
    x = list(map(float, x_str.split()))
    y = list(map(float, y_str.split()))
    if len(x)!=len(y):messagebox.showerror("Error","num of xs and num of ys must be equal"); return
    n = len(x)

    if n < 2:
        messagebox.showerror("Error", "Please enter at least two x points.")
        return

    h = np.diff(x)[0]

    frame = tk.Frame(parent); frame.pack(pady=5, fill='both', expand=True)
    output_text = tk.Text(frame, height=10, width=35); output_text.pack(side='left', padx=5, fill='both', expand=True)
    plot_frame = tk.Frame(frame); plot_frame.pack(side='left', padx=5, fill='both', expand=True)

    diff_table = [y[:]]
    for j in range(1, n):
        diff_table.append([diff_table[j-1][i+1] - diff_table[j-1][i] for i in range(n-j)])

    col_width = 8 
    column_names = ['x'] + [f'f(x)'] + [f'D^{i}' for i in range(1, n)]
    header = "".join(f"{name:<{col_width}}" for name in column_names)
    separator = "-" * (col_width * (n + 1))

    output_text.insert(tk.END, "Forward Difference Table:\n")
    output_text.insert(tk.END, header + "\n")
    output_text.insert(tk.END, separator + "\n")
    
    for i in range(n):
        row_str = f"{round(x[i], 4):<{col_width}.4f}" 
        
        for j in range(n):
            if i < len(diff_table[j]): 
                value = diff_table[j][i]
                row_str += f"{round(value, 4):<{col_width}.4f}" 
            else:
                row_str += " " * col_width
                
        output_text.insert(tk.END, row_str.strip() + "\n")
        
    output_text.insert(tk.END, f"\nh = {h}\n")

    approx_val = None
    if x_eval != '':
        xp = float(x_eval)
        s = (xp - x[0]) / h
        approx_val = diff_table[0][0]; sprod = 1.0
        for k in range(1, n):
            sprod *= (s - (k-1))
            approx_val += (sprod/math.factorial(k)) * diff_table[k][0]

    if approx_val is not None:
        output_text.insert(tk.END, f"Approximation at x={xp}: {approx_val}\n")

    fig, ax = plt.subplots(figsize=(5,4))
    ax.scatter(x, y, color='red', label='Data Points')
    x_vals = np.linspace(x[0], x[-1], 400); y_vals = []
    for xv in x_vals:
        s = (xv - x[0]) / h
        val = diff_table[0][0]; sprod = 1.0
        for k in range(1, n):
            sprod *= (s - (k-1))
            val += (sprod/math.factorial(k)) * diff_table[k][0]
        y_vals.append(val)
    ax.plot(x_vals, y_vals, '-', color='blue', label='Forward polynomial')
    if approx_val is not None:
        ax.scatter([xp], [approx_val], color='green', label=f'x={xp}')
    ax.set_title('Forward Difference'); ax.grid(True); ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame); canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)




# -----------------------------------
# 7️⃣ Backward Difference (without spacing check)
# -----------------------------------
def backward_difference_gui(x_str, y_str, x_eval='', parent=None):
    x = list(map(float, x_str.split()))
    y = list(map(float, y_str.split()))
    if len(x)!=len(y):messagebox.showerror("Error","num of xs and num of ys must be equal"); return
    n = len(x)

    if n < 2:
        messagebox.showerror("Error", "Please enter at least two x points.")
        return

    h = np.diff(x)[0]

    frame = tk.Frame(parent); frame.pack(pady=5, fill='both', expand=True)
    output_text = tk.Text(frame, height=14, width=35); output_text.pack(side='left', padx=5, fill='both', expand=True)
    plot_frame = tk.Frame(frame); plot_frame.pack(side='left', padx=5, fill='both', expand=True)

    diff_table = [y[:]]
    for j in range(1, n):
        prev = diff_table[j-1]
        diff_table.append([prev[i] - prev[i-1] for i in range(1, len(prev))])

    approx_val = None
    if x_eval != '':
        xp = float(x_eval)
        s = (xp - x[-1]) / h
        approx_val = diff_table[0][-1]; sprod = 1.0
        for k in range(1, n):
            sprod *= (s + (k-1))
            approx_val += (sprod/math.factorial(k)) * diff_table[k][-1]

    # طباعة الجدول
    column_names = ['x', 'f(x)'] + [f'Delta^{i}' for i in range(1, n)]
    col_width = 12
    
    header = "".join(f"{name:<{col_width}}" for name in column_names)
    separator = "-" * (col_width * n + 2 * col_width)
    
    output_text.insert(tk.END, "Backward Difference Table:\n")
    output_text.insert(tk.END, header + "\n")
    output_text.insert(tk.END, separator + "\n")
    
    for i in range(n):
        row_str = f"{x[i]:<{col_width}.4f}{diff_table[0][i]:<{col_width}.4f}"
        
        for j in range(1, n):
            if i >= j: 
                value = diff_table[j][i - j]
                row_str += f"{value:<{col_width}.4f}"
            else:
                row_str += " " * col_width
        
        output_text.insert(tk.END, row_str.strip() + "\n")
        
    output_text.insert(tk.END, f"\nh = {h}\n")
    
    if approx_val is not None:
        output_text.insert(tk.END, f"Approximation at x={xp}: {approx_val}\n")

    fig, ax = plt.subplots(figsize=(5,4))
    ax.scatter(x, y, color='blue', label='Data Points')
    x_vals = np.linspace(x[0], x[-1], 400); y_vals = []
    for xv in x_vals:
        s = (xv - x[-1]) / h
        val = diff_table[0][-1]; sprod = 1.0
        for k in range(1, n):
            sprod *= (s + (k-1))
            val += (sprod/math.factorial(k)) * diff_table[k][-1]
        y_vals.append(val)
    ax.plot(x_vals, y_vals, '-', color='purple', label='Backward polynomial')
    if approx_val is not None:
        ax.scatter([xp], [approx_val], color='green', label=f'x={xp}')
    ax.set_title('Backward Difference'); ax.grid(True); ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame); canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)


# -----------------------------------
# 8️⃣ Interpolation (Auto choose method)
# -----------------------------------
def interpolation_gui(x_str, y_str, x_eval, parent=None):
    xs = list(map(float, x_str.split()))
    ys = list(map(float, y_str.split()))
    xp = float(x_eval)
    diffs = np.diff(xs)

    # If X not equally spaced → Newton Divided Differences
    if not np.allclose(diffs, diffs[0]):
        newton_gui(x_str, y_str, x_eval, parent=parent)
    else:
        h = diffs[0]
        # If xp closer to first x → Forward Difference
        if abs(xp - xs[0]) <= abs(xp - xs[-1]):
            forward_difference_gui(x_str, y_str, x_eval, parent=parent)
        else:
            backward_difference_gui(x_str, y_str, x_eval, parent=parent)


# -----------------------------------
# Tkinter Main GUI
# -----------------------------------
root = tk.Tk()
root.title('✨ Numerical Analysis Program ✨')
root.geometry('500x550')
root.resizable(False, False)

ttk.Label(root, text='✨ Numerical Analysis Program ✨', font=('Helvetica',16,'bold')).pack(pady=20)
button_frame = tk.Frame(root); button_frame.pack(pady=10)

# Methods list (Interpolation replaces Newton/Forward/Backward)
methods = [
    ('Bisection Method', bisection_method_gui),
    ('Taylor Method', taylor_method_gui),
    ('Neville Interpolation', neville_gui),
    ('Lagrange Interpolation', lagrange_gui),
    ('Newton Divided Difference', interpolation_gui)  # unified option
]

style = ttk.Style()
style.configure('Big.TButton', font=('Helvetica',10,'bold'), padding=10)

for name, func in methods:
    ttk.Button(
        button_frame,
        text=name,
        width=30,
        command=lambda f=func: create_method_window(f, root),
        style='Big.TButton'
    ).pack(pady=10)


# -----------------------------------
# Create method window dynamically
# -----------------------------------
def create_method_window(func, master):
    win = tk.Toplevel(master)
    win.title(func.__name__)
    win.geometry('700x600')
    win.resizable(True, True)

    style_sub = ttk.Style()
    style_sub.configure('SubBig.TButton', font=('Helvetica',12,'bold'), padding=8)

    if 'bisection' in func.__name__:
        entries = {}
        for label in ['f(x) =','a =','b =','Tolerance =']:
            ttk.Label(win, text=label, font=('Helvetica',12)).pack(pady=3)
            entries[label] = ttk.Entry(win, width=35, font=('Helvetica',12))
            entries[label].pack(pady=3)
        ttk.Button(win, text='Compute', command=lambda: func(
            entries['f(x) ='].get(), entries['a ='].get(), entries['b ='].get(),
            tol=float(entries['Tolerance ='].get()) if entries['Tolerance ='].get() else 1e-6,
            parent=win), style='SubBig.TButton').pack(pady=10)

    elif 'taylor' in func.__name__:
        labels = ['f(x) =','Degree =','x0 =','Point for error(optional) =','x_eval (optional)=']
        entries = {}
        for lbl in labels:
            ttk.Label(win, text=lbl, font=('Helvetica',12)).pack(pady=3)
            entries[lbl] = ttk.Entry(win, width=35, font=('Helvetica',12))
            entries[lbl].pack(pady=3)
        ttk.Button(win, text='Compute', command=lambda: func(
            entries['f(x) ='].get(), entries['Degree ='].get(), entries['x0 ='].get(),
            entries['Point for error(optional) ='].get(), entries['x_eval (optional)='].get(),
            parent=win), style='SubBig.TButton').pack(pady=10)

    else:
        labels = ['X values (space separated) =','Y values (space separated) =','x_eval =']
        if 'neville' in func.__name__: labels[2] = 'Point x ='
        entries = {}
        for lbl in labels:
            ttk.Label(win, text=lbl, font=('Helvetica',12)).pack(pady=3)
            entries[lbl] = ttk.Entry(win, width=35, font=('Helvetica',12))
            entries[lbl].pack(pady=3)
        ttk.Button(win, text='Compute', command=lambda: func(
            entries[labels[0]].get(), entries[labels[1]].get(), entries[labels[2]].get(),
            parent=win), style='SubBig.TButton').pack(pady=10)


root.mainloop()