import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression

demand_growth = 0.02
supply_response = 0.05

# Диференціальні рівняння
def demand_supply_system(t, y):
    demand, supply = y
    dD_dt = demand_growth * demand
    dS_dt = supply_response * (demand - supply)
    return [dD_dt, dS_dt]

demand_0 = 100
supply_0 = 90
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 100)

# Розв'язок системи
y0 = [demand_0, supply_0]
sol = solve_ivp(demand_supply_system, t_span, y0, t_eval=t_eval)

time = sol.t
demand_dyn = sol.y[0]
supply_dyn = sol.y[1]

X = time.reshape(-1, 1)
demand_trend_model = LinearRegression().fit(X, demand_dyn)
supply_trend_model = LinearRegression().fit(X, supply_dyn)

demand_trend = demand_trend_model.predict(X)
supply_trend = supply_trend_model.predict(X)

plt.figure(figsize=(10, 5))
plt.plot(time, demand_dyn, label="Попит (диференц. модель)", linestyle='-', color='blue')
plt.plot(time, supply_dyn, label="Пропозиція (диференц. модель)", linestyle='-', color='green')
plt.plot(time, demand_trend, label="Попит (трендова екстраполяція)", linestyle='--', color='blue')
plt.plot(time, supply_trend, label="Пропозиція (трендова екстраполяція)", linestyle='--', color='green')
plt.xlabel("Час")
plt.ylabel("Кількість")
plt.legend()
plt.title("Динаміка попиту та пропозиції електроенергії")
plt.grid()
plt.show()
