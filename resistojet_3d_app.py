import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import LinearSegmentedColormap
import time
from resistojet_3d import create_3d_visualization, animate_particles, create_temperature_distribution_plot, create_velocity_gradient_plot
import plotly.graph_objects as go

# Настройка страницы
st.set_page_config(
    page_title="3D-симуляция резистоджета",
    page_icon="🚀",
    layout="wide"
)

# Заголовок
st.title("3D-симуляция тепловых процессов и движения частиц в резистоджете")
st.markdown("""
Эта интерактивная модель позволяет симулировать тепловые процессы и движение частиц в резистивном ракетном двигателе 
с регенеративным охлаждением. Вы можете изменять параметры и наблюдать их влияние на результаты.
""")

# Боковая панель с параметрами
st.sidebar.header("Параметры модели")

# Геометрические параметры
st.sidebar.subheader("Геометрия")
chamber_height = st.sidebar.slider("Высота камеры (мм)", 20.0, 100.0, 40.0, 1.0) / 1000
chamber_inner_radius = st.sidebar.slider("Внутренний радиус камеры (мм)", 5.0, 25.0, 10.0, 0.5) / 1000
chamber_wall_thickness = st.sidebar.slider("Толщина стенки камеры (мм)", 0.5, 5.0, 1.0, 0.1) / 1000
cooling_gap = st.sidebar.slider("Зазор охлаждения (мм)", 0.5, 5.0, 1.0, 0.1) / 1000
cooling_wall_thickness = st.sidebar.slider("Толщина стенки охлаждающей рубашки (мм)", 0.5, 5.0, 1.0, 0.1) / 1000

# Рабочие параметры
st.sidebar.subheader("Рабочие параметры")
mass_flow_rate = st.sidebar.slider("Массовый расход (г/с)", 0.1, 5.0, 1.0, 0.1) / 1000
pressure = st.sidebar.slider("Давление (МПа)", 0.1, 10.0, 1.0, 0.1) * 1e6
heater_power = st.sidebar.slider("Мощность нагревателя (Вт)", 50, 500, 200, 10)
initial_temp = st.sidebar.slider("Начальная температура (K)", 100, 500, 300, 10)

# Параметры визуализации
st.sidebar.subheader("Параметры визуализации")
num_particles = st.sidebar.slider("Количество частиц", 10, 100, 50, 5)
animate = st.sidebar.checkbox("Анимировать движение частиц", True)

# Расчет параметров модели
@st.cache_data
def calculate_model_parameters(chamber_height, chamber_inner_radius, chamber_wall_thickness, 
                              cooling_gap, cooling_wall_thickness, mass_flow_rate, 
                              pressure, heater_power, initial_temp, num_stations=20):
    """
    Расчет параметров модели резистоджета.
    """
    # Расчет геометрических параметров
    chamber_inner_diameter = 2 * chamber_inner_radius
    chamber_outer_radius = chamber_inner_radius + chamber_wall_thickness
    cooling_jacket_inner_radius = chamber_outer_radius
    cooling_jacket_outer_radius = cooling_jacket_inner_radius + cooling_gap
    shell_inner_radius = cooling_jacket_outer_radius
    shell_outer_radius = shell_inner_radius + cooling_wall_thickness
    
    # Площади поперечного сечения
    A_chamber = np.pi * chamber_inner_radius**2
    A_cooling = np.pi * (cooling_jacket_outer_radius**2 - cooling_jacket_inner_radius**2)
    
    # Создаем профиль сопла (упрощенно)
    nozzle_profile = np.zeros(num_stations+1)
    for i in range(num_stations+1):
        z_normalized = i / num_stations
        if z_normalized < 0.4:
            # Постепенное сужение
            nozzle_profile[i] = chamber_inner_radius * (1 - 0.3 * (z_normalized / 0.4))
        else:
            # Расширение после сужения
            nozzle_profile[i] = chamber_inner_radius * 0.7 + chamber_inner_radius * 0.8 * ((z_normalized - 0.4) / 0.6)
    
    # Расчет температур (упрощенно)
    # Температура рабочей жидкости (водород) увеличивается от входа к выходу
    T_chamber_fluid = np.linspace(initial_temp, initial_temp + heater_power / (mass_flow_rate * 14300), num_stations+1)
    
    # Температура стенки камеры немного ниже температуры рабочей жидкости
    T_chamber_wall = T_chamber_fluid - 50
    
    # Температура охлаждающей жидкости увеличивается от входа к выходу
    T_cooling_fluid = np.linspace(initial_temp, initial_temp + 200, num_stations+1)
    
    return (nozzle_profile, T_chamber_fluid, T_chamber_wall, T_cooling_fluid, 
            chamber_outer_radius, cooling_jacket_inner_radius, cooling_jacket_outer_radius, 
            shell_inner_radius, shell_outer_radius, A_chamber, A_cooling)

# Расчет параметров модели
(nozzle_profile, T_chamber_fluid, T_chamber_wall, T_cooling_fluid, 
 chamber_outer_radius, cooling_jacket_inner_radius, cooling_jacket_outer_radius, 
 shell_inner_radius, shell_outer_radius, A_chamber, A_cooling) = calculate_model_parameters(
    chamber_height, chamber_inner_radius, chamber_wall_thickness, 
    cooling_gap, cooling_wall_thickness, mass_flow_rate, 
    pressure, heater_power, initial_temp
)

# Создание 3D-визуализации
fig = create_3d_visualization(
    chamber_height, chamber_inner_radius, chamber_wall_thickness, 
    cooling_gap, cooling_wall_thickness, nozzle_profile, 
    T_chamber_fluid, T_chamber_wall, T_cooling_fluid
)

# Анимация движения частиц
if animate:
    fig = animate_particles(fig)

# Отображение 3D-визуализации
st.plotly_chart(fig, use_container_width=True)

# Создание и отображение визуализации градиента скорости
st.header("Визуализация градиента скорости")
st.write("""
Ниже представлена визуализация градиента скорости в резистоджете. 
Цветовая шкала показывает скорость потока в м/с, а контурные линии - области с одинаковой скоростью.
""")

velocity_fig = create_velocity_gradient_plot(chamber_height, chamber_inner_radius, nozzle_profile)
st.plotly_chart(velocity_fig, use_container_width=True)

# Создание и отображение визуализации распределения температуры
st.header("Визуализация распределения температуры")
st.write("""
Ниже представлена визуализация распределения температуры в резистоджете.
Цветовая шкала показывает температуру в Кельвинах.
""")

temp_fig = create_temperature_distribution_plot(
    chamber_height, chamber_inner_radius, chamber_wall_thickness, 
    cooling_gap, cooling_wall_thickness, 
    T_chamber_fluid, T_chamber_wall, T_cooling_fluid
)
st.plotly_chart(temp_fig, use_container_width=True)

# Добавление пояснения
st.write("""
### Пояснение к визуализации

#### 3D-модель движения частиц
На левом графике показана 3D-модель движения частиц в резистоджете:
- **Цвет траектории**: соответствует числу Маха (от синего для дозвуковых скоростей до красного для сверхзвуковых)
- **Форма сопла**: показана полупрозрачной серой поверхностью
- **Движение частиц**: частицы движутся от входа к выходу сопла, ускоряясь в сужающейся части

#### Распределение числа Маха
На правом графике показано распределение числа Маха в осевом сечении:
- **Синий цвет**: дозвуковой поток (M < 1)
- **Зеленый цвет**: околозвуковой поток (M ≈ 1)
- **Желтый/оранжевый цвет**: сверхзвуковой поток (1 < M < 3)
- **Красный цвет**: высокоскоростной сверхзвуковой поток (M > 3)

Контурные линии показывают области с одинаковым числом Маха, что позволяет лучше видеть градиенты.
""")

# Добавление информации о параметрах модели
st.write("""
### Параметры модели

#### Геометрические параметры
- Высота камеры: {:.1f} мм
- Внутренний радиус камеры: {:.1f} мм
- Толщина стенки камеры: {:.1f} мм
- Зазор охлаждения: {:.1f} мм
- Толщина стенки охлаждающей рубашки: {:.1f} мм

#### Рабочие параметры
- Массовый расход: {:.1f} г/с
- Давление: {:.1f} МПа
- Мощность нагревателя: {:.0f} Вт
- Начальная температура: {:.0f} K

#### Расчетные параметры
- Максимальная температура рабочей жидкости: {:.0f} K
- Максимальная температура стенки: {:.0f} K
- Максимальное число Маха: {:.1f}
""".format(
    chamber_height*1000, chamber_inner_radius*1000, chamber_wall_thickness*1000,
    cooling_gap*1000, cooling_wall_thickness*1000,
    mass_flow_rate*1000, pressure/1e6, heater_power, initial_temp,
    np.max(T_chamber_fluid), np.max(T_chamber_wall), 3.7
))

# Добавление информации о проекте
st.sidebar.markdown("""
### О проекте
Эта модель демонстрирует 3D-визуализацию движения частиц и распределения числа Маха в резистоджете.

Модель учитывает:
- Геометрию сопла
- Тепловые процессы
- Ускорение потока
- Изменение числа Маха

Для более детального анализа используйте основную модель.
""") 