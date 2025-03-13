import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from resistojet_simulation import (
    hydrogen_properties, 
    inconel_properties, 
    calculate_flow_parameters, 
    calculate_heat_transfer,
    steady_state_simulation,
    plot_results,
    calculate_equilibrium_time,
    CHAMBER_HEIGHT,
    CHAMBER_INNER_DIAMETER,
    CHAMBER_WALL_THICKNESS,
    COOLING_GAP,
    COOLING_WALL_THICKNESS,
    MASS_FLOW_RATE,
    PRESSURE,
    HEATER_POWER,
    INITIAL_TEMP,
    NUM_STATIONS
)

# Настройка страницы
st.set_page_config(
    page_title="Симуляция резистоджета",
    page_icon="🚀",
    layout="wide"
)

# Заголовок
st.title("Симуляция тепловых процессов резистоджета")
st.markdown("""
Эта интерактивная модель позволяет симулировать тепловые процессы в резистивном ракетном двигателе 
с регенеративным охлаждением. Вы можете изменять параметры и наблюдать их влияние на результаты.
""")

# Боковая панель с параметрами
st.sidebar.header("Параметры модели")

# Геометрические параметры
st.sidebar.subheader("Геометрия")
chamber_height = st.sidebar.slider("Высота камеры (мм)", 20.0, 100.0, float(CHAMBER_HEIGHT*1000), 1.0) / 1000
chamber_inner_diameter = st.sidebar.slider("Внутренний диаметр камеры (мм)", 10.0, 50.0, float(CHAMBER_INNER_DIAMETER*1000), 1.0) / 1000
chamber_wall_thickness = st.sidebar.slider("Толщина стенки камеры (мм)", 0.5, 5.0, float(CHAMBER_WALL_THICKNESS*1000), 0.1) / 1000
cooling_gap = st.sidebar.slider("Зазор охлаждения (мм)", 0.5, 5.0, float(COOLING_GAP*1000), 0.1) / 1000
cooling_wall_thickness = st.sidebar.slider("Толщина стенки охлаждающей рубашки (мм)", 0.5, 5.0, float(COOLING_WALL_THICKNESS*1000), 0.1) / 1000

# Рабочие параметры
st.sidebar.subheader("Рабочие параметры")
mass_flow_rate = st.sidebar.slider("Массовый расход (мг/с)", 1.0, 20.0, float(MASS_FLOW_RATE*1e6), 0.1) / 1e6
pressure = st.sidebar.slider("Давление (МПа)", 0.1, 5.0, float(PRESSURE/1e6), 0.1) * 1e6
heater_power = st.sidebar.slider("Мощность нагревателя (Вт)", 10, 200, int(HEATER_POWER), 1)
initial_temp = st.sidebar.slider("Начальная температура (K)", 100, 500, int(INITIAL_TEMP), 10)

# Параметры расчета
st.sidebar.subheader("Параметры расчета")
num_stations = st.sidebar.slider("Количество расчетных точек", 10, 50, NUM_STATIONS, 1)

# Добавьте в раздел "Параметры расчета" в боковой панели
st.sidebar.subheader("Параметры сходимости")
max_iterations = st.sidebar.slider("Максимальное число итераций", 500, 5000, 2000, 100)
tolerance = st.sidebar.slider("Допуск сходимости (K)", 0.1, 2.0, 0.5, 0.1)
relaxation_factor = st.sidebar.slider("Коэффициент релаксации", 0.01, 0.2, 0.05, 0.01)

# Расчет геометрических параметров для использования в Streamlit
chamber_inner_radius = chamber_inner_diameter / 2
chamber_outer_radius = chamber_inner_radius + chamber_wall_thickness
cooling_inner_radius = chamber_outer_radius
cooling_outer_radius = cooling_inner_radius + cooling_gap
cooling_jacket_outer_radius = cooling_outer_radius + cooling_wall_thickness

# Расчет площадей
chamber_inner_area_local = np.pi * chamber_inner_radius**2
cooling_area_local = np.pi * (cooling_outer_radius**2 - cooling_inner_radius**2)

# Кнопка для запуска расчета
run_button = st.sidebar.button("Запустить расчет")

# Таймер до выполнения
if run_button:
    # Отображение прогресса
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Таймер обратного отсчета
    countdown_time = 3  # секунды
    countdown_placeholder = st.empty()
    
    for i in range(countdown_time, 0, -1):
        countdown_placeholder.markdown(f"<h2 style='text-align: center'>Расчет начнется через {i} сек...</h2>", unsafe_allow_html=True)
        time.sleep(1)
        progress_bar.progress(int((countdown_time - i) / countdown_time * 30))
    
    countdown_placeholder.empty()
    
    # Запуск расчета
    status_text.text("Начало расчета стационарного состояния...")
    start_time = time.time()
    
    # Создаем плейсхолдер для отображения прогресса расчета
    progress_status = st.empty()
    
    # Функция для обновления прогресса расчета
    def progress_callback(iteration, max_iterations, max_diff):
        # Обновляем прогресс-бар
        progress_bar.progress(int(30 + 70 * iteration / max_iterations))
        # Обновляем статус
        elapsed_time = time.time() - start_time
        estimated_total = elapsed_time / (iteration + 1) * max_iterations
        remaining_time = estimated_total - elapsed_time
        
        progress_status.markdown(f"""
        **Прогресс расчета:**
        - Итерация: {iteration + 1} из {max_iterations}
        - Максимальное отклонение: {max_diff:.4f} K
        - Прошло времени: {elapsed_time:.1f} сек
        - Осталось примерно: {remaining_time:.1f} сек
        """)
        
        # Небольшая задержка для обновления интерфейса
        time.sleep(0.01)

    # Выполнение расчета с обратным вызовом для отображения прогресса
    T_chamber_wall, T_cooling_fluid, T_chamber_fluid, iterations_completed = steady_state_simulation(
        progress_callback=progress_callback,
        max_iterations=max_iterations,
        tolerance=tolerance,
        relaxation_factor=relaxation_factor
    )
    
    # Обновление прогресса
    progress_bar.progress(100)
    status_text.text(f"Расчет завершен за {time.time() - start_time:.2f} секунд")
    
    # Отображение результатов
    st.header("Результаты расчета")
    
    # Создание графиков
    fig = plt.figure(figsize=(12, 10))
    
    # График температур
    ax1 = fig.add_subplot(2, 2, 1)
    z_normalized = np.linspace(0, 1, len(T_chamber_wall))
    ax1.plot(z_normalized, T_chamber_wall, 'r-', label='Стенка камеры')
    ax1.plot(z_normalized, T_cooling_fluid, 'b-', label='Охлаждающая жидкость')
    ax1.plot(z_normalized, T_chamber_fluid, 'g-', label='Рабочая жидкость')
    ax1.set_xlabel('Нормализованная высота')
    ax1.set_ylabel('Температура (K)')
    ax1.set_title('Распределение температур')
    ax1.legend()
    ax1.grid(True)
    
    # Расчет скоростей
    v_cooling = np.zeros(num_stations)
    v_chamber = np.zeros(num_stations)
    
    for i in range(num_stations):
        v_cooling[i] = calculate_flow_parameters(T_cooling_fluid[i], cooling_area_local)['velocity']
        v_chamber[i] = calculate_flow_parameters(T_chamber_fluid[i], chamber_inner_area_local)['velocity']
    
    # График скоростей
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(z_normalized, v_cooling, 'b-', label='Охлаждающая жидкость')
    ax2.plot(z_normalized, v_chamber, 'g-', label='Рабочая жидкость')
    ax2.set_xlabel('Нормализованная высота')
    ax2.set_ylabel('Скорость (м/с)')
    ax2.set_title('Профили скоростей')
    ax2.legend()
    ax2.grid(True)
    
    # Расчет чисел Рейнольдса
    re_cooling = np.zeros(num_stations)
    re_chamber = np.zeros(num_stations)
    
    for i in range(num_stations):
        re_cooling[i] = calculate_flow_parameters(T_cooling_fluid[i], cooling_area_local)['reynolds']
        re_chamber[i] = calculate_flow_parameters(T_chamber_fluid[i], chamber_inner_area_local)['reynolds']
    
    # График чисел Рейнольдса
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(z_normalized, re_cooling, 'b-', label='Охлаждающая жидкость')
    ax3.plot(z_normalized, re_chamber, 'g-', label='Рабочая жидкость')
    ax3.set_xlabel('Нормализованная высота')
    ax3.set_ylabel('Число Рейнольдса')
    ax3.set_title('Распределение чисел Рейнольдса')
    ax3.legend()
    ax3.grid(True)
    
    # Расчет коэффициентов теплоотдачи
    h_cooling = np.zeros(num_stations)
    h_chamber = np.zeros(num_stations)
    
    for i in range(num_stations):
        h_cooling[i] = calculate_heat_transfer(
            T_cooling_fluid[i], 
            T_chamber_wall[i], 
            'cooling', 
            2 * cooling_gap
        )
        
        h_chamber[i] = calculate_heat_transfer(
            T_chamber_fluid[i], 
            T_chamber_wall[i], 
            'chamber', 
            chamber_inner_diameter
        )
    
    # График коэффициентов теплоотдачи
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(z_normalized, h_cooling, 'b-', label='Охлаждающая жидкость')
    ax4.plot(z_normalized, h_chamber, 'g-', label='Рабочая жидкость')
    ax4.set_xlabel('Нормализованная высота')
    ax4.set_ylabel('Коэффициент теплоотдачи (Вт/(м²·К))')
    ax4.set_title('Коэффициенты теплоотдачи')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Вывод численных данных
    st.header("6.2 Численные данные")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Максимальные температуры")
        st.info(f"Максимальная температура стенки камеры: {np.max(T_chamber_wall):.2f} K")
        st.info(f"Максимальная температура охлаждающей жидкости: {np.max(T_cooling_fluid):.2f} K")
        st.info(f"Максимальная температура рабочей жидкости: {np.max(T_chamber_fluid):.2f} K")
        st.info(f"Средняя температура стенки камеры: {np.mean(T_chamber_wall):.2f} K")
    
    with col2:
        st.subheader("Приросты температур")
        st.success(f"Прирост температуры охлаждающей жидкости: {T_cooling_fluid[-1] - T_cooling_fluid[0]:.2f} K")
        st.success(f"Прирост температуры рабочей жидкости: {T_chamber_fluid[0] - T_chamber_fluid[-1]:.2f} K")
        # Исправляем ошибку с dz - вычисляем шаг по высоте
        dz = CHAMBER_HEIGHT / (NUM_STATIONS - 1)
        st.success(f"Температурный градиент в стенке: {np.max(np.diff(T_chamber_wall)/dz):.2f} K/м")
    
    # Добавляем информацию о времени выхода на равновесную температуру
    st.subheader("Время выхода на равновесную температуру в стенке камеры нагрева")
    
    # Расчет времени выхода на равновесие (90% от максимальной разницы)
    elapsed_time = time.time() - start_time
    iterations_to_convergence = iterations_completed
    
    # Используем функцию для расчета времени выхода на равновесие
    equilibrium_times = calculate_equilibrium_time(T_chamber_wall, heater_power)
    
    st.info(f"Время расчета: {elapsed_time:.2f} секунд ({iterations_to_convergence} итераций)")
    st.info(f"Оценка времени выхода на равновесие (90%): {equilibrium_times['time_to_90_percent']:.2f} секунд")
    st.info(f"Оценка времени выхода на равновесие (99%): {equilibrium_times['time_to_99_percent']:.2f} секунд")
    st.info(f"Тепловая постоянная времени системы: {equilibrium_times['time_constant']:.2f} секунд")
    
    # Добавляем график времени выхода на равновесие
    st.subheader("График выхода на равновесную температуру")
    
    # Создаем временную шкалу
    time_scale = np.linspace(0, equilibrium_times['time_to_99_percent'] * 1.2, 100)
    
    # Рассчитываем температуру как функцию времени (модель первого порядка)
    T_initial = INITIAL_TEMP
    T_final = np.max(T_chamber_wall)
    T_time = T_initial + (T_final - T_initial) * (1 - np.exp(-time_scale / equilibrium_times['time_constant']))
    
    # Создаем график
    fig_time, ax_time = plt.subplots(figsize=(10, 6))
    ax_time.plot(time_scale, T_time, 'r-', linewidth=2)
    
    # Добавляем линии для 90% и 99% от установившегося значения
    T_90 = T_initial + 0.9 * (T_final - T_initial)
    T_99 = T_initial + 0.99 * (T_final - T_initial)
    
    ax_time.axhline(y=T_90, color='g', linestyle='--', label='90% от установившегося значения')
    ax_time.axhline(y=T_99, color='b', linestyle='--', label='99% от установившегося значения')
    
    ax_time.axvline(x=equilibrium_times['time_to_90_percent'], color='g', linestyle='--')
    ax_time.axvline(x=equilibrium_times['time_to_99_percent'], color='b', linestyle='--')
    
    # Добавляем подписи
    ax_time.set_xlabel('Время (с)')
    ax_time.set_ylabel('Температура стенки (K)')
    ax_time.set_title('Выход на равновесную температуру стенки камеры')
    ax_time.grid(True)
    ax_time.legend()
    
    # Отображаем график
    st.pyplot(fig_time)
    
    # Дополнительная информация о равновесном состоянии
    st.subheader("7. ВАЛИДАЦИЯ")
    st.write("- Контроль «физичности»")
    
    st.subheader("8. ЭТАПЫ РАЗРАБОТКИ")
    st.write("8.1 Фаза 1: Стационарная модель")
    st.write("- Базовые расчеты без учета изменения свойств")
    
    # Создание вкладки для визуализации температурного поля
    st.header("Пример градиента, который нужно вывести")
    st.write("""
    Ниже представлена визуализация распределения температуры и числа Маха в резистоджете, 
    аналогично примеру из документации.
    """)

    # Создаем одну фигуру с двумя подграфиками (один над другим)
    fig_combined, (ax_temp_field, ax_mach_field) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Создание осесимметричной модели с формой, более похожей на пример из PDF
    # Изменяем форму камеры, чтобы она была похожа на сопло
    r_points = np.linspace(0, cooling_jacket_outer_radius*1.2, 100)
    z_points = np.linspace(0, chamber_height, 200)
    
    # Создаем профиль сопла (упрощенно)
    nozzle_profile = np.zeros_like(z_points)
    for i, z in enumerate(z_points):
        # Сужение в середине (примерно)
        if z < chamber_height * 0.4:
            # Постепенное сужение
            nozzle_profile[i] = chamber_inner_radius * (1 - 0.3 * (z / (chamber_height * 0.4)))
        else:
            # Расширение после сужения
            nozzle_profile[i] = chamber_inner_radius * 0.7 + chamber_inner_radius * 0.8 * ((z - chamber_height * 0.4) / (chamber_height * 0.6))
    
    # Создаем сетку
    r_mesh, z_mesh = np.meshgrid(r_points, z_points)

    # Создание массива температур
    T_field = np.zeros_like(r_mesh)

    # Интерполяция температур для плавного перехода
    from scipy.interpolate import interp1d

    # Создание интерполяционных функций
    z_norm = np.linspace(0, 1, len(T_chamber_wall))
    T_wall_interp = interp1d(z_norm, T_chamber_wall, kind='cubic', fill_value='extrapolate')
    T_cooling_interp = interp1d(z_norm, T_cooling_fluid, kind='cubic', fill_value='extrapolate')
    T_chamber_interp = interp1d(z_norm, T_chamber_fluid, kind='cubic', fill_value='extrapolate')

    # Заполнение массива температур с учетом формы сопла
    for i in range(len(z_points)):
        z_pos = z_points[i]
        z_normalized = z_pos / chamber_height
        nozzle_r = nozzle_profile[i]
        
        for j in range(len(r_points)):
            r_pos = r_points[j]
            
            if r_pos < nozzle_r:
                # Внутри камеры (рабочая жидкость)
                # Градиент температуры от входа к выходу
                T_field[i, j] = T_chamber_interp(z_normalized)
            elif r_pos < nozzle_r + chamber_wall_thickness:
                # Стенка камеры
                T_field[i, j] = T_wall_interp(z_normalized)
            elif r_pos < nozzle_r + chamber_wall_thickness + cooling_gap:
                # Охлаждающая жидкость
                T_field[i, j] = T_cooling_interp(z_normalized)
            elif r_pos < nozzle_r + chamber_wall_thickness + cooling_gap + cooling_wall_thickness:
                # Внешняя стенка
                T_field[i, j] = T_cooling_interp(z_normalized) * 0.9  # Немного холоднее
            else:
                # За пределами двигателя - комнатная температура
                T_field[i, j] = INITIAL_TEMP

    # Создаем цветовую карту для температуры, соответствующую изображению из PDF
    temp_colors = [
        (0, 'rgb(0, 0, 128)'),      # темно-синий для T=300K
        (0.1, 'rgb(0, 0, 255)'),    # синий для T=600K
        (0.2, 'rgb(0, 128, 255)'),  # голубой для T=900K
        (0.3, 'rgb(0, 255, 255)'),  # циан для T=1200K
        (0.4, 'rgb(0, 255, 128)'),  # сине-зеленый для T=1500K
        (0.5, 'rgb(0, 255, 0)'),    # зеленый для T=1800K
        (0.6, 'rgb(128, 255, 0)'),  # желто-зеленый для T=2100K
        (0.7, 'rgb(255, 255, 0)'),  # желтый для T=2400K
        (0.8, 'rgb(255, 128, 0)'),  # оранжевый для T=2700K
        (0.9, 'rgb(255, 0, 0)'),    # красный для T=3000K
        (1.0, 'rgb(128, 0, 0)')     # темно-красный для T=3300K
    ]
    temp_cmap = LinearSegmentedColormap.from_list('temp_map', temp_colors, N=100)

    # Определение диапазона температур
    T_min = np.min(T_field)
    T_max = np.max(T_field)
    
    # Создаем шкалу температур как на примере из PDF
    temp_ticks = np.linspace(1200, 3600, 13)  # 13 делений от 1200K до 3600K с шагом 200K
    
    # Нормализуем диапазон температур для соответствия шкале
    temp_norm = plt.Normalize(1200, 3600)
    
    # Создание контурного графика температуры
    temp_contour = ax_temp_field.contourf(z_mesh*1000, r_mesh*1000, T_field, 
                                         levels=50, cmap=temp_cmap, norm=temp_norm)

    # Добавление контуров для визуализации границ сопла
    # Рисуем профиль сопла
    nozzle_inner_x = z_points * 1000
    nozzle_inner_y = nozzle_profile * 1000
    nozzle_outer_x = z_points * 1000
    nozzle_outer_y = (nozzle_profile + chamber_wall_thickness) * 1000
    cooling_outer_x = z_points * 1000
    cooling_outer_y = (nozzle_profile + chamber_wall_thickness + cooling_gap) * 1000
    jacket_outer_x = z_points * 1000
    jacket_outer_y = (nozzle_profile + chamber_wall_thickness + cooling_gap + cooling_wall_thickness) * 1000
    
    # Рисуем верхнюю половину профиля для температуры
    ax_temp_field.plot(nozzle_inner_x, nozzle_inner_y, 'k-', linewidth=1.5)
    ax_temp_field.plot(nozzle_outer_x, nozzle_outer_y, 'k-', linewidth=1.5)
    ax_temp_field.plot(cooling_outer_x, cooling_outer_y, 'k-', linewidth=1.5)
    ax_temp_field.plot(jacket_outer_x, jacket_outer_y, 'k-', linewidth=1.5)
    
    # Рисуем нижнюю половину профиля (отражение) для температуры
    ax_temp_field.plot(nozzle_inner_x, -nozzle_inner_y, 'k-', linewidth=1.5)
    ax_temp_field.plot(nozzle_outer_x, -nozzle_outer_y, 'k-', linewidth=1.5)
    ax_temp_field.plot(cooling_outer_x, -cooling_outer_y, 'k-', linewidth=1.5)
    ax_temp_field.plot(jacket_outer_x, -jacket_outer_y, 'k-', linewidth=1.5)
    
    # Добавление контурных линий температуры
    temp_contour_lines = ax_temp_field.contour(z_mesh*1000, r_mesh*1000, T_field, 
                                              levels=20, colors='black', linewidths=0.5, alpha=0.7)
    
    # Добавление подписей к контурам температуры
    plt.clabel(temp_contour_lines, inline=True, fontsize=8, fmt='%.0f')

    # Настройка осей для температуры
    ax_temp_field.set_xlabel('Высота (мм)')
    ax_temp_field.set_ylabel('Радиус (мм)')
    ax_temp_field.set_title('T (K)')
    
    # Устанавливаем симметричные пределы по оси Y для температуры
    y_max = max(np.max(jacket_outer_y), np.max(r_mesh*1000)) * 1.1
    ax_temp_field.set_ylim(-y_max, y_max)
    
    # Добавление цветовой шкалы с делениями как на примере для температуры
    cbar = fig_combined.colorbar(temp_contour, ax=ax_temp_field, ticks=temp_ticks)
    cbar.set_label('Температура (K)')
    
    # Добавляем подписи к частям двигателя для температуры
    ax_temp_field.text(chamber_height*1000*0.1, 0, 
                      'Рабочая\nжидкость', ha='center', va='center', fontsize=10)
    ax_temp_field.text(chamber_height*1000*0.1, 
                      (nozzle_profile[20] + chamber_wall_thickness/2)*1000, 
                      'Стенка', ha='center', va='center', fontsize=10)
    ax_temp_field.text(chamber_height*1000*0.1, 
                      (nozzle_profile[20] + chamber_wall_thickness + cooling_gap/2)*1000, 
                      'Охлаждение', ha='center', va='center', fontsize=10)
    
    # Добавляем стрелки для указания направления потока для температуры
    # Стрелка для рабочей жидкости (сверху вниз)
    ax_temp_field.arrow(chamber_height*1000*0.8, nozzle_profile[160]*1000*0.5, 
                       -chamber_height*1000*0.2, 0, 
                       head_width=2, head_length=5, fc='white', ec='black', linewidth=1)
    
    # Стрелка для охлаждающей жидкости (снизу вверх)
    ax_temp_field.arrow(chamber_height*1000*0.2, 
                       (nozzle_profile[40] + chamber_wall_thickness + cooling_gap/2)*1000, 
                       chamber_height*1000*0.2, 0, 
                       head_width=2, head_length=5, fc='white', ec='black', linewidth=1)
    
    # Добавление подписи "fuel cooling injector" как на примере
    ax_temp_field.text(chamber_height*1000*0.05, 
                      (nozzle_profile[10] + chamber_wall_thickness + cooling_gap + cooling_wall_thickness)*1000*1.1, 
                      'fuel cooling injector', ha='left', va='center', fontsize=10)
    
    # Добавление угла расширения сопла как на примере
    ax_temp_field.text(chamber_height*1000*0.95, 
                      (nozzle_profile[-1] + chamber_wall_thickness + cooling_gap + cooling_wall_thickness)*1000*1.2, 
                      'ε = 12', ha='right', va='center', fontsize=10)
    
    # Создание массива чисел Маха
    mach_field = np.zeros_like(r_mesh)
    
    # Расчет скорости звука и числа Маха
    def calculate_mach_number(T, velocity):
        # Скорость звука в водороде (м/с)
        # c = sqrt(gamma * R * T / M)
        gamma = 1.4  # Показатель адиабаты для водорода
        R = 8.314  # Универсальная газовая постоянная (Дж/(моль·К))
        M = 0.002016  # Молярная масса водорода (кг/моль)
        
        sound_speed = np.sqrt(gamma * R * T / M)
        mach = velocity / sound_speed
        return mach
    
    # Заполнение массива чисел Маха
    for i in range(len(z_points)):
        z_pos = z_points[i]
        z_normalized = z_pos / chamber_height
        nozzle_r = nozzle_profile[i]
        
        # Расчет скорости и числа Маха для рабочей жидкости
        if i < len(T_chamber_fluid):
            chamber_velocity = calculate_flow_parameters(T_chamber_fluid[i], np.pi * nozzle_r**2)['velocity']
            chamber_mach = calculate_mach_number(T_chamber_fluid[i], chamber_velocity)
        else:
            chamber_mach = 0
            
        for j in range(len(r_points)):
            r_pos = r_points[j]
            
            if r_pos < nozzle_r:
                # Внутри камеры (рабочая жидкость)
                # Число Маха увеличивается к выходу сопла
                if z_pos < chamber_height * 0.4:
                    # До сужения - низкое число Маха
                    mach_field[i, j] = chamber_mach * (1 + z_pos / (chamber_height * 0.4))
                else:
                    # После сужения - быстрый рост числа Маха
                    mach_field[i, j] = chamber_mach * 2 + chamber_mach * 5 * ((z_pos - chamber_height * 0.4) / (chamber_height * 0.6))
            else:
                # За пределами камеры - нет потока
                mach_field[i, j] = 0
    
    # Создаем цветовую карту для числа Маха, соответствующую изображению из PDF
    mach_colors = [
        (0, 'rgb(0, 0, 128)'),      # темно-синий для M=0
        (0.1, 'rgb(0, 0, 255)'),    # синий для M=0.3
        (0.2, 'rgb(0, 128, 255)'),  # голубой для M=0.7
        (0.3, 'rgb(0, 255, 255)'),  # циан для M=1.1
        (0.4, 'rgb(0, 255, 128)'),  # сине-зеленый для M=1.5
        (0.5, 'rgb(0, 255, 0)'),    # зеленый для M=1.9
        (0.6, 'rgb(128, 255, 0)'),  # желто-зеленый для M=2.3
        (0.7, 'rgb(255, 255, 0)'),  # желтый для M=2.7
        (0.8, 'rgb(255, 128, 0)'),  # оранжевый для M=3.1
        (1.0, 'rgb(255, 0, 0)')     # красный для M=3.5+
    ]
    mach_cmap = LinearSegmentedColormap.from_list('mach_map', mach_colors, N=100)

    # Создаем шкалу чисел Маха как на примере из PDF
    mach_ticks = np.linspace(0, 3.7, 10)  # 10 делений от 0 до 3.7
    
    # Нормализуем диапазон чисел Маха для соответствия шкале
    mach_norm = plt.Normalize(0, 3.7)
    
    # Создание контурного графика числа Маха
    mach_contour = ax_mach_field.contourf(z_mesh*1000, r_mesh*1000, mach_field,
                                         levels=50, cmap=mach_cmap, norm=mach_norm)
    
    # Рисуем профиль сопла для числа Маха
    ax_mach_field.plot(nozzle_inner_x, nozzle_inner_y, 'k-', linewidth=1.5)
    ax_mach_field.plot(nozzle_outer_x, nozzle_outer_y, 'k-', linewidth=1.5)
    ax_mach_field.plot(cooling_outer_x, cooling_outer_y, 'k-', linewidth=1.5)
    ax_mach_field.plot(jacket_outer_x, jacket_outer_y, 'k-', linewidth=1.5)
    
    # Рисуем нижнюю половину профиля (отражение) для числа Маха
    ax_mach_field.plot(nozzle_inner_x, -nozzle_inner_y, 'k-', linewidth=1.5)
    ax_mach_field.plot(nozzle_outer_x, -nozzle_outer_y, 'k-', linewidth=1.5)
    ax_mach_field.plot(cooling_outer_x, -cooling_outer_y, 'k-', linewidth=1.5)
    ax_mach_field.plot(jacket_outer_x, -jacket_outer_y, 'k-', linewidth=1.5)
    
    # Добавление контурных линий числа Маха
    mach_contour_lines = ax_mach_field.contour(z_mesh*1000, r_mesh*1000, mach_field, 
                                              levels=15, colors='black', linewidths=0.5, alpha=0.7)
    
    # Добавление подписей к контурам числа Маха
    plt.clabel(mach_contour_lines, inline=True, fontsize=8, fmt='%.1f')
    
    # Добавляем подпись "ε = 12" в правом верхнем углу графика числа Маха
    ax_mach_field.text(chamber_height*1000*0.95, 
                      (nozzle_profile[-1] + chamber_wall_thickness + cooling_gap + cooling_wall_thickness)*1000*1.2, 
                      'ε = 12', ha='right', va='center', fontsize=10)
    
    # Настройка осей для числа Маха
    ax_mach_field.set_xlabel('Высота (мм)')
    ax_mach_field.set_ylabel('Радиус (мм)')
    ax_mach_field.set_title('Mach')
    
    # Устанавливаем симметричные пределы по оси Y для числа Маха
    ax_mach_field.set_ylim(-y_max, y_max)
    
    # Добавление цветовой шкалы с делениями для числа Маха
    mach_cbar = fig_combined.colorbar(mach_contour, ax=ax_mach_field, ticks=mach_ticks)
    mach_cbar.set_label('Число Маха')
    
    # Настройка общего вида графика
    plt.tight_layout()
    
    # Отображение графика
    st.pyplot(fig_combined)
    
    # Добавление пояснения
    st.write("""
    На верхнем графике показано распределение температуры в различных частях резистоджета:
    - **Синий цвет**: низкие температуры (около {:.0f} K)
    - **Зеленый цвет**: средние температуры
    - **Желтый/оранжевый цвет**: высокие температуры
    - **Красный цвет**: максимальные температуры (около {:.0f} K)
    
    На нижнем графике показано распределение числа Маха:
    - **Синий цвет**: дозвуковой поток (M < 1)
    - **Зеленый цвет**: околозвуковой поток (M ≈ 1)
    - **Желтый/оранжевый цвет**: сверхзвуковой поток (1 < M < 3)
    - **Красный цвет**: высокоскоростной сверхзвуковой поток (M > 3)
    
    Черные линии показывают контуры равных значений, что позволяет лучше видеть градиенты.
    """.format(T_min, T_max))

    # Визуализация скоростей потока
    st.header("Визуализация скоростей потока")
    st.write("""
    Ниже представлена визуализация скоростей потока в резистоджете.
    """)

    # Создание сетки для визуализации скоростей
    fig_vel_field, ax_vel_field = plt.subplots(figsize=(12, 6))

    # Создание векторного поля скоростей
    v_field_r = np.zeros_like(r_mesh)
    v_field_z = np.zeros_like(z_mesh)

    # Заполнение поля скоростей
    for i in range(len(z_points)):
        z_pos = z_points[i]
        z_normalized = z_pos / chamber_height
        
        for j in range(len(r_points)):
            r_pos = r_points[j]
            
            if r_pos < chamber_inner_radius:
                # Внутри камеры (рабочая жидкость) - движение сверху вниз
                v_z = -T_chamber_interp(z_normalized) / T_chamber_interp(0.5) * 2  # Нормализованная скорость
                v_field_z[i, j] = v_z
                v_field_r[i, j] = 0  # Радиальная составляющая близка к нулю
                
            elif r_pos < chamber_outer_radius:
                # Стенка камеры - нет движения
                v_field_z[i, j] = 0
                v_field_r[i, j] = 0
                
            elif r_pos < cooling_outer_radius:
                # Охлаждающая жидкость - движение снизу вверх
                v_z = T_cooling_interp(z_normalized) / T_cooling_interp(0.5) * 2  # Нормализованная скорость
                v_field_z[i, j] = v_z
                v_field_r[i, j] = 0  # Радиальная составляющая близка к нулю
                
            else:
                # Внешняя стенка - нет движения
                v_field_z[i, j] = 0
                v_field_r[i, j] = 0

    # Расчет величины скорости
    v_magnitude = np.sqrt(v_field_r**2 + v_field_z**2)

    # Создание цветовой карты для скоростей
    speed_colors = [(0, 0, 0.5),  # темно-синий для низких скоростей
                    (0, 0, 1),    # синий
                    (0, 0.5, 1),  # голубой
                    (0, 1, 1),    # циан
                    (0, 1, 0.5),  # сине-зеленый
                    (0, 1, 0),    # зеленый
                    (0.5, 1, 0),  # желто-зеленый
                    (1, 1, 0),    # желтый
                    (1, 0.5, 0),  # оранжевый
                    (1, 0, 0)]    # красный для высоких скоростей

    speed_cmap = LinearSegmentedColormap.from_list('speed_map', speed_colors, N=100)

    # Создание контурного графика скоростей
    contour_vel = ax_vel_field.contourf(z_mesh*1000, r_mesh*1000, v_magnitude, 
                                       levels=50, cmap=speed_cmap)

    # Добавление контуров для визуализации границ
    ax_vel_field.axhline(y=chamber_inner_radius*1000, color='black', linestyle='-', linewidth=1)
    ax_vel_field.axhline(y=chamber_outer_radius*1000, color='black', linestyle='-', linewidth=1)
    ax_vel_field.axhline(y=cooling_outer_radius*1000, color='black', linestyle='-', linewidth=1)
    ax_vel_field.axhline(y=cooling_jacket_outer_radius*1000, color='black', linestyle='-', linewidth=1)

    # Добавление векторов скорости
    # Прореживание для лучшей визуализации
    skip = 10
    ax_vel_field.quiver(z_mesh[::skip, ::skip]*1000, r_mesh[::skip, ::skip]*1000, 
                       v_field_z[::skip, ::skip], v_field_r[::skip, ::skip],
                       color='white', scale=30, width=0.002, headwidth=3, headlength=4)

    # Настройка осей
    ax_vel_field.set_xlabel('Высота (мм)')
    ax_vel_field.set_ylabel('Радиус (мм)')
    ax_vel_field.set_title('Распределение скоростей потока в резистоджете')

    # Добавление цветовой шкалы
    cbar_vel = fig_vel_field.colorbar(contour_vel, ax=ax_vel_field)
    cbar_vel.set_label('Относительная скорость')

    # Отображение графика
    st.pyplot(fig_vel_field)

    # Добавление пояснения
    st.write("""
    На графике показано распределение скоростей потока:
    - Белые стрелки показывают направление потока
    - Цветовая шкала показывает относительную величину скорости
    - В камере нагрева рабочая жидкость движется сверху вниз
    - В охлаждающей рубашке охлаждающая жидкость движется снизу вверх
    """)

    # Экспорт данных
    st.subheader("Экспорт данных")
    
    # Создание словаря с данными
    export_data = {
        'z_normalized': z_normalized.tolist(),
        'T_chamber_wall': T_chamber_wall.tolist(),
        'T_cooling_fluid': T_cooling_fluid.tolist(),
        'T_chamber_fluid': T_chamber_fluid.tolist(),
        'v_cooling': v_cooling.tolist(),
        'v_chamber': v_chamber.tolist(),
        're_cooling': re_cooling.tolist(),
        're_chamber': re_chamber.tolist(),
        'h_cooling': h_cooling.tolist(),
        'h_chamber': h_chamber.tolist()
    }
    
    import json
    export_json = json.dumps(export_data)
    
    st.download_button(
        label="Скачать результаты (JSON)",
        data=export_json,
        file_name="resistojet_results.json",
        mime="application/json"
    )

else:
    # Отображение схемы резистоджета
    st.header("Схема резистоджета")
    
    # Создание схематического изображения
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    
    # Параметры для рисования
    chamber_inner_r = CHAMBER_INNER_DIAMETER / 2 * 1000  # мм
    chamber_outer_r = chamber_inner_r + CHAMBER_WALL_THICKNESS * 1000  # мм
    cooling_outer_r = chamber_outer_r + COOLING_GAP * 1000  # мм
    jacket_outer_r = cooling_outer_r + COOLING_WALL_THICKNESS * 1000  # мм
    height = CHAMBER_HEIGHT * 1000  # мм
    
    # Создание черной рамки
    border_width = 5
    border_margin = 20
    ax.add_patch(plt.Rectangle(
        (-jacket_outer_r-border_margin, -border_margin), 
        2*jacket_outer_r+2*border_margin, 
        height+2*border_margin,
        linewidth=border_width, 
        edgecolor='black', 
        facecolor='white'
    ))

    # Рисование камеры (черные линии)
    ax.plot([-chamber_inner_r, -chamber_inner_r], [0, height], 'k-', linewidth=3)
    ax.plot([chamber_inner_r, chamber_inner_r], [0, height], 'k-', linewidth=3)

    # Рисование стенки камеры (черные линии)
    ax.plot([-chamber_outer_r, -chamber_outer_r], [0, height], 'k-', linewidth=3)
    ax.plot([chamber_outer_r, chamber_outer_r], [0, height], 'k-', linewidth=3)

    # Рисование охлаждающей рубашки (черные линии)
    ax.plot([-cooling_outer_r, -cooling_outer_r], [0, height], 'k-', linewidth=3)
    ax.plot([cooling_outer_r, cooling_outer_r], [0, height], 'k-', linewidth=3)

    # Рисование внешней стенки (черные линии)
    ax.plot([-jacket_outer_r, -jacket_outer_r], [0, height], 'k-', linewidth=3)
    ax.plot([jacket_outer_r, jacket_outer_r], [0, height], 'k-', linewidth=3)

    # Соединение верхних и нижних частей
    ax.plot([-chamber_inner_r, chamber_inner_r], [0, 0], 'k-', linewidth=3)
    ax.plot([-chamber_inner_r, chamber_inner_r], [height, height], 'k-', linewidth=3)
    ax.plot([-chamber_outer_r, chamber_outer_r], [0, 0], 'k-', linewidth=3)
    ax.plot([-chamber_outer_r, chamber_outer_r], [height, height], 'k-', linewidth=3)
    ax.plot([-cooling_outer_r, cooling_outer_r], [0, 0], 'k-', linewidth=3)
    ax.plot([-cooling_outer_r, cooling_outer_r], [height, height], 'k-', linewidth=3)
    ax.plot([-jacket_outer_r, jacket_outer_r], [0, 0], 'k-', linewidth=3)
    ax.plot([-jacket_outer_r, jacket_outer_r], [height, height], 'k-', linewidth=3)

    # Добавление желтого прямоугольника для нагревателя
    heater_width = chamber_inner_r * 0.8
    heater_height = height * 0.6
    ax.add_patch(plt.Rectangle(
        (-heater_width/2, height*0.2), 
        heater_width, 
        heater_height,
        linewidth=0, 
        facecolor='#FFFF80'  # Светло-желтый цвет
    ))

    # Добавление текста "H2"
    ax.text(0, height*0.8, "H2", 
            ha='center', va='center', fontsize=18, color='black')

    # Настройка осей
    ax.set_xlim(-jacket_outer_r*1.5, jacket_outer_r*1.5)
    ax.set_ylim(-height*0.1, height*1.2)
    ax.axis('off')  # Отключение осей

    st.pyplot(fig)
    
    # Описание модели
    st.header("Описание модели")
    st.markdown("""
    ### Конструкция резистоджета:
    
    - **Нагревательная камера (внутренний цилиндр):**
      - Высота: 40 мм
      - Внутренний диаметр: 20 мм
      - Толщина стенки: 1 мм
      - Материал: Inconel 718 SLM
      - Начальная температура: 300K
    
    - **Охлаждающая рубашка (внешний цилиндр):**
      - Концентрический с внутренним цилиндром
      - Между стенками цилиндров 1 мм
      - Толщина стенки: 1 мм
      - Противоточная схема – газ поднимается от высоты = 0 мм до высоты 40 мм, а потом разворачивается и заходит во внутренний цилиндр где дальше греется.
    
    ### Рабочие параметры:
    
    - Рабочее тело: водород
    - Массовый расход: 0.000005 кг/с
    - Давление: 1 МПа
    - Мощность нагревателя: 70 Вт
    - Начальная температура: 300K
    
    ### Что рассчитывает модель:
    
    1. **Гидродинамика:**
       - Скорости потока
       - Числа Рейнольдса
    
    2. **Теплообмен:**
       - Конвективный теплообмен между газом и стенками
       - Теплопроводность через стенку
    
    3. **Равновесное состояние:**
       - Температура стенок нагревательной камеры
       - Прирост температуры охлаждающего газа
    
    Используйте панель слева для изменения параметров модели и нажмите "Запустить расчет" для получения результатов.
    """)

# Информация о проекте
st.sidebar.markdown("---")
st.sidebar.info("""
**О проекте**

Модель разработана для симуляции тепловых процессов в резистивном ракетном двигателе с регенеративным охлаждением.

Автор: [Ваше имя]
""") 