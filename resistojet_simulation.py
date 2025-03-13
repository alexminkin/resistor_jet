import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import time

# Константы
NUM_STATIONS = 20  # Количество расчетных точек
CHAMBER_HEIGHT = 0.04  # м
CHAMBER_INNER_DIAMETER = 0.02  # м
CHAMBER_WALL_THICKNESS = 0.001  # м
COOLING_GAP = 0.001  # м
COOLING_WALL_THICKNESS = 0.001  # м
MASS_FLOW_RATE = 0.000005  # кг/с
PRESSURE = 1e6  # Па (1 МПа)
HEATER_POWER = 70  # Вт
INITIAL_TEMP = 300  # K
CHAMBER_ROUGHNESS = 0.8e-6  # м
COOLING_ROUGHNESS = 50e-6  # м (типичное значение для необработанного металла после SLM)

# Геометрические параметры
chamber_inner_radius = CHAMBER_INNER_DIAMETER / 2
chamber_outer_radius = chamber_inner_radius + CHAMBER_WALL_THICKNESS
cooling_inner_radius = chamber_outer_radius
cooling_outer_radius = cooling_inner_radius + COOLING_GAP
cooling_jacket_outer_radius = cooling_outer_radius + COOLING_WALL_THICKNESS

chamber_inner_area = np.pi * chamber_inner_radius**2
cooling_area = np.pi * (cooling_outer_radius**2 - cooling_inner_radius**2)

# Расчетные точки по высоте
z_points = np.linspace(0, CHAMBER_HEIGHT, NUM_STATIONS)
dz = CHAMBER_HEIGHT / (NUM_STATIONS - 1)

# 1. Модуль свойств материалов
def hydrogen_properties(T):
    """
    Возвращает свойства водорода в зависимости от температуры.
    Интерполяция данных для диапазона 300K-4000K.
    
    Параметры:
    T : float или array_like
        Температура в Кельвинах
    
    Возвращает:
    dict: Словарь со свойствами (плотность, теплоемкость, теплопроводность, вязкость)
    """
    # Табличные данные для водорода (примерные значения)
    T_data = np.array([300, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000])
    
    # Плотность (кг/м³) при давлении 1 МПа
    rho_data = np.array([0.8, 0.48, 0.24, 0.16, 0.12, 0.096, 0.08, 0.069, 0.06])
    
    # Теплоемкость (Дж/(кг·К))
    cp_data = np.array([14300, 14500, 15000, 15500, 16000, 16300, 16500, 16700, 16800])
    
    # Теплопроводность (Вт/(м·К))
    k_data = np.array([0.18, 0.26, 0.42, 0.52, 0.60, 0.65, 0.72, 0.78, 0.84])
    
    # Динамическая вязкость (Па·с)
    mu_data = np.array([8.9e-6, 12.2e-6, 19.9e-6, 26.0e-6, 31.0e-6, 35.0e-6, 38.5e-6, 41.5e-6, 44.0e-6])
    
    # Создание интерполяционных функций
    rho_interp = interp1d(T_data, rho_data, kind='cubic', fill_value='extrapolate')
    cp_interp = interp1d(T_data, cp_data, kind='cubic', fill_value='extrapolate')
    k_interp = interp1d(T_data, k_data, kind='cubic', fill_value='extrapolate')
    mu_interp = interp1d(T_data, mu_data, kind='cubic', fill_value='extrapolate')
    
    # Расчет свойств для заданной температуры
    rho = rho_interp(T)
    cp = cp_interp(T)
    k = k_interp(T)
    mu = mu_interp(T)
    
    return {
        'density': rho,
        'specific_heat': cp,
        'thermal_conductivity': k,
        'viscosity': mu
    }

def inconel_properties(T):
    """
    Возвращает свойства Inconel 718 в зависимости от температуры.
    Интерполяция данных для диапазона 300K-1300K.
    
    Параметры:
    T : float или array_like
        Температура в Кельвинах
    
    Возвращает:
    dict: Словарь со свойствами (теплопроводность, теплоемкость)
    """
    # Табличные данные для Inconel 718 (примерные значения)
    T_data = np.array([300, 500, 700, 900, 1100, 1300])
    
    # Теплопроводность (Вт/(м·К))
    k_data = np.array([11.4, 13.8, 16.2, 18.6, 21.0, 23.4])
    
    # Теплоемкость (Дж/(кг·К))
    cp_data = np.array([435, 460, 485, 510, 535, 560])
    
    # Плотность (кг/м³) - примерно постоянная
    rho = 8190
    
    # Создание интерполяционных функций
    k_interp = interp1d(T_data, k_data, kind='cubic', fill_value='extrapolate')
    cp_interp = interp1d(T_data, cp_data, kind='cubic', fill_value='extrapolate')
    
    # Расчет свойств для заданной температуры
    k = k_interp(T)
    cp = cp_interp(T)
    
    return {
        'thermal_conductivity': k,
        'specific_heat': cp,
        'density': rho
    }

# 2. Гидродинамический модуль
def calculate_flow_parameters(T, area, mass_flow=MASS_FLOW_RATE, pressure=PRESSURE):
    """
    Расчет параметров потока.
    
    Параметры:
    T : float или array_like
        Температура газа в Кельвинах
    area : float
        Площадь поперечного сечения в м²
    mass_flow : float
        Массовый расход в кг/с
    pressure : float
        Давление в Па
    
    Возвращает:
    dict: Словарь с параметрами потока (скорость, число Рейнольдса)
    """
    props = hydrogen_properties(T)
    
    # Расчет скорости
    velocity = mass_flow / (props['density'] * area)
    
    # Расчет гидравлического диаметра
    if area == chamber_inner_area:
        hydraulic_diameter = CHAMBER_INNER_DIAMETER
        roughness = CHAMBER_ROUGHNESS
    else:  # cooling area
        hydraulic_diameter = 2 * COOLING_GAP
        roughness = COOLING_ROUGHNESS
    
    # Расчет числа Рейнольдса
    reynolds = props['density'] * velocity * hydraulic_diameter / props['viscosity']
    
    # Расчет коэффициента трения (для ламинарного потока)
    if reynolds < 2300:
        friction_factor = 64 / reynolds
    else:
        # Уравнение Блазиуса для турбулентного потока
        friction_factor = 0.316 * reynolds**(-0.25)
    
    # Коррекция для шероховатости (уравнение Колбрука-Уайта)
    if reynolds > 4000:
        def colebrook_white(f):
            return 1/np.sqrt(f) + 2*np.log10(roughness/(3.7*hydraulic_diameter) + 2.51/(reynolds*np.sqrt(f)))
        
        # Итерационное решение
        f_old = friction_factor
        for _ in range(10):
            f_new = (1/(-2*np.log10(roughness/(3.7*hydraulic_diameter) + 2.51/(reynolds*np.sqrt(f_old)))))**2
            if abs(f_new - f_old) < 1e-6:
                break
            f_old = f_new
        
        friction_factor = f_old
    
    return {
        'velocity': velocity,
        'reynolds': reynolds,
        'friction_factor': friction_factor
    }

# 3. Тепловой модуль
def calculate_heat_transfer(T_fluid, T_wall, area_type, hydraulic_diameter):
    """
    Расчет коэффициента теплоотдачи.
    
    Параметры:
    T_fluid : float
        Температура жидкости в Кельвинах
    T_wall : float
        Температура стенки в Кельвинах
    area_type : str
        Тип области ('chamber' или 'cooling')
    hydraulic_diameter : float
        Гидравлический диаметр в м
    
    Возвращает:
    float: Коэффициент теплоотдачи в Вт/(м²·К)
    """
    props = hydrogen_properties(T_fluid)
    
    if area_type == 'chamber':
        area = chamber_inner_area
        roughness = CHAMBER_ROUGHNESS
    else:  # cooling
        area = cooling_area
        roughness = COOLING_ROUGHNESS
    
    flow_params = calculate_flow_parameters(T_fluid, area)
    reynolds = flow_params['reynolds']
    
    # Расчет числа Прандтля
    prandtl = props['specific_heat'] * props['viscosity'] / props['thermal_conductivity']
    
    # Расчет числа Нуссельта
    if reynolds < 2300:  # Ламинарный поток
        # Уравнение для ламинарного потока в трубе
        nusselt = 3.66 + (0.0668 * (hydraulic_diameter / CHAMBER_HEIGHT) * reynolds * prandtl) / (1 + 0.04 * ((hydraulic_diameter / CHAMBER_HEIGHT) * reynolds * prandtl)**(2/3))
    else:  # Турбулентный поток
        # Уравнение Диттуса-Бёлтера
        nusselt = 0.023 * reynolds**0.8 * prandtl**0.4
    
    # Коэффициент теплоотдачи
    h = nusselt * props['thermal_conductivity'] / hydraulic_diameter
    
    return h

# 4. Модель стационарного состояния
def steady_state_simulation(progress_callback=None, max_iterations=2000, tolerance=0.5, relaxation_factor=0.05):
    """
    Расчет стационарного состояния системы.
    
    Параметры:
    progress_callback : function, optional
        Функция обратного вызова для отображения прогресса расчета
    max_iterations : int, optional
        Максимальное количество итераций (по умолчанию 2000)
    tolerance : float, optional
        Допуск для сходимости в Кельвинах (по умолчанию 0.5)
    relaxation_factor : float, optional
        Коэффициент релаксации для улучшения сходимости (по умолчанию 0.05)
    
    Возвращает:
    tuple: (T_chamber_wall, T_cooling_fluid, T_chamber_fluid, iterations_completed)
        Температуры стенки камеры, охлаждающей жидкости, рабочей жидкости и количество выполненных итераций
    """
    # Начальные условия
    T_cooling_fluid = np.ones(NUM_STATIONS) * INITIAL_TEMP
    T_chamber_wall = np.ones(NUM_STATIONS) * INITIAL_TEMP
    T_chamber_fluid = np.ones(NUM_STATIONS) * INITIAL_TEMP
    
    # Распределение мощности нагревателя по станциям
    heater_power_distribution = np.ones(NUM_STATIONS) * HEATER_POWER / NUM_STATIONS
    
    # Инициализация переменной для отслеживания предыдущего отклонения
    prev_max_diff = float('inf')
    
    for iteration in range(max_iterations):
        # Сохраняем предыдущие значения для проверки сходимости
        T_chamber_wall_prev = T_chamber_wall.copy()
        
        # 1. Расчет для охлаждающей жидкости (снизу вверх)
        for i in range(NUM_STATIONS-1):
            # Свойства жидкости
            props_cooling = hydrogen_properties(T_cooling_fluid[i])
            
            # Коэффициент теплоотдачи
            h_cooling = calculate_heat_transfer(
                T_cooling_fluid[i], 
                T_chamber_wall[i], 
                'cooling', 
                2 * COOLING_GAP
            )
            
            # Площадь теплообмена для участка
            cooling_heat_transfer_area = 2 * np.pi * chamber_outer_radius * (CHAMBER_HEIGHT / NUM_STATIONS)
            
            # Тепловой поток от стенки к охлаждающей жидкости
            q_cooling = h_cooling * cooling_heat_transfer_area * (T_chamber_wall[i] - T_cooling_fluid[i])
            
            # Изменение температуры охлаждающей жидкости
            dT_cooling = q_cooling / (MASS_FLOW_RATE * props_cooling['specific_heat'])
            
            # Обновление температуры охлаждающей жидкости
            T_cooling_fluid[i+1] = T_cooling_fluid[i] + dT_cooling
        
        # 2. Расчет для рабочей жидкости в камере (сверху вниз)
        # Начальная температура рабочей жидкости равна конечной температуре охлаждающей жидкости
        T_chamber_fluid[-1] = T_cooling_fluid[-1]
        
        for i in range(NUM_STATIONS-1, 0, -1):
            # Свойства рабочей жидкости
            props_chamber = hydrogen_properties(T_chamber_fluid[i])
            
            # Коэффициент теплоотдачи
            h_chamber = calculate_heat_transfer(
                T_chamber_fluid[i], 
                T_chamber_wall[i], 
                'chamber', 
                CHAMBER_INNER_DIAMETER
            )
            
            # Площадь теплообмена для участка
            chamber_heat_transfer_area = 2 * np.pi * chamber_inner_radius * (CHAMBER_HEIGHT / NUM_STATIONS)
            
            # Тепловой поток от нагревателя и стенки к рабочей жидкости
            q_heater = heater_power_distribution[i]
            q_wall = h_chamber * chamber_heat_transfer_area * (T_chamber_wall[i] - T_chamber_fluid[i])
            q_total = q_heater + q_wall
            
            # Изменение температуры рабочей жидкости
            dT_chamber = q_total / (MASS_FLOW_RATE * props_chamber['specific_heat'])
            
            # Обновление температуры рабочей жидкости
            T_chamber_fluid[i-1] = T_chamber_fluid[i] + dT_chamber
        
        # 3. Расчет теплового баланса стенки
        for i in range(NUM_STATIONS):
            # Коэффициенты теплоотдачи
            h_cooling = calculate_heat_transfer(
                T_cooling_fluid[i], 
                T_chamber_wall[i], 
                'cooling', 
                2 * COOLING_GAP
            )
            
            h_chamber = calculate_heat_transfer(
                T_chamber_fluid[i], 
                T_chamber_wall[i], 
                'chamber', 
                CHAMBER_INNER_DIAMETER
            )
            
            # Площади теплообмена
            cooling_area = 2 * np.pi * chamber_outer_radius * (CHAMBER_HEIGHT / NUM_STATIONS)
            chamber_area = 2 * np.pi * chamber_inner_radius * (CHAMBER_HEIGHT / NUM_STATIONS)
            
            # Тепловые потоки
            q_cooling = h_cooling * cooling_area * (T_chamber_wall[i] - T_cooling_fluid[i])
            q_chamber = h_chamber * chamber_area * (T_chamber_wall[i] - T_chamber_fluid[i])
            q_heater = heater_power_distribution[i]
            
            # Баланс тепла для стенки
            q_net = q_heater - q_cooling - q_chamber
            
            # Свойства материала стенки
            props_wall = inconel_properties(T_chamber_wall[i])
            
            # Масса участка стенки
            wall_volume = np.pi * (chamber_outer_radius**2 - chamber_inner_radius**2) * (CHAMBER_HEIGHT / NUM_STATIONS)
            wall_mass = props_wall['density'] * wall_volume
            
            # Изменение температуры стенки с улучшенной релаксацией
            dT_wall = q_net / (wall_mass * props_wall['specific_heat']) * relaxation_factor
            
            # Обновление температуры стенки
            T_chamber_wall[i] += dT_wall
        
        # Проверка сходимости
        max_diff = np.max(np.abs(T_chamber_wall - T_chamber_wall_prev))
        
        # Вызов функции обратного вызова для отображения прогресса
        if progress_callback is not None:
            try:
                progress_callback(iteration, max_iterations, max_diff)
            except Exception as e:
                print(f"Ошибка при вызове функции обратного вызова: {e}")
        
        # Адаптивная релаксация - уменьшаем фактор если колебания увеличиваются
        if iteration > 0 and max_diff > prev_max_diff:
            relaxation_factor *= 0.95  # Уменьшаем фактор релаксации
        
        # Сохраняем текущее отклонение для следующей итерации
        prev_max_diff = max_diff
        
        if max_diff < tolerance:
            print(f"Сходимость достигнута за {iteration+1} итераций")
            break
    
    if iteration == max_iterations - 1:
        print("Предупреждение: максимальное количество итераций достигнуто без сходимости")
        print(f"Текущее максимальное отклонение: {max_diff:.4f} K")
    
    # Возвращаем результаты и количество выполненных итераций
    return T_chamber_wall, T_cooling_fluid, T_chamber_fluid, iteration + 1

# 5. Визуализация результатов
def plot_results(T_chamber_wall, T_cooling_fluid, T_chamber_fluid):
    """
    Визуализация результатов расчета.
    
    Параметры:
    T_chamber_wall : array_like
        Температура стенки камеры
    T_cooling_fluid : array_like
        Температура охлаждающей жидкости
    T_chamber_fluid : array_like
        Температура рабочей жидкости в камере
    """
    z_normalized = z_points / CHAMBER_HEIGHT
    
    plt.figure(figsize=(12, 8))
    
    # График температур
    plt.subplot(2, 2, 1)
    plt.plot(z_normalized, T_chamber_wall, 'r-', label='Стенка камеры')
    plt.plot(z_normalized, T_cooling_fluid, 'b-', label='Охлаждающая жидкость')
    plt.plot(z_normalized, T_chamber_fluid, 'g-', label='Рабочая жидкость')
    plt.xlabel('Нормализованная высота')
    plt.ylabel('Температура (K)')
    plt.title('Распределение температур')
    plt.legend()
    plt.grid(True)
    
    # График скоростей
    plt.subplot(2, 2, 2)
    
    # Расчет скоростей
    v_cooling = np.zeros(NUM_STATIONS)
    v_chamber = np.zeros(NUM_STATIONS)
    
    for i in range(NUM_STATIONS):
        v_cooling[i] = calculate_flow_parameters(T_cooling_fluid[i], cooling_area)['velocity']
        v_chamber[i] = calculate_flow_parameters(T_chamber_fluid[i], chamber_inner_area)['velocity']
    
    plt.plot(z_normalized, v_cooling, 'b-', label='Охлаждающая жидкость')
    plt.plot(z_normalized, v_chamber, 'g-', label='Рабочая жидкость')
    plt.xlabel('Нормализованная высота')
    plt.ylabel('Скорость (м/с)')
    plt.title('Профили скоростей')
    plt.legend()
    plt.grid(True)
    
    # График чисел Рейнольдса
    plt.subplot(2, 2, 3)
    
    # Расчет чисел Рейнольдса
    re_cooling = np.zeros(NUM_STATIONS)
    re_chamber = np.zeros(NUM_STATIONS)
    
    for i in range(NUM_STATIONS):
        re_cooling[i] = calculate_flow_parameters(T_cooling_fluid[i], cooling_area)['reynolds']
        re_chamber[i] = calculate_flow_parameters(T_chamber_fluid[i], chamber_inner_area)['reynolds']
    
    plt.plot(z_normalized, re_cooling, 'b-', label='Охлаждающая жидкость')
    plt.plot(z_normalized, re_chamber, 'g-', label='Рабочая жидкость')
    plt.xlabel('Нормализованная высота')
    plt.ylabel('Число Рейнольдса')
    plt.title('Распределение чисел Рейнольдса')
    plt.legend()
    plt.grid(True)
    
    # График коэффициентов теплоотдачи
    plt.subplot(2, 2, 4)
    
    # Расчет коэффициентов теплоотдачи
    h_cooling = np.zeros(NUM_STATIONS)
    h_chamber = np.zeros(NUM_STATIONS)
    
    for i in range(NUM_STATIONS):
        h_cooling[i] = calculate_heat_transfer(
            T_cooling_fluid[i], 
            T_chamber_wall[i], 
            'cooling', 
            2 * COOLING_GAP
        )
        
        h_chamber[i] = calculate_heat_transfer(
            T_chamber_fluid[i], 
            T_chamber_wall[i], 
            'chamber', 
            CHAMBER_INNER_DIAMETER
        )
    
    plt.plot(z_normalized, h_cooling, 'b-', label='Охлаждающая жидкость')
    plt.plot(z_normalized, h_chamber, 'g-', label='Рабочая жидкость')
    plt.xlabel('Нормализованная высота')
    plt.ylabel('Коэффициент теплоотдачи (Вт/(м²·К))')
    plt.title('Коэффициенты теплоотдачи')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('resistojet_results.png', dpi=300)
    plt.show()
    
    # Вывод численных данных
    print(f"Максимальная температура стенки камеры: {np.max(T_chamber_wall):.2f} K")
    print(f"Максимальная температура охлаждающей жидкости: {np.max(T_cooling_fluid):.2f} K")
    print(f"Максимальная температура рабочей жидкости: {np.max(T_chamber_fluid):.2f} K")
    print(f"Прирост температуры охлаждающей жидкости: {T_cooling_fluid[-1] - T_cooling_fluid[0]:.2f} K")

def calculate_equilibrium_time(T_chamber_wall, heater_power=HEATER_POWER):
    """
    Расчет времени выхода на равновесную температуру в стенке камеры нагрева.
    
    Параметры:
    T_chamber_wall : array_like
        Температура стенки камеры
    heater_power : float
        Мощность нагревателя в Вт
        
    Возвращает:
    dict: Словарь с временами выхода на равновесие (90% и 99%)
    """
    # Оценка времени выхода на равновесие в реальной системе
    # Предполагаем, что тепловая инерция стенки пропорциональна массе и теплоемкости
    props_wall = inconel_properties(np.mean(T_chamber_wall))
    wall_volume = np.pi * (chamber_outer_radius**2 - chamber_inner_radius**2) * CHAMBER_HEIGHT
    wall_mass = props_wall['density'] * wall_volume
    wall_heat_capacity = wall_mass * props_wall['specific_heat']
    
    # Оценка времени выхода на равновесие (в секундах)
    # Используем простую модель первого порядка
    time_constant = wall_heat_capacity / (heater_power * 0.8)  # предполагаем, что 80% мощности идет на нагрев стенки
    time_to_90_percent = -time_constant * np.log(0.1)  # время до достижения 90% от установившегося значения
    time_to_99_percent = -time_constant * np.log(0.01)  # время до достижения 99% от установившегося значения
    
    return {
        'time_constant': time_constant,
        'time_to_90_percent': time_to_90_percent,
        'time_to_99_percent': time_to_99_percent
    }

# Основная функция
def main():
    start_time = time.time()
    
    print("Начало расчета стационарного состояния...")
    T_chamber_wall, T_cooling_fluid, T_chamber_fluid, iterations_completed = steady_state_simulation()
    
    print(f"Расчет завершен за {time.time() - start_time:.2f} секунд")
    print(f"Количество выполненных итераций: {iterations_completed}")
    
    plot_results(T_chamber_wall, T_cooling_fluid, T_chamber_fluid)

if __name__ == "__main__":
    main() 