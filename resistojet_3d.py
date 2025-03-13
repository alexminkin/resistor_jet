import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import time

def create_particle_trajectories(chamber_height, chamber_inner_radius, nozzle_profile, 
                                 T_chamber_fluid, num_particles=50, num_steps=100):
    """
    Создает траектории частиц в резистоджете на основе профиля сопла и температуры.
    
    Args:
        chamber_height: Высота камеры (м)
        chamber_inner_radius: Внутренний радиус камеры (м)
        nozzle_profile: Профиль сопла (массив радиусов)
        T_chamber_fluid: Температура рабочей жидкости (массив)
        num_particles: Количество частиц
        num_steps: Количество шагов для каждой траектории
        
    Returns:
        x, y, z: Координаты частиц для каждого шага
        u, v, w: Компоненты скорости для каждого шага
        mach: Число Маха для каждого шага
    """
    # Создаем начальные позиции частиц (равномерно распределенные по входу)
    initial_z = np.zeros(num_particles)
    initial_r = np.linspace(0.1*chamber_inner_radius, 0.9*chamber_inner_radius, num_particles)
    initial_theta = np.random.uniform(0, 2*np.pi, num_particles)
    
    # Преобразуем в декартовы координаты
    initial_x = initial_r * np.cos(initial_theta)
    initial_y = initial_r * np.sin(initial_theta)
    
    # Создаем массивы для хранения траекторий
    x = np.zeros((num_particles, num_steps))
    y = np.zeros((num_particles, num_steps))
    z = np.zeros((num_particles, num_steps))
    u = np.zeros((num_particles, num_steps))
    v = np.zeros((num_particles, num_steps))
    w = np.zeros((num_particles, num_steps))
    mach = np.zeros((num_particles, num_steps))
    
    # Устанавливаем начальные позиции
    x[:, 0] = initial_x
    y[:, 0] = initial_y
    z[:, 0] = initial_z
    
    # Константы для расчета скорости звука
    gamma = 1.4  # Показатель адиабаты для водорода
    R = 8.314  # Универсальная газовая постоянная (Дж/(моль·К))
    M = 0.002016  # Молярная масса водорода (кг/моль)
    
    # Шаг по времени
    dt = 0.0001  # секунды
    
    # Интерполяция профиля сопла
    z_points = np.linspace(0, chamber_height, len(nozzle_profile))
    
    # Расчет траекторий
    for p in range(num_particles):
        for step in range(1, num_steps):
            # Текущие координаты
            current_z = z[p, step-1]
            current_r = np.sqrt(x[p, step-1]**2 + y[p, step-1]**2)
            current_theta = np.arctan2(y[p, step-1], x[p, step-1])
            
            # Проверка, находится ли частица внутри сопла
            if current_z >= chamber_height or current_z < 0:
                # Частица вышла из сопла, копируем последнюю позицию
                x[p, step:] = x[p, step-1]
                y[p, step:] = y[p, step-1]
                z[p, step:] = z[p, step-1]
                u[p, step:] = u[p, step-1]
                v[p, step:] = v[p, step-1]
                w[p, step:] = w[p, step-1]
                mach[p, step:] = mach[p, step-1]
                break
            
            # Находим ближайший индекс в профиле сопла
            z_idx = int(current_z / chamber_height * (len(z_points) - 1))
            z_idx = max(0, min(z_idx, len(z_points) - 1))
            
            # Получаем радиус сопла в текущей позиции
            nozzle_r = nozzle_profile[z_idx]
            
            # Если частица вышла за пределы сопла, отражаем ее
            if current_r > nozzle_r:
                # Отражение от стенки (упрощенно)
                reflection_angle = np.arctan2(current_z, current_r)
                current_theta = current_theta + np.pi
                current_r = nozzle_r * 0.95  # Немного отступаем от стенки
            
            # Расчет температуры в текущей позиции (интерполяция)
            z_normalized = current_z / chamber_height
            z_normalized = max(0, min(z_normalized, 0.999))  # Ограничиваем для безопасности
            temp_idx = int(z_normalized * (len(T_chamber_fluid) - 1))
            current_temp = T_chamber_fluid[temp_idx]
            
            # Расчет скорости звука
            sound_speed = np.sqrt(gamma * R * current_temp / M)
            
            # Расчет скорости (зависит от положения в сопле)
            # В начале сопла скорость низкая, в сужении увеличивается, в расширении - еще больше
            if current_z < chamber_height * 0.4:
                # До сужения - низкая скорость
                velocity_z = 50 + 150 * (current_z / (chamber_height * 0.4))
                velocity_r = 0  # Радиальная скорость близка к нулю
            else:
                # После сужения - быстрый рост скорости
                velocity_z = 200 + 1800 * ((current_z - chamber_height * 0.4) / (chamber_height * 0.6))
                # Небольшая радиальная скорость из-за расширения сопла
                velocity_r = 50 * (current_r / nozzle_r) * ((current_z - chamber_height * 0.4) / (chamber_height * 0.6))
            
            # Расчет числа Маха
            current_mach = velocity_z / sound_speed
            
            # Преобразование в декартовы компоненты скорости
            velocity_x = velocity_r * np.cos(current_theta)
            velocity_y = velocity_r * np.sin(current_theta)
            
            # Обновление позиции с использованием скорости
            x[p, step] = x[p, step-1] + velocity_x * dt
            y[p, step] = y[p, step-1] + velocity_y * dt
            z[p, step] = z[p, step-1] + velocity_z * dt
            
            # Сохранение компонентов скорости
            u[p, step] = velocity_x
            v[p, step] = velocity_y
            w[p, step] = velocity_z
            
            # Сохранение числа Маха
            mach[p, step] = current_mach
    
    return x, y, z, u, v, w, mach

def create_3d_visualization(chamber_height, chamber_inner_radius, chamber_wall_thickness, 
                           cooling_gap, cooling_wall_thickness, nozzle_profile, 
                           T_chamber_fluid, T_chamber_wall, T_cooling_fluid):
    """
    Создает 3D-визуализацию резистоджета с траекториями частиц и распределением числа Маха.
    
    Args:
        chamber_height: Высота камеры (м)
        chamber_inner_radius: Внутренний радиус камеры (м)
        chamber_wall_thickness: Толщина стенки камеры (м)
        cooling_gap: Зазор охлаждения (м)
        cooling_wall_thickness: Толщина стенки охлаждающей рубашки (м)
        nozzle_profile: Профиль сопла (массив радиусов)
        T_chamber_fluid: Температура рабочей жидкости (массив)
        T_chamber_wall: Температура стенки камеры (массив)
        T_cooling_fluid: Температура охлаждающей жидкости (массив)
        
    Returns:
        fig: Объект Plotly Figure с 3D-визуализацией
    """
    # Создаем траектории частиц
    x, y, z, u, v, w, mach_values = create_particle_trajectories(
        chamber_height, chamber_inner_radius, nozzle_profile, T_chamber_fluid
    )
    
    # Создаем фигуру с двумя подграфиками (3D-визуализация и 2D-срез)
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'xy'}]],
        subplot_titles=('3D-модель движения частиц', 'Распределение числа Маха (срез)')
    )
    
    # Создаем цветовую шкалу для числа Маха
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
    
    # Добавляем траектории частиц
    for p in range(len(x)):
        # Используем число Маха для цвета
        fig.add_trace(
            go.Scatter3d(
                x=z[p], y=x[p], z=y[p],
                mode='lines',
                line=dict(
                    color=mach_values[p],
                    colorscale=mach_colors,
                    width=3,
                    colorbar=dict(
                        title='Число Маха',
                        thickness=20,
                        len=0.5
                    ),
                    cmin=0,
                    cmax=3.7
                ),
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Создаем геометрию сопла (внутренняя поверхность)
    theta = np.linspace(0, 2*np.pi, 50)
    z_points = np.linspace(0, chamber_height, len(nozzle_profile))
    
    r_mesh, theta_mesh = np.meshgrid(nozzle_profile, theta)
    z_mesh, _ = np.meshgrid(z_points, theta)
    
    x_mesh = r_mesh * np.cos(theta_mesh)
    y_mesh = r_mesh * np.sin(theta_mesh)
    
    # Добавляем поверхность сопла
    fig.add_trace(
        go.Surface(
            x=z_mesh, y=x_mesh, z=y_mesh,
            colorscale='Greys',
            opacity=0.3,
            showscale=False
        ),
        row=1, col=1
    )
    
    # Настройка 3D-графика
    fig.update_scenes(
        aspectmode='data',
        xaxis_title='Высота (м)',
        yaxis_title='X (м)',
        zaxis_title='Y (м)',
        row=1, col=1
    )
    
    # Создаем 2D-срез для визуализации числа Маха
    # Создаем сетку для визуализации
    r_points = np.linspace(0, chamber_inner_radius, 50)
    z_points = np.linspace(0, chamber_height, 100)
    r_mesh, z_mesh = np.meshgrid(r_points, z_points)
    
    # Создаем массив чисел Маха
    mach_field = np.zeros_like(r_mesh)
    
    # Заполняем массив чисел Маха
    for i in range(len(z_points)):
        z_pos = z_points[i]
        z_normalized = z_pos / chamber_height
        
        # Находим соответствующий индекс в профиле сопла
        nozzle_idx = min(int(z_normalized * len(nozzle_profile)), len(nozzle_profile) - 1)
        
        for j in range(len(r_points)):
            r_pos = r_points[j]
            
            if r_pos < nozzle_profile[nozzle_idx]:
                # Внутри камеры (рабочая жидкость)
                if z_pos < chamber_height * 0.4:
                    # До сужения - низкое число Маха
                    mach_field[i, j] = 0.2 + 0.8 * (z_pos / (chamber_height * 0.4))
                else:
                    # После сужения - быстрый рост числа Маха
                    mach_field[i, j] = 1.0 + 2.7 * ((z_pos - chamber_height * 0.4) / (chamber_height * 0.6))
            else:
                # За пределами камеры - нет потока
                mach_field[i, j] = 0
    
    # Добавляем контурный график числа Маха
    fig.add_trace(
        go.Contour(
            z=mach_field,
            x=z_points*1000,  # переводим в мм
            y=r_points*1000,  # переводим в мм
            colorscale=mach_colors,
            contours=dict(
                start=0,
                end=3.7,
                size=0.2,
                showlabels=True
            ),
            colorbar=dict(
                title='Число Маха',
                thickness=20,
                len=0.5,
                y=0.5,
                yanchor='middle'
            )
        ),
        row=1, col=2
    )
    
    # Добавляем профиль сопла на 2D-график
    fig.add_trace(
        go.Scatter(
            x=z_points*1000,
            y=nozzle_profile*1000,
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Добавляем отраженный профиль сопла (для симметрии)
    fig.add_trace(
        go.Scatter(
            x=z_points*1000,
            y=-nozzle_profile*1000,
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Настройка 2D-графика
    fig.update_xaxes(
        title_text='Высота (мм)',
        row=1, col=2
    )
    fig.update_yaxes(
        title_text='Радиус (мм)',
        row=1, col=2
    )
    
    # Общие настройки
    fig.update_layout(
        title='3D-модель движения частиц и распределение числа Маха',
        height=800,
        width=1200
    )
    
    return fig

def animate_particles(fig, num_frames=50):
    """
    Создает анимацию движения частиц.
    
    Args:
        fig: Объект Plotly Figure с 3D-визуализацией
        num_frames: Количество кадров анимации
        
    Returns:
        fig: Объект Plotly Figure с анимацией
    """
    # Получаем данные о траекториях из графика
    traces = []
    for trace in fig.data:
        if isinstance(trace, go.Scatter3d):
            traces.append(trace)
    
    # Создаем кадры анимации
    frames = []
    for frame in range(num_frames):
        frame_data = []
        
        for trace in traces:
            # Получаем координаты
            x = trace.x
            y = trace.y
            z = trace.z
            
            # Определяем, сколько точек показывать в текущем кадре
            points_to_show = int(len(x) * (frame + 1) / num_frames)
            
            # Создаем новый след с ограниченным числом точек
            new_trace = go.Scatter3d(
                x=x[:points_to_show],
                y=y[:points_to_show],
                z=z[:points_to_show],
                mode='lines',
                line=dict(
                    color=trace.line.color,
                    colorscale=trace.line.colorscale,
                    width=trace.line.width
                ),
                showlegend=False
            )
            
            frame_data.append(new_trace)
        
        frames.append(go.Frame(data=frame_data, name=f'frame{frame}'))
    
    # Добавляем кадры к фигуре
    fig.frames = frames
    
    # Добавляем элементы управления анимацией
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], dict(frame=dict(duration=0, redraw=True), mode='immediate')]
                    )
                ],
                x=0.1,
                y=0,
                xanchor='right',
                yanchor='bottom'
            )
        ]
    )
    
    return fig

def create_temperature_distribution_plot(chamber_height, chamber_inner_radius, chamber_wall_thickness, 
                                        cooling_gap, cooling_wall_thickness, 
                                        T_chamber_fluid, T_chamber_wall, T_cooling_fluid):
    """
    Создает визуализацию распределения температуры в резистоджете, похожую на Figure 2.
    
    Args:
        chamber_height: Высота камеры (м)
        chamber_inner_radius: Внутренний радиус камеры (м)
        chamber_wall_thickness: Толщина стенки камеры (м)
        cooling_gap: Зазор охлаждения (м)
        cooling_wall_thickness: Толщина стенки охлаждающей рубашки (м)
        T_chamber_fluid: Температура рабочей жидкости (массив)
        T_chamber_wall: Температура стенки камеры (массив)
        T_cooling_fluid: Температура охлаждающей жидкости (массив)
        
    Returns:
        fig: Объект Plotly Figure с визуализацией распределения температуры
    """
    # Создаем фигуру
    fig = go.Figure()
    
    # Определяем размеры в мм
    chamber_height_mm = chamber_height * 1000
    chamber_inner_radius_mm = chamber_inner_radius * 1000
    chamber_wall_thickness_mm = chamber_wall_thickness * 1000
    cooling_gap_mm = cooling_gap * 1000
    cooling_wall_thickness_mm = cooling_wall_thickness * 1000
    
    # Рассчитываем радиальные позиции
    chamber_outer_radius_mm = chamber_inner_radius_mm + chamber_wall_thickness_mm
    cooling_outer_radius_mm = chamber_outer_radius_mm + cooling_gap_mm
    shell_outer_radius_mm = cooling_outer_radius_mm + cooling_wall_thickness_mm
    
    # Создаем сетку для высоты
    height_points = np.linspace(0, chamber_height_mm, len(T_chamber_fluid))
    
    # Создаем цветовую шкалу для температуры
    temp_colors = [
        (0, 'rgb(70, 0, 150)'),     # фиолетовый для 0K
        (0.1, 'rgb(0, 0, 255)'),    # синий для 200K
        (0.2, 'rgb(0, 100, 255)'),  # голубой для 400K
        (0.3, 'rgb(0, 200, 255)'),  # циан для 600K
        (0.4, 'rgb(0, 255, 200)'),  # сине-зеленый для 800K
        (0.6, 'rgb(100, 255, 0)'),  # желто-зеленый для 1000K
        (0.8, 'rgb(255, 200, 0)'),  # желтый для 1200K
        (1.0, 'rgb(255, 0, 0)')     # красный для 1400K+
    ]
    
    # Находим глобальный минимум и максимум температуры
    min_temp = min(np.min(T_chamber_fluid), np.min(T_chamber_wall), np.min(T_cooling_fluid))
    max_temp = max(np.max(T_chamber_fluid), np.max(T_chamber_wall), np.max(T_cooling_fluid))
    
    # Создаем массивы для визуализации
    # Камера (рабочая жидкость)
    chamber_x = []
    chamber_y = []
    chamber_temp = []
    
    # Стенка камеры
    wall_x = []
    wall_y = []
    wall_temp = []
    
    # Охлаждающая жидкость
    cooling_x = []
    cooling_y = []
    cooling_temp = []
    
    # Внешняя стенка
    shell_x = []
    shell_y = []
    shell_temp = []
    
    # Заполняем массивы данными
    for i, h in enumerate(height_points):
        # Камера (рабочая жидкость) - левая сторона
        chamber_x.append(-chamber_inner_radius_mm)
        chamber_y.append(h)
        chamber_temp.append(T_chamber_fluid[i])
        
        # Камера (рабочая жидкость) - правая сторона
        chamber_x.append(chamber_inner_radius_mm)
        chamber_y.append(h)
        chamber_temp.append(T_chamber_fluid[i])
        
        # Стенка камеры - левая сторона
        wall_x.append(-chamber_outer_radius_mm)
        wall_y.append(h)
        wall_temp.append(T_chamber_wall[i])
        
        # Стенка камеры - правая сторона
        wall_x.append(chamber_outer_radius_mm)
        wall_y.append(h)
        wall_temp.append(T_chamber_wall[i])
        
        # Охлаждающая жидкость - левая сторона
        cooling_x.append(-cooling_outer_radius_mm)
        cooling_y.append(h)
        cooling_temp.append(T_cooling_fluid[i])
        
        # Охлаждающая жидкость - правая сторона
        cooling_x.append(cooling_outer_radius_mm)
        cooling_y.append(h)
        cooling_temp.append(T_cooling_fluid[i])
        
        # Внешняя стенка - левая сторона
        shell_x.append(-shell_outer_radius_mm)
        shell_y.append(h)
        shell_temp.append(300)  # Предполагаем постоянную температуру внешней стенки
        
        # Внешняя стенка - правая сторона
        shell_x.append(shell_outer_radius_mm)
        shell_y.append(h)
        shell_temp.append(300)  # Предполагаем постоянную температуру внешней стенки
    
    # Добавляем визуализацию рабочей жидкости
    fig.add_trace(
        go.Heatmap(
            x=chamber_x,
            y=chamber_y,
            z=chamber_temp,
            colorscale=temp_colors,
            zmin=min_temp,
            zmax=max_temp,
            showscale=False
        )
    )
    
    # Добавляем визуализацию стенки камеры
    fig.add_trace(
        go.Heatmap(
            x=wall_x,
            y=wall_y,
            z=wall_temp,
            colorscale=temp_colors,
            zmin=min_temp,
            zmax=max_temp,
            showscale=False
        )
    )
    
    # Добавляем визуализацию охлаждающей жидкости
    fig.add_trace(
        go.Heatmap(
            x=cooling_x,
            y=cooling_y,
            z=cooling_temp,
            colorscale=temp_colors,
            zmin=min_temp,
            zmax=max_temp,
            showscale=False
        )
    )
    
    # Добавляем визуализацию внешней стенки
    fig.add_trace(
        go.Heatmap(
            x=shell_x,
            y=shell_y,
            z=shell_temp,
            colorscale=temp_colors,
            zmin=min_temp,
            zmax=max_temp,
            showscale=True,
            colorbar=dict(
                title='Температура (K)',
                thickness=20,
                len=0.5,
                y=0.5,
                yanchor='middle'
            )
        )
    )
    
    # Добавляем вертикальные линии для обозначения границ
    # Левая сторона
    fig.add_trace(
        go.Scatter(
            x=[-chamber_inner_radius_mm, -chamber_inner_radius_mm],
            y=[0, chamber_height_mm],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[-chamber_outer_radius_mm, -chamber_outer_radius_mm],
            y=[0, chamber_height_mm],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[-cooling_outer_radius_mm, -cooling_outer_radius_mm],
            y=[0, chamber_height_mm],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[-shell_outer_radius_mm, -shell_outer_radius_mm],
            y=[0, chamber_height_mm],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        )
    )
    
    # Правая сторона
    fig.add_trace(
        go.Scatter(
            x=[chamber_inner_radius_mm, chamber_inner_radius_mm],
            y=[0, chamber_height_mm],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[chamber_outer_radius_mm, chamber_outer_radius_mm],
            y=[0, chamber_height_mm],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[cooling_outer_radius_mm, cooling_outer_radius_mm],
            y=[0, chamber_height_mm],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[shell_outer_radius_mm, shell_outer_radius_mm],
            y=[0, chamber_height_mm],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        )
    )
    
    # Горизонтальные линии (верх и низ)
    fig.add_trace(
        go.Scatter(
            x=[-shell_outer_radius_mm, shell_outer_radius_mm],
            y=[0, 0],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[-shell_outer_radius_mm, shell_outer_radius_mm],
            y=[chamber_height_mm, chamber_height_mm],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        )
    )
    
    # Настройка графика
    fig.update_layout(
        title='Figure 2: Temperature Distribution (Fluid + Wall)',
        xaxis_title='Radial Position (mm)',
        yaxis_title='Axial Height (mm)',
        width=800,
        height=800,
        xaxis=dict(
            range=[-shell_outer_radius_mm-1, shell_outer_radius_mm+1],
            gridcolor='lightgray',
            zerolinecolor='lightgray',
            showgrid=True
        ),
        yaxis=dict(
            range=[0, chamber_height_mm],
            gridcolor='lightgray',
            zerolinecolor='lightgray',
            showgrid=True
        ),
        plot_bgcolor='white'
    )
    
    return fig

def create_velocity_gradient_plot(chamber_height, chamber_inner_radius, nozzle_profile):
    """
    Создает визуализацию градиента скорости в резистоджете.
    
    Args:
        chamber_height: Высота камеры (м)
        chamber_inner_radius: Внутренний радиус камеры (м)
        nozzle_profile: Профиль сопла (массив радиусов)
        
    Returns:
        fig: Объект Plotly Figure с визуализацией градиента скорости
    """
    # Создаем фигуру
    fig = go.Figure()
    
    # Определяем размеры в мм
    chamber_height_mm = chamber_height * 1000
    chamber_inner_radius_mm = chamber_inner_radius * 1000
    
    # Создаем сетку для расчета скорости
    z_points = np.linspace(0, chamber_height_mm, 100)
    r_points = np.linspace(-chamber_inner_radius_mm, chamber_inner_radius_mm, 50)
    z_mesh, r_mesh = np.meshgrid(z_points, r_points)
    
    # Создаем массив скоростей
    velocity_field = np.zeros_like(z_mesh)
    
    # Заполняем массив скоростей
    for i in range(len(r_points)):
        r_pos = abs(r_points[i])
        
        for j in range(len(z_points)):
            z_pos = z_points[j]
            z_normalized = z_pos / chamber_height_mm
            
            # Находим соответствующий индекс в профиле сопла
            nozzle_idx = min(int(z_normalized * len(nozzle_profile)), len(nozzle_profile) - 1)
            nozzle_r_mm = nozzle_profile[nozzle_idx] * 1000
            
            if r_pos < nozzle_r_mm:
                # Внутри камеры (рабочая жидкость)
                if z_pos < chamber_height_mm * 0.4:
                    # До сужения - низкая скорость
                    velocity_field[i, j] = 50 + 150 * (z_pos / (chamber_height_mm * 0.4))
                else:
                    # После сужения - быстрый рост скорости
                    velocity_field[i, j] = 200 + 1800 * ((z_pos - chamber_height_mm * 0.4) / (chamber_height_mm * 0.6))
                
                # Добавляем радиальную зависимость (скорость ниже у стенок)
                velocity_field[i, j] *= (1 - 0.3 * (r_pos / nozzle_r_mm)**2)
            else:
                # За пределами камеры - нет потока
                velocity_field[i, j] = 0
    
    # Создаем цветовую шкалу для скорости
    velocity_colors = [
        (0, 'rgb(70, 0, 150)'),     # фиолетовый для 0 м/с
        (0.1, 'rgb(0, 0, 255)'),    # синий для низкой скорости
        (0.2, 'rgb(0, 100, 255)'),  # голубой
        (0.3, 'rgb(0, 200, 255)'),  # циан
        (0.4, 'rgb(0, 255, 200)'),  # сине-зеленый
        (0.6, 'rgb(100, 255, 0)'),  # желто-зеленый
        (0.8, 'rgb(255, 200, 0)'),  # желтый
        (1.0, 'rgb(255, 0, 0)')     # красный для высокой скорости
    ]
    
    # Добавляем контурный график скорости
    fig.add_trace(
        go.Contour(
            z=velocity_field,
            x=z_points,
            y=r_points,
            colorscale=velocity_colors,
            contours=dict(
                start=0,
                end=2000,
                size=100,
                showlabels=True,
                labelfont=dict(size=10, color='black')
            ),
            colorbar=dict(
                title='Скорость (м/с)',
                thickness=20,
                len=0.5,
                y=0.5,
                yanchor='middle'
            )
        )
    )
    
    # Добавляем профиль сопла
    z_profile = np.linspace(0, chamber_height_mm, len(nozzle_profile))
    r_profile = nozzle_profile * 1000
    
    fig.add_trace(
        go.Scatter(
            x=z_profile,
            y=r_profile,
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=z_profile,
            y=-r_profile,
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        )
    )
    
    # Настройка графика
    fig.update_layout(
        title='Velocity Gradient Distribution',
        xaxis_title='Axial Height (mm)',
        yaxis_title='Radial Position (mm)',
        width=800,
        height=600,
        xaxis=dict(
            range=[0, chamber_height_mm],
            gridcolor='lightgray',
            zerolinecolor='lightgray',
            showgrid=True
        ),
        yaxis=dict(
            range=[-chamber_inner_radius_mm*1.2, chamber_inner_radius_mm*1.2],
            gridcolor='lightgray',
            zerolinecolor='lightgray',
            showgrid=True
        ),
        plot_bgcolor='white'
    )
    
    # Добавляем текст "ε = 12" в правом верхнем углу
    fig.add_annotation(
        x=chamber_height_mm * 0.95,
        y=chamber_inner_radius_mm * 0.9,
        text="ε = 12",
        showarrow=False,
        font=dict(size=14, color="black")
    )
    
    return fig 