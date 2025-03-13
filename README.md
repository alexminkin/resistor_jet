# Симуляция резистоджета

Этот проект представляет собой интерактивную симуляцию тепловых процессов и движения частиц в резистивном ракетном двигателе (резистоджете) с регенеративным охлаждением.

## Возможности

- Визуализация температурных градиентов в различных частях резистоджета
- Визуализация распределения числа Маха в сопле
- 3D-модель движения частиц с анимацией
- Интерактивное изменение параметров модели
- Расчет тепловых и газодинамических характеристик

## Требования

- Python 3.8 или выше
- Библиотеки, указанные в файле `requirements.txt`

## Установка

1. Клонируйте репозиторий:
```
git clone https://github.com/yourusername/resistojet-simulation.git
cd resistojet-simulation
```

2. Установите зависимости:
```
pip install -r requirements.txt
```

## Запуск

### Основное приложение с визуализацией температуры и числа Маха

```
streamlit run resistojet_streamlit.py
```

### 3D-визуализация движения частиц

```
streamlit run resistojet_3d_app.py
```

## Использование

1. Используйте слайдеры в боковой панели для изменения параметров модели:
   - Геометрические параметры (высота камеры, радиус, толщина стенок)
   - Рабочие параметры (массовый расход, давление, мощность нагревателя)
   - Параметры визуализации

2. Наблюдайте за изменениями в визуализации в реальном времени

3. Изучайте распределение температуры и числа Маха с помощью цветовых шкал и контурных линий

4. В 3D-визуализации используйте мышь для вращения, масштабирования и перемещения модели

## Структура проекта

- `resistojet_streamlit.py` - основное приложение с визуализацией температуры и числа Маха
- `resistojet_3d.py` - модуль для 3D-визуализации движения частиц
- `resistojet_3d_app.py` - приложение с 3D-визуализацией
- `resistojet_simulation.py` - модуль с физической моделью резистоджета
- `requirements.txt` - список зависимостей

## Физическая модель

Модель учитывает:
- Теплопередачу между рабочей жидкостью, стенками и охлаждающей жидкостью
- Изменение свойств водорода в зависимости от температуры
- Ускорение потока в сопле и изменение числа Маха
- Регенеративное охлаждение

## Лицензия

MIT 