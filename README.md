# Magnetometer Calibration (RLSM + Genetic Algorithm)

The project implements and compares two calibration algorithms for a three-axis magnetoresistive sensor: a recursive least squares method (RLSM) and a genetic algorithm (GA). Both estimate the sensor's zero biases and scale factor errors from raw ADC readings recorded while rotating the sensor in the Earth's magnetic field, and compensate the readings so that the magnitude of the measured field vector becomes close to constant (a unit sphere after normalization).

## Sensor

The data comes from the three-axis magnetoresistive sensor HMC1043L (Honeywell), part of the M-2T magnetic compass used in the BISNS-2T integrated navigation system. The sensor is built on AMR (Anisotropic Magnetoresistive) technology: thin-film permalloy (Ni-Fe) resistors form three Wheatstone bridges arranged orthogonally along the X, Y and Z axes. An external magnetic field along a sensitive axis changes the bridge resistance and produces a differential output voltage proportional to the field projection.

## Deviation methodology

The sensor rotates in the Earth's constant magnetic field. For an ideal three-axis magnetometer, the measurements would lie exactly on a sphere:

```math
(H_1')^2 + (H_2')^2 + (H_3')^2 = H_E^2
```

where `H_E` is the magnitude of the Earth's field vector at the test location. In practice, sensor errors turn this sphere into an offset ellipsoid of revolution, which gives the following relation:

```math
\left( \frac{H_1 - \Delta H_1}{1 + \Delta K_1} \right)^2 + \left( \frac{H_2 - \Delta H_2}{1 + \Delta K_2} \right)^2 + \left( \frac{H_3 - \Delta H_3}{1 + \Delta K_3} \right)^2 = 1^2
\tag{2}
```

General deviation work procedure for this magnetic compass:

1. Mounting check: verify the compass installation meets deviation minimization requirements (non-magnetic clearance from power cables and ferromagnetic masses).
2. Functional check: verify all compass units, including the inertial sensors, are operational.
3. Prepare source data: collect current magnetic declination data for the test area.
4. Set up measurement equipment: prepare instruments for recording current coordinates and orientation angles (roll, pitch) for later correction.
5. Take baseline readings: record initial compass deviations with the aircraft's electrical equipment de-energized to establish reference (zero) values.
6. Measure on different headings: record compass readings on a sequence of heading angles and compare against reference values.
7. Statistical processing: repeat measurements on each heading multiple times to remove random errors.
8. Calibration: run the compass calibration procedure.
9. Analyze results: build a residual deviation table from the collected data and compute correction coefficients.
10. Integrate corrections: load the computed correction coefficients into the onboard equipment complex (OEC) software for automatic deviation compensation.
11. Verification: repeat the deviation work with the corrections applied to confirm accuracy.
12. Record data: document the final deviation correction values for use during operation.

## Mathematical description

### Full sensor error model

```math
\begin{bmatrix}
H_1 \\
H_2 \\
H_3
\end{bmatrix}
=
\begin{bmatrix}
H_1^{ideal} \\
H_2^{ideal} \\
H_3^{ideal}
\end{bmatrix}
+
\begin{bmatrix}
\Delta H_1 \\
\Delta H_2 \\
\Delta H_3
\end{bmatrix}
+
\begin{bmatrix}
\Delta K_1 & \mu_{1,2} & \mu_{1,3} \\
\mu_{2,1} & \Delta K_2 & \mu_{2,3} \\
\mu_{3,1} & \mu_{3,2} & \Delta K_3
\end{bmatrix}
\begin{bmatrix}
H_1^{ideal} \\
H_2^{ideal} \\
H_3^{ideal}
\end{bmatrix}
+
\begin{bmatrix}
\delta H_1 \\
\delta H_2 \\
\delta H_3
\end{bmatrix}
```

where `H_1, H_2, H_3` are the measured field projections on the sensor's sensitive axes, `H_1^{ideal}, H_2^{ideal}, H_3^{ideal}` are the ideal (true) Earth field projections, `Delta H_1, Delta H_2, Delta H_3` are the constant zero-offset errors, `Delta K_1, Delta K_2, Delta K_3` are the scale factor errors, `mu_{i,j}` are the errors caused by axis misalignment and non-orthogonality, and `delta H_1, delta H_2, delta H_3` are residual errors (quantization noise, temperature and pressure drift, etc).

Dropping the second-order (cross-axis) terms gives the simplified model actually used for calibration:

```math
\begin{bmatrix}
H_1 \\
H_2 \\
H_3
\end{bmatrix}
=
\begin{bmatrix}
H_1^{ideal} \\
H_2^{ideal} \\
H_3^{ideal}
\end{bmatrix}
+
\begin{bmatrix}
\Delta H_1 \\
\Delta H_2 \\
\Delta H_3
\end{bmatrix}
+
\begin{bmatrix}
\Delta K_1 & 0 & 0 \\
0 & \Delta K_2 & 0 \\
0 & 0 & \Delta K_3
\end{bmatrix}
\begin{bmatrix}
H_1^{ideal} \\
H_2^{ideal} \\
H_3^{ideal}
\end{bmatrix}
+
\begin{bmatrix}
\delta H_1 \\
\delta H_2 \\
\delta H_3
\end{bmatrix}
\tag{3}
```

### Measurement model and estimator synthesis

```math
\overline{Z} = [H]\overline{X} + \overline{V}
\tag{4}
```

Expanding the sphere equation and substituting variables turns it into a linear measurement equation in a 6-dimensional state space:

```math
X^T = \{C_1, C_2, C_3, C_4, C_5, C_6\}
\tag{6}
```

```math
C_1 = \Delta H_1
\tag{7}
```

```math
C_2 = \frac{(1 + \Delta K_1)^2}{(1 + \Delta K_2)^2}
\tag{8}
```

```math
C_3 = \Delta H_2 \frac{(1 + \Delta K_1)^2}{(1 + \Delta K_2)^2}
\tag{9}
```

```math
C_4 = \frac{(1 + \Delta K_1)^2}{(1 + \Delta K_3)^2}
\tag{10}
```

```math
C_5 = \Delta H_3 \frac{(1 + \Delta K_1)^2}{(1 + \Delta K_3)^2}
\tag{11}
```

```math
C_6 = \Delta H_1^2 + \Delta H_2^2 \frac{(1 + \Delta K_1)^2}{(1 + \Delta K_2)^2} + \Delta H_3^2 \frac{(1 + \Delta K_1)^2}{(1 + \Delta K_3)^2} - H_E^2 (1 + \Delta K_1)^2
\tag{12}
```

The number of measurement points used to form this equation is set at the compass software design stage and must not be smaller than the dimension of the state vector, i.e. six. Given the large number of measurements required, the estimator uses the recursive least squares method (RLSM).

### Recursive least squares equations

```math
[P]_{k+1} = [P]_k [H]_{k+1}^T \left([E] + [H]_{k+1} [P]_k [H]_{k+1}^T\right)^{-1} [H]_{k+1} [P]_k
\tag{13}
```

```math
\widehat{X}_{k+1} = \widehat{X}_k + [P]_k [H]_{k+1}^T \left(\overline{Z}_{k+1} - [H]_{k+1} \widehat{X}_k\right)
\tag{14}
```

where `k, k+1` are the estimation steps, `Z_{k+1}` is the measurement vector of the current iteration, `H_{k+1}` is the measurement matrix of the current iteration, `E` is the identity matrix, `P_k, P_{k+1}` are the covariance matrices of the current and previous steps, and `X_k, X_{k+1}` are the state vector estimates.

The initial covariance matrix and state vector are chosen from prior knowledge of the sensor's accuracy characteristics. For the HMC1043 sensor used here, the initial values are:

```math
P_0 =
\begin{bmatrix}
0.5 & 0 & 0 & 0 & 0 & 0 \\
0 & 1.5 & 0 & 0 & 0 & 0 \\
0 & 0 & 1.5 & 0 & 0 & 0 \\
0 & 0 & 0 & 1.5 & 0 & 0 \\
0 & 0 & 0 & 0 & 1.5 & 0 \\
0 & 0 & 0 & 0 & 0 & 20
\end{bmatrix}
\tag{15}
```

```math
\widehat{X}_0^T = \{0, \; 1, \; 0, \; 1, \; 0, \; -1\}
```

The magnetometer readings are corrected as follows:

```math
H_{1,2,3}^c = \left(H_{1,2,3} - \Delta\widehat{H}_{1,2,3}\right) \cdot \left(1 + \Delta\widehat{K}_{1,2,3}\right)
\tag{16}
```

where `H_{1,2,3}^c` are the field projections corrected for the estimated sensor errors and deviation.

### Calibration quality metrics

Both algorithms are compared using the deviation of the field vector's magnitude from unity. An ideal calibrated magnetometer should satisfy:

```math
|\overline{H}| = \sqrt{H_x^2 + H_y^2 + H_z^2} \approx 1
\tag{17}
```

for all measurements (assuming the mean field magnitude is normalized to 1). The mean squared deviation of the vector magnitude from unity is used as the quality metric:

```math
J = \frac{1}{N} \sum_{i=1}^{N} \left(\sqrt{H_{x,i}^2 + H_{y,i}^2 + H_{z,i}^2} - 1\right)^2
\tag{18}
```

The smaller `J` is, the better the calibration. For clarity, the mean absolute error (MAE) and the maximum deviation are also computed:

```math
MAE = \frac{1}{N} \sum_{i=1}^{N} \left|\sqrt{H_{x,i}^2 + H_{y,i}^2 + H_{z,i}^2} - 1\right|
\tag{19}
```

```math
\Delta_{max} = \max_i \left|\sqrt{H_{x,i}^2 + H_{y,i}^2 + H_{z,i}^2} - 1\right|
\tag{20}
```

### Genetic algorithm

As an alternative calibration method, a genetic algorithm (GA) is implemented. Unlike RLSM, the GA requires no linearization of the model and works directly with the loss function, evaluating the compensation quality by how far the compensated data deviates from a sphere.

The algorithm operates on a population of N individuals, each representing a vector of calibration parameters `x = [dH1, dH2, dH3, dK1, dK2, dK3]`. Each individual's quality is evaluated by the loss function:

```math
f(x) = \frac{1}{N} \sum_{i=1}^{N} \left(|H_i^{comp}| - \overline{r}\right)^2
\tag{22}
```

Each generation performs three operations. Tournament selection: out of k randomly chosen individuals, the one with the smaller `f` wins. BLX-alpha crossover: for two parents A and B, each gene of the child is:

```math
x_g^{child} = A_g + u \cdot (B_g - A_g), \qquad u \sim \mathcal{U}(-\alpha,\; 1+\alpha)
\tag{23}
```

Mutation: each gene is shifted by a random amount with a given probability:

```math
x_g^{new} = x_g + \sigma \cdot \mathcal{N}(0,1)
\tag{24}
```

where `sigma` decreases with the generation number, allowing a broad search early on and fine-tuning later. Iterations run until the maximum number of generations is reached or the population converges.

## Project structure

```
.
├── sensors_data/
│   └── magnetic_data.txt      raw ADC readings recorded during sensor rotation
├── parser.py                  parse_H: loads magnetic_data.txt into a numpy array
├── math_model.py               RLSM, GA, fitness, compensate, apply_error_model,
│                                tournament, crossover, mutate
├── plots.py                    plot_sphere_comparison, plot_hK_history, plot_convergence
└── main.py                     entry point, runs both algorithms and builds all plots
```

## Building and running

### Dependencies

```bash
pip install numpy matplotlib
```

### Run

```bash
python main.py
```

`main.py` reads `sensors_data/magnetic_data.txt` (comma-separated H1, H2, H3 readings, one triple per line), scales the raw ADC counts down by 1000, and then:

1. Runs RLSM (`math_model.RLSM`) with convergence threshold `eps = 0.01` on the trace of `P`, printing the estimated biases `dH` and scale factors `dK`.
2. Compensates the data with the RLSM estimate and prints the mean/std of the field vector magnitude before and after.
3. Plots RLSM convergence (`plot_convergence`), the bias/scale factor history (`plot_hK_history`), and a 3D/2D sphere comparison before and after calibration (`plot_sphere_comparison`).
4. Runs the genetic algorithm (`math_model.GA`, population 32, 50 generations, mutation probability 0.15) and prints its estimate of `dH`, `dK`.
5. Compensates the data with the GA estimate, prints the mean/std of the field vector magnitude, and plots the GA convergence and the sphere comparison before/after.

Expected input file format (`sensors_data/magnetic_data.txt`), one line per measurement:

```
H1, H2, H3
H1, H2, H3
...
```

## Results

(place photos and plots from the test runs here)

RLSM covariance trace (convergence of the algorithm):

<img width="500" alt="image" src="https://github.com/user-attachments/assets/78289329-d159-4b3e-bb33-1ffe4beb124a" />

Scale factors and zero offsets over time:

<img width="500" alt="image" src="https://github.com/user-attachments/assets/142db508-de9e-4f49-b41d-ab9c3a3a5200" />

Data before and after RLSM calibration:

<img width="600" alt="image" src="https://github.com/user-attachments/assets/66bcf2ae-ad37-413e-9223-8e98da025228" />

Genetic algorithm loss function:

<img width="500" alt="image" src="https://github.com/user-attachments/assets/705a03da-c325-4b93-9ccc-002dfad2b39e" />

Data before and after GA calibration:

<img width="600" alt="image" src="https://github.com/user-attachments/assets/4125ffcb-f7e9-4091-92cb-e6e2065c9601" />

### Numerical results

RLSM estimate: `dH = [-0.2019, 0.2850, -0.2410]`, `dK = [0, 0.2037, 0.1325]`.

GA estimate: `dH = [-0.2492, 0.2980, -0.2331]`, `dK = [-0.0375, -0.0460, -0.0707]`.

RMS deviation of the field vector magnitude from unity:

| Stage | RMS deviation |
|---|---|
| Before calibration | 0.2358 |
| After RLSM | 0.1396 |
| After GA | 0.1213 |

RLSM reduces the RMS deviation from the unit sphere by 86.04%, while the genetic algorithm reduces it by 87.87%. Both methods solve the calibration problem successfully; the genetic algorithm produced a somewhat more accurate estimate than RLSM on this dataset.

## Conclusion

The technical characteristics of the HMC1043 magnetoresistive sensor used in the M-2T magnetic compass of the BISNS-2T integrated navigation system were studied, along with the general deviation work procedure aimed at reducing measurement errors. Based on the three-axis magnetometer error model, a measurement equation was formed and an optimal estimator was synthesized using the recursive least squares method. A genetic algorithm was additionally implemented to refine the error model parameters. The genetic algorithm provided a slightly higher calibration accuracy than RLSM on the recorded dataset, while both methods are viable for practical deviation work.

---
---

# Калибровка магнитного компаса (РМНК + генетический алгоритм)

Проект реализует и сравнивает два алгоритма калибровки трёхосного магниторезистивного датчика: рекуррентный метод наименьших квадратов (РМНК) и генетический алгоритм (ГА). Оба метода оценивают сдвиги нуля и ошибки масштабных коэффициентов датчика по сырым отсчётам АЦП, записанным при вращении датчика в магнитном поле Земли, и корректируют показания так, чтобы модуль измеренного вектора поля стал близок к постоянному значению (единичная сфера после нормировки).

## Датчик

Данные получены с трёхосного магниторезистивного датчика HMC1043L (Honeywell), входящего в состав магнитного компаса М-2Т комплексной навигационной системы БИСНС-2Т. Датчик выполнен по технологии AMR (Anisotropic Magnetoresistive): тонкоплёночные резисторы из пермаллоя (Ni-Fe) образуют три моста Уитстона, расположенных ортогонально по осям X, Y и Z. При воздействии внешнего магнитного поля вдоль чувствительной оси сопротивление плеч моста меняется, формируя дифференциальное выходное напряжение, пропорциональное проекции поля.

## Методика девиационных работ

Датчик вращается в постоянном магнитном поле Земли. Измерения идеального трёхосного магнитометра описываются уравнением сферы:

```math
(H_1')^2 + (H_2')^2 + (H_3')^2 = H_E^2
```

где `H_E` - модуль вектора напряжённости магнитного поля Земли в точке проведения девиационных работ. Наличие погрешностей приводит к тому, что реальные измерения описываются уравнением эллипсоида вращения со смещённым центром:

```math
\left( \frac{H_1 - \Delta H_1}{1 + \Delta K_1} \right)^2 + \left( \frac{H_2 - \Delta H_2}{1 + \Delta K_2} \right)^2 + \left( \frac{H_3 - \Delta H_3}{1 + \Delta K_3} \right)^2 = 1^2
\tag{2}
```

Обобщённая методика проведения девиационных работ для рассматриваемого магнитного компаса:

1. Проверка монтажа: убедиться в соответствии установки МК требованиям по минимизации девиации (наличие немагнитных промежутков до силовых кабелей и ферромагнитных масс).
2. Функциональный контроль: провести проверку работоспособности всех узлов МК, включая инерциальные датчики.
3. Подготовка исходных данных: собрать и подготовить актуальные сведения о магнитном склонении в районе проведения работ.
4. Настройка измерительного оборудования: подготовить приборы для фиксации текущих координат и углов пространственной ориентации (крен, тангаж) с целью последующей коррекции показаний.
5. Снятие базовых показаний: выполнить первичные замеры отклонений МК при обесточенном электрооборудовании самолёта для определения эталонных (нулевых) значений.
6. Измерения на разных курсах: произвести последовательную фиксацию показаний МК на различных курсовых углах с последующим расчётом и сравнением с эталонными значениями.
7. Статистическая обработка: провести многократные повторные замеры на каждом курсовом угле для исключения случайных погрешностей.
8. Калибровка: выполнить процедуру калибровки магнитного компаса.
9. Анализ результатов: по полученным данным составить карту (таблицу) остаточной девиации и рассчитать поправочные коэффициенты.
10. Интеграция поправок: внести вычисленные корректирующие коэффициенты в программное обеспечение комплекса бортового оборудования (КБО) для автоматической компенсации девиации.
11. Верификация: произвести контрольные девиационные работы с уже внесёнными коррективами для подтверждения точности.
12. Регистрация данных: задокументировать итоговые значения девиационных поправок для последующего использования в процессе эксплуатации БПЛА.

## Математическое описание

### Полная модель ошибок датчика

```math
\begin{bmatrix}
H_1 \\
H_2 \\
H_3
\end{bmatrix}
=
\begin{bmatrix}
H_1^{ИД} \\
H_2^{ИД} \\
H_3^{ИД}
\end{bmatrix}
+
\begin{bmatrix}
\Delta H_1 \\
\Delta H_2 \\
\Delta H_3
\end{bmatrix}
+
\begin{bmatrix}
\Delta K_1 & \mu_{1,2} & \mu_{1,3} \\
\mu_{2,1} & \Delta K_2 & \mu_{2,3} \\
\mu_{3,1} & \mu_{3,2} & \Delta K_3
\end{bmatrix}
\begin{bmatrix}
H_1^{ИД} \\
H_2^{ИД} \\
H_3^{ИД}
\end{bmatrix}
+
\begin{bmatrix}
\delta H_1 \\
\delta H_2 \\
\delta H_3
\end{bmatrix}
```

где `H_1, H_2, H_3` - измерения проекций напряжённости магнитного поля на чувствительные оси датчика, `H_1^{ИД}, H_2^{ИД}, H_3^{ИД}` - идеальные значения проекций напряжённости поля Земли, `Delta H_1, Delta H_2, Delta H_3` - постоянные составляющие ошибок ("сдвиг нуля"), `Delta K_1, Delta K_2, Delta K_3` - ошибки масштабных коэффициентов, `mu_{i,j}` - ошибки, вызванные несоосностью и неортогональностью чувствительных осей, `delta H_1, delta H_2, delta H_3` - остаточные погрешности датчика (ошибки оцифровки, температурный и барометрический дрейф и т.д.).

Исключая погрешности второго порядка малости, получаем упрощённую модель, которая используется для калибровки:

```math
\begin{bmatrix}
H_1 \\
H_2 \\
H_3
\end{bmatrix}
=
\begin{bmatrix}
H_1^{ИД} \\
H_2^{ИД} \\
H_3^{ИД}
\end{bmatrix}
+
\begin{bmatrix}
\Delta H_1 \\
\Delta H_2 \\
\Delta H_3
\end{bmatrix}
+
\begin{bmatrix}
\Delta K_1 & 0 & 0 \\
0 & \Delta K_2 & 0 \\
0 & 0 & \Delta K_3
\end{bmatrix}
\begin{bmatrix}
H_1^{ИД} \\
H_2^{ИД} \\
H_3^{ИД}
\end{bmatrix}
+
\begin{bmatrix}
\delta H_1 \\
\delta H_2 \\
\delta H_3
\end{bmatrix}
\tag{3}
```

### Модель измерений и синтез оценивателя

```math
\overline{Z} = [H]\overline{X} + \overline{V}
\tag{4}
```

Раскрывая квадраты уравнения сферы и вводя замену переменных, получаем линейное уравнение измерений в 6-мерном пространстве состояния:

```math
X^T = \{C_1, C_2, C_3, C_4, C_5, C_6\}
\tag{6}
```

```math
C_1 = \Delta H_1
\tag{7}
```

```math
C_2 = \frac{(1 + \Delta K_1)^2}{(1 + \Delta K_2)^2}
\tag{8}
```

```math
C_3 = \Delta H_2 \frac{(1 + \Delta K_1)^2}{(1 + \Delta K_2)^2}
\tag{9}
```

```math
C_4 = \frac{(1 + \Delta K_1)^2}{(1 + \Delta K_3)^2}
\tag{10}
```

```math
C_5 = \Delta H_3 \frac{(1 + \Delta K_1)^2}{(1 + \Delta K_3)^2}
\tag{11}
```

```math
C_6 = \Delta H_1^2 + \Delta H_2^2 \frac{(1 + \Delta K_1)^2}{(1 + \Delta K_2)^2} + \Delta H_3^2 \frac{(1 + \Delta K_1)^2}{(1 + \Delta K_3)^2} - H_E^2 (1 + \Delta K_1)^2
\tag{12}
```

Количество точек измерения для формирования уравнения определяется на этапе разработки ПО МК; их число не должно быть меньше размерности вектора состояния, то есть шести. Ввиду большого количества измерений применяется рекуррентный метод наименьших квадратов (РМНК).

### Уравнения рекуррентного МНК

```math
[P]_{k+1} = [P]_k [H]_{k+1}^T \left([E] + [H]_{k+1} [P]_k [H]_{k+1}^T\right)^{-1} [H]_{k+1} [P]_k
\tag{13}
```

```math
\widehat{X}_{k+1} = \widehat{X}_k + [P]_k [H]_{k+1}^T \left(\overline{Z}_{k+1} - [H]_{k+1} \widehat{X}_k\right)
\tag{14}
```

где `k, k+1` - такт оценивания, `Z_{k+1}` - вектор измерения текущего шага, `H_{k+1}` - матрица измерений текущего шага, `E` - единичная матрица, `P_k, P_{k+1}` - матрицы ковариаций текущего и предыдущего шагов, `X_k, X_{k+1}` - оценки вектора состояния.

Начальные значения матрицы ковариации и вектора оценки выбираются исходя из априорной информации о точностных характеристиках датчика. Для датчика HMC1043 начальные значения:

```math
P_0 =
\begin{bmatrix}
0.5 & 0 & 0 & 0 & 0 & 0 \\
0 & 1.5 & 0 & 0 & 0 & 0 \\
0 & 0 & 1.5 & 0 & 0 & 0 \\
0 & 0 & 0 & 1.5 & 0 & 0 \\
0 & 0 & 0 & 0 & 1.5 & 0 \\
0 & 0 & 0 & 0 & 0 & 20
\end{bmatrix}
\tag{15}
```

```math
\widehat{X}_0^T = \{0, \; 1, \; 0, \; 1, \; 0, \; -1\}
```

Поправка показаний магнитометра осуществляется следующим образом:

```math
H_{1,2,3}^c = \left(H_{1,2,3} - \Delta\widehat{H}_{1,2,3}\right) \cdot \left(1 + \Delta\widehat{K}_{1,2,3}\right)
\tag{16}
```

где `H_{1,2,3}^c` - измерения проекции напряжённости магнитного поля с учётом оценок погрешностей датчика и девиации.

### Критерии качества калибровки

Для сравнения алгоритмов используется отклонение модуля вектора поля от единицы. Идеальный калиброванный магнитометр должен удовлетворять условию:

```math
|\overline{H}| = \sqrt{H_x^2 + H_y^2 + H_z^2} \approx 1
\tag{17}
```

для всех измерений (при нормировке среднего модуля поля к 1). В качестве меры качества используется среднеквадратичное отклонение модуля вектора от единицы:

```math
J = \frac{1}{N} \sum_{i=1}^{N} \left(\sqrt{H_{x,i}^2 + H_{y,i}^2 + H_{z,i}^2} - 1\right)^2
\tag{18}
```

Чем меньше значение `J`, тем выше качество калибровки. Для наглядности также вычисляется среднее абсолютное отклонение (MAE) и максимальное отклонение:

```math
MAE = \frac{1}{N} \sum_{i=1}^{N} \left|\sqrt{H_{x,i}^2 + H_{y,i}^2 + H_{z,i}^2} - 1\right|
\tag{19}
```

```math
\Delta_{max} = \max_i \left|\sqrt{H_{x,i}^2 + H_{y,i}^2 + H_{z,i}^2} - 1\right|
\tag{20}
```

### Генетический алгоритм

В качестве альтернативного метода калибровки реализован генетический алгоритм (ГА). В отличие от РМНК, ГА не требует линеаризации модели и работает непосредственно с функцией потерь, оценивая качество компенсации по отклонению скомпенсированных данных от сферы.

Алгоритм оперирует популяцией из N особей, каждая из которых представляет собой вектор калибровочных параметров `x = [dH1, dH2, dH3, dK1, dK2, dK3]`. Качество каждой особи оценивается функцией потерь:

```math
f(x) = \frac{1}{N} \sum_{i=1}^{N} \left(|H_i^{cm}| - \overline{r}\right)^2
\tag{22}
```

На каждом поколении выполняются три операции. Турнирная селекция: из случайно выбранных k особей побеждает та, у которой меньше `f`. Скрещивание по методу BLX-alpha: для двух родителей A и B потомок по каждому гену:

```math
x_g^{child} = A_g + u \cdot (B_g - A_g), \qquad u \sim \mathcal{U}(-\alpha,\; 1+\alpha)
\tag{23}
```

Мутация: каждый ген с заданной вероятностью сдвигается на случайную величину:

```math
x_g^{new} = x_g + \sigma \cdot \mathcal{N}(0,1)
\tag{24}
```

где `sigma` убывает с номером поколения, обеспечивая сначала широкий поиск, затем уточнение. Итерации выполняются до достижения максимального числа поколений или выполнения условия сходимости популяции.

## Структура проекта

```
.
├── sensors_data/
│   └── magnetic_data.txt      сырые отсчёты АЦП, записанные при вращении датчика
├── parser.py                  parse_H: загружает magnetic_data.txt в numpy-массив
├── math_model.py               RLSM, GA, fitness, compensate, apply_error_model,
│                                tournament, crossover, mutate
├── plots.py                    plot_sphere_comparison, plot_hK_history, plot_convergence
└── main.py                     точка входа, запускает оба алгоритма и строит графики
```

## Сборка и запуск

### Зависимости

```bash
pip install numpy matplotlib
```

### Запуск

```bash
python main.py
```

`main.py` читает `sensors_data/magnetic_data.txt` (значения H1, H2, H3 через запятую, по одной тройке на строку), масштабирует сырые отсчёты АЦП делением на 1000, после чего:

1. Запускает РМНК (`math_model.RLSM`) с порогом сходимости `eps = 0.01` по следу матрицы `P`, печатает оценки сдвигов `dH` и масштабных коэффициентов `dK`.
2. Компенсирует данные полученной оценкой РМНК и печатает среднее/СКО модуля вектора поля до и после.
3. Строит график сходимости РМНК (`plot_convergence`), историю сдвигов и масштабных коэффициентов (`plot_hK_history`) и сравнение сферы данных до/после калибровки (`plot_sphere_comparison`).
4. Запускает генетический алгоритм (`math_model.GA`, популяция 32, 50 поколений, вероятность мутации 0.15), печатает его оценки `dH`, `dK`.
5. Компенсирует данные оценкой ГА, печатает среднее/СКО модуля вектора поля и строит график сходимости ГА и сравнение сферы до/после.

Ожидаемый формат входного файла (`sensors_data/magnetic_data.txt`), по одной строке на измерение:

```
H1, H2, H3
H1, H2, H3
...
```

## Результаты работы

(здесь разместить фотографии и графики, полученные в ходе испытаний)

След матрицы ковариации P (сходимость алгоритма РМНК):

<img width="500" alt="image" src="https://github.com/user-attachments/assets/78289329-d159-4b3e-bb33-1ffe4beb124a" />

Масштабные коэффициенты и сдвиги нуля с течением времени:

<img width="500" alt="image" src="https://github.com/user-attachments/assets/142db508-de9e-4f49-b41d-ab9c3a3a5200" />

Данные до и после калибровки РМНК:

<img width="600" alt="image" src="https://github.com/user-attachments/assets/66bcf2ae-ad37-413e-9223-8e98da025228" />

Функция потерь генетического алгоритма:

<img width="500" alt="image" src="https://github.com/user-attachments/assets/705a03da-c325-4b93-9ccc-002dfad2b39e" />

Данные до и после калибровки ГА:

<img width="600" alt="image" src="https://github.com/user-attachments/assets/4125ffcb-f7e9-4091-92cb-e6e2065c9601" />

### Численные результаты

Оценка РМНК: `dH = [-0.2019, 0.2850, -0.2410]`, `dK = [0, 0.2037, 0.1325]`.

Оценка ГА: `dH = [-0.2492, 0.2980, -0.2331]`, `dK = [-0.0375, -0.0460, -0.0707]`.

СКО отклонения модуля вектора поля от единицы:

| Этап | СКО |
|---|---|
| До калибровки | 0.2358 |
| После РМНК | 0.1396 |
| После ГА | 0.1213 |

РМНК демонстрирует точность 86.04% по СКО от вектора единицы, а генетический алгоритм, 87.87%. Оба метода успешно решают задачу калибровки; генетический алгоритм показал несколько более высокую точность оценок по сравнению с РМНК на данном наборе данных.

## Заключение

В ходе работы изучены технические характеристики магниторезистивного датчика HMC1043, применяемого в магнитном компасе М-2Т комплексной навигационной системы БИСНС-2Т, рассмотрены принципы функционирования магнитного компаса и обобщённый порядок выполнения девиационных работ. На основе математической модели погрешностей трёхосного магнитометра сформировано уравнение измерений и синтезирован оптимальный оцениватель с использованием рекуррентного метода наименьших квадратов. Дополнительно реализован генетический алгоритм для уточнения параметров модели ошибок. Генетический алгоритм обеспечил несколько более высокую точность оценок по сравнению с РМНК на записанном наборе данных, при этом оба метода пригодны для практических девиационных работ.
