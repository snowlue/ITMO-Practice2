Алгоритм описан в статье псевдокодом:
```
Algorithm Augmented_MCL(χ_{t-1}, u_t, z_t, m):
    static w_slow, w_fast
    χ_bar_t = χ_t = ∅
    for m = 1 to M:
        x_t^[m] = sample_motion_model(u_t, x_{t-1}^[m])
        w_t^[m] = measurement_model(z_t, x_t^[m], m)
        χ_bar_t = χ_bar_t + <x_t^[m], w_t^[m]>
        w_avg = w_avg + (1/M) * w_t^[m]
    endfor
    w_slow = w_slow + α_slow * (w_avg - w_slow)
    w_fast = w_fast + α_fast * (w_avg - w_fast)
    for m = 1 to M:
        with probability max(0.0, 1.0 - w_fast/w_slow):
            add random pose to χ_t
        else:
            draw i ∈ {1,...,N} with probability proportional ∝ w_t^[i]
            add x_t^[i] to χ_t
        endwith
    endfor
    return χ_t
```
Перед нами адаптивный вариант MCL, который добавляет случайные точки, являющиеся предположением о положении робота. Количество случайных точек определяется путем сравнения краткосрочной и долгосрочной вероятности результатов измерений сенсора.

Для реализации алгоритма требуется `motion_model` и `measurement_model` — модель движения и модель измерения. 

Модель движения описана в статье так:
```
Algorithm sample_motion_model_velocity(u_t, x_{t-1}):
    v_hat = v + sample(α₁|v| + α₂|ω|)
    ω_hat = ω + sample(α₃|v| + α₄|ω|)
    γ_hat = sample(α₅|v| + α₆|ω|)
    
    x' = x - (v_hat / ω_hat) * sin(θ) + (v_hat / ω_hat) * sin(θ + ω_hat * Δt)
    y' = y + (v_hat / ω_hat) * cos(θ) - (v_hat / ω_hat) * cos(θ + ω_hat * Δt)
    θ' = θ + ω_hat * Δt + γ_hat * Δt
    
    return x_i = (x', y', θ')ᵀ
```
Выше алгоритм для сэмплинга позиции $x_t = (x\,',\ y\,',\ \theta\,')^T$ из предыдущей $x_{t-1} = (x,\ y,\ \theta)^T$ и контрольной $u_t=(v,\ \omega)^T$. Конечная ориентация изменяется с помощью дополнительного случайного члена $\hat{\gamma}$. Переменные от $\alpha_1$ до $\alpha_6$ являются параметрами шума движения. Функция `sample(b)` генерирует случайную выборку из распределения с нулевым центром с дисперсией $b$. Мы используем выборку с нормальным распределением:
```
Algorithm sample_normal_distribution(b):
    return (b / 6) * sum_{i=1 to 12} rand(-1, 1)
```

В качестве модели измерения (measurement), которая будет оценивать качество измерения и улучшать алгоритм, используем модель известного соответствия, которая будет использовать ориентиры (landmarks) и уже известную карту и таким образом уточнять положение робота:
```
Algorithm sample_landmark_model_known_correspondence(f_t^i, c_t^i, m):
    j = c_t^i
    γ_hat = rand(0, 2π)
    r_hat = r_t^i + sample(σ_r²)
    φ_hat = φ_t^i + sample(σ_φ²)
    x = m_{j,x} + r_hat * cos(γ_hat)
    y = m_{j,y} + r_hat * sin(γ_hat)
    θ = γ_hat - π - φ_hat
    return (x, y, θ)ᵀ
```

