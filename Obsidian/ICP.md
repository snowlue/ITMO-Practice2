# Iterative Closest Point
- Использует основное облако точек $S_{n-1}$ и движимое облако точек $S_n$
- Возвращает новое облако — преобразование, которое приводит движимое облако к основному
- Используется как альтернатива одометрии, чтобы было понятно, как робот повернулся и сдвинулся
- Новое облако-преобразование получается путём итеративного сдвига и поворота движимого облако точек к основному, пока не будет удовлетворён threshold
- Задача: $\underset{R, t_n}{\min}||S_{n-1}-S_n||$, где $R$ — поворот, а $t_n$ — сдвиг
$$\left[\begin{array}{l}
	\left[\begin{array}{l}
		p_i = S_{n_i} \\
		q = \underset{m\times2}{S_{n-1}} \\
		q_i = \min||p_i-q||_2 \\
		\overline{p} = \text{mean}(S_n) \\
		\overline{q} = \text{mean}(S_{n-1}) \\
		\text{Cov} =\sum(p_i-\overline{p})(q_i-\overline{q})
	\end{array}\right. \leftarrow\text{для всех точек}\\
	u, \Sigma, v = \text{SVD}(\text{Cov}) \\
	R = uv^T \\
	t = R\overline{p} - \overline{q} \\
	p_i = Rp_i + t
\end{array}\right.$$

```python

```
