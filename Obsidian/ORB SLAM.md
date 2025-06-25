## 1. Входные топики и параметры камеры
- **Топики**  
  - `/camera/color/image_raw` (sensor_msgs/Image)  
  - `/camera/camera_info` (sensor_msgs/CameraInfo)  
- **Параметры**:
Внутр. матрица `K` и коэффициенты искажения из `CameraInfo`  

## 2. Детектирование признаков и ArUco
1. **ArUco (OpenCV)**  
   - Чтение цветного кадра -> `cv2.aruco.detectMarkers`  
   - Оценка позы маркера -> начальная установка масштаба и ориентации
```python
corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, K, distCoeffs)
```
2. Для каждого найденного маркера iii строим однородную матрицу:

$$T_{\text{cam}\to i} = \begin{bmatrix} R(rvec_i) & tvec_i \\ 0\;0\;0 & 1 \end{bmatrix},$$

где $R(rvec_i)$ получается из `cv2.Rodrigues`.
## 3. Локализация камеры (когда есть известные маркеры)
Пусть в кадре есть хотя бы один маркер $i$ с ID, уже занесённым в `map_markers`:

$Tmap→i$ из карты, и  $Tcam→i$ из детекции. 

Тогда текущую позу камеры в карте вычисляем по формуле:
$$T_{map→cam}=T_{map→i}  (T_{cam→i})^{−1}.$$
Если в кадре несколько известных маркеров, можно усреднить или решить задачу наименьших квадратов:
 $$\min_{T\in SE(3)} \sum_{i\in\text{known}} \bigl\|\,T\,T_{\text{cam}\to i}\;-\;T_{\text{map}\to i}\bigr\|^2.$$
## 4. Добавление новых маркеров в карту

Для каждого обнаруженного маркера $j$ с ID, которого нет в `map_markers`:

1. Берём только что вычисленную $T_{\text{map}\to \text{cam}}.$  
2. Строим    
$$T_{\text{map}\to j} = T_{\text{map}\to \text{cam}} \;\; T_{\text{cam}\to j}.$$
3. Сохраняем.
## 6. Публикация через tf2



- **Триангуляция новых точек**
- **Интеграция с ICP и AMCL**

