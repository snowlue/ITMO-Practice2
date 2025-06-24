## 1. Входные топики и параметры камеры
- **Топики**  
  - `/camera/color/image_raw` (sensor_msgs/Image)  
  - `/camera/depth/image_rect_raw` (sensor_msgs/Image)  
  - `/camera/camera_info` (sensor_msgs/CameraInfo)  
- **Параметры**  
  - Внутр. матрица `K` и коэффициенты искажения из `CameraInfo`  
  - Для RGB-D сразу имеем глубину, поэтому масштаб известен

## 2. Детектирование признаков и ArUco
1. **ArUco (OpenCV)**  
   - Чтение цветного кадра -> `cv2.aruco.detectMarkers`  
   - Оценка позы маркера -> начальная установка масштаба и ориентации
1. **ORB (cv2.ORB_create)**  
   - Детектор FAST + дескриптор BRIEF  
   - На каждом кадре:  
```python
corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, K, distCoeffs)
```

## 3. Сопоставление дескрипторов  
```python
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_prev, des_cur)
```

- **Триангуляция новых точек**
- **Интеграция с ICP и AMCL**