# NeRF Data Collector for ROS 2 (Jazzy)

## Opis

`nerf_data_collector` to węzeł ROS 2 napisany w Pythonie, który umożliwia zbieranie danych (obrazów RGB i pozycji kamery) w celu przygotowania zbioru danych kompatybilnego z [NeRF Studio](https://nerf.studio/). Dane te mogą być bezpośrednio wykorzystane do trenowania modeli NeRF.

Zebrane dane zawierają:
- Obrazy RGB (`.jpg`)
- Macierze transformacji (pozy i orientacje kamery w układzie NeRF)
- Plik `transforms.json` zgodny z wymaganiami NeRF Studio

## Parametry

| Parametr            | Domyślna wartość        | Opis |
|---------------------|--------------------------|------|
| `image_topic`       | `/rgb/image_raw`         | Topik z obrazem RGB |
| `camera_info_topic` | `/rgb/camera_info`       | Topik z parametrami kamery |
| `source_frame`      | `base_link`              | Ramka odniesienia robota |
| `target_frame`      | `azure_rgb`              | Ramka kamery RGB |
| `output_dir`        | `nerf_data`              | Katalog do zapisu danych |
| `collection_rate`   | `10.0`                   | Częstotliwość zbierania danych (co ile zapisywana jest jedna klatka)

## Uruchomienie

```bash
colcon build

source install/setup.bash

ros2 run mlinpl nerf_data_collector
```

## Działanie

1. Węzeł nasłuchuje na podanych topikach (`Image`, `CameraInfo`) i czeka na pierwsze dane kalibracyjne kamery.
2. Po odebraniu parametrów kamery rozpoczyna zbieranie danych.
3. Co 10. odebrany obraz zostaje zapisany wraz z odpowiadającą mu transformacją TF.
4. Zebrane dane są zapisywane w strukturze zgodnej z NeRF Studio:
   ```
   nerf_data/
   ├── images/
   │   ├── frame_000000.jpg
   │   ├── ...
   └── transforms.json
   ```

## Zatrzymanie i zapis

Aby zakończyć zbieranie danych i zapisać plik `transforms.json`, przerwij program np. przez `Ctrl+C`.

## Konwersja układów współrzędnych

Transformacje z ROS (układ ENU) są konwertowane do układu NeRF Studio (OpenGL-style camera-to-world). W tym celu używana jest macierz:

```python
ros_to_nerf = np.array([
    [1, 0,  0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0,  0, 1]
])
```

## Przykładowy plik `transforms.json`

```json
{
  "camera_model": "OPENCV",
  "fl_x": 600.0,
  "fl_y": 600.0,
  "cx": 320.0,
  "cy": 240.0,
  "w": 640,
  "h": 480,
  "frames": [
    {
      "file_path": "images/frame_000000.jpg",
      "transform_matrix": [[...], [...], [...], [...]]
    }
  ]
}
```
