# Suite de Inventario FMCG — Lote Económico, Stock de Seguridad y Punto de Reorden

Aplicación web (Streamlit) del modelo de inventario de la dimensión 2 del caso FMCG
(*Modelamiento Predictivo de Datos*, Universidad de Lima). Calcula por categoría la política de inventario
(σ_DL, EOQ, factor k, stock de seguridad, punto de reorden, nivel de servicio y costo anual) con el
**fill rate** como métrica objetivo.

[![Abrir la app](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://modelo-de-inventario-kv4qjhxqy68vfkho3lqd7t.streamlit.app)

> Reemplaza `https://TU-APP.streamlit.app` por la URL que te entregue Streamlit al desplegar (paso 3).

## Qué hace

- Construye la demanda mensual por categoría en tres escenarios (A: suma cruda · B: media diaria × días · C: σ diario × √días).
- Deriva el costo de ordenar con la curva de intercambio de Silver, Pyke & Thomas (2017, §4.10).
- Resuelve el factor k con la función de pérdida normal estándar G(k) (bisección) para el fill rate objetivo.
- Muestra tablas, composición de varianza, sensibilidad al fill rate y calibración del número de reposiciones.

Por defecto usa el dataset del caso incluido en `data/`. También permite subir otro CSV/Excel
(el separador `;` o `,` se detecta automáticamente).

## Contenido del repositorio

| Archivo | Para qué sirve |
|---|---|
| `modeloInventario_FichaTecnica2.py` | La aplicación (archivo principal) |
| `requirements.txt` | Librerías que instala el servidor |
| `data/extended_fmcg_demand_forecasting.csv` | Dataset del caso (se carga por defecto) |
| `.streamlit/config.toml` | Ajustes del servidor (tamaño de carga, sin telemetría) |

## Ejecutarla en tu computadora

```bash
pip install -r requirements.txt
streamlit run modeloInventario_FichaTecnica2.py
```

## Publicarla como sitio web desde GitHub (gratis, Streamlit Community Cloud)

GitHub Pages solo sirve páginas estáticas y no puede ejecutar Python, por eso la app se despliega
en Streamlit Community Cloud, que lee el código directamente de este repositorio.

1. **Sube el repositorio a GitHub** (cuenta propia; un repositorio público es lo más simple):
   - Por la web: *New repository* → nombre `inventario-fmcg-streamlit` → *uploading an existing file* →
     arrastra **todo el contenido de esta carpeta** (incluidas `.streamlit/` y `data/`) → *Commit changes*.
   - O por terminal, dentro de esta carpeta:
     ```bash
     git init -b main
     git add .
     git commit -m "App de inventario FMCG"
     git remote add origin https://github.com/TU-USUARIO/inventario-fmcg-streamlit.git
     git push -u origin main
     ```
2. Entra a <https://share.streamlit.io> e inicia sesión con GitHub.
3. **Create app** → *Deploy a public app from GitHub* y completa:
   - Repository: `TU-USUARIO/inventario-fmcg-streamlit`
   - Branch: `main`
   - Main file path: `modeloInventario_FichaTecnica2.py`
   - *Advanced settings* → Python **3.12** (o 3.11).
4. **Deploy**. En unos minutos obtendrás una URL del tipo `https://<nombre>.streamlit.app`.
   Pégala en el botón de arriba de este README y comparte el enlace.

Cada vez que hagas *push* a `main`, la web se actualiza sola. Tras un periodo de inactividad la app se
"duerme" y al abrirla aparece un botón para despertarla (tarda unos segundos): conviene abrirla un rato
antes de una exposición.

## Nota sobre el dataset

El archivo de `data/` es el dataset de Kaggle usado en el curso. Si prefieres no redistribuirlo, elimina la
carpeta `data/`: la app seguirá funcionando con un dataset sintético equivalente o con el archivo que se
suba desde la barra lateral.
