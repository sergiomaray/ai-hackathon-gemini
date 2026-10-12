# AI Spatial Optimization with Gemini & LlamaIndex 🗺️🤖

Soluciones a retos de optimización espacial en cuadrículas 2D desarrolladas para **Hackathons de Inteligencia Artificial (Universidad de Oviedo)**.

Ambos proyectos combinan **algoritmos clásicos de filtrado espacial y heurísticas** con la capacidad de razonamiento de **Google Gemini 2.5 Flash** mediante flujos de trabajo autónomos orientados a eventos (**LlamaIndex Workflows**).

---

## 📌 Arquitectura Híbrida

La colocación óptima de infraestructuras sobre matrices discretas es un problema combinatorio de alta complejidad. El enfoque utilizado desacopla el cálculo determinista del razonamiento heurístico:

1. **Pre-filtrado algorítmico (Python):**
   - Filtrado estricto de casillas válidas (adyacencia a carreteras, zonas de exclusión).
   - Cálculo de densidad espacial y dispersión geográfica (**Non-Maximum Suppression / NMS**) para evitar concentrar recursos en un solo punto.
2. **Razonamiento guiado (Google Gemini):**
   - El modelo recibe el mapa de cuadrícula, las coordenadas críticas y una lista priorizada de candidatos válidos.
   - Analiza la distribución global y devuelve la combinación en formato JSON estructurado.
3. **Validación estricta y puntuación en Python:**
   - Se validan todas las reglas del problema (coordenadas únicas, límites del mapa, coberturas mínimas).
   - Se evalúa la función objetivo (distancia Manhattan).
4. **Bucle de retroalimentación (Feedback Loop):**
   - Si la solución infringe alguna regla o no bate el récord actual, el error o la puntuación se inyectan en el prompt del siguiente intento para guiar al LLM hacia una mejor solución.

---

## 📂 Estructura del Repositorio

```text
ai-hackathon-gemini/
├── README.md                           # Documentación principal
├── requirements.txt                    # Dependencias de Python
├── .gitignore                          # Exclusiones de Git (claves, caches, salidas)
├── .env.example                        # Plantilla para la API Key de Gemini
├── LICENSE                             # Licencia MIT
├── waste-container-placer/             # Reto 1: Contenedores urbanos
│   ├── container_placer.py             # Script principal del agente
│   ├── entradas/                       # Mapas oficiales del hackathon (1.txt a 4.txt)
│   └── salidas/                        # Mapas optimizados resultantes (generados)
└── power-grid-optimizer/               # Reto 2: Red eléctrica e industrias
    ├── power_grid_optimizer.py         # Script principal del agente
    ├── entradas/                       # Mapas de entrada (ejemplo_sintetico.txt)
    └── salidas/                        # Mapas optimizados resultantes (generados)
```

---

##Proyectos Incluidos

### 1. Urban Infrastructure Placer (`waste-container-placer/`)
Optimiza la colocación de **6 contenedores (`C`)** para minimizar la distancia caminada por los residentes de todas las viviendas (`O`).

* **Leyenda del mapa:**
  * `O` : Viviendas residenciales.
  * `X` : Carreteras públicas.
  * `-` : Casilla disponible.
  * `C` : Contenedor (colocado por el agente).
* **Reglas:**
  * Cada contenedor debe situarse en una casilla vacía `-` y tocar al menos una carretera `X` (8 vecinos).
  * Los 6 contenedores deben ubicarse en casillas distintas.
* **Mapas incluidos:** `1.txt`, `2.txt`, `3.txt`, `4.txt` (desde 20x21 hasta 76x75 casillas).

### 2. Power Grid Optimizer (`power-grid-optimizer/`)
Optimiza la ubicación de **6 transformadores eléctricos (`C`)** para abastecer industrias prioritarias (`T`) minimizando la distancia a hospitales (`O`) e industrias.

* **Leyenda del mapa:**
  * `T` : Industrias (requieren abastecimiento redundante).
  * `O` : Hospitales.
  * `X` : Carreteras.
  * `E` : Edificios protegidos / Zonas de exclusión.
  * `-` : Casilla disponible.
  * `C` : Transformador (colocado por el agente).
* **Reglas:**
  * Cada transformador debe ubicarse en casilla `-`, tocar una carretera `X` y **NO** tocar ningún edificio restringido `E`.
  * **Regla crítica de industria:** Cada industria `T` debe tener al menos 2 transformadores distintos a distancia Manhattan $\le 3$.
* **Objetivo:** Minimizar la suma de distancias Manhattan de cada hospital `O` e industria `T` a su transformador más cercano.

---

## Instalación y Uso

### 1. Requisitos previos
* Python 3.10 o superior.
* API Key de Google Gemini (gratuita en [Google AI Studio](https://aistudio.google.com/)).

### 2. Clonar e instalar dependencias
```bash
git clone https://github.com/sergiomaray/ai-hackathon-gemini.git
cd ai-hackathon-gemini

pip install -r requirements.txt
```

### 3. Configurar API Key
Copia la plantilla `.env.example` a `.env` y añade tu clave:
```bash
cp .env.example .env
```
Edita `.env`:
```env
GOOGLE_API_KEY=tu_api_key_aqui
```

### 4. Ejecución de los scripts

**Contenedores de residuos:**
```bash
# Ejecutar con el mapa 1 (o 2.txt, 3.txt, 4.txt)
python waste-container-placer/container_placer.py 1.txt

# Personalizar número de intentos del LLM
python waste-container-placer/container_placer.py 4.txt --intentos 4
```

**Optimizador de red eléctrica:**
```bash
python power-grid-optimizer/power_grid_optimizer.py ejemplo_sintetico.txt --intentos 4
```

El resultado guardará el mapa final con las casillas `C` colocadas en la carpeta `salidas/` correspondiente y mostrará la puntuación y el tiempo transcurrido por pantalla.



---

##Tecnologías

* **Python 3**
* **Google Gemini API:** `gemini-2.5-flash` para razonamiento rápido y preciso.
* **LlamaIndex Workflows:** Flujos orientados a eventos (`StartEvent`, `GenerationEvent`, `EvaluationEvent`, `StopEvent`).
* **Algoritmos y Heurísticas:** Distancia Manhattan, Non-Maximum Suppression (NMS), análisis de adyacencia espacial 8-vecinos, extracción de JSON resiliente.

---

##Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
