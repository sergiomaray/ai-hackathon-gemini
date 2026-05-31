import asyncio
import sys
import json
import re
from collections import deque
from pathlib import Path
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
from llama_index.core.workflow import (
    Workflow, Event, StartEvent, StopEvent, step
)

# --- CONFIGURACIÓN ---
NOMBRE_MAPA = "4.txt"   # Nombre del archivo en la carpeta 'entradas'
CANTIDAD_C = 6          # Número de contenedores a colocar
MAX_INTENTOS = 4        # Intentos del LLM para elegir la mejor combinación

# --- CONFIGURACIÓN DEL MODELO ---
try:
    # IMPORTANTE: Pon aquí tu NUEVA API KEY
    llm = GoogleGenAI(
        model="gemini-2.5-flash", 
        api_key="Insert your API key", 
        temperature=0.2  # Temperatura baja para máxima precisión
    )
    Settings.llm = llm
except Exception as e:
    print(f"Error LLM: {e}")
    sys.exit(1)

# --- EVENTOS (Transportan el estado entre pasos) ---
class GenerationEvent(Event):
    feedback: str
    intento_actual: int
    best_score: float
    best_coords: list
    candidatos: list 

class EvaluationEvent(Event):
    coordenadas_json: str
    intento_actual: int
    best_score: float
    best_coords: list
    candidatos: list

# --- FUNCIONES PURAS (Lógica de Python) ---

def cargar_mapa(ruta):
    with open(ruta, "r") as f:
        lines = [line.rstrip() for line in f.readlines() if line.strip()]
    return [list(line) for line in lines]

def mapa_a_string_simple(mapa):
    """Convierte el mapa a texto simple con números de fila."""
    res = []
    ancho = len(str(len(mapa)))
    for i, fila in enumerate(mapa):
        res.append(f"{i:0{ancho}d} | {''.join(fila)}")
    return "\n".join(res)

def obtener_candidatos_por_densidad(mapa):
    """
    ESTRATEGIA INTELIGENTE V2: DENSIDAD CON DISPERSIÓN.
    1. Calcula densidad de casas.
    2. Ordena de mejor a peor.
    3. Selecciona candidatos asegurando que estén lejos unos de otros.
    """
    filas = len(mapa)
    cols = len(mapa[0])
    todos_candidatos = []
    
    # Radio pequeño para calcular la densidad local (qué tan bueno es el sitio)
    radio_densidad = 2 
    
    # 1. RECOLECTAR TODOS LOS SITIOS VÁLIDOS Y SU CALIDAD
    for r in range(filas):
        for c in range(cols):
            if mapa[r][c] == '-':
                tiene_x = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < filas and 0 <= nc < cols and mapa[nr][nc] == 'X':
                            tiene_x = True
                            break
                
                if tiene_x:
                    casas_cerca = 0
                    r_min, r_max = max(0, r-radio_densidad), min(filas, r+radio_densidad+1)
                    c_min, c_max = max(0, c-radio_densidad), min(cols, c+radio_densidad+1)
                    
                    for ir in range(r_min, r_max):
                        for ic in range(c_min, c_max):
                            if mapa[ir][ic] == 'O':
                                casas_cerca += 1
                    
                    todos_candidatos.append({'coord': [r, c], 'score': casas_cerca})
    
    # Ordenar por densidad (de más casas a menos)
    todos_candidatos.sort(key=lambda x: x['score'], reverse=True)
    
    # 2. SELECCIÓN CON DISPERSIÓN (Non-Maximum Suppression)
    seleccionados = []
    # Distancia mínima para no repetir zonas (ej. 15 casillas de distancia Manhattan)
    distancia_minima = 15 
    
    for cand in todos_candidatos:
        r1, c1 = cand['coord']
        
        # Comprobar si está lejos de los ya seleccionados
        es_lejos = True
        for sel in seleccionados:
            r2, c2 = sel
            dist = abs(r1 - r2) + abs(c1 - c2) # Distancia Manhattan
            if dist < distancia_minima:
                es_lejos = False
                break
        
        if es_lejos:
            seleccionados.append(cand['coord'])
            
        # Si ya tenemos 50, paramos
        if len(seleccionados) >= 50:
            break
            
    # Si no llegamos a 50 con la restricción de distancia, rellenamos con los mejores restantes
    # (aunque estén cerca) para que el LLM tenga opciones
    if len(seleccionados) < 50:
        for cand in todos_candidatos:
            if cand['coord'] not in seleccionados:
                seleccionados.append(cand['coord'])
                if len(seleccionados) >= 50:
                    break
                    
    return seleccionados

def aplicar_coordenadas(mapa_orig, coordenadas):
    import copy
    nuevo_mapa = copy.deepcopy(mapa_orig)
    filas = len(nuevo_mapa)
    cols = len(nuevo_mapa[0])
    
    for coord in coordenadas:
        if not isinstance(coord, list) or len(coord) != 2:
            return None, "Formato incorrecto"
        r, c = coord
        if 0 <= r < filas and 0 <= c < cols:
            nuevo_mapa[r][c] = 'C'
    return nuevo_mapa, None

def calcular_distancia_total(mapa):
    filas, cols = len(mapa), len(mapa[0])
    casas = []
    q = deque()
    visitado = [[-1]*cols for _ in range(filas)]
    
    for r in range(filas):
        for c in range(cols):
            val = mapa[r][c]
            if val == 'O':
                casas.append((r,c))
            elif val == 'C':
                q.append((r,c,0))
                visitado[r][c] = 0
    
    if not q: return float('inf') 

    while q:
        r, c, dist = q.popleft()
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < filas and 0 <= nc < cols and visitado[nr][nc] == -1:
                visitado[nr][nc] = dist + 1
                q.append((nr, nc, dist + 1))
    
    total = 0
    for hr, hc in casas:
        d = visitado[hr][hc]
        if d == -1: return float('inf')
        total += d
    return total

def validar_reglas_basicas(mapa_orig, coordenadas):
    errores = []
    if len(coordenadas) != CANTIDAD_C:
        return [f"Debes enviar exactamente {CANTIDAD_C} coordenadas."]
    
    filas, cols = len(mapa_orig), len(mapa_orig[0])
    for r, c in coordenadas:
        if not (0 <= r < filas and 0 <= c < cols):
            errores.append(f"({r},{c}) fuera de rango.")
        elif mapa_orig[r][c] != '-':
            errores.append(f"({r},{c}) ocupado por '{mapa_orig[r][c]}'.")
            
    return errores

# --- AGENTE / WORKFLOW ---

class CoordinateAgent(Workflow):
    
    def __init__(self, mapa_original, **kwargs):
        super().__init__(**kwargs)
        self.mapa_original = mapa_original
        self.mapa_str = mapa_a_string_simple(mapa_original)
        self.rows = len(mapa_original)
        self.cols = len(mapa_original[0])
        
        # PASO CRÍTICO: Pre-calcular los mejores sitios antes de empezar
        print("📊 Analizando densidad de casas en el mapa...")
        self.todos_candidatos = obtener_candidatos_por_densidad(mapa_original)

    @step
    async def start(self, ev: StartEvent) -> GenerationEvent:
        total = len(self.todos_candidatos)
        print(f"🗺️ Mapa {self.rows}x{self.cols}. Sitios válidos totales: {total}")
        
        # SELECCIÓN DE CANDIDATOS:
        # Pasamos al LLM solo los 50 mejores (los que tienen más casas cerca).
        # Esto elimina el ruido y asegura que el LLM se enfoque en zonas importantes.
        limit = 50
        candidatos_prompt = self.todos_candidatos[:limit]
        
        print(f"✨ Seleccionados los {len(candidatos_prompt)} candidatos con mayor densidad para el LLM.")

        return GenerationEvent(
            feedback="Inicia el proceso.", 
            intento_actual=1,
            best_score=float('inf'),
            best_coords=[],
            candidatos=candidatos_prompt
        )

    @step
    async def generator(self, ev: GenerationEvent) -> EvaluationEvent | StopEvent:
        if ev.intento_actual > MAX_INTENTOS + 1:
             return StopEvent(result="MAX_INTENTOS_EXCEEDED")
        
        print(f"\n🧠 Intento {ev.intento_actual}/{MAX_INTENTOS}...")
        
        prompt = f"""
MAPA DE LA CIUDAD:
{self.mapa_str}

LEYENDA MAPA:

'O' = Casas (Tu objetivo es estar cerca de ellas)
'X' = Carretera
'-' = Espacio vacío
'C' = Contenedor

TU TAREA:
Selecciona EXACTAMENTE {CANTIDAD_C} coordenadas de la LISTA DE CANDIDATOS proporcionada abajo.
Estos candidatos son los mejores sitios disponibles (zonas con muchas casas).

OBJETIVO:
Elige la combinación estratégica de 3 coordenadas que minimice la distancia global a TODAS las casas.
No los pongas todos juntos; intenta cubrir el mapa.

LISTA DE MEJORES CANDIDATOS (Ordenados por calidad):
{ev.candidatos}

FORMATO RESPUESTA JSON:
{{"contenedores": [[fila, col], [fila, col], [fila, col]]}}

FEEDBACK ANTERIOR:
{ev.feedback}
"""
        # Llamada directa (sin bucle de reintento, usa API Key nueva)
        response = await Settings.llm.acomplete(prompt)
        
        return EvaluationEvent(
            coordenadas_json=response.text, 
            intento_actual=ev.intento_actual,
            best_score=ev.best_score,
            best_coords=ev.best_coords,
            candidatos=ev.candidatos
        )

    @step
    async def evaluator(self, ev: EvaluationEvent) -> GenerationEvent | StopEvent:
        current_best_score = ev.best_score
        current_best_coords = ev.best_coords
        
        # 1. Parsear JSON
        coords = []
        try:
            texto = ev.coordenadas_json
            match = re.search(r'\{.*\}', texto, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                coords = data.get("contenedores", [])
            else:
                raise ValueError("No se encontró JSON")
        except Exception as e:
            if ev.intento_actual < MAX_INTENTOS:
                return GenerationEvent(
                    feedback="Error de formato JSON. Devuelve solo el JSON estricto.", 
                    intento_actual=ev.intento_actual+1,
                    best_score=current_best_score,
                    best_coords=current_best_coords,
                    candidatos=ev.candidatos
                )
            coords = []

        # 2. Validar y Puntuar
        score = float('inf')
        errores = validar_reglas_basicas(self.mapa_original, coords)
        
        if not errores:
            # Reconstruir mapa temporal para calcular distancias BFS
            mapa_con_c, _ = aplicar_coordenadas(self.mapa_original, coords)
            score = calcular_distancia_total(mapa_con_c)
            print(f"✅ Válido. Coords: {coords} -> Score: {score}")

            if score < current_best_score:
                print(f"🚀 ¡NUEVO RÉCORD! ({score})")
                current_best_score = score
                current_best_coords = coords
                feedback = f"Récord: {score}. Intenta mejorar eligiendo otros candidatos de la lista para cubrir mejor las zonas lejanas."
            else:
                feedback = f"Solución válida ({score}) pero no supera tu récord ({current_best_score}). Prueba una distribución diferente."
        else:
            print(f"❌ Inválido: {coords}")
            feedback = "Coordenadas inválidas. Asegúrate de elegir SOLO de la lista proporcionada."

        # 3. Finalizar si es el último intento
        if ev.intento_actual >= MAX_INTENTOS:
            print("\n🏁 Fin de intentos. Generando resultado final...")
            if not current_best_coords:
                print("⚠️ El agente falló en encontrar una solución válida.")
                return StopEvent(result=None)
            
            mapa_final, _ = aplicar_coordenadas(self.mapa_original, current_best_coords)
            mapa_str = "\n".join(["".join(fila) for fila in mapa_final])
            print(f"🏆 TERMINADO. Mejor Score: {current_best_score}")
            return StopEvent(result=mapa_str)
        
        # Siguiente vuelta
        return GenerationEvent(
            feedback=feedback, 
            intento_actual=ev.intento_actual+1,
            best_score=current_best_score,
            best_coords=current_best_coords,
            candidatos=ev.candidatos
        )

# --- EJECUCIÓN ---
async def main():
    ruta = Path(__file__).parent / "entradas" / NOMBRE_MAPA
    if not ruta.exists():
        print(f"No encuentro {ruta}. Crea carpeta 'entradas'.")
        return

    mapa_orig = cargar_mapa(ruta)
    
    # Timeout seguro de 600s
    agent = CoordinateAgent(mapa_original=mapa_orig, timeout=600, verbose=False)
    
    try:
        resultado = await agent.run()
        if resultado:
            ruta_salida = Path(__file__).parent / "salidas" / NOMBRE_MAPA
            ruta_salida.parent.mkdir(exist_ok=True)
            with open(ruta_salida, "w") as f:
                f.write(str(resultado))
            print(f"💾 Guardado en {ruta_salida}")
    except Exception as e:
        print(f"Error fatal: {e}")

if __name__ == "__main__":
    asyncio.run(main())