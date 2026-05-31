import asyncio
import sys
import json
import re
import time
import ast
from datetime import datetime
from collections import deque
from pathlib import Path
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
from llama_index.core.workflow import Workflow, Event, StartEvent, StopEvent, step

# --- VARIABLES GLOBALES ---
NOMBRE_MAPA = "1.txt"
CANTIDAD_C = 6          
MAX_INTENTOS = 4        

local_llm = GoogleGenAI(
        model="gemini-2.5-flash", 
        api_key="Insert your API key", 
        temperature=0.15 
    )
Settings.llm = local_llm


# --- Eventos simplificados ---
class GenerationEvent(Event):
    feedback: str
    intento_actual: int
    best_score: float
    best_coords: list
    candidatos: list 
    ts_reales: list

class EvaluationEvent(Event):
    coordenadas_json: str
    intento_actual: int
    best_score: float
    best_coords: list
    candidatos: list
    ts_reales: list

# ---FUNCIONES PURAS DE LÓGICA Y VALIDACIÓN---
def cargar_mapa(ruta):
    with open(ruta, "r") as f:
        lines = [line.rstrip() for line in f.readlines() if line.strip()]
    return [list(line) for line in lines]

def mapa_a_string_simple(mapa):
    res = []
    ancho = len(str(len(mapa)))
    for i, fila in enumerate(mapa):
        res.append(f"{i:0{ancho}d} | {''.join(fila)}")
    return "\n".join(res)

def distancia_minima(mapa, start, targets):
    filas, cols = len(mapa), len(mapa[0])
    targets_set = set(tuple(t) for t in targets)
    visitado = [[False]*cols for _ in range(filas)]
    queue = deque([(start[0], start[1], 0)])
    visitado[start[0]][start[1]] = True
    while queue:
        i, j, d = queue.popleft()
        if (i, j) in targets_set:
            return d
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < filas and 0 <= nj < cols and not visitado[ni][nj]:
                visitado[ni][nj] = True
                queue.append((ni, nj, d+1))
        
    return float('inf')

def obtener_candidatos_estrategicos(mapa):
    """
    1. Regla: '-' + Adyacente a 'X' + NO Adyacente a 'E'.
    2. Priorizar cercanía a 'T' (Industria) y luego a 'O' (Hospital).
    """
    filas = len(mapa)
    cols = len(mapa[0])
    candidatos = []
    
    # Localizar puntos de interés para calcular puntuación
    torres = [(r,c) for r in range(filas) for c in range(cols) if mapa[r][c] == 'T']
    hospitales = [(r,c) for r in range(filas) for c in range(cols) if mapa[r][c] == 'O']

    for r in range(filas):
        for c in range(cols):
            # 1. SOLO MIRAMOS LOS '-'
            if mapa[r][c] == '-':
                tiene_x = False
                tiene_e = False
                
                # Revisar 8 vecinos
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < filas and 0 <= nc < cols:
                            vecino = mapa[nr][nc]
                            if vecino == 'X': tiene_x = True
                            if vecino == 'E': tiene_e = True
                
                # CONDICIÓN Toca X y NO toca E
                if tiene_x and not tiene_e:
                    # 2. CALCULAR IMPORTANCIA
                    score = 0
                    
                    # Bonus si está cerca de una Industria 'T' (Radio 3)
                    for tr, tc in torres:
                        dist = abs(r-tr) + abs(c-tc)
                        if dist <= 3:
                            score += 50 # 
                    
                    # Bonus por estar cerca de Hospitales 'O' (Radio 5)
                    for hr, hc in hospitales:
                        dist = abs(r-hr) + abs(c-hc)
                        if dist <= 5:
                            score += 10
                    
                    candidatos.append({'coord': [r, c], 'score': score})
    
    # Ordenar: Los que salvan a las 'T' van primero
    candidatos.sort(key=lambda x: x['score'], reverse=True)
    return [c['coord'] for c in candidatos]

def aplicar_coordenadas(mapa_orig, coordenadas):
    import copy
    nuevo_mapa = copy.deepcopy(mapa_orig)
    filas = len(nuevo_mapa)
    cols = len(nuevo_mapa[0])
    for coord in coordenadas:
        if (len(coord) == 2 and isinstance(coord, list)):
            r, c = coord
            if 0 <= r < filas and 0 <= c < cols:
                nuevo_mapa[r][c] = 'C'
    return nuevo_mapa

def validar_y_puntuar(mapa_orig, coordenadas):
    """
    Ejecuta todas las validaciones del reto.
    Devuelve: (es_valido, lista_errores, puntuacion)
    """
    errores = []
    filas, cols = len(mapa_orig), len(mapa_orig[0])
    
    if len(coordenadas) != CANTIDAD_C:
        return False, [f"Cantidad incorrecta: {len(coordenadas)} vs {CANTIDAD_C}"], float('inf')

    
    mapa_temp = aplicar_coordenadas(mapa_orig, coordenadas)
    
    # Cada T debe tener 2 Cs a distancia <= 3
    torres = [(r,c) for r in range(filas) for c in range(cols) if mapa_orig[r][c] == 'T']
    transformadores = coordenadas 
    
    # Pre-calculamos candidatos válidos para sugerir en caso de error
    
    for tr, tc in torres:
        c_cercanos = 0
        for cr, cc in transformadores:
            d = distancia_minima(mapa_temp, [tr, tc], [[cr, cc]])
            if d <= 3:
                c_cercanos += 1
        
        if c_cercanos < 2:
            # si no cumple requisitos de 'T' le damos sugerencias
            sugerencias = []
            for r in range(max(0, tr-3), min(filas, tr+4)):
                for c in range(max(0, tc-3), min(cols, tc+4)):
                    if abs(r-tr) + abs(c-tc) <= 3:
                        if mapa_orig[r][c] == '-': 
                             # Verificar reglas básicas (X y no E) 
                             tiene_x = False; tiene_e = False
                             for dr in [-1,0,1]:
                                 for dc in [-1,0,1]:
                                     if dr==0 and dc==0: continue
                                     nr,nc = r+dr, c+dc
                                     if 0<=nr<filas and 0<=nc<cols:
                                         if mapa_orig[nr][nc]=='X': tiene_x=True
                                         if mapa_orig[nr][nc]=='E': tiene_e=True
                             if tiene_x and not tiene_e:
                                 sugerencias.append(f"[{r},{c}]")
            
            pista = f" (Sugerencias válidas cerca: {', '.join(sugerencias[:5])})" if sugerencias else ""
            errores.append(f"Regla VIOLADA: La T en ({tr},{tc}) tiene {c_cercanos} Cs cerca (necesita 2).{pista}")
    if errores:
        return False, errores, float('inf')

    # Si pasa las reglas, calculamos la puntuación total
    hospitales = [(r,c) for r in range(filas) for c in range(cols) if mapa_orig[r][c] == 'O']
    total_dist = 0
    
    # Distancia O -> C más cercano
    for hr, hc in hospitales:
        d = distancia_minima(mapa_temp, [hr, hc], transformadores)
        if d == float('inf'): return False, ["Hospital inalcanzable"], float('inf')
        total_dist += d
        
    # Distancia T -> C más cercano
    for tr, tc in torres:
        d = distancia_minima(mapa_temp, [tr, tc], transformadores)
        if d == float('inf'): return False, ["Industria inalcanzable"], float('inf')
        total_dist += d
        
    return True, [], total_dist

# --- Workflow simplificado (3 steps) ---
class EnergyAgent(Workflow):
    
    def __init__(self, mapa_original, **kwargs):
        super().__init__(**kwargs)
        self.mapa_original = mapa_original
        self.mapa_str = mapa_a_string_simple(mapa_original)
        self.rows = len(mapa_original)
        self.cols = len(mapa_original[0])
        
        
        print("· Analizamos candidatos")
        self.todos_candidatos = obtener_candidatos_estrategicos(mapa_original)

    # Step 1: Analizar problema completo
    @step
    async def start(self, ev: StartEvent) -> GenerationEvent:
        # 1. Obtener candidatos
        self.todos_candidatos = obtener_candidatos_estrategicos(self.mapa_original)
        
        # 2. Rescate Inteligente (Lógica de conjuntos)
        candidatos_finales = set()
        torres = [(r,c) for r in range(self.rows) for c in range(self.cols) if self.mapa_original[r][c] == 'T']
        
        # Rescatar vecinos de las T
        for tr, tc in torres:
            vecinos = [c for c in self.todos_candidatos if (abs(c[0]-tr) + abs(c[1]-tc)) <= 3]
            for v in vecinos[:8]: # Aseguramos 8 opciones por T
                candidatos_finales.add(tuple(v))
        
        filas = len(self.mapa_original)

        if filas < 20:
            LIMIT_FINAL = 60 
        elif filas < 30:
            LIMIT_FINAL = 100
        elif filas < 40:
            LIMIT_FINAL = 140
        else:
            LIMIT_FINAL = 200

        for cand in self.todos_candidatos:
            if len(candidatos_finales) >= LIMIT_FINAL: break
            candidatos_finales.add(tuple(cand))
            
        lista_final = list(candidatos_finales)
        
        print(f"· Mapa {self.rows}x{self.cols}, mandamos {len(lista_final)} candidatos al LLM")

        print(f"· Hay {len(torres)} Industrias T. Enviando coordenadas exactas al LLM.")

        return GenerationEvent(
            feedback="INICIO.", 
            intento_actual=1,
            best_score=float('inf'),
            best_coords=[],
            candidatos=lista_final,
            ts_reales=torres
        )

    # Step 2: Generar y validar solución (con reintentos)
    @step
    async def generator(self, ev: GenerationEvent) -> EvaluationEvent | StopEvent:
        if ev.intento_actual > MAX_INTENTOS:
             return StopEvent(result="MAX_INTENTOS_EXCEEDED")
        
        print(f"\n- INTENTO {ev.intento_actual}/{MAX_INTENTOS}...")

        lista_ts_str = str(ev.ts_reales)

        prompt = f"""
MAPA DE REFERENCIA:
{self.mapa_str}

OBJETIVOS CRÍTICOS (COORDENADAS REALES):
LISTA DE INDUSTRIAS 'T': {lista_ts_str}

TU TAREA:
Selecciona EXACTAMENTE {CANTIDAD_C} coordenadas de la LISTA DE CANDIDATOS.

INSTRUCCIONES DE RAZONAMIENTO (OBLIGATORIO):
1. Primero, PARA CADA 'T' de la lista, escribe explícitamente qué 2 candidatos vas a usar para cubrirla.
   Ejemplo: "Para T(4,13) usaré (3,13) y (5,13)".
2. Verifica que no estás dejando ninguna T sin cubrir.
3. Si el feedback anterior dice que falló una T específica, dale prioridad absoluta.
4. Al final, genera el bloque JSON.

LISTA DE CANDIDATOS DISPONIBLES:
{ev.candidatos}

FEEDBACK DEL INTENTO ANTERIOR:
{ev.feedback}

FORMATO RESPUESTA (El JSON debe ir al final):
Razonamiento aquí...
```json
{{"transformadores": [[fila, col], [fila, col]]}}

FEEDBACK ANTERIOR:
{ev.feedback}
"""
        response = await Settings.llm.acomplete(prompt)
        return EvaluationEvent(
            coordenadas_json=response.text, 
            intento_actual=ev.intento_actual,
            best_score=ev.best_score,
            best_coords=ev.best_coords,
            candidatos=ev.candidatos,
            ts_reales=ev.ts_reales
        )

    # Step 3: Evalúa la solucion y finaliza
    @step
    async def evaluator(self, ev: EvaluationEvent) -> GenerationEvent | StopEvent:
        current_best_score = ev.best_score
        current_best_coords = ev.best_coords
        
        coords = []
        raw_text = ev.coordenadas_json
        
        

        try:
            # 1. Limpieza inicial
            text_clean = raw_text.replace("```json", "").replace("```", "").strip()
            
            # limpiamos todo
            match = re.search(r'\{\s*["\']transformadores["\']\s*:\s*\[.*?\]\s*\}', text_clean, re.DOTALL)
            
            if match:
                json_candidate = match.group(0)
                # Corrección de comillas simples a dobles
                json_candidate = json_candidate.replace("'", '"')
                
                try:
                    data = json.loads(json_candidate)
                except:
                    # si no va parseamos con python
                    data = ast.literal_eval(json_candidate)
                
                coords = data.get("transformadores", [])
            else:
                # Si sigue fallando, búsqueda genérica pero solo del bloque final
                matches = list(re.finditer(r'\{.*\}', text_clean, re.DOTALL))
                if matches:
                    # Probamos el último bloque encontrado (suele ser el resultado final)
                    last_match = matches[-1].group(0).replace("'", '"')
                    data = json.loads(last_match)
                    coords = data.get("transformadores", [])
                else:
                    raise ValueError("No se encontró la clave 'transformadores' en la respuesta")

            if not isinstance(coords, list): raise ValueError("No es una lista")

        except Exception as e:
            print(f"⚠️ Error de Parseo en intento {ev.intento_actual}: {e}")
            if ev.intento_actual < MAX_INTENTOS:
                return GenerationEvent(
                    feedback=f"ERROR DE FORMATO: {str(e)}. NO ESCRIBAS CÓDIGO PYTHON. Devuelve SOLO el JSON final.", 
                    intento_actual=ev.intento_actual+1,
                    best_score=current_best_score,
                    best_coords=current_best_coords,
                    candidatos=ev.candidatos,
                    ts_reales=ev.ts_reales
                )
            coords = []

        # --- VALIDACIÓN ---
        es_valido, errores, score = validar_y_puntuar(self.mapa_original, coords)
        
        feedback = ""
        if es_valido:
            print(f"✅ Intento {ev.intento_actual} VÁLIDO. Score: {score}")
            if score < current_best_score:
                print(f"🏆 ¡NUEVO RÉCORD! ({score})")
                current_best_score = score
                current_best_coords = coords
                feedback = f"Récord ({score}). Intenta bajarlo más."
            else:
                feedback = f"Válido ({score}) pero peor que récord ({current_best_score})."
        else:
            print(f"❌ Intento {ev.intento_actual} INVÁLIDO: {errores[0] if errores else 'Error desconocido'}")
            feedback = "ERRORES:\n" + "\n".join(errores[:3])

        # --- CIERRE ---
        if ev.intento_actual >= MAX_INTENTOS:
            if not current_best_coords:
                print("💀 Fracaso total.")
                return StopEvent(result=None)
            
            mapa_final = aplicar_coordenadas(self.mapa_original, current_best_coords)
            mapa_str = "\n".join(["".join(fila) for fila in mapa_final])
            
            return StopEvent(result={
                "mapa_texto": mapa_str,
                "score": current_best_score,
                "coords": current_best_coords
            })
        
        return GenerationEvent(
            feedback=feedback, 
            intento_actual=ev.intento_actual+1,
            best_score=current_best_score,
            best_coords=current_best_coords,
            candidatos=ev.candidatos,
            ts_reales=ev.ts_reales
        )

# --- Ejecución ---
async def main(): 
    # Tiempo inicial
    tiempo_inicio = time.time()
    timestamp_inicio = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🕐 Inicio del programa: {timestamp_inicio}\n")

    ruta = Path(__file__).parent / "entradas" / NOMBRE_MAPA
    if not ruta.exists():
        print(f"Falta {ruta}")
        return

    mapa_orig = cargar_mapa(ruta)
    
    agent = EnergyAgent(mapa_original=mapa_orig, timeout=600, verbose=False)
    
    try:
        resultado = await agent.run()
        
        if resultado:
            # 2. Output en fichero
            ruta_salida = Path(__file__).parent / "salidas" / NOMBRE_MAPA
            ruta_salida.parent.mkdir(exist_ok=True)
            with open(ruta_salida, "w") as f:
                f.write(str(resultado["mapa_texto"]))
            
            print("\n" + "="*50)
            print("MAPA GENERADO POR LLM:")
            print("="*50)
            print(resultado["mapa_texto"])
            print(f"\n✅ Mapa guardado en {ruta_salida}")
            
            # 3. Output de Puntuación y Tiempo (Formato del Profesor)
          
            print(f"\nSuma total de pasos (Hospitales O + Industrias T) a los transformadores C más cercanos: {resultado['score']}")
            # Tiempo final
            
            tiempo_fin = time.time()
            timestamp_fin = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            tiempo_transcurrido = tiempo_fin - tiempo_inicio
            print(f"\n🕐 Fin del programa: {timestamp_fin}")
            print(f"⏱️  Tiempo transcurrido: {tiempo_transcurrido:.2f} segundos")
            
    except Exception as e:
        print(f"Error fatal: {e}")

if __name__ == "__main__":
    asyncio.run(main())