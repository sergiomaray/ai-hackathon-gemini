"""
Power Grid Optimizer — Hackathon de IA (Universidad de Oviedo)

Coloca CANTIDAD_C transformadores eléctricos 'C' en un mapa de cuadrícula para
abastecer a las industrias 'T' minimizando la distancia a hospitales e industrias.

Leyenda del mapa:
    '-'  Espacio vacío
    'X'  Carretera
    'E'  Edificio restringido
    'T'  Industria
    'O'  Hospital
    'C'  Transformador (lo añade el programa)

Reglas de colocación:
    1. Exactamente CANTIDAD_C transformadores, en coordenadas distintas.
    2. Un transformador solo puede ir en una casilla '-'.
    3. Debe tocar (8 vecinos) al menos una carretera 'X'.
    4. No puede tocar (8 vecinos) ningún edificio restringido 'E'.
    5. Cada industria 'T' necesita al menos MIN_C_POR_T transformadores distintos
       a distancia <= RADIO_T.

Puntuación (menor es mejor):
    Suma de distancias de cada hospital 'O' y cada industria 'T' a su
    transformador más cercano. El movimiento es libre por toda la cuadrícula,
    así que la distancia es la distancia Manhattan |Δfila| + |Δcolumna|.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from llama_index.core.workflow import Workflow, Event, StartEvent, StopEvent, step

# --- VARIABLES GLOBALES ---
MAPA_POR_DEFECTO = "ejemplo_sintetico.txt"
CANTIDAD_C = 6
MAX_INTENTOS = 4
MODELO = "gemini-2.5-flash"
TEMPERATURA = 0.15
RADIO_T = 3          # Distancia máxima de un transformador a la industria que abastece
MIN_C_POR_T = 2      # Transformadores necesarios por industria
CANDIDATOS_POR_T = 8 # Candidatos que se "rescatan" alrededor de cada industria

BASE = Path(__file__).resolve().parent
VECINOS_8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


# --- Eventos simplificados ---
class GenerationEvent(Event):
    feedback: str
    intento_actual: int
    best_score: float
    best_coords: list
    candidatos: list
    ts_reales: list


class EvaluationEvent(Event):
    respuesta_llm: str
    intento_actual: int
    best_score: float
    best_coords: list
    candidatos: list
    ts_reales: list


# --- FUNCIONES PURAS DE LÓGICA Y VALIDACIÓN ---

def cargar_mapa(ruta):
    """Lee el mapa ignorando líneas vacías y comprueba que sea rectangular."""
    with open(ruta, "r", encoding="utf-8") as f:
        lineas = [linea.rstrip() for linea in f if linea.strip()]
    if not lineas:
        raise ValueError(f"El mapa {ruta} está vacío")
    anchos = {len(linea) for linea in lineas}
    if len(anchos) != 1:
        raise ValueError(f"El mapa {ruta} no es rectangular (anchos de fila: {sorted(anchos)})")
    return [list(linea) for linea in lineas]


def mapa_a_string_simple(mapa):
    """Convierte el mapa a texto con números de fila y una cabecera de columnas (módulo 10)."""
    ancho = len(str(len(mapa)))
    cabecera = " " * (ancho + 3) + "".join(str(c % 10) for c in range(len(mapa[0])))
    filas = [f"{i:0{ancho}d} | {''.join(fila)}" for i, fila in enumerate(mapa)]
    return "\n".join([cabecera] + filas)


def posiciones(mapa, simbolo):
    return [(r, c) for r, fila in enumerate(mapa) for c, valor in enumerate(fila) if valor == simbolo]


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def toca(mapa, r, c, simbolo):
    """True si alguna de las 8 casillas vecinas de (r, c) contiene `simbolo`."""
    filas, cols = len(mapa), len(mapa[0])
    return any(
        0 <= r + dr < filas and 0 <= c + dc < cols and mapa[r + dr][c + dc] == simbolo
        for dr, dc in VECINOS_8
    )


def es_posicion_valida(mapa, r, c):
    """Regla: '-' + Adyacente a 'X' + NO Adyacente a 'E'."""
    return mapa[r][c] == "-" and toca(mapa, r, c, "X") and not toca(mapa, r, c, "E")


def obtener_candidatos_estrategicos(mapa):
    """
    1. Regla: '-' + Adyacente a 'X' + NO Adyacente a 'E'.
    2. Priorizar cercanía a 'T' (Industria) y luego a 'O' (Hospital).
    """
    torres = posiciones(mapa, "T")
    hospitales = posiciones(mapa, "O")
    candidatos = []

    for r in range(len(mapa)):
        for c in range(len(mapa[0])):
            if es_posicion_valida(mapa, r, c):
                # Bonus si está cerca de una Industria 'T' (Radio 3)
                score = 50 * sum(1 for t in torres if manhattan((r, c), t) <= RADIO_T)
                # Bonus por estar cerca de Hospitales 'O' (Radio 5)
                score += 10 * sum(1 for h in hospitales if manhattan((r, c), h) <= 5)
                candidatos.append(([r, c], score))

    # Ordenar: Los que salvan a las 'T' van primero
    candidatos.sort(key=lambda x: x[1], reverse=True)
    return [coord for coord, _ in candidatos]


def limite_candidatos(filas):
    if filas < 20:
        return 60
    if filas < 30:
        return 100
    if filas < 40:
        return 140
    return 200


def seleccionar_candidatos_prompt(mapa, todos_candidatos):
    """
    Rescate inteligente: garantiza CANDIDATOS_POR_T opciones alrededor de cada
    industria y rellena con los mejores restantes, conservando el orden de prioridad.
    """
    elegidos = {}  # dict ordenado usado como "set ordenado"
    for t in posiciones(mapa, "T"):
        vecinos = [c for c in todos_candidatos if manhattan(c, t) <= RADIO_T]
        for v in vecinos[:CANDIDATOS_POR_T]:
            elegidos[tuple(v)] = None

    limite = max(limite_candidatos(len(mapa)), len(elegidos))
    for cand in todos_candidatos:
        if len(elegidos) >= limite:
            break
        elegidos.setdefault(tuple(cand), None)

    orden = {tuple(c): i for i, c in enumerate(todos_candidatos)}
    return [list(c) for c in sorted(elegidos, key=orden.__getitem__)]


def aplicar_coordenadas(mapa_orig, coordenadas):
    """Devuelve una copia del mapa con una 'C' en cada coordenada (ya validada)."""
    nuevo_mapa = [fila[:] for fila in mapa_orig]
    for r, c in coordenadas:
        nuevo_mapa[r][c] = "C"
    return nuevo_mapa


def extraer_coordenadas(texto, clave):
    """
    Busca en la respuesta del LLM todos los objetos JSON que contengan `clave`
    y devuelve el valor del ÚLTIMO (el razonamiento puede incluir ejemplos antes).
    Si no hay JSON válido, reintenta cambiando comillas simples por dobles.
    """
    decoder = json.JSONDecoder()
    for variante in (texto, texto.replace("'", '"')):
        encontrados = []
        i = variante.find("{")
        while i != -1:
            try:
                obj, fin = decoder.raw_decode(variante, i)
            except json.JSONDecodeError:
                i = variante.find("{", i + 1)
                continue
            if isinstance(obj, dict) and clave in obj:
                encontrados.append(obj[clave])
            i = variante.find("{", fin)
        if encontrados:
            return encontrados[-1]
    raise ValueError(f"No se encontró un JSON con la clave '{clave}' en la respuesta")


def normalizar_coordenadas(datos):
    """Convierte la salida del LLM en una lista de [fila, col] enteros sin lanzar excepciones."""
    if not isinstance(datos, list):
        return [], [f"Se esperaba una lista de coordenadas y llegó: {datos!r}"]
    coords, errores = [], []
    for item in datos:
        if (
            isinstance(item, (list, tuple)) and len(item) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and float(v).is_integer() for v in item)
        ):
            coords.append([int(item[0]), int(item[1])])
        else:
            errores.append(f"Elemento inválido {item!r}: cada coordenada debe ser [fila, col] con enteros.")
    return coords, errores


def validar_y_puntuar(mapa_orig, datos):
    """
    Ejecuta todas las validaciones del reto.
    Devuelve: (es_valido, lista_errores, puntuacion, coordenadas_normalizadas)
    """
    coords, errores = normalizar_coordenadas(datos)
    if errores:
        return False, errores, float("inf"), coords

    if len(coords) != CANTIDAD_C:
        errores.append(f"Cantidad incorrecta: enviaste {len(coords)} coordenadas y deben ser exactamente {CANTIDAD_C}.")

    repetidas = sorted({tuple(c) for c in coords if coords.count(c) > 1})
    if repetidas:
        errores.append(f"Coordenadas repetidas: {[list(r) for r in repetidas]}. Cada transformador debe ir en una casilla distinta.")

    filas, cols = len(mapa_orig), len(mapa_orig[0])
    for r, c in coords:
        if not (0 <= r < filas and 0 <= c < cols):
            errores.append(f"[{r},{c}] está fuera del mapa ({filas}x{cols}).")
        elif mapa_orig[r][c] != "-":
            errores.append(f"[{r},{c}] está ocupada por '{mapa_orig[r][c]}' (solo se permite '-').")
        elif not toca(mapa_orig, r, c, "X"):
            errores.append(f"[{r},{c}] no toca ninguna carretera 'X'.")
        elif toca(mapa_orig, r, c, "E"):
            errores.append(f"[{r},{c}] toca un edificio restringido 'E'.")

    if errores:
        return False, errores, float("inf"), coords

    # Cada T debe tener MIN_C_POR_T transformadores distintos a distancia <= RADIO_T
    torres = posiciones(mapa_orig, "T")
    for t in torres:
        cercanos = sum(1 for c in coords if manhattan(c, t) <= RADIO_T)
        if cercanos < MIN_C_POR_T:
            # si no cumple requisitos de 'T' le damos sugerencias
            sugerencias = [
                f"[{r},{c}]"
                for r in range(max(0, t[0] - RADIO_T), min(filas, t[0] + RADIO_T + 1))
                for c in range(max(0, t[1] - RADIO_T), min(cols, t[1] + RADIO_T + 1))
                if manhattan((r, c), t) <= RADIO_T and es_posicion_valida(mapa_orig, r, c) and [r, c] not in coords
            ]
            pista = f" (Sugerencias válidas cerca: {', '.join(sugerencias[:5])})" if sugerencias else ""
            errores.append(
                f"Regla VIOLADA: La T en ({t[0]},{t[1]}) tiene {cercanos} Cs cerca (necesita {MIN_C_POR_T}).{pista}"
            )
    if errores:
        return False, errores, float("inf"), coords

    # Si pasa las reglas, calculamos la puntuación total:
    # distancia de cada hospital 'O' e industria 'T' al transformador más cercano
    objetivos = posiciones(mapa_orig, "O") + torres
    total = sum(min(manhattan(p, c) for c in coords) for p in objetivos)
    return True, [], total, coords


def construir_prompt(mapa_str, ts_reales, candidatos, best_coords, best_score, feedback):
    mejor = (
        f"{best_coords} con puntuación {int(best_score)}"
        if best_coords else "Todavía no hay ninguna solución válida."
    )
    return f"""
Eres un experto en optimización espacial.

MAPA DE REFERENCIA (fila | contenido; la primera línea indica la columna módulo 10):
{mapa_str}

LEYENDA: '-' vacío, 'X' carretera, 'E' edificio restringido, 'T' industria, 'O' hospital.

REGLAS (todas obligatorias):
1. Coloca EXACTAMENTE {CANTIDAD_C} transformadores 'C' en coordenadas DISTINTAS.
2. Elige SOLO coordenadas de la LISTA DE CANDIDATOS (ya cumplen: casilla '-', tocan 'X', no tocan 'E').
3. Cada industria 'T' necesita al menos {MIN_C_POR_T} transformadores distintos a distancia Manhattan <= {RADIO_T}.

OBJETIVO:
Minimizar la suma de distancias Manhattan de cada hospital 'O' y cada industria 'T'
a su transformador más cercano.

OBJETIVOS CRÍTICOS (COORDENADAS REALES):
LISTA DE INDUSTRIAS 'T' [fila, col]: {ts_reales}

LISTA DE CANDIDATOS DISPONIBLES [fila, col] (de mayor a menor prioridad):
{candidatos}

MEJOR SOLUCIÓN HASTA AHORA:
{mejor}

FEEDBACK DEL INTENTO ANTERIOR:
{feedback}

INSTRUCCIONES DE RAZONAMIENTO (OBLIGATORIO):
1. Primero, PARA CADA 'T' de la lista, escribe explícitamente qué {MIN_C_POR_T} candidatos vas a usar para cubrirla.
   Ejemplo: "Para T(4,13) usaré (3,13) y (5,13)".
2. Verifica que no estás dejando ninguna T sin cubrir.
3. Si el feedback anterior dice que falló una T específica, dale prioridad absoluta.
4. Usa los transformadores restantes para acercarte a los hospitales.
5. Al final, genera el bloque JSON.

FORMATO RESPUESTA (el JSON debe ir al final, con exactamente {CANTIDAD_C} pares [fila, col]):
Razonamiento aquí...
```json
{{"transformadores": [[fila, col], [fila, col], ...]}}
```
"""


# --- Workflow simplificado (3 steps) ---
class EnergyAgent(Workflow):

    def __init__(self, mapa_original, modelo_llm, max_intentos=MAX_INTENTOS, **kwargs):
        super().__init__(**kwargs)
        self.mapa_original = mapa_original
        self.modelo_llm = modelo_llm
        self.max_intentos = max_intentos
        self.mapa_str = mapa_a_string_simple(mapa_original)
        self.rows = len(mapa_original)
        self.cols = len(mapa_original[0])

    # Step 1: Analizar problema completo
    @step
    async def start(self, ev: StartEvent) -> GenerationEvent:
        print("· Analizamos candidatos")
        todos = obtener_candidatos_estrategicos(self.mapa_original)
        candidatos = seleccionar_candidatos_prompt(self.mapa_original, todos)
        torres = [list(t) for t in posiciones(self.mapa_original, "T")]

        print(f"· Mapa {self.rows}x{self.cols}, mandamos {len(candidatos)} candidatos al LLM")
        print(f"· Hay {len(torres)} Industrias T. Enviando coordenadas exactas al LLM.")

        return GenerationEvent(
            feedback="Primer intento.",
            intento_actual=1,
            best_score=float("inf"),
            best_coords=[],
            candidatos=candidatos,
            ts_reales=torres,
        )

    # Step 2: Generar solución
    @step
    async def generator(self, ev: GenerationEvent) -> EvaluationEvent:
        print(f"\n- INTENTO {ev.intento_actual}/{self.max_intentos}...")
        prompt = construir_prompt(self.mapa_str, ev.ts_reales, ev.candidatos, ev.best_coords, ev.best_score, ev.feedback)
        response = await self.modelo_llm.acomplete(prompt)
        return EvaluationEvent(
            respuesta_llm=response.text or "",
            intento_actual=ev.intento_actual,
            best_score=ev.best_score,
            best_coords=ev.best_coords,
            candidatos=ev.candidatos,
            ts_reales=ev.ts_reales,
        )

    # Step 3: Evalúa la solución y decide si reintentar o finalizar
    @step
    async def evaluator(self, ev: EvaluationEvent) -> GenerationEvent | StopEvent:
        current_best_score = ev.best_score
        current_best_coords = ev.best_coords

        try:
            datos = extraer_coordenadas(ev.respuesta_llm, "transformadores")
        except ValueError as e:
            print(f"⚠️ Error de Parseo en intento {ev.intento_actual}: {e}")
            es_valido, score, coords = False, float("inf"), []
            errores = [f"ERROR DE FORMATO: {e}. NO ESCRIBAS CÓDIGO PYTHON. Termina con el bloque JSON pedido."]
        else:
            # --- VALIDACIÓN ---
            es_valido, errores, score, coords = validar_y_puntuar(self.mapa_original, datos)

        if es_valido:
            print(f"✅ Intento {ev.intento_actual} VÁLIDO. Score: {score}")
            if score < current_best_score:
                print(f"🏆 ¡NUEVO RÉCORD! ({score})")
                current_best_score = score
                current_best_coords = coords
                feedback = f"Récord ({score}). Intenta bajarlo más."
            else:
                feedback = f"Válido ({score}) pero peor que récord ({int(current_best_score)})."
        else:
            print(f"❌ Intento {ev.intento_actual} INVÁLIDO: {errores[0]}")
            feedback = "ERRORES:\n" + "\n".join(errores[:5])

        # --- CIERRE ---
        if ev.intento_actual >= self.max_intentos:
            if not current_best_coords:
                print("💀 Fracaso total.")
                return StopEvent(result=None)

            mapa_final = aplicar_coordenadas(self.mapa_original, current_best_coords)
            return StopEvent(result={
                "mapa_texto": "\n".join("".join(fila) for fila in mapa_final),
                "score": int(current_best_score),
                "coords": current_best_coords,
            })

        return GenerationEvent(
            feedback=feedback,
            intento_actual=ev.intento_actual + 1,
            best_score=current_best_score,
            best_coords=current_best_coords,
            candidatos=ev.candidatos,
            ts_reales=ev.ts_reales,
        )


# --- Ejecución ---

def resolver_ruta_mapa(nombre):
    """Acepta una ruta existente o el nombre de un fichero dentro de 'entradas/'."""
    ruta = Path(nombre)
    return ruta if ruta.is_file() else BASE / "entradas" / nombre


def crear_llm(modelo):
    """Crea el cliente de Gemini leyendo la API key de GOOGLE_API_KEY (o de un fichero .env)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Falta la variable de entorno GOOGLE_API_KEY. "
            "Copia .env.example a .env y pon tu clave de Google AI Studio."
        )
    from llama_index.llms.google_genai import GoogleGenAI
    return GoogleGenAI(model=modelo, api_key=api_key, temperature=TEMPERATURA)


async def main(argv=None):
    parser = argparse.ArgumentParser(description="Coloca transformadores para abastecer industrias y hospitales.")
    parser.add_argument("mapa", nargs="?", default=MAPA_POR_DEFECTO,
                        help=f"Fichero de 'entradas/' o ruta a un mapa (por defecto: {MAPA_POR_DEFECTO})")
    parser.add_argument("--intentos", type=int, default=MAX_INTENTOS, help="Número de intentos del LLM")
    parser.add_argument("--modelo", default=MODELO, help="Modelo de Gemini a usar")
    args = parser.parse_args(argv)

    # Tiempo inicial
    tiempo_inicio = time.time()
    print(f"🕐 Inicio del programa: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    ruta = resolver_ruta_mapa(args.mapa)
    if not ruta.is_file():
        print(f"Falta {ruta}")
        return 1

    try:
        mapa_orig = cargar_mapa(ruta)
        llm = crear_llm(args.modelo)
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        return 1

    agent = EnergyAgent(mapa_original=mapa_orig, modelo_llm=llm, max_intentos=args.intentos,
                        timeout=600, verbose=False)

    try:
        resultado = await agent.run()
    except Exception as e:
        print(f"Error fatal: {e}")
        return 1

    if not resultado:
        return 1

    # Output en fichero
    ruta_salida = BASE / "salidas" / ruta.name
    ruta_salida.parent.mkdir(exist_ok=True)
    ruta_salida.write_text(resultado["mapa_texto"] + "\n", encoding="utf-8")

    print("\n" + "=" * 50)
    print("MAPA GENERADO POR LLM:")
    print("=" * 50)
    print(resultado["mapa_texto"])
    print(f"\n✅ Mapa guardado en {ruta_salida}")

    # Output de Puntuación y Tiempo (Formato del Profesor)
    print(f"\nSuma total de pasos (Hospitales O + Industrias T) a los transformadores C más cercanos: {resultado['score']}")
    print(f"\n🕐 Fin del programa: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"⏱️  Tiempo transcurrido: {time.time() - tiempo_inicio:.2f} segundos")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))