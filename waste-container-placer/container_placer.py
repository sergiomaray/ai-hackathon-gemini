"""
Waste Container Placer — Hackathon de IA (Universidad de Oviedo)

Coloca CANTIDAD_C contenedores 'C' en un mapa de cuadrícula para minimizar la
distancia total que recorren los vecinos de todas las casas 'O'.

Leyenda del mapa:
    '-'  Espacio vacío
    'X'  Carretera
    'O'  Casa
    'C'  Contenedor (lo añade el programa)

Reglas de colocación:
    1. Exactamente CANTIDAD_C contenedores, en coordenadas distintas.
    2. Un contenedor solo puede ir en una casilla '-'.
    3. Un contenedor debe tocar (8 vecinos) al menos una carretera 'X'.

Puntuación (menor es mejor):
    Suma, para cada casa 'O', de la distancia a su contenedor más cercano.
    El movimiento es libre por toda la cuadrícula, así que la distancia es la
    distancia Manhattan |Δfila| + |Δcolumna|.

Enfoque:
    Python calcula candidatos (densidad de casas + dispersión), Gemini elige la
    combinación, Python la valida y puntúa, y el resultado vuelve a Gemini como
    feedback durante varios intentos (LlamaIndex Workflow).
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

# --- CONFIGURACIÓN ---
MAPA_POR_DEFECTO = "4.txt"   # Nombre del archivo en la carpeta 'entradas'
CANTIDAD_C = 6               # Número de contenedores a colocar
MAX_INTENTOS = 4             # Intentos del LLM para elegir la mejor combinación
MODELO = "gemini-2.5-flash"
TEMPERATURA = 0.2            # Temperatura baja para máxima precisión
LIMITE_CANDIDATOS = 50       # Candidatos que se envían al LLM
RADIO_DENSIDAD = 2           # Radio para calcular la densidad local de casas
DISTANCIA_DISPERSION = 15    # Distancia Manhattan mínima entre candidatos dispersos

BASE = Path(__file__).resolve().parent
VECINOS_8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


# --- EVENTOS (Transportan el estado entre pasos) ---
class GenerationEvent(Event):
    feedback: str
    intento_actual: int
    best_score: float
    best_coords: list
    candidatos: list


class EvaluationEvent(Event):
    respuesta_llm: str
    intento_actual: int
    best_score: float
    best_coords: list
    candidatos: list


# --- FUNCIONES PURAS (Lógica de Python) ---

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


def toca(mapa, r, c, simbolo):
    """True si alguna de las 8 casillas vecinas de (r, c) contiene `simbolo`."""
    filas, cols = len(mapa), len(mapa[0])
    return any(
        0 <= r + dr < filas and 0 <= c + dc < cols and mapa[r + dr][c + dc] == simbolo
        for dr, dc in VECINOS_8
    )


def es_posicion_valida(mapa, r, c):
    return mapa[r][c] == "-" and toca(mapa, r, c, "X")


def obtener_candidatos_por_densidad(mapa):
    """
    ESTRATEGIA: DENSIDAD CON DISPERSIÓN.
    1. Calcula la densidad de casas alrededor de cada sitio válido.
    2. Ordena de mejor a peor.
    3. Selecciona candidatos asegurando que estén lejos unos de otros
       (Non-Maximum Suppression) y, si faltan, rellena con los mejores restantes.
    """
    filas, cols = len(mapa), len(mapa[0])
    todos = []

    # 1. RECOLECTAR TODOS LOS SITIOS VÁLIDOS Y SU CALIDAD
    for r in range(filas):
        for c in range(cols):
            if es_posicion_valida(mapa, r, c):
                casas_cerca = sum(
                    1
                    for ir in range(max(0, r - RADIO_DENSIDAD), min(filas, r + RADIO_DENSIDAD + 1))
                    for ic in range(max(0, c - RADIO_DENSIDAD), min(cols, c + RADIO_DENSIDAD + 1))
                    if mapa[ir][ic] == "O"
                )
                todos.append(([r, c], casas_cerca))

    # Ordenar por densidad (de más casas a menos)
    todos.sort(key=lambda x: x[1], reverse=True)

    # 2. SELECCIÓN CON DISPERSIÓN (Non-Maximum Suppression)
    seleccionados = []
    for coord, _ in todos:
        if all(abs(coord[0] - s[0]) + abs(coord[1] - s[1]) >= DISTANCIA_DISPERSION for s in seleccionados):
            seleccionados.append(coord)
        if len(seleccionados) >= LIMITE_CANDIDATOS:
            break

    # 3. Si no llegamos al límite con la restricción de distancia, rellenamos con
    # los mejores restantes (aunque estén cerca) para que el LLM tenga opciones
    vistos = {tuple(s) for s in seleccionados}
    for coord, _ in todos:
        if len(seleccionados) >= LIMITE_CANDIDATOS:
            break
        if tuple(coord) not in vistos:
            seleccionados.append(coord)
            vistos.add(tuple(coord))

    return seleccionados


def aplicar_coordenadas(mapa_orig, coordenadas):
    """Devuelve una copia del mapa con una 'C' en cada coordenada (ya validada)."""
    nuevo_mapa = [fila[:] for fila in mapa_orig]
    for r, c in coordenadas:
        nuevo_mapa[r][c] = "C"
    return nuevo_mapa


def calcular_distancia_total(mapa_orig, coordenadas):
    """Suma de distancias Manhattan de cada casa 'O' a su contenedor más cercano."""
    if not coordenadas:
        return float("inf")
    return sum(
        min(abs(hr - r) + abs(hc - c) for r, c in coordenadas)
        for hr, hc in posiciones(mapa_orig, "O")
    )


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
        errores.append(f"Coordenadas repetidas: {[list(r) for r in repetidas]}. Cada contenedor debe ir en una casilla distinta.")

    filas, cols = len(mapa_orig), len(mapa_orig[0])
    for r, c in coords:
        if not (0 <= r < filas and 0 <= c < cols):
            errores.append(f"[{r},{c}] está fuera del mapa ({filas}x{cols}).")
        elif mapa_orig[r][c] != "-":
            errores.append(f"[{r},{c}] está ocupada por '{mapa_orig[r][c]}' (solo se permite '-').")
        elif not toca(mapa_orig, r, c, "X"):
            errores.append(f"[{r},{c}] no toca ninguna carretera 'X'.")

    if errores:
        return False, errores, float("inf"), coords
    return True, [], calcular_distancia_total(mapa_orig, coords), coords


def construir_prompt(mapa_str, candidatos, best_coords, best_score, feedback):
    mejor = (
        f"{best_coords} con puntuación {int(best_score)}"
        if best_coords else "Todavía no hay ninguna solución válida."
    )
    return f"""
Eres un experto en optimización espacial.

MAPA DE LA CIUDAD (fila | contenido; la primera línea indica la columna módulo 10):
{mapa_str}

LEYENDA MAPA:
'O' = Casas (Tu objetivo es estar cerca de ellas)
'X' = Carretera
'-' = Espacio vacío
'C' = Contenedor

TU TAREA:
Selecciona EXACTAMENTE {CANTIDAD_C} coordenadas DISTINTAS de la LISTA DE CANDIDATOS.
Todos los candidatos ya cumplen las reglas de colocación (casilla '-' junto a una carretera 'X').

OBJETIVO:
Elige la combinación de {CANTIDAD_C} coordenadas que minimice la suma de distancias
Manhattan de TODAS las casas 'O' a su contenedor más cercano.
No los pongas todos juntos; reparte los contenedores para cubrir todo el mapa.

LISTA DE CANDIDATOS [fila, col] (de mayor a menor densidad de casas):
{candidatos}

MEJOR SOLUCIÓN HASTA AHORA:
{mejor}

FEEDBACK DEL INTENTO ANTERIOR:
{feedback}

FORMATO DE RESPUESTA:
Razona brevemente y termina con un único bloque JSON con exactamente {CANTIDAD_C} pares [fila, col]:
```json
{{"contenedores": [[fila, col], [fila, col], ...]}}
```
"""


# --- AGENTE / WORKFLOW ---

class CoordinateAgent(Workflow):

    def __init__(self, mapa_original, modelo_llm, max_intentos=MAX_INTENTOS, **kwargs):
        super().__init__(**kwargs)
        self.mapa_original = mapa_original
        self.modelo_llm = modelo_llm
        self.max_intentos = max_intentos
        self.mapa_str = mapa_a_string_simple(mapa_original)
        self.rows = len(mapa_original)
        self.cols = len(mapa_original[0])

    @step
    async def start(self, ev: StartEvent) -> GenerationEvent:
        # PASO CRÍTICO: Pre-calcular los mejores sitios antes de empezar
        print("📊 Analizando densidad de casas en el mapa...")
        candidatos = obtener_candidatos_por_densidad(self.mapa_original)
        print(f"🗺️ Mapa {self.rows}x{self.cols}. Enviando {len(candidatos)} candidatos (densos y dispersos) al LLM.")

        return GenerationEvent(
            feedback="Primer intento.",
            intento_actual=1,
            best_score=float("inf"),
            best_coords=[],
            candidatos=candidatos,
        )

    @step
    async def generator(self, ev: GenerationEvent) -> EvaluationEvent:
        print(f"\n🧠 Intento {ev.intento_actual}/{self.max_intentos}...")
        prompt = construir_prompt(self.mapa_str, ev.candidatos, ev.best_coords, ev.best_score, ev.feedback)
        response = await self.modelo_llm.acomplete(prompt)

        return EvaluationEvent(
            respuesta_llm=response.text or "",
            intento_actual=ev.intento_actual,
            best_score=ev.best_score,
            best_coords=ev.best_coords,
            candidatos=ev.candidatos,
        )

    @step
    async def evaluator(self, ev: EvaluationEvent) -> GenerationEvent | StopEvent:
        current_best_score = ev.best_score
        current_best_coords = ev.best_coords

        # 1. Parsear JSON
        try:
            datos = extraer_coordenadas(ev.respuesta_llm, "contenedores")
        except ValueError as e:
            es_valido, errores, score, coords = False, [f"ERROR DE FORMATO: {e}. Termina con el bloque JSON pedido."], float("inf"), []
        else:
            # 2. Validar y Puntuar
            es_valido, errores, score, coords = validar_y_puntuar(self.mapa_original, datos)

        if es_valido:
            print(f"✅ Válido. Coords: {coords} -> Score: {score}")
            if score < current_best_score:
                print(f"🚀 ¡NUEVO RÉCORD! ({score})")
                current_best_score = score
                current_best_coords = coords
                feedback = f"Récord: {score}. Intenta mejorar eligiendo otros candidatos para cubrir mejor las zonas lejanas."
            else:
                feedback = f"Solución válida ({score}) pero no supera tu récord ({int(current_best_score)}). Prueba una distribución diferente."
        else:
            print(f"❌ Inválido: {errores[0]}")
            feedback = "ERRORES:\n" + "\n".join(errores[:5])

        # 3. Finalizar si es el último intento
        if ev.intento_actual >= self.max_intentos:
            print("\n🏁 Fin de intentos. Generando resultado final...")
            if not current_best_coords:
                print("⚠️ El agente falló en encontrar una solución válida.")
                return StopEvent(result=None)

            mapa_final = aplicar_coordenadas(self.mapa_original, current_best_coords)
            return StopEvent(result={
                "mapa_texto": "\n".join("".join(fila) for fila in mapa_final),
                "score": int(current_best_score),
                "coords": current_best_coords,
            })

        # Siguiente vuelta
        return GenerationEvent(
            feedback=feedback,
            intento_actual=ev.intento_actual + 1,
            best_score=current_best_score,
            best_coords=current_best_coords,
            candidatos=ev.candidatos,
        )


# --- EJECUCIÓN ---

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
    parser = argparse.ArgumentParser(description="Coloca contenedores minimizando la distancia a las casas.")
    parser.add_argument("mapa", nargs="?", default=MAPA_POR_DEFECTO,
                        help=f"Fichero de 'entradas/' o ruta a un mapa (por defecto: {MAPA_POR_DEFECTO})")
    parser.add_argument("--intentos", type=int, default=MAX_INTENTOS, help="Número de intentos del LLM")
    parser.add_argument("--modelo", default=MODELO, help="Modelo de Gemini a usar")
    args = parser.parse_args(argv)

    tiempo_inicio = time.time()
    print(f"🕐 Inicio del programa: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    ruta = resolver_ruta_mapa(args.mapa)
    if not ruta.is_file():
        print(f"No encuentro {ruta}. Pon el mapa en la carpeta 'entradas'.")
        return 1

    try:
        mapa_orig = cargar_mapa(ruta)
        llm = crear_llm(args.modelo)
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        return 1

    # Timeout seguro de 600s
    agent = CoordinateAgent(mapa_original=mapa_orig, modelo_llm=llm, max_intentos=args.intentos,
                            timeout=600, verbose=False)

    try:
        resultado = await agent.run()
    except Exception as e:
        print(f"Error fatal: {e}")
        return 1

    if not resultado:
        return 1

    ruta_salida = BASE / "salidas" / ruta.name
    ruta_salida.parent.mkdir(exist_ok=True)
    ruta_salida.write_text(resultado["mapa_texto"] + "\n", encoding="utf-8")

    print("\n" + "=" * 50)
    print("MAPA GENERADO POR LLM:")
    print("=" * 50)
    print(resultado["mapa_texto"])
    print(f"\n💾 Guardado en {ruta_salida}")
    print(f"\nSuma total de pasos de las casas O al contenedor C más cercano: {resultado['score']}")
    print(f"\n🕐 Fin del programa: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"⏱️  Tiempo transcurrido: {time.time() - tiempo_inicio:.2f} segundos")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))