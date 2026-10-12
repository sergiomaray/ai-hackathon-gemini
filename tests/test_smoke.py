"""
Smoke tests para verificar la lógica algorítmica, validación y parsing
de ambos proyectos sin necesidad de llamar a la API de Gemini.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "waste-container-placer"))
sys.path.insert(0, str(ROOT / "power-grid-optimizer"))

import container_placer as cp
import power_grid_optimizer as pgo


def test_waste_container_placer():
    print("-> Probando Waste Container Placer...")
    mapa_path = ROOT / "waste-container-placer" / "entradas" / "1.txt"
    assert mapa_path.is_file(), f"Falta el mapa de prueba {mapa_path}"

    mapa = cp.cargar_mapa(mapa_path)
    assert len(mapa) == 20 and len(mapa[0]) == 21, "Dimensiones incorrectas de mapa 1.txt"

    candidatos = cp.obtener_candidatos_por_densidad(mapa)
    assert len(candidatos) > 0, "No se encontraron candidatos por densidad"
    assert len(candidatos) <= 50, "El número de candidatos excede el límite esperado"

    # 1. Prueba de validación positiva
    valido, errores, score, coords = cp.validar_y_puntuar(mapa, candidatos[:6])
    assert valido, f"Se esperaba válido pero falló con: {errores}"
    assert len(errores) == 0
    assert score < float("inf") and score > 0
    assert len(coords) == 6

    # 2. Prueba de validación negativa: menos de 6
    valido_mal, errores_mal, _, _ = cp.validar_y_puntuar(mapa, candidatos[:5])
    assert not valido_mal
    assert any("Cantidad incorrecta" in e for e in errores_mal)

    # 3. Prueba de validación negativa: coordenadas duplicadas
    dups = [candidatos[0], candidatos[0]] + candidatos[2:6]
    valido_dup, errores_dup, _, _ = cp.validar_y_puntuar(mapa, dups)
    assert not valido_dup
    assert any("repetidas" in e for e in errores_dup)

    # 4. Prueba de parser JSON
    mock_resp = """Aquí el razonamiento previo...
```json
{
  "contenedores": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
}
```"""
    parsed = cp.extraer_coordenadas(mock_resp, "contenedores")
    assert len(parsed) == 6
    assert parsed[0] == [1, 2]
    print("   [OK] Waste Container Placer funciona correctamente.")


def test_power_grid_optimizer():
    print("-> Probando Power Grid Optimizer...")
    mapa_path = ROOT / "power-grid-optimizer" / "entradas" / "ejemplo_sintetico.txt"
    assert mapa_path.is_file(), f"Falta el mapa de prueba {mapa_path}"

    mapa = pgo.cargar_mapa(mapa_path)
    assert len(mapa) == 20 and len(mapa[0]) == 20, "Dimensiones incorrectas de mapa sintético"

    candidatos = pgo.obtener_candidatos_estrategicos(mapa)
    assert len(candidatos) > 0, "No se encontraron candidatos estratégicos"

    prompt_cands = pgo.seleccionar_candidatos_prompt(mapa, candidatos)
    assert len(prompt_cands) > 0

    # Construir una solución válida para cubrir las T
    torres = pgo.posiciones(mapa, "T")
    coords_validas = []
    for t in torres:
        cercanos = [c for c in prompt_cands if pgo.manhattan(c, t) <= pgo.RADIO_T and c not in coords_validas]
        coords_validas.extend(cercanos[:pgo.MIN_C_POR_T])

    resto = [c for c in prompt_cands if c not in coords_validas]
    coords_validas.extend(resto[:pgo.CANTIDAD_C - len(coords_validas)])

    valido, errores, score, _ = pgo.validar_y_puntuar(mapa, coords_validas)
    assert valido, f"Se esperaba solución válida pero dio errores: {errores}"
    assert score < float("inf") and score > 0

    # Prueba negativa: colocar sobre edificio 'E'
    e_pos = pgo.posiciones(mapa, "E")[0]
    coords_invalidas = [list(e_pos)] + coords_validas[1:]
    valido_e, errores_e, _, _ = pgo.validar_y_puntuar(mapa, coords_invalidas)
    assert not valido_e
    assert any("ocupada por 'E'" in e for e in errores_e)

    # Prueba parser JSON
    mock_resp = """Razonamiento...
```json
{
  "transformadores": [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]
}
```"""
    parsed = pgo.extraer_coordenadas(mock_resp, "transformadores")
    assert len(parsed) == 6
    assert parsed[-1] == [10, 11]
    print("   [OK] Power Grid Optimizer funciona correctamente.")


if __name__ == "__main__":
    test_waste_container_placer()
    test_power_grid_optimizer()
    print("\nTODOS LOS TESTS HAN PASADO CON EXITO.")
