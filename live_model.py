import math

def live_goal_model(
    xg_A: float,
    xg_B: float,
    minuto: int,
    goles_A: int,
    goles_B: int,
    local: bool = True,
    cuota_A: float = 2.0,
    cuota_B: float = 3.0,
    # Estadisticas reales en vivo (opcionales, se estiman si no se dan)
    tiros_A: int = None,
    tiros_B: int = None,
    tiros_arco_A: int = None,
    tiros_arco_B: int = None,
    ataques_A: int = None,
    ataques_B: int = None,
    # Parametros de estimacion si no hay stats reales
    shots_per_xg_A: float = 3.8,
    shots_per_xg_B: float = 4.2,
    sot_ratio_A: float = 0.38,
    sot_ratio_B: float = 0.35,
    attacks_per_shot_A: float = 2.1,
    attacks_per_shot_B: float = 2.3
) -> dict:
    """
    Modelo EN VIVO: estima la probabilidad del proximo gol
    considerando el minuto actual, marcador real y stats reales.

    Params:
        xg_A / xg_B     : Expected Goals para cada equipo (partido completo, pre-partido)
        minuto          : Minuto actual del partido (1-90+)
        goles_A / goles_B: Marcador real actual
        local           : True si equipo A es local
        cuota_A/B       : Cuotas en vivo actuales
        tiros_*/ataques_*: Stats reales en vivo (si no se dan, se estiman proporcionalmente)

    Returns:
        dict con probabilidades, EV, motivacion y contexto del partido
    """
    if minuto <= 0 or minuto > 120:
        raise ValueError("El minuto debe estar entre 1 y 120")

    minutos_restantes = max(90 - minuto, 1)
    fraccion_jugada = minuto / 90

    # --- Stats en vivo: usar reales si existen, sino estimar segun fraccion del partido ---
    tiros_A = tiros_A if tiros_A is not None else round(xg_A * shots_per_xg_A * fraccion_jugada)
    tiros_B = tiros_B if tiros_B is not None else round(xg_B * shots_per_xg_B * fraccion_jugada)
    tiros_arco_A = tiros_arco_A if tiros_arco_A is not None else round(tiros_A * sot_ratio_A)
    tiros_arco_B = tiros_arco_B if tiros_arco_B is not None else round(tiros_B * sot_ratio_B)
    ataques_A = ataques_A if ataques_A is not None else round(tiros_A * attacks_per_shot_A)
    ataques_B = ataques_B if ataques_B is not None else round(tiros_B * attacks_per_shot_B)

    # --- Potencia Ofensiva base (igual que pregame) ---
    PO_A = 0.25 * tiros_arco_A + 0.20 * tiros_A + 0.30 * ataques_A + 0.25 * xg_A
    PO_B = 0.25 * tiros_arco_B + 0.20 * tiros_B + 0.30 * ataques_B + 0.25 * xg_B

    # --- Ajuste por localia ---
    if local:
        PO_A *= 1.12
    else:
        PO_B *= 1.08

    # --- Motivacion dinamica segun marcador real y minuto ---
    # Sigmoide centrada en min 55: efecto se activa mas en 2da mitad
    # Si va perdiendo -> sube su motivacion
    dif_A = goles_B - goles_A   # positivo si A va perdiendo
    dif_B = goles_A - goles_B   # positivo si B va perdiendo
    sigmoid = 1 / (1 + math.exp(-(minuto - 55) / 12))

    MP_A = 1 + (dif_A * 0.08) * sigmoid
    MP_B = 1 + (dif_B * 0.08) * sigmoid

    # --- Ajuste por desesperacion (ultimos 15 min, equipo perdiendo) ---
    if minuto >= 75:
        if dif_A > 0:
            MP_A *= 1 + (0.04 * dif_A)   # A empuja mas si va perdiendo al final
        if dif_B > 0:
            MP_B *= 1 + (0.04 * dif_B)

    # --- Poder de Ataque final ---
    PA_A = PO_A * MP_A
    PA_B = PO_B * MP_B

    # --- Probabilidad del proximo gol ---
    PNG_A = PA_A / (PA_A + PA_B)
    PNG_B = 1.0 - PNG_A

    # --- Expected Value ---
    EV_A = PNG_A * cuota_A - 1
    EV_B = PNG_B * cuota_B - 1

    # --- Contexto del partido ---
    if goles_A > goles_B:
        estado = f"A gana {goles_A}-{goles_B}"
    elif goles_B > goles_A:
        estado = f"B gana {goles_B}-{goles_A}"
    else:
        estado = f"Empate {goles_A}-{goles_B}"

    return {
        "Minuto": minuto,
        "Minutos_restantes": minutos_restantes,
        "Marcador_actual": f"{goles_A}-{goles_B}",
        "Estado": estado,
        "Prob_A": round(PNG_A, 4),
        "Prob_B": round(PNG_B, 4),
        "EV_A": round(EV_A, 4),
        "EV_B": round(EV_B, 4),
        "Motivacion": {
            "MP_A": round(MP_A, 4),
            "MP_B": round(MP_B, 4),
        },
        "Estadisticas_en_vivo": {
            "Tiros_A": tiros_A,
            "Tiros_B": tiros_B,
            "Tiros_arco_A": tiros_arco_A,
            "Tiros_arco_B": tiros_arco_B,
            "Ataques_A": ataques_A,
            "Ataques_B": ataques_B,
        }
    }


if __name__ == "__main__":
    print("=== EN VIVO - Min 60, marcador 0-1 ===")
    result = live_goal_model(
        xg_A=1.6, xg_B=1.1,
        minuto=60, goles_A=0, goles_B=1,
        local=True, cuota_A=2.5, cuota_B=2.8
    )
    for k, v in result.items():
        print(f"{k}: {v}")
