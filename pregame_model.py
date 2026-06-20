import math

def pregame_goal_model(
    xg_A: float,
    xg_B: float,
    local: bool = True,
    cuota_A: float = 2.0,
    cuota_B: float = 3.0,
    # Parametros ajustables
    shots_per_xg_A: float = 3.8,
    shots_per_xg_B: float = 4.2,
    sot_ratio_A: float = 0.38,
    sot_ratio_B: float = 0.35,
    attacks_per_shot_A: float = 2.1,
    attacks_per_shot_B: float = 2.3
) -> dict:
    """
    Modelo PRE-PARTIDO - estima la probabilidad del proximo gol
    usando xG esperado para todo el partido y estadisticas derivadas.
    Se corre ANTES de que empiece el partido (minuto = 0).
    No hay marcador real, no hay motivacion dinamica.

    Params:
        xg_A / xg_B     : Expected Goals para cada equipo (partido completo)
        local           : True si el equipo A juega de local
        cuota_A/B       : Cuotas de apuesta para calcular Expected Value
        shots_per_xg_*  : Tiros estimados por unidad de xG
        sot_ratio_*     : Ratio tiros al arco / tiros totales
        attacks_per_shot_* : Ataques por tiro

    Returns:
        dict con probabilidades, EV y estadisticas estimadas
    """
    # --- Estadisticas derivadas del xG ---
    tiros_A = round(xg_A * shots_per_xg_A)
    tiros_B = round(xg_B * shots_per_xg_B)
    tiros_arco_A = round(tiros_A * sot_ratio_A)
    tiros_arco_B = round(tiros_B * sot_ratio_B)
    ataques_A = round(tiros_A * attacks_per_shot_A)
    ataques_B = round(tiros_B * attacks_per_shot_B)

    # --- Potencia Ofensiva (pesos: 25% SoT, 20% tiros, 30% ataques, 25% xG) ---
    PO_A = 0.25 * tiros_arco_A + 0.20 * tiros_A + 0.30 * ataques_A + 0.25 * xg_A
    PO_B = 0.25 * tiros_arco_B + 0.20 * tiros_B + 0.30 * ataques_B + 0.25 * xg_B

    # --- Ajuste por localia ---
    if local:
        PO_A *= 1.12   # ventaja local para A
    else:
        PO_B *= 1.08   # ventaja local para B

    # --- Minuto 0: no hay motivacion dinamica, MP es neutro ---
    # MP_A = MP_B = 1.0 (marcador 0-0, sin presion)
    PA_A = PO_A  # Poder de Ataque final
    PA_B = PO_B

    # --- Probabilidad del proximo gol ---
    PNG_A = PA_A / (PA_A + PA_B)
    PNG_B = 1.0 - PNG_A

    # --- Expected Value ---
    EV_A = PNG_A * cuota_A - 1
    EV_B = PNG_B * cuota_B - 1

    return {
        "Prob_A": round(PNG_A, 4),
        "Prob_B": round(PNG_B, 4),
        "EV_A": round(EV_A, 4),
        "EV_B": round(EV_B, 4),
        "Estadisticas_estimadas": {
            "Tiros_A": tiros_A,
            "Tiros_B": tiros_B,
            "Tiros_arco_A": tiros_arco_A,
            "Tiros_arco_B": tiros_arco_B,
            "Ataques_A": ataques_A,
            "Ataques_B": ataques_B,
        }
    }


if __name__ == "__main__":
    result = pregame_goal_model(xg_A=1.6, xg_B=1.1, local=True, cuota_A=2.0, cuota_B=3.5)
    print("=== PRE-PARTIDO ===")
    for k, v in result.items():
        print(f"{k}: {v}")
