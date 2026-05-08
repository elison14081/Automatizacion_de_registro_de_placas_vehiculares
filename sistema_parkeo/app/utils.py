from datetime import datetime

def calcular_monto(hora_entrada: datetime, hora_salida: datetime, tarifa_hora: float = 3.5) -> float:
    """
    Calcula el monto a pagar según la diferencia en horas.
    """
    tiempo = hora_salida - hora_entrada
    horas = tiempo.total_seconds() / 3600
    horas_redondeadas = int(horas) + (1 if horas % 1 > 0 else 0)
    monto = horas_redondeadas * tarifa_hora
    return monto
