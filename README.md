# Cartera de inversión automatizada (Alpaca)

Proyecto de **trading algorítmico en paper trading** con Python: dos estrategias (bajo riesgo / largo plazo y alto riesgo / corto plazo), rebalanceo periódico, stops y reportes por Telegram.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Alpaca](https://img.shields.io/badge/Broker-Alpaca-000000)

## Qué hace

- **Optimización de cartera** con datos históricos (Sharpe, ganancias, etc.).
- **Top N activos** con filtros para evitar activos en caída fuerte.
- **Rebalanceo** y **stop loss / trailing** con estado persistente.
- **Reportes** (mini y completos) configurables; opcional **Telegram**.

> ⚠️ **Paper trading por defecto.** No constituye asesoramiento financiero.

## Stack

| Área        | Tecnología                          |
|------------|--------------------------------------|
| Lenguaje   | Python 3                             |
| Broker API | [Alpaca](https://alpaca.markets/)   |
| Datos      | pandas, numpy                        |
| Ops        | `.env`, scripts autónomos          |

## Requisitos

```bash
pip install -r requirements.txt
```

## Configuración

1. Copia variables de entorno:
   ```bash
   cp .env.example .env
   ```
2. Edita `.env` con tus claves de **Alpaca Paper** (y Telegram si lo usas).
3. **No subas** el archivo `.env` a Git (ya está en `.gitignore`).

## Cómo ejecutar (local)

```bash
python alpaca_trading_low_risk.py    # largo plazo / bajo riesgo
python alpaca_trading_high_risk.py   # corto plazo / alto riesgo
```

## Estructura (principal)

| Archivo | Rol |
|---------|-----|
| `alpaca_trading_low_risk.py` / `alpaca_trading_high_risk.py` | Bots principales |
| `alpaca_trading_helpers.py` | Config compartida, stops, reportes, estado |
| `cartera_largo_plazo.py` / `cartera_alto_riesgo_corto_plazo.py` | Análisis de cartera |

## Licencia y uso

Uso educativo y de portafolio. El autor no se responsabiliza del uso en cuentas reales.

---

### Para recruiters / revisión rápida

- Revisa los scripts anteriores para ver **arquitectura**, **gestión de riesgo** y **integración API**.
- En entrevistas puedes explicar: optimización de portafolio, rebalanceo, cooldowns, persistencia de estado y diferencias entre estrategias.
