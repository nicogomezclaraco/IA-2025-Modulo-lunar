# Proyecto Lunar Lander con Agente DQN

Este proyecto implementa un agente DQN (Deep Q-Network) para el entorno Lunar Lander, utilizando un entorno personalizado basado en `gymnasium`. Incluye scripts para entrenar el agente y evaluarlo tanto en modo visual como en modo estadístico.

---

## Contenido

- Código fuente en `src/` para el entorno y el agente DQN.
- Documentos del proyecto en `docs/`.
- Documento de notebook del proyecto en `notebooks/`.
- Modelos guardados en `saved_models/`.
- Scripts guardados en `scripts/`:
- [`trainDQN.py`](#trainDQNpy): Entrena el agente DQN con parámetros configurables.
- [`testReward.py`](#testRewardpy): Evalúa el agente entrenado, con opción visual o estadística.

---

## Requisitos y dependencias

El proyecto usa Python 3.12+ y las siguientes librerías principales:

- `gymnasium` para entornos de simulación.
- `numpy` para cálculos numéricos.
- `matplotlib` para graficar resultados.
- `tensorflow` para la red neuronal del agente DQN.

### Instalación rápida

Puedes instalar las dependencias con pip:

```bash
pip install gymnasium numpy matplotlib tensorflow


### Ejecución scripts

Para ejecutar ambos scripts se puede realizar estando en la carpeta raiz del proyecto, y en la terminal ejecutar los siguientes comandos para ejecutar el entrenamiento del modelo y el test para comprobar el modelo:
```bash
python -m scripts.trainDQN --plot --episodes 500

```bash
python -m scripts.testReward --visual
