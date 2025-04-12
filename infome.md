## **Informe del Proyecto**

En este proyecto se desarrolló un agente inteligente para el juego del Hex utilizando el algoritmo **MinMax con poda Alfa-Beta**. La implementación se centra en dos aspectos principales: la **generación de las próximas jugadas** y la **evaluación del estado del tablero**.

---

### Heurística para la Generación de Próximas Jugadas (`next_moves`)
La función `next_moves(self, board)` no genera simplemente las celdas adyacentes a jugadas existentes. En cambio, **ordena todos los movimientos posibles según su cercanía al centro del tablero**, lo cual estadísticamente favorece posiciones más influyentes.

- Se utiliza la fórmula:  
    \[
    (x - c)^2 + (y - c)^2
    \]  
    donde \(c\) es la coordenada del centro del tablero.
    
- Los movimientos se ordenan según esta distancia y luego se seleccionan de forma **estocástica** (aleatoria) priorizando los más cercanos al centro, utilizando una distribución exponencial con λ = 1/2.

---

### Heurísticas de Evaluación del Tablero (`evaluate_board`)

#### 1. Estimación del Camino Más Corto (`shortest_path`)

Calcula una estimación del camino más corto que permitiría al jugador conectar sus dos lados opuestos del tablero.

- Se utilizan conjuntos disjuntos (**Disjoint Set**) para identificar componentes conectadas del jugador.
- Luego, se construye un grafo con **NetworkX** conectando estas componentes entre sí, así como con nodos virtuales de entrada y salida.
- Se emplea `nx.shortest_path_length()` para hallar el camino mínimo ponderado entre bordes virtuales.

**Estructuras utilizadas:**
- `DisjointSet` (conjuntos disjuntos).
- Grafo `nx.Graph()` de NetworkX.


#### 2. Juego Cercano al Centro (`center_plays`)

La función `center_plays` premia las piezas propias ubicadas cerca del centro del tablero, basándose en las filas y columnas centrales (y adyacentes si el tablero lo permite).


#### 3. Interrupción de Puentes Rivales (`break_rival_bridges`)

La función `break_rival_bridges` detecta configuraciones del rival que puedan representar puentes (cercanía de varias fichas rivales adyacentes a una del jugador), y estima cuántos de estos puentes podrían ser interrumpidos.

- Se cuenta la cantidad de fichas rivales adyacentes a cada ficha del jugador.
- Se usa la combinación matemática \(\binom{c}{2}\) para estimar la cantidad de conexiones posibles que se pueden romper.


#### 4. Influencia del Rival sobre Casillas Vacías (`rival_influence`)

Esta heurística penaliza casillas vacías que están fuertemente influenciadas por el rival (rodeadas por fichas rivales), ya que representan puntos peligrosos para la expansión del oponente.

---

### Evaluación Global del Tablero

La función `evaluate_board` combina las heurísticas anteriores utilizando ponderaciones específicas:
- **3 veces** el valor de `shortest_path`
- **10 veces** el valor de `center_plays`
- **3 veces** el valor de `break_rival_bridges`
- **6 veces** el valor de `rival_influence`

Esta combinación busca equilibrar la valoración entre la posición ofensiva (caminos propios y control del centro) y la defensiva (interrupción de jugadas rivales y penalización por influencia enemiga).

---
### Profundidad de Búsqueda Dinámica (`calculate_depth`)

La función `calculate_depth(board)` determina la profundidad de la búsqueda en función del **número de movimientos posibles**:
- Se establece un límite dinámico:  
  - Si hay más de 50 jugadas posibles, la profundidad es 1.
  - Entre 21 y 50, la profundidad es 2.
  - Entre 11 y 20, la profundidad es 3.
  - Entre 6 y 10, la profundidad es 4.
  - Y si es menor o igual a 5, la profundidad es 7.
- **Objetivo:**  
  - A medida que el tablero se llena y disminuye la cantidad de movimientos posibles, se incrementa la profundidad de búsqueda, permitiendo una evaluación más fina en etapas avanzadas del juego.
- Además, se guarda una **jugada aleatoria** como respaldo en caso de que se acabe el tiempo de cómputo (`random_move`).
