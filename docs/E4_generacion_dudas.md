# E4 · Generación con Transformers: dudas y respuestas

Dudas que surgieron al diseñar e implementar `notebooks/07_generation_transformers.ipynb`, con su respuesta. Están agrupadas por tema y, dentro de cada tema, en el orden en que aparecieron.

---

## 1. Diseño de la evaluación

### ¿Por qué no usamos BLEU para medir la memorización?

BLEU sí lo usamos, pero para la **diversidad** (self-BLEU dentro de cada conjunto generado). Para comparar con train no sirve:

- BLEU cuenta cuántos n-gramas de hasta 4 palabras del texto aparecen en las referencias.
- Con los ~10.000 prompts de train como referencias, casi cualquier secuencia de 4 palabras aparece en alguno ("I want you to", "you are going to"…). Todos los textos sacarían una puntuación alta y la métrica no distinguiría nada.
- Con **8-gramas** pasa lo contrario: una secuencia de 8 palabras idéntica a una de train casi nunca es casualidad, es copia. Es la misma idea que BLEU, pero con n-gramas más largos, que es lo que hace falta para detectar copias. Por eso la métrica de copia literal es el % de 8-gramas del texto que aparecen en train.

### ¿Qué es distinct-n y por qué depende de la longitud?

- **distinct-n** = n-gramas distintos / n-gramas totales en todo el conjunto de textos. Por ejemplo, distinct-2 = 0,6 significa que el 60 % de los pares de palabras consecutivas no se repite. Cuanto más alto, más variado.
- **Depende de la longitud** porque, cuanto más texto hay, más se repiten las palabras y expresiones comunes ("the", "you are"…): el número de n-gramas distintos crece más despacio que el total. Si el modelo genera textos más cortos que los reales, parecería "más diverso" solo por ser más corto.
- Por eso distinct-n y self-BLEU se calculan sobre textos **truncados a 200 palabras** y con **el mismo número de textos** que la referencia de val real.

### ¿Recomiendo early stopping evaluando cada N pasos?

**Sí.** Con ~155 pasos por época, evaluar solo al final de cada época da 2 o 3 mediciones: muy pocas para ver cuándo la loss de validación empieza a subir, que en generación significa que el modelo empieza a memorizar. Evaluando cada 50 pasos hay unas 10 mediciones, y nos quedamos con el mejor checkpoint (`load_best_model_at_end`). Cuesta poco: guardar un checkpoint es guardar solo el adaptador (unos MB), y evaluar los 507 ejemplos de val tarda unos 3,4 min en la T4.

---

## 2. Datos, tokens especiales y máscara de la loss

### ¿Qué problema había con que el token de padding fuera el mismo que el `<eos>`?

- Para entrenar por lotes, todos los textos de un batch tienen que medir lo mismo, así que a los cortos se les añaden tokens de relleno (padding). Esas posiciones se marcan con label `-100` para que no cuenten en la loss.
- En Qwen3.5 base, el token de relleno **es el mismo** que el `<eos>` (`<|endoftext|>`, id 248044).
- Si el código marcara con `-100` "todas las posiciones cuyo token sea 248044", también borraría el `<eos>` real del final de cada texto. El modelo nunca aprendería a terminar y generaría hasta agotar los 1.024 tokens.

### ¿Cómo se ha solucionado? ¿Cómo sabe el modelo cuándo dejar de generar?

- TRL construye los `labels` **antes** del padding, con el `<eos>` real incluido. Al formar el batch, rellena los `labels` con `-100` **en las posiciones de relleno**, sin mirar qué token hay.
- Lo comprobamos en la celda 3.4: el `<eos>` real tiene label 248044 y el relleno tiene `-100` y `attention_mask = 0`, aunque los dos son el mismo token.
- Así, el modelo aprende que **después del texto viene `<eos>`**. Al generar, cuando produce ese token, `generate` para. Si en un lote un texto termina antes que los demás, su fila se rellena con padding (otra vez 248044), y cortamos cada texto en el primer `<eos>`.

### ¿Qué significa que los textos largos "se truncan sin `<eos>`" y que "sale solo"?

El orden de las operaciones de TRL lo hace automáticamente:

1. añade el `<eos>` al final del texto: `texto … <eos>`;
2. trunca a 1.024 tokens, quedándose con el principio;
3. si el texto cabe, conserva su `<eos>`; si no cabe, el corte se lleva el final, y con él el `<eos>`.

Así, el modelo no aprende a terminar los textos de golpe a mitad de frase. Con el tokenizador de Qwen, 1.024 tokens cubren entero el 87 % de los jailbreaks de train y el ~92 % de los normales.

---

## 3. Modelo y técnicas de ajuste

### ¿Qué es bf16?

Es un formato para guardar números decimales en 16 bits, en vez de los 32 del formato estándar (fp32). Un número tiene bits de *exponente* (rango: cuánto de grande puede ser) y bits de *mantisa* (precisión):

| formato | bits | exponente | mantisa | idea |
|---|---|---|---|---|
| fp32 | 32 | 8 | 23 | estándar: preciso, pero ocupa el doble |
| **bf16** | 16 | **8** | 7 | el mismo rango que fp32, menos precisión |
| fp16 | 16 | 5 | 10 | más precisión, pero desborda por encima de ~65.000 |

En redes neuronales, esa pérdida de precisión apenas afecta. A cambio, el modelo ocupa la mitad de memoria y, en GPUs con bf16 nativo, los cálculos son mucho más rápidos. Qwen3.5 se publicó en bf16. **La L4 tiene bf16 nativo** (arquitectura Ampere o posterior, capacidad de cómputo ≥ 8.0). **La T4 no** (Turing, 7.5): lo emula, así que funciona, pero más despacio. Además, una operación de TRL no existía en la T4 y hubo que indicarle que usara su alternativa en fp32 (celda 2.1).

### ¿Qué significa que transformers "envuelve" las funciones costosas de DeltaNet?

El código del modelo siempre llama a una función con el mismo nombre (por ejemplo, `chunk_gated_delta_rule`). transformers mete esa función dentro de otra (el "envoltorio"). Al importar el módulo de Qwen3.5, el envoltorio mira si está instalado `flash-linear-attention`:

- si lo está, las llamadas se redirigen a su versión optimizada, escrita como kernel de GPU en Triton;
- si no, se usa la versión en PyTorch puro: hace lo mismo, pero mucho más lenta.

La celda 2.2 mira dentro del envoltorio qué versión se eligió. Como la elección se hace al importar, `flash-linear-attention` tiene que instalarse antes de cargar nada (Paso 0).

### ¿Por qué no ponemos LoRA en `lm_head`? ¿Qué son los embeddings "atados"?

- **Embeddings (entrada):** convierten cada token en un vector.
- **`lm_head` (salida):** convierte el vector final en una puntuación para cada uno de los 248.000 tokens del vocabulario.
- En Qwen3.5 pequeño, las dos son **la misma matriz**, compartida o "atada" (*weight tying*). Así se ahorran 254 M de parámetros.

No la entrenamos porque:

1. **No añadimos tokens nuevos.** La cabecera es texto normal ("Type: jailbreak"), y todos sus tokens ya tienen un buen vector. Con tokens especiales nuevos sí habría que entrenarla.
2. **Lo que queremos adaptar es el estilo**, y eso se aprende en las capas internas, que es donde va LoRA.
3. **Es enorme y está compartida:** tocarla cambiaría a la vez cómo el modelo lee y cómo escribe.

Además, `target_modules="all-linear"` de PEFT excluye `lm_head` por defecto.

### ¿Qué es el gradient (activation) checkpointing?

- Para el backward, el modelo necesita los valores intermedios del forward de cada capa (las **activaciones**). Normalmente se guardan todos, y ocupan mucha memoria: crecen con las capas × la longitud × el tamaño oculto.
- Con *checkpointing*, en el forward solo se guarda **la entrada de cada capa**. En el backward, **se recalcula el forward** de cada capa cuando hace falta.
- Es cambiar memoria por tiempo. En la T4 con el 0.8B: sin checkpointing, 5,0 GB y 2,2 s por ejemplo; con checkpointing, 1,9 GB y 3,0 s por ejemplo (~35 % más lento).
- En transformers 5 el parámetro se llama `activation_checkpointing`. `SFTConfig` lo activa por defecto, así que hay que desactivarlo explícitamente. No lo usamos con el 0.8B porque cabe sin él.

### ¿Qué es el adaptador?

Son **las matrices de LoRA**, guardadas aparte del modelo. En cada capa adaptada hay dos matrices pequeñas, A y B: 10,8 M de parámetros en el 0.8B, un fichero de 43 MB (`adapter/adapter_model.safetensors`). El modelo base no se modifica ni se guarda. Para usar el modelo ajustado se carga el base (de Hugging Face) y se le "enchufa" el adaptador con `PeftModel.from_pretrained`. Se llama adaptador porque adapta el modelo base sin tocar sus pesos. Un mismo base puede tener varios adaptadores.

---

## 4. Configuración del entrenamiento

### ¿Qué batch usamos? ¿16? ¿Por qué se habla de "batch 1"?

**El batch efectivo es 16**, conseguido como **1 × 16**:

- **Batch por dispositivo = 1:** la GPU procesa un ejemplo cada vez (forward y backward) y va **acumulando** los gradientes.
- **Acumulación = 16:** tras 16 ejemplos, el optimizador hace **una** actualización de los pesos con la suma.

Matemáticamente es casi lo mismo que un batch de 16 de golpe, pero en la GPU solo hay un ejemplo cada vez. En la T4, con 2 ejemplos a la vez ocupaba 8,2 GB y no era más rápido, y con 4 no cabía.

### ¿Qué son las sondas?

Son **mini-entrenamientos de prueba** (celda 4.2): 3 pasos con los ejemplos más largos (1.024 tokens, el peor caso), para varias combinaciones de batch por dispositivo (1, 2, 4…) con y sin checkpointing. De cada una se mide **la memoria máxima** y **el tiempo por ejemplo**. El notebook elige la combinación más rápida que deje al menos un 15 % de la memoria libre. El resultado se guarda en Drive (`train_setups.json`), con la GPU y el modelo como clave, para no repetirlo.

### ¿De dónde salen las 3 épocas y los ~465 pasos?

- **3 épocas:** de la decisión inicial ("2–3 épocas con early stopping"). Es el máximo; el early stopping para antes si la loss de val no mejora en 3 evaluaciones seguidas (150 pasos).
- **465 pasos:** 2.478 ejemplos de train ÷ 16 por paso ≈ 155 pasos por época × 3 épocas ≈ 465. Un "paso" es una actualización de los pesos. Con una evaluación cada 50 pasos, más la inicial, salen unas 10 evaluaciones.

---

## 5. Métricas y evaluación (Pasos 6–8)

### ¿Qué hace la celda 6.1?

Reentrena el mejor clasificador superficial de E3 (TF-IDF + LinearSVC), con la misma tokenización de E2 y los mismos hiperparámetros de Optuna, usando todo train. Se usa para dos cosas:

1. **En val real**, para la **referencia**: detecta el 47 % de los jailbreaks reales (igual que en E3) y marca como jailbreak al 2 % de los normales.
2. **En los textos generados** (Pasos 7–8), para medir la **fidelidad a la clase**.

El 47 % es el techo razonable: si solo detecta la mitad de los jailbreaks reales, no se puede exigir que detecte el 100 % de los generados. El clasificador Transformer de la parte de clasificación sería una referencia más fiable.

### ¿Por qué usamos una implementación propia de self-BLEU en vez de la de NLTK?

Por **velocidad**; el resultado es idéntico. El self-BLEU compara cada texto con todos los demás. Con NLTK, para 200 textos son 200 llamadas, y en cada una vuelve a contar los n-gramas de las otras 199 referencias. Son decenas de miles de recuentos repetidos, y hacen falta para muchos conjuntos (12 del barrido, 6 finales y las referencias). La versión propia cuenta los n-gramas de cada texto **una sola vez** y guarda, para cada n-grama, los dos recuentos más altos del conjunto, que es lo que hace falta para el máximo "entre los demás". Comprobado con 40 textos: las dos dan 0,143443.

### ¿Tiene sentido ejecutar el Paso 6 sin el 2B entrenado? ¿Es solo con val?

Sí: **el Paso 6 no usa ningún modelo de lenguaje**. Hace dos cosas:

1. Define las funciones de las métricas.
2. Calcula las **referencias con los prompts reales de val**.

Train también se usa: para entrenar el clasificador de E3, para buscar el texto más parecido (memorización) y como muestra real del discriminador. Solo necesita CPU. Las métricas de los textos generados se calculan en los Pasos 7 y 8 con estas mismas funciones.

### ¿Los Pasos 7.1 y 7.2 son solo con el 0.8B?

- **7.1** solo define las funciones de generación, válidas para cualquier modelo.
- **7.2 (barrido)** usa solo el **0.8B ajustado**, que es más barato. Con su tabla se elige `GEN_CONFIG` (7.3).
- **7.4** aplica esa configuración a los tres: **base 0.8B, 0.8B ajustado y 2B ajustado**.

Se supone que la mejor configuración del 0.8B también vale para el 2B. Es razonable (misma familia, mismo tokenizador, mismos datos), pero hay que mencionarlo en la memoria como simplificación.

### ¿El Paso 8 repite las métricas del Paso 6 con las generaciones?

En parte. **8.1** aplica las funciones del Paso 6 a los conjuntos finales (base 0.8B, 0.8B ajustado y 2B ajustado), junto a la referencia de val real. Además añade:

- **8.2 Perplejidad por clase:** lo bien que cada modelo (base y ajustado, 0.8B y 2B) predice los prompts **reales** de val, jailbreaks y normales por separado. Usa los modelos, no sus generaciones.
- **8.3** histograma de memorización; **8.4** palabras que delatan a los textos generados; **8.5** ejemplos.

---

## 6. Uso del notebook en Colab

### ¿Qué hay que subir a Drive?

Los dos CSV de E2 en `Mi unidad/jailbreak/data/interim/` (`train_clean.csv` y `val_clean.csv`), y el notebook. Todo lo que el notebook produce va a `Mi unidad/jailbreak/generation/`, que crea él mismo.

### ¿Hay que ejecutar todo otra vez cada vez que se sube una versión nueva?

Depende de si la sesión de Colab sigue viva. Lo que está en memoria (variables, modelo cargado, paquetes) vive en la máquina de Colab, no en el archivo:

- **Sesión viva** (sigue "Conectado"): basta con ejecutar las celdas nuevas.
- **Sesión reiniciada:** hay que ejecutar desde el Paso 0. Lo costoso (entrenamientos, sondas, generaciones, referencias de métricas) se guarda en Drive y se salta si ya existe, así que volver al punto anterior cuesta pocos minutos.

### ¿Qué dependencias necesita Colab? ¿Hay que añadirlas a uv en local?

- Colab ya trae `torch`, `pandas`, `numpy`, `scikit-learn`, `nltk` y `matplotlib`. El Paso 0 instala o actualiza `transformers>=5.2`, `peft`, `trl` y `flash-linear-attention`. No instalamos `causal-conv1d`: solo acelera una convolución pequeña y en Colab suele haber que compilarlo, lo que tarda más de 20 min.
- En el `pyproject.toml` local se añadieron `transformers`, `peft` y `trl`, que documentan lo que usa el proyecto. `flash-linear-attention` no, porque necesita GPU NVIDIA en Linux.

### ¿Nos vale la T4?

Para depurar, sí. Para los entrenamientos completos es lenta, porque emula el bf16: el 0.8B tarda varias horas y el 2B, bastantes más. La L4 tiene bf16 nativo y debería ser bastante más rápida.

### ¿Funciona si después se ejecuta con la L4?

Sí. Lo único ligado al hardware es el resultado de las sondas (`train_setups.json`), cuya clave es GPU + modelo: en la L4 se vuelven a hacer las sondas y se elige un batch adecuado para ella. El arreglo de TRL para la T4 solo se activa en GPUs sin bf16 nativo. Los adaptadores, las generaciones y las métricas no dependen de la GPU. Conviene evitar solo **cambiar de GPU a mitad de un mismo entrenamiento**: funcionaría, pero el orden de los ejemplos en el resto de esa época cambiaría un poco.

### ¿Se puede usar "Ejecutar todas"? ¿Qué celdas dependen de que cambiemos algo?

Sí. Con `TRAIN_2B = False` (celda 5.2), "Ejecutar todas" hace los Pasos 0–4, entrena el 0.8B, salta el 2B, calcula las métricas (Paso 6), hace el barrido (7.2) y **se para en la 7.3** a propósito. Hay que cambiar dos cosas a mano:

- **`GEN_CONFIG`** (7.3): se elige al ver la tabla del barrido.
- **`TRAIN_2B`** (5.2): se pone a `True` cuando se decide entrenar el 2B. Después se vuelven a ejecutar la 7.4 y el Paso 8; lo ya hecho se salta.
