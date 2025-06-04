# Projeto de Coloração de Grafos

Este repositório contém os algoritmos de grafos implementados em Python, incluindo busca (BFS, DFS), cálculo de menor caminho (Dijkstra) e, principalmente, diferentes heurísticas de coloração de grafos não ponderados e não direcionados.

## Conteúdo

- `diegoS_Coloracao_v4.py`: implementação principal que carrega um grafo de um arquivo `.txt`, executa algoritmos clássicos (BFS, DFS, Dijkstra) e diferentes métodos de coloração (`brute`, `naive`, `welsh`, `dsatur`).
- Arquivos de grafo de exemplo (`k5.txt`, `kquase5.txt`, `k33.txt`, `r1000-234-234.txt`, `r250-66-65.txt`, `C4000-260-X.txt`), que seguem o formato:
  ```
  n m D P
  u1 v1 [peso1]
  u2 v2 [peso2]
  …
  ```
  Onde:
  - `n`: número de vértices
  - `m`: número de arestas
  - `D`: 0 ou 1, indicando grafo não direcionado (0) ou direcionado (1)
  - `P`: 0 ou 1, indicando não ponderado (0) ou ponderado (1)
  - As linhas seguintes definem cada aresta: vértice de origem, vértice de destino e, se houver, peso (float).

## Pré-requisitos

- Python 3.8 ou superior
- Bibliotecas Python:
  - `networkx`
  - `matplotlib`

Você pode instalar as dependências com:
```bash
pip install networkx matplotlib
```

## Como usar

Na raiz do repositório, execute:
```bash
python diegoS_Coloracao_v4.py <caminho_para_arquivo_de_grafo> [opções]
```

### Parâmetros obrigatórios

- `filepath`  
  Caminho para o arquivo `.txt` que contém a definição do grafo.

### Opções (flags)

- `--rep {list,matrix,both}`  
  Escolhe a representação interna para busca (BFS/DFS/Dijkstra):  
  - `list`: usa lista de adjacência  
  - `matrix`: usa matriz de adjacência  
  - `both`: executa em ambas (padrão: `both`)

- `--start N`  
  Vértice de origem para as buscas clássicas ou início da coloração (inteiro entre `0` e `n–1`).  
  Padrão: `0`.

- `--plot`  
  Se presente, exibe gráficos dos resultados de BFS, DFS e Dijkstra (apenas para representação `list` ou `matrix`).

- `--coloring-method {brute,naive,welsh,dsatur,all}`  
  Escolhe o método de coloração:  
  - `brute`: força bruta (testa combinações de 2..n cores; recomendado apenas para n<=10)  
  - `naive`: heurística ingênua (ordem natural 0..n-1)  
  - `welsh`: heurística Welsh–Powell (ordem decrescente de grau)  
  - `dsatur`: heurística DSATUR (maior saturação)  
  - `all`: executa todas as heurísticas acima, uma a uma.

- `--plot-coloring`  
  Se presente, exibe o grafo colorido com legendas e setas que indicam a ordem de coloração.

## Descrição detalhada dos argumentos

1. **`filepath`**  
   - Tipo: `string`  
   - Exemplo: `k5.txt`, `C4000-260-X.txt`  
   - Descrição: caminho para o arquivo de texto que define o grafo. Deve estar no formato especificado (primeira linha com `n m D P`).

2. **`--rep`**  
   - Tipo: `string` (`list`, `matrix` ou `both`)  
   - Padrão: `both`  
   - Descrição: define se os algoritmos clássicos (BFS, DFS, Dijkstra) serão executados usando lista de adjacência (`list`), matriz de adjacência (`matrix`) ou ambos (`both`).  
   - Uso:  
     - `--rep list` → apenas lista de adjacência  
     - `--rep matrix` → apenas matriz de adjacência  
     - `--rep both` → executa em ambas representações

3. **`--start`**  
   - Tipo: `inteiro`  
   - Padrão: `0`  
   - Descrição: vértice de referência para buscas clássicas (origem) ou heurísticas de coloração (início).  
   - Uso:  
     - `--start 2` → inicia buscas ou colorações a partir do vértice 2.

4. **`--plot`**  
   - Tipo: `flag` (quando presente, assume valor `True`)  
   - Descrição: exibe os gráficos de BFS, DFS e Dijkstra usando `matplotlib`. Só faz sentido se não estiver apenas colorindo (ou seja, sem `--coloring-method`, ou antes de chamar a coloração).

5. **`--coloring-method`**  
   - Tipo: `string` (`brute`, `naive`, `welsh`, `dsatur` ou `all`)  
   - Descrição: seleciona qual heurística de coloração será usada.  
   - Uso:  
     - `--coloring-method naive`  
     - `--coloring-method dsatur`  
     - `--coloring-method all` (executa todas as 4 heurísticas em sequência)

6. **`--plot-coloring`**  
   - Tipo: `flag`  
   - Descrição: exibe o resultado da coloração em um gráfico, com vértices coloridos, legendas e setas mostrando a ordem de coloração.

## Casos de teste (exemplos de execução)

> **Observação**: ajuste os caminhos de arquivos conforme o local dos seus `.txt`. A execução pode demorar para instâncias grandes ou com heurística `brute`.

### Grafos pequenos (exemplo: `k5.txt`, 5 vértices)

```bash
python diegoS_Coloracao_v4.py k5.txt --coloring-method naive --plot-coloring
python diegoS_Coloracao_v4.py k5.txt --coloring-method brute --plot-coloring
python diegoS_Coloracao_v4.py k5.txt --coloring-method welsh --plot-coloring
python diegoS_Coloracao_v4.py k5.txt --coloring-method dsatur --plot-coloring
```

### Grafos quase completos (exemplo: `kquase5.txt`)

```bash
python diegoS_Coloracao_v4.py kquase5.txt --coloring-method naive --plot-coloring
python diegoS_Coloracao_v4.py kquase5.txt --coloring-method brute --plot-coloring
python diegoS_Coloracao_v4.py kquase5.txt --coloring-method welsh --plot-coloring
python diegoS_Coloracao_v4.py kquase5.txt --coloring-method dsatur --plot-coloring
```

### Ciclo de 33 vértices (exemplo: `k33.txt`)

```bash
python diegoS_Coloracao_v4.py k33.txt --coloring-method naive --plot-coloring
python diegoS_Coloracao_v4.py k33.txt --coloring-method brute --plot-coloring
python diegoS_Coloracao_v4.py k33.txt --coloring-method welsh --plot-coloring
python diegoS_Coloracao_v4.py k33.txt --coloring-method dsatur --plot-coloring
```

### Grafos randômicos maiores (exemplos: `r1000-234-234.txt`, `r250-66-65.txt`)

```bash
python diegoS_Coloracao_v4.py r1000-234-234.txt --coloring-method naive
python diegoS_Coloracao_v4.py r1000-234-234.txt --coloring-method brute
python diegoS_Coloracao_v4.py r1000-234-234.txt --coloring-method welsh
python diegoS_Coloracao_v4.py r1000-234-234.txt --coloring-method dsatur

python diegoS_Coloracao_v4.py r250-66-65.txt --coloring-method naive
python diegoS_Coloracao_v4.py r250-66-65.txt --coloring-method brute
python diegoS_Coloracao_v4.py r250-66-65.txt --coloring-method welsh
python diegoS_Coloracao_v4.py r250-66-65.txt --coloring-method dsatur
```

### Grafo grande (exemplo: `C4000-260-X.txt`), iniciando a coloração no vértice 2

```bash
python diegoS_Coloracao_v4.py C4000-260-X.txt --coloring-method dsatur --start 2
python diegoS_Coloracao_v4.py C4000-260-X.txt --coloring-method naive --start 2
python diegoS_Coloracao_v4.py C4000-260-X.txt --coloring-method welsh --start 2
```

## Explicação rápida de cada método de coloração

1. **`brute` (Força Bruta)**  
   Tenta todas as combinações possíveis de cores (de 2 até n) para encontrar a coloração mínima. Inviável para grafos com mais de ~10 vértices, pois escala exponencialmente.

2. **`naive` (Heurística Ingênua)**  
   Percorre os vértices na ordem natural (`0, 1, 2, …`) ou começando em `--start`. A cada vértice, atribui a menor cor não utilizada pelos vizinhos já coloridos (greedy simples).

3. **`welsh` (Welsh–Powell)**  
   Ordena todos os vértices por grau decrescente e depois “rota” a lista para que o vértice `--start` fique primeiro. Em seguida, colore greedily nesta ordem.

4. **`dsatur` (DSATUR)**  
   Inicia colorindo o vértice `--start` com cor 0. A cada passo, escolhe o vértice não colorido que possui menor disponibilidade de cores (maior “saturação” = número de cores distintas nos vizinhos). Em caso de empate, escolhe o vértice de maior grau.

## Saída esperada

Para cada heurística, a saída padrão mostrará algo como:

```
--- Coloração ---
Naive: tempo=0.123 ms cores=χ
#######################################################################

Ordem de coloração: v2->v0->v1->…
#######################################################################

################# ATRIBUIÇÕES  #########################################

########################################################################

Atribuições na ordem de coloração:
v2:0, v0:1, v1:0, …
```

Se `--plot-coloring` for ativado, um gráfico interativo aparecerá com:

- Vértices coloridos (usando colormap categórico)
- Bordas destacando vértices de maior grau
- Setas indicando a ordem de coloração (se aplicável)
- Legenda de cores (Cor 1, Cor 2, …)

## Estrutura resumida do script

- **Leitura do grafo (`load_from_file`)**:  
  Lê número de vértices, arestas, se é direcionado e/ou ponderado. Carrega lista e matriz de adjacência.  
- **Algoritmos clássicos** (comentados por padrão se apenas coloração):  
  - `bfs(graph, start, rep)`  
  - `dfs(graph, start, rep)`  
  - `dijkstra(graph, start, rep)`  
- **Funções de coloração**:  
  - `brute_force_coloring(graph, start)`  
  - `heuristic_naive(graph, start)`  
  - `welsh_powell_coloring(graph, start)`  
  - `dsatur_coloring(graph, start)`  
- **Funções de plotagem**  
  - `plot_all_results(...)`: plota BFS, DFS e Dijkstra lado a lado  
  - `plot_coloring(...)`: plota grafo colorido com legendas e setas

## Observações finais

- Para instâncias de grafos grandes, a heurística `brute` pode demorar muito ou não terminar em tempo viável.  
- Ao usar `--plot` ou `--plot-coloring`, certifique-se de que o ambiente permita a abertura de janelas (`matplotlib` interativo).  
- O parâmetro `--start` influencia tanto buscas (origem) quanto colorações (vértice inicial). Para busca clássica, `--start` indica o vértice de onde a busca começa; para coloração, indica qual vértice deve ser colorido primeiro.

---

**Autor:** Diego Silva  
**Data:** Junho de 2025  
