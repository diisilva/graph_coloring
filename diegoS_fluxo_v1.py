#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
graph_algorithms.py
-------------------
Leitura de um grafo a partir de um arquivo de texto e execução de algoritmos:
 - BFS (Busca em Largura)
 - DFS (Busca em Profundidade)
 - Dijkstra (Menor Caminho)

Novas funcionalidades de coloração de grafos (não ponderados, não direcionados):
 - brute: Força bruta (testa combinações de 2..n cores; recomendado apenas para n<=10)
 - naive: Heurística ingênua (ordem natural 0..n-1)
 - welsh: Heurística Welsh–Powell (ordem decrescente de grau)
 - dsatur: Heurística DSATUR (maior saturação)

Adicionadas funcionalidades de Fluxo Máximo e Busca Local:
 - ford_fulkerson: Algoritmo de Ford-Fulkerson com DFS para caminho aumentante
 - local_search_max_flow: Hill-climbing flipando direções de arestas

Uso geral:
    python graph_algorithms.py <arquivo> [--rep list|matrix|both] [--plot] [--start VERTICE]
        # executa BFS, DFS e Dijkstra

Para fluxo máximo e busca local:
    python graph_algorithms.py <arquivo> --maxflow [--source S] [--sink T]
    python graph_algorithms.py <arquivo> --local-search [--source S] [--sink T] [--max-steps N]

Para executar apenas um método de coloração:
    python graph_algorithms.py .\slides.txt --coloring-method brute
    python graph_algorithms.py .\slides.txt --coloring-method naive
    python graph_algorithms.py .\slides.txt --coloring-method welsh
    python graph_algorithms.py .\slides.txt --coloring-method dsatur
    python graph_algorithms.py .\slides.txt --coloring-method all
    python graph_algorithms.py .\slides.txt --coloring-method welsh --plot-coloring
    python graph_algorithms.py .\slides.txt --coloring-method brute --plot-coloring

Para executar fluxo maximo 

# 2) Fluxo máximo simples
python diegoS_fluxo_v1.py .\grafos_testados\slides.txt --maxflow

# 3) Fluxo máximo com registro de etapas
python diegoS_fluxo_v1.py .\grafos_testados\slides.txt --maxflow --flow-steps

# 4) Fluxo máximo a partir de outro vértice de origem
python diegoS_fluxo_v1.py .\grafos_testados\slides.txt --maxflow --source 2

# 5) Fluxo máximo até outro vértice de destino
python diegoS_fluxo_v1.py .\grafos_testados\slides.txt --maxflow --sink 3

# 6) Busca local para otimizar fluxo (origem=0, destino=ultimo)
python diegoS_fluxo_v1.py .\grafos_testados\slides.txt --local-search

# 7) Busca local com limite de iterações (ex.: 20)
python diegoS_fluxo_v1.py .\grafos_testados\slides.txt --local-search --max-steps 20

Flags de coloração:
    --plot-coloring    # exibe gráficos das colorações
"""

from __future__ import annotations
import argparse
import math
import sys
import time
import itertools
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple, Optional
import heapq
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

INF: float = math.inf

def shorten_sequence(seq: List[int], max_len: int = 6) -> str:
    if len(seq) <= max_len:
        return "->".join(map(str, seq))
    return "->".join(map(str, seq[:max_len])) + "..."

class Graph:
    """
    Grafo com lista e matriz de adjacência, suportando grafos direcionados/não e ponderados/não.
    Vértices numerados de 0 a n-1.
    """
    def __init__(self) -> None:
        self.n: int = 0
        self.m: int = 0
        self.directed: bool = False
        self.weighted: bool = False
        self.adj_list: Dict[int, List[Tuple[int, float]]] = {}
        self.adj_matrix: List[List[float]] = []
        self.negative_weight: bool = False

    def load_from_file(self, filename: str | Path) -> None:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                header = f.readline().split()
                if len(header) != 4:
                    raise ValueError("Cabeçalho inválido: V A D P")
                self.n, self.m = map(int, header[:2])
                self.directed = bool(int(header[2]))
                self.weighted = bool(int(header[3]))
                self.adj_list = {i: [] for i in range(self.n)}
                self.adj_matrix = [[0 if i == j else INF for j in range(self.n)] 
                                   for i in range(self.n)]
                count = 0
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.split()
                    expected = 3 if self.weighted else 2
                    if len(parts) != expected:
                        raise ValueError(f"Aresta inválida: {line.strip()}")
                    u, v = map(int, parts[:2])
                    if not (0 <= u < self.n and 0 <= v < self.n):
                        raise ValueError(f"Vértice fora de 0..{self.n-1}: {u},{v}")
                    w = float(parts[2]) if self.weighted else 1.0
                    if self.weighted and w < 0:
                        self.negative_weight = True
                    self._add_edge(u, v, w)
                    count += 1

                if count != self.m:
                    print(f"Aviso: lidas {count} arestas; header dizia {self.m}. Prosseguindo com {count} arestas.")
                    self.m = count
        except Exception as e:
            sys.exit(f"Erro ao ler '{filename}': {e}")

    def _add_edge(self, u: int, v: int, w: float) -> None:
        self.adj_list[u].append((v, w))
        self.adj_matrix[u][v] = w
        if not self.directed:
            self.adj_list[v].append((u, w))
            self.adj_matrix[v][u] = w

    def get_neighbors(self, u: int, rep: str = "list") -> List[Tuple[int, float]]:
        if rep == "list":
            return self.adj_list[u]
        if rep == "matrix":
            return [(v, w) for v, w in enumerate(self.adj_matrix[u]) if w != INF and v != u]
        raise ValueError("Rep deve ser 'list' ou 'matrix'.")

    def copy(self) -> Graph:
        new = Graph()
        new.n = self.n
        new.m = self.m
        new.directed = self.directed
        new.weighted = self.weighted
        new.negative_weight = self.negative_weight
        new.adj_list = {u: list(self.adj_list[u]) for u in self.adj_list}
        new.adj_matrix = [row[:] for row in self.adj_matrix]
        return new

# -------------------------------------------------------------------------
# Algoritmos clássicos
# -------------------------------------------------------------------------

def bfs(graph: Graph, start: int, rep: str) -> List[int]:
    visited = {start}
    order: List[int] = []
    q: Deque[int] = deque([start])
    while q:
        u = q.popleft()
        order.append(u)
        for v, _ in graph.get_neighbors(u, rep):
            if v not in visited:
                visited.add(v)
                q.append(v)
    return order

def dfs(graph: Graph, start: int, rep: str) -> List[int]:
    visited = set()
    order: List[int] = []
    def _dfs(u: int):
        visited.add(u)
        order.append(u)
        for v, _ in graph.get_neighbors(u, rep):
            if v not in visited:
                _dfs(v)
    _dfs(start)
    return order

def dijkstra(graph: Graph, start: int, rep: str) -> Tuple[Dict[int, float], Dict[int, List[int]]]:
    dist = {i: INF for i in range(graph.n)}
    dist[start] = 0.0
    path = {i: None for i in range(graph.n)}
    path[start] = [start]
    pq: List[Tuple[float, int]] = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph.get_neighbors(u, rep):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                path[v] = path[u] + [v]
                heapq.heappush(pq, (nd, v))
    return dist, path

# -------------------------------------------------------------------------
# Fluxo Máximo e Busca Local
# -------------------------------------------------------------------------

def ford_fulkerson(graph: Graph, source: int, sink: int, record_steps: bool = False):
    # inicializa residual como antes
    residual = [
        [w if w != INF else 0.0 for w in row]
        for row in graph.adj_matrix
    ]
    n = graph.n
    max_flow = 0.0
    augmentations = 0
    start_time = time.perf_counter()

    # listas para registro
    visited_list_all: List[List[int]] = []
    augmenting_paths: List[List[int]] = []

    def dfs_path(u: int, t: int, visited: set[int]) -> Optional[List[int]]:
        if u == t:
            return [u]
        visited.add(u)
        for v in range(n):
            if v not in visited and residual[u][v] > 0:
                p = dfs_path(v, t, visited)
                if p:
                    return [u] + p
        return None

    while True:
        visited = set()
        path = dfs_path(source, sink, visited)
        # registra os vértices que o DFS tocou
        visited_list_all.append(sorted(visited))
        if not path:
            break
        augmentations += 1
        augmenting_paths.append(path)

        # mesma lógica de atualização de residual...
        flow = min(residual[path[i]][path[i+1]] for i in range(len(path)-1))
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            residual[u][v] -= flow
            residual[v][u] += flow
        max_flow += flow

    elapsed = (time.perf_counter() - start_time) * 1000
    if record_steps:
        # retorna também os passos de desenho se for o caso
        return max_flow, augmentations, elapsed, augmenting_paths, visited_list_all, steps
    return max_flow, augmentations, elapsed, augmenting_paths, visited_list_all


def local_search_max_flow(graph: Graph, source: int, sink: int, max_steps: int = 100) -> Tuple[float, float, int]:
    original = ford_fulkerson(graph, source, sink)
    current = original
    g_cur = graph.copy()
    steps = 0
    improved = True

    while improved and steps < max_steps:
        improved = False
        best_flow = current
        best_graph = None

        for u in range(g_cur.n):
            for v, w in enumerate(g_cur.adj_matrix[u]):
                if g_cur.directed and w > 0:
                    neighbor = g_cur.copy()
                    # flip u->v
                    neighbor.adj_matrix[u][v] = 0
                    neighbor.adj_list[u] = [(x,cw) for (x,cw) in neighbor.adj_list[u] if x!=v]
                    neighbor.adj_matrix[v][u] = w
                    neighbor.adj_list[v].append((u,w))

                    f = ford_fulkerson(neighbor, source, sink)
                    if f > best_flow:
                        best_flow = f
                        best_graph = neighbor

        if best_graph:
            g_cur = best_graph
            current = best_flow
            steps += 1
            improved = True

    return original, current, steps

# -------------------------------------------------------------------------
# Plot resultados clássicos
# -------------------------------------------------------------------------

def plot_all_results(graph: Graph, rep: str, bfs_path: List[int], dfs_path: List[int], dijk_result=None) -> None:
    G = nx.DiGraph() if graph.directed else nx.Graph()
    G.add_nodes_from(range(graph.n))
    for u in range(graph.n):
        for v, w in graph.get_neighbors(u, rep):
            if not graph.directed and u > v:
                continue
            G.add_edge(u, v, weight=w) if graph.weighted else G.add_edge(u, v)

    pos = nx.spring_layout(G)
    fig, axes = plt.subplots(1, 3, figsize=(18,6))
    fig.suptitle(f"Resultados em {rep}")

    # BFS
    axes[0].set_title("BFS")
    nx.draw(G, pos, ax=axes[0], node_color="lightblue", with_labels=True)
    nx.draw_networkx_edges(G, pos, ax=axes[0],
        edgelist=list(zip(bfs_path, bfs_path[1:])), edge_color="red", width=2)

    # DFS
    axes[1].set_title("DFS")
    nx.draw(G, pos, ax=axes[1], node_color="lightgreen", with_labels=True)
    nx.draw_networkx_edges(G, pos, ax=axes[1],
        edgelist=list(zip(dfs_path, dfs_path[1:])), edge_color="red", width=2)

    # Dijkstra
    axes[2].set_title("Dijkstra")
    nx.draw(G, pos, ax=axes[2], node_color="lightcoral", with_labels=True)
    if dijk_result:
        _, paths = dijk_result
        target = list(paths.keys())[1] if len(paths)>1 else 0
        path = paths[target]
        if path:
            nx.draw_networkx_edges(G, pos, ax=axes[2],
                edgelist=list(zip(path, path[1:])), edge_color="red", width=2)

    plt.show()

# -------------------------------------------------------------------------
# Métodos de coloração
# -------------------------------------------------------------------------

def brute_force_coloring(graph: Graph, start: int) -> Tuple[Dict[int,int],float,int,Optional[List[int]]]:
    start_time = time.perf_counter()
    for k in range(2, graph.n+1):
        for assign in itertools.product(range(k), repeat=graph.n):
            valid = True
            for u in range(graph.n):
                for v,_ in graph.get_neighbors(u):
                    if assign[u]==assign[v]:
                        valid=False
                        break
                if not valid:
                    break
            if valid:
                elapsed = time.perf_counter()-start_time
                return ({i:assign[i] for i in range(graph.n)}, elapsed, k, None)
    elapsed = time.perf_counter()-start_time
    return ({}, elapsed, graph.n, None)

def greedy_coloring(graph: Graph, order: List[int]) -> Tuple[Dict[int,int],float,int,List[int]]:
    start_time = time.perf_counter()
    colors:Dict[int,int] = {}
    for u in order:
        forbidden = {colors[v] for v,_ in graph.get_neighbors(u) if v in colors}
        c = 0
        while c in forbidden: c+=1
        colors[u]=c
    elapsed = time.perf_counter()-start_time
    return colors, elapsed, max(colors.values())+1, order

def heuristic_naive(graph: Graph, start: int):
    order = [start]+[i for i in range(graph.n) if i!=start]
    return greedy_coloring(graph, order)

def welsh_powell_coloring(graph: Graph, start: int):
    base = sorted(range(graph.n), key=lambda u: len(graph.get_neighbors(u)), reverse=True)
    if start in base:
        idx = base.index(start)
        order = base[idx:]+base[:idx]
    else:
        order = base
    return greedy_coloring(graph, order)

def dsatur_coloring(graph: Graph, start: int):
    start_time = time.perf_counter()
    colors:Dict[int,int]={}
    sat = {u:set() for u in range(graph.n)}
    deg = {u:len(graph.get_neighbors(u)) for u in range(graph.n)}
    u0 = start
    colors[u0]=0
    for v,_ in graph.get_neighbors(u0):
        sat[v].add(0)
    order_seq=[u0]
    while len(colors)<graph.n:
        u = max((x for x in range(graph.n) if x not in colors),
                key=lambda x:(len(sat[x]), deg[x]))
        forbidden = sat[u]
        c=0
        while c in forbidden: c+=1
        colors[u]=c
        order_seq.append(u)
        for w,_ in graph.get_neighbors(u):
            if w not in colors:
                sat[w].add(c)
    elapsed = time.perf_counter()-start_time
    return colors, elapsed, max(colors.values())+1, order_seq

def plot_coloring(graph: Graph, colors: Dict[int,int], k:int, elapsed:float,
                  order:Optional[List[int]], title_base:str) -> None:
    G = nx.DiGraph() if graph.directed else nx.Graph()
    G.add_nodes_from(range(graph.n))
    for u in range(graph.n):
        for v,_ in graph.get_neighbors(u):
            if not graph.directed and u>v: continue
            G.add_edge(u,v)
    pos = nx.spring_layout(G)
    degrees = dict(G.degree())
    max_deg = max(degrees.values())
    border = [3 if degrees[n]==max_deg else 1 for n in G.nodes()]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_title(f"{title_base} (k={k}, t={elapsed:.4f}s)")
    cmap = plt.cm.get_cmap('tab20', k)
    nx.draw_networkx_nodes(G, pos, ax=ax,
        node_color=[colors[n] for n in G.nodes()],
        cmap=cmap, node_size=500, linewidths=border, edgecolors='black')
    labels = {n:f"v{n}" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_color='white', ax=ax)
    if order:
        for u,v in zip(order, order[1:]):
            ax.annotate("", xy=pos[v], xytext=pos[u],
                        arrowprops=dict(arrowstyle="->", lw=1, color="gray"))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
    for idx in sorted(set(colors.values())):
        ax.scatter([],[], c=[cmap(idx)], label=f"Cor {idx+1}", s=100)
    ax.legend(title="Legenda de cores", loc='upper right', bbox_to_anchor=(1.3,1))
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------
# Impressão de resultados
# -------------------------------------------------------------------------

def _print_results(name: str, result) -> None:
    print(f"\n=== {name} ===")
    if isinstance(result, list):
        print("Ordem:", result)
    else:
        dist, paths = result
        for v in sorted(dist):
            d = dist[v] if dist[v]!=INF else "∞"
            p = "-".join(map(str, paths[v])) if paths[v] else "-"
            print(f"v{v}: dist={d} path={p}")

# -------------------------------------------------------------------------
# Função principal
# -------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="BFS, DFS, Dijkstra, Fluxo Máximo e Busca Local"
    )
    parser.add_argument("filepath", help="Arquivo .txt do grafo")
    parser.add_argument("--rep", choices=["list","matrix","both"], default="both",
                        help="Representação (lista/matriz)")
    parser.add_argument("--start", type=int, default=0,
                        help="Vértice origem / início da coloração")
    parser.add_argument("--plot", action="store_true",
                        help="Exibe gráficos de BFS/DFS/Dijkstra")
    parser.add_argument("--coloring-method", choices=["brute","naive","welsh","dsatur","all"],
                        help="Método de coloração")
    parser.add_argument("--plot-coloring", action="store_true",
                        help="Exibe gráficos de coloração")
    parser.add_argument("--maxflow", action="store_true",
                        help="Executa fluxo máximo Ford-Fulkerson")
    parser.add_argument("--flow-steps", action="store_true",
                        help="Salva imagens de cada etapa do Ford–Fulkerson")
    parser.add_argument("--local-search", action="store_true",
                        help="Executa busca local para otimização do fluxo")
    parser.add_argument("--source", type=int, default=0,
                        help="Vértice de origem para fluxo")
    parser.add_argument("--sink", type=int, nargs="?", default=None,
                        help="Vértice de destino para fluxo")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Passos máximos na busca local")
    args = parser.parse_args()

    g = Graph()
    g.load_from_file(args.filepath)
    if args.flow_steps and not args.maxflow:
        args.maxflow = True
    if (args.maxflow or args.local_search) and args.sink is None:
        args.sink = g.n - 1

    # Execução de fluxo máximo
    if args.maxflow:
        # chama a função e desempacota tudo
        if args.flow_steps:
            mf, aug, ms, aug_paths, visited_list, steps = ford_fulkerson(
                g, args.source, args.sink, record_steps=True
            )
        else:
            mf, aug, ms, aug_paths, visited_list = ford_fulkerson(
                g, args.source, args.sink, record_steps=False
            )

        print("\n=== Fluxo Máximo (Ford–Fulkerson) ===\n")
        print(f"Fluxo Máximo = {mf} em {aug} caminhos aumentantes (t={ms:.2f} ms)")
        print(f"Origem: {args.source}, Destino: {args.sink}\n")

        # imprime vértices visitados em cada DFS
        print("Vértices visitados por DFS em cada iteração:")
        for i, vis in enumerate(visited_list, start=1):
            print(f"  Iter {i}: {vis}")
        print()

        # imprime cada caminho aumentante
        print("Caminhos aumentantes encontrados:")
        for i, p in enumerate(aug_paths, start=1):
            seq = "→".join(map(str, p))
            cap = min([g.adj_matrix[p[j]][p[j+1]] for j in range(len(p)-1)])
            print(f"  Caminho {i}: {seq} (capacidade = {cap})")
        print()

        if args.flow_steps:
            from pathlib import Path
            out_dir = Path("flow_steps")
            out_dir.mkdir(exist_ok=True)
            print(f"Passos gravados: {len(steps)}")

            for i, (G_step, pos, labels) in enumerate(steps, start=1):
                fig, ax = plt.subplots(figsize=(4, 3))
                nx.draw_networkx(G_step, pos, ax=ax, with_labels=True, node_size=300)

                # ajuste manual dos limites para incluir todos os nós
                xs = [x for x, y in pos.values()]
                ys = [y for x, y in pos.values()]
                xpad = (max(xs) - min(xs)) * 0.1
                ypad = (max(ys) - min(ys)) * 0.1
                ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
                ax.set_ylim(min(ys) - ypad, max(ys) + ypad)

                # rótulos de arestas com fallback
                for (u, v), txt in labels.items():
                    try:
                        nx.draw_networkx_edge_labels(
                            G_step,
                            pos,
                            edge_labels={(u, v): txt},
                            ax=ax,
                            font_size=8,
                            rotate=False,
                            clip_on=True,
                        )
                    except Exception:
                        x1, y1 = pos[u]; x2, y2 = pos[v]
                        ax.text(
                            (x1 + x2) / 2,
                            (y1 + y2) / 2,
                            txt,
                            fontsize=8,
                            ha="center",
                            va="center",
                            clip_on=True,
                        )
                ax.axis("off")
                fname = out_dir / f"step_{i:02d}.png"
                fig.savefig(fname, dpi=150)
                plt.close(fig)

            print(f"\nImagens salvas em ./{out_dir}/step_01.png … step_{len(steps):02d}.png")
            sys.exit(0)

    # Execução de busca local
    if args.local_search:
        orig, opt, steps_ls = local_search_max_flow(
            g, args.source, args.sink, args.max_steps
        )
        # Desempacotar apenas os três primeiros valores de cada tupla
        flow_o, aug_o, time_o, *rest_o = orig
        flow_p, aug_p, time_p, *rest_p = opt

        print(f"\n=== Otimização por Busca Local ===\n")

        # Fluxo Original detalhado
        print(f"Fluxo Original = {flow_o} em {aug_o} caminhos aumentantes (t={time_o:.2f} ms)\n")

        # Fluxo Otimizado detalhado
        print(f"Fluxo Otimizado = {flow_p} em {aug_p} caminhos aumentantes (t={time_p:.2f} ms)\n")

        # Contagem de passos da busca local
        print(f"Número de passos de busca local: {steps_ls}")

    # Execução clássica (BFS, DFS, Dijkstra)
    reps = [args.rep] if args.rep != "both" else ["list", "matrix"]
    '''
    for rep in reps:
        ...
    '''

    # Coloração
    if args.coloring_method:
        methods = {
            "brute": ("FORÇA BRUTA", brute_force_coloring),
            "naive": ("Naive", heuristic_naive),
            "welsh": ("WELSH–POWELL", welsh_powell_coloring),
            "dsatur": ("DSATUR", dsatur_coloring),
        }
        selected = (
            methods
            if args.coloring_method == "all"
            else {args.coloring_method: methods[args.coloring_method]}
        )
        print("\n--- Coloração ---")
        for key, (label, func) in selected.items():
            colors, elapsed, k, order = func(g, args.start)
            print(f"{label}: tempo={elapsed*1000:.3f} ms cores={k}")
            print(
                "Ordem de coloração:",
                shorten_sequence(order, len(order)) if order else "",
            )
            print(
                "Atribuições:",
                ", ".join(f"v{u}:{colors[u]}" for u in (order or sorted(colors))),
            )

if __name__ == "__main__":
    main()

