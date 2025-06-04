#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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

Uso geral:
    python graph_algorithms.py <arquivo> [--rep list|matrix|both] [--plot] [--start VERTICE]
        # executa BFS, DFS e Dijkstra

Para executar apenas um método de coloração:
    python graph_algorithms.py .\slides.txt --coloring-method brute
    python graph_algorithms.py .\slides.txt --coloring-method naive
    python graph_algorithms.py .\slides.txt --coloring-method welsh
    python graph_algorithms.py .\slides.txt --coloring-method dsatur
    python graph_algorithms.py .\slides.txt --coloring-method all
    python graph_algorithms.py .\slides.txt --coloring-method welsh --plot-coloring
    python graph_algorithms.py .\slides.txt --coloring-method brute --plot-coloring

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

                # Ajuste: aceitar quantas arestas forem lidas, mesmo que difiram do header
                if count != self.m:
                    print(f"Aviso: lidas {count} arestas; header dizia {self.m}. "
                          f"Prosseguindo com {count} arestas.")
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
# Plot resultados clássicos
# -------------------------------------------------------------------------

def plot_all_results(graph: Graph, rep: str, bfs_path: List[int], dfs_path: List[int], dijk_result=None) -> None:
    G = nx.DiGraph() if graph.directed else nx.Graph()
    G.add_nodes_from(range(graph.n))
    for u in range(graph.n):
        for v, w in graph.get_neighbors(u, rep):
            if not graph.directed and u > v:
                continue
            if graph.weighted:
                G.add_edge(u, v, weight=w)
            else:
                G.add_edge(u, v)
    pos = nx.spring_layout(G)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Resultados em {rep}")
    # BFS
    axes[0].set_title("BFS")
    nx.draw(G, pos, ax=axes[0], node_color="lightblue", with_labels=True)
    nx.draw_networkx_edges(G, pos, ax=axes[0], edgelist=list(zip(bfs_path, bfs_path[1:])), edge_color="red", width=2)
    # DFS
    axes[1].set_title("DFS")
    nx.draw(G, pos, ax=axes[1], node_color="lightgreen", with_labels=True)
    nx.draw_networkx_edges(G, pos, ax=axes[1], edgelist=list(zip(dfs_path, dfs_path[1:])), edge_color="red", width=2)
    # Dijkstra
    axes[2].set_title("Dijkstra")
    nx.draw(G, pos, ax=axes[2], node_color="lightcoral", with_labels=True)
    if dijk_result:
        _, paths = dijk_result
        target = list(paths.keys())[1]  # exemplo: segundo vértice
        path = paths[target]
        if path:
            nx.draw_networkx_edges(G, pos, ax=axes[2], edgelist=list(zip(path, path[1:])), edge_color="red", width=2)
    plt.show()

# -------------------------------------------------------------------------
# Funções de coloração com --start
# -------------------------------------------------------------------------

def brute_force_coloring(graph: Graph, start: int) -> Tuple[Dict[int, int], float, int, Optional[List[int]]]:
    """
    A força bruta não depende de ordem, mas recebe `start` para manter consistência de assinatura.
    """
    start_time = time.perf_counter()
    # Mantemos exatamente como era; `start` é ignorado internamente.
    for k in range(2, graph.n + 1):
        for assign in itertools.product(range(k), repeat=graph.n):
            valid = True
            for u in range(graph.n):
                for v, _ in graph.get_neighbors(u):
                    if assign[u] == assign[v]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                elapsed = time.perf_counter() - start_time
                return ({i: assign[i] for i in range(graph.n)}, elapsed, k, None)
    elapsed = time.perf_counter() - start_time
    return ({}, elapsed, graph.n, None)

def greedy_coloring(graph: Graph, order: List[int]) -> Tuple[Dict[int, int], float, int, List[int]]:
    start_time = time.perf_counter()
    colors: Dict[int, int] = {}
    for u in order:
        forbidden = {colors[v] for v, _ in graph.get_neighbors(u) if v in colors}
        c = 0
        while c in forbidden:
            c += 1
        colors[u] = c
    elapsed = time.perf_counter() - start_time
    return colors, elapsed, max(colors.values()) + 1, order

def heuristic_naive(graph: Graph, start: int):
    # ordem natural, mas começando em `start`
    order = [start] + [i for i in range(graph.n) if i != start]
    return greedy_coloring(graph, order)

def welsh_powell_coloring(graph: Graph, start: int):
    # ordem por grau decrescente
    base = sorted(range(graph.n), key=lambda u: len(graph.get_neighbors(u)), reverse=True)
    # rotaciona para `start` primeiro
    if start in base:
        idx = base.index(start)
        order = base[idx:] + base[:idx]
    else:
        order = base
    return greedy_coloring(graph, order)

def dsatur_coloring(graph: Graph, start: int):
    start_time = time.perf_counter()
    print(f"[DEBUG] dsatur_coloring: vértice inicial = {start}")
    colors: Dict[int, int] = {}
    sat: Dict[int, set[int]] = {u: set() for u in range(graph.n)}
    deg = {u: len(graph.get_neighbors(u)) for u in range(graph.n)}
    # inicia em `start`, não necessariamente o de maior grau
    u0 = start
    colors[u0] = 0
    for v, _ in graph.get_neighbors(u0):
        sat[v].add(0)
    order_sequence: List[int] = [u0]
    while len(colors) < graph.n:
        u = max((x for x in range(graph.n) if x not in colors),
                key=lambda x: (len(sat[x]), deg[x]))
        forbidden = sat[u]
        c = 0
        while c in forbidden:
            c += 1
        colors[u] = c
        order_sequence.append(u)
        for w, _ in graph.get_neighbors(u):
            if w not in colors:
                sat[w].add(c)
    elapsed = time.perf_counter() - start_time
    #print(f"[DEBUG] ordem de coloração DSATUR: {order_sequence}")
    return colors, elapsed, max(colors.values()) + 1, order_sequence

def plot_coloring(graph: Graph, colors: Dict[int, int], k: int, elapsed: float,
                  order: Optional[List[int]], title_base: str) -> None:
    """
    Plota o grafo colorido, com:
     - legenda de cores (numeração cromática)
     - título com k e tempo
     - setas indicando ordem de coloração (se fornecida)
     - vértices de maior grau destacados
    """
    G = nx.DiGraph() if graph.directed else nx.Graph()
    G.add_nodes_from(range(graph.n))
    for u in range(graph.n):
        for v, _ in graph.get_neighbors(u):
            if not graph.directed and u > v:
                continue
            G.add_edge(u, v)

    # layout
    pos = nx.spring_layout(G)

    # destacando vértices de maior grau
    degrees = dict(G.degree())
    max_deg = max(degrees.values())
    node_border = [3 if degrees[n] == max_deg else 1 for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(6,6))
    title = f"{title_base} (k={k}, t={elapsed:.4f}s)"
    ax.set_title(title)

    # nós coloridos
    cmap = plt.cm.get_cmap('tab20', max(colors.values())+1)
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=[colors[n] for n in G.nodes()],
        cmap=cmap,
        node_size=500,
        linewidths=node_border,
        edgecolors='black'
    )
    # labels v0,v1,...
    labels = {n: f"v{n}" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_color='white', ax=ax)

    # setas/linhas de coloração
    if order:
        for u, v in zip(order, order[1:]):
            ax.annotate("",
                        xy=pos[v], xycoords='data',
                        xytext=pos[u], textcoords='data',
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1))
    # arestas
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)

    # legenda de cores
    for color_idx in sorted(set(colors.values())):
        ax.scatter([], [], c=[cmap(color_idx)], label=f"Cor {color_idx+1}", s=100)
    ax.legend(title="Legenda de cores", loc='upper right', bbox_to_anchor=(1.3,1))

    ax.axis('off')
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------
# Impressão genérica
# -------------------------------------------------------------------------

def _print_results(name: str, result) -> None:
    print(f"\n=== {name} ===")
    if isinstance(result, list):
        print("Ordem:", result)
    else:
        dist, paths = result
        for v in sorted(dist):
            p = "-".join(map(str, paths[v])) if paths[v] else "-"
            d = dist[v] if dist[v] != INF else "∞"
            print(f"v{v}: dist={d} path={p}")

# -------------------------------------------------------------------------
# Função principal
# -------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="BFS, DFS, Dijkstra e Coloração de Grafos")
    parser.add_argument("filepath", help="Arquivo .txt do grafo")
    parser.add_argument("--rep", choices=["list", "matrix", "both"], default="both",
                        help="Representação (lista/matriz)")
    parser.add_argument("--start", type=int, default=0, help="Vértice origem / início da coloração")
    parser.add_argument("--plot", action="store_true", help="Exibe gráficos de BFS/DFS/Dijkstra")
    parser.add_argument("--coloring-method", choices=["brute", "naive", "welsh", "dsatur", "all"],
                        help="Método de coloração")
    parser.add_argument("--plot-coloring", action="store_true", help="Exibe gráficos de coloração")
    args = parser.parse_args()

    g = Graph()
    g.load_from_file(args.filepath)
    reps = [args.rep] if args.rep != "both" else ["list", "matrix"]
    '''
    # Execução clássica
    for rep in reps:
        print(f"\n--- Representação: {rep} ---")
        bfs_r = bfs(g, args.start, rep)
        dfs_r = dfs(g, args.start, rep)
        _print_results(f"BFS ({rep})", bfs_r)
        _print_results(f"DFS ({rep})", dfs_r)
        if g.negative_weight:
            print(f"Dijkstra ({rep}) pulado: peso negativo detectado")
            dijk_r = None
        else:
            dijk_r = dijkstra(g, args.start, rep)
            _print_results(f"Dijkstra ({rep})", dijk_r)
        if args.plot and rep == reps[0]:
            plot_all_results(g, rep, bfs_r, dfs_r, dijk_r)
    '''
    # Coloração
    if args.coloring_method:
        methods = {
            'brute':  ('FORÇA BRUTA', brute_force_coloring),
            'naive':  ('Naive',        heuristic_naive),
            'welsh':  ('WELSH–POWEL',  welsh_powell_coloring),
            'dsatur': ('DSATUR',       dsatur_coloring)
        }
        selected = methods if args.coloring_method == 'all' else {
            args.coloring_method: methods[args.coloring_method]
        }

        print("\n--- Coloração ---")
        for key, (label, func) in selected.items():
            try:
                colors, elapsed, k, order = func(g, args.start)
                # Impressão do resultado de coloração
                print(f"{label}: tempo={elapsed*1000:.3f} ms cores={k}")
                print('#######################################################################\n')
                if order:
                    print("Ordem de coloração:", shorten_sequence(order, len(order)))
                print('#######################################################################\n')
                print('################# ATRIBUIÇÕES  #########################################\n')
                print('########################################################################\n')
                # imprime na ordem de coloração
                if order:
                    print("Atribuições na ordem de coloração:")
                    print(", ".join(f"v{u}:{colors[u]}" for u in order))

                if args.plot_coloring:
                    plot_coloring(g, colors, k, elapsed, order, label)
            except Exception as e:
                print(f"{label} falhou: {e}")

if __name__ == "__main__":
    main()
