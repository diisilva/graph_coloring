#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-----------------
Implementação de algoritmos de Árvore Geradora Mínima (MST):
 - Prim
 - Kruskal

Uso:
    python agm_v1.py <arquivo> [--algorithm prim|kruskal|all] [--start VERTICE_PRIM]

Argumentos:
    arquivo             Caminho para arquivo de grafo (.txt) com formato:
                        n m directed_flag weighted_flag
                        u v [w]
    --algorithm         Algoritmo(s) a executar (default: all)
    --start             Vértice inicial para Prim (default: 0)

Saídas exibidas:
 - Tempo de execução (formatado automaticamente em µs, ms ou s)
 - Soma das arestas da MST
 - Lista de arestas da MST
 - Média de peso das arestas (informação adicional útil)

Testes Realizados:
python agm_v1.py grafos_testados\r250-66-65.txt --algorithm kruskal
python agm_v1.py grafos_testados\r250-66-65.txt --algorithm prim --start 4
python agm_v1.py grafos_testados\slides_modificado.txt --algorithm all
"""

import argparse
import sys
import time
import heapq

# -------------------------------------------------------------------------
# Utilitários
# -------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    """
    Formata o tempo em µs, ms ou s conforme magnitude.
    """
    us = seconds * 1e6
    ms = seconds * 1e3
    if us < 1000:
        return f"{us:.2f} µs"
    if ms < 1000:
        return f"{ms:.2f} ms"
    return f"{seconds:.4f} s"

# -------------------------------------------------------------------------
# Estrutura de Grafo
# -------------------------------------------------------------------------

class Graph:
    """
    Grafo não direcionado, suportando pesos.
    """
    def __init__(self):
        self.n = 0
        self.m = 0
        self.directed = False
        self.weighted = False
        self.adj_list = {}        # {u: [(v, w), ...], ...}
        self.edge_list = []       # [(w, u, v), ...]

    def load_from_file(self, filepath: str) -> None:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                header = f.readline().split()
                if len(header) != 4:
                    raise ValueError('Cabeçalho inválido: esperado 4 valores')
                self.n, self.m = map(int, header[:2])
                self.directed = bool(int(header[2]))
                self.weighted = bool(int(header[3]))
                # Inicializa estruturas
                self.adj_list = {i: [] for i in range(self.n)}
                self.edge_list = []
                # Lê arestas
                count = 0
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.split()
                    expected = 3 if self.weighted else 2
                    if len(parts) != expected:
                        raise ValueError(f'Aresta inválida: {line.strip()}')
                    u, v = map(int, parts[:2])
                    w = float(parts[2]) if self.weighted else 1.0
                    # Adiciona
                    self.adj_list[u].append((v, w))
                    self.adj_list[v].append((u, w))
                    self.edge_list.append((w, u, v))
                    count += 1
                if count != self.m:
                    print(f'Aviso: lidas {count} arestas; header dizia {self.m}. Ajustando.')
                    self.m = count
        except Exception as e:
            sys.exit(f"Erro ao ler '{filepath}': {e}")
        if self.directed:
            sys.exit('MST requer grafo não direcionado (directed_flag=0)')

# -------------------------------------------------------------------------
# Kruskal com Union-Find
# -------------------------------------------------------------------------

class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, u: int) -> int:
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u: int, v: int) -> bool:
        ru, rv = self.find(u), self.find(v)
        if ru == rv:
            return False
        if self.rank[ru] < self.rank[rv]:
            self.parent[ru] = rv
        else:
            self.parent[rv] = ru
            if self.rank[ru] == self.rank[rv]:
                self.rank[ru] += 1
        return True


def kruskal(graph: Graph):
    """
    Executa Kruskal e retorna (arestas_mst, soma_pesos, tempo_sec).
    """
    start = time.perf_counter()
    uf = UnionFind(graph.n)
    edges = sorted(graph.edge_list, key=lambda x: x[0])
    mst = []
    total = 0.0
    for w, u, v in edges:
        if uf.union(u, v):
            mst.append((u, v, w))
            total += w
            if len(mst) == graph.n - 1:
                break
    elapsed = time.perf_counter() - start
    return mst, total, elapsed

# -------------------------------------------------------------------------
# Prim com Min-Heap
# -------------------------------------------------------------------------

def prim(graph: Graph, start_vertex: int = 0):
    """
    Executa Prim e retorna (arestas_mst, soma_pesos, tempo_sec).
    """
    start = time.perf_counter()
    visited = [False] * graph.n
    visited[start_vertex] = True
    heap = []
    for v, w in graph.adj_list[start_vertex]:
        heapq.heappush(heap, (w, start_vertex, v))
    mst = []
    total = 0.0
    while heap and len(mst) < graph.n - 1:
        w, u, v = heapq.heappop(heap)
        if visited[v]:
            continue
        visited[v] = True
        mst.append((u, v, w))
        total += w
        for nxt, wgt in graph.adj_list[v]:
            if not visited[nxt]:
                heapq.heappush(heap, (wgt, v, nxt))
    elapsed = time.perf_counter() - start
    return mst, total, elapsed

# -------------------------------------------------------------------------
# Impressão de resultados
# -------------------------------------------------------------------------

def print_results(name: str, mst, total: float, elapsed: float) -> None:
    print(f"\n=== {name} ===")
    print(f"Tempo: {format_time(elapsed)}")
    print(f"Soma dos pesos: {total:.2f}")
    avg = total / len(mst) if mst else 0
    print(f"Arestas na MST: {len(mst)} (média de peso = {avg:.2f})")
    print("Lista de arestas (u, v, peso):")
    for u, v, w in mst:
        print(f"  {u} -- {v} [w={w}]")

# -------------------------------------------------------------------------
# Função Principal
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Executa algoritmos de MST (Prim, Kruskal) em grafos não direcionados.')
    parser.add_argument('filepath', help='Arquivo de grafo (.txt)')
    parser.add_argument('--algorithm', choices=['prim', 'kruskal', 'all'],
                        default='all', help='Algoritmo a executar')
    parser.add_argument('--start', type=int, default=0,
                        help='Vértice inicial para Prim')
    args = parser.parse_args()

    g = Graph()
    g.load_from_file(args.filepath)

    if args.algorithm in ('kruskal', 'all'):
        mst_k, total_k, t_k = kruskal(g)
        print_results('Kruskal', mst_k, total_k, t_k)

    if args.algorithm in ('prim', 'all'):
        mst_p, total_p, t_p = prim(g, args.start)
        print_results('Prim', mst_p, total_p, t_p)

if __name__ == '__main__':
    main()
