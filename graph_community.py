# -*- coding: utf-8 -*-
"""
圖譜社群偵測與視覺化模組
實現類似 Microsoft GraphRAG 的社群聚類視覺化
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx
from loguru import logger

try:
    import community as community_louvain  # python-louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    logger.warning("python-louvain 未安裝，將使用備用社群偵測方法")


@dataclass
class Community:
    """社群資料結構"""
    id: int
    name: str = ""
    nodes: Set[str] = field(default_factory=set)
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    center_node: str = ""
    color: str = "#888888"
    size: int = 0


class GraphCommunityDetector:
    """圖譜社群偵測器"""

    def __init__(self, graph: nx.Graph):
        """
        初始化社群偵測器

        Args:
            graph: NetworkX 圖譜（會轉換為無向圖進行社群偵測）
        """
        # 轉換為無向圖
        if isinstance(graph, nx.DiGraph) or isinstance(graph, nx.MultiDiGraph):
            self.graph = graph.to_undirected()
        else:
            self.graph = graph

        self.original_graph = graph
        self.communities: Dict[int, Community] = {}
        self.node_to_community: Dict[str, int] = {}

    def detect_communities(self, resolution: float = 1.0) -> Dict[int, Community]:
        """
        偵測社群

        Args:
            resolution: 解析度參數，越大社群越小

        Returns:
            社群字典
        """
        if LOUVAIN_AVAILABLE:
            return self._detect_louvain(resolution)
        else:
            return self._detect_label_propagation()

    def _detect_louvain(self, resolution: float = 1.0) -> Dict[int, Community]:
        """使用 Louvain 演算法偵測社群"""
        logger.info("使用 Louvain 演算法偵測社群...")

        # 執行 Louvain 演算法
        partition = community_louvain.best_partition(
            self.graph,
            resolution=resolution,
            random_state=42
        )

        # 整理社群資料
        community_nodes = defaultdict(set)
        for node, comm_id in partition.items():
            community_nodes[comm_id].add(node)
            self.node_to_community[node] = comm_id

        # 建立社群物件
        colors = self._generate_colors(len(community_nodes))

        for comm_id, nodes in community_nodes.items():
            community = Community(
                id=comm_id,
                nodes=nodes,
                size=len(nodes),
                color=colors[comm_id % len(colors)]
            )

            # 找出中心節點（度數最高）
            max_degree = 0
            for node in nodes:
                degree = self.graph.degree(node)
                if degree > max_degree:
                    max_degree = degree
                    community.center_node = node

            # 生成社群名稱和關鍵字
            community.name, community.keywords = self._generate_community_info(nodes)

            self.communities[comm_id] = community

        logger.success(f"偵測到 {len(self.communities)} 個社群")
        return self.communities

    def _detect_label_propagation(self) -> Dict[int, Community]:
        """使用標籤傳播演算法（備用方法）"""
        logger.info("使用標籤傳播演算法偵測社群...")

        communities_generator = nx.community.label_propagation_communities(self.graph)

        colors = self._generate_colors(100)  # 預設顏色

        for comm_id, nodes in enumerate(communities_generator):
            nodes_set = set(nodes)

            for node in nodes_set:
                self.node_to_community[node] = comm_id

            community = Community(
                id=comm_id,
                nodes=nodes_set,
                size=len(nodes_set),
                color=colors[comm_id % len(colors)]
            )

            # 找出中心節點
            max_degree = 0
            for node in nodes_set:
                degree = self.graph.degree(node)
                if degree > max_degree:
                    max_degree = degree
                    community.center_node = node

            community.name, community.keywords = self._generate_community_info(nodes_set)
            self.communities[comm_id] = community

        logger.success(f"偵測到 {len(self.communities)} 個社群")
        return self.communities

    def _generate_community_info(self, nodes: Set[str]) -> Tuple[str, List[str]]:
        """生成社群名稱和關鍵字"""
        # 統計節點類型
        type_counts = defaultdict(int)
        names = []

        for node in nodes:
            if self.original_graph.has_node(node):
                data = self.original_graph.nodes[node]
                node_type = data.get('node_type', '未知')
                type_counts[node_type] += 1

                name = data.get('name', '')
                if name and node_type == '職能基準':
                    names.append(name)

        # 主要類型作為社群名稱的一部分
        main_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else '混合'

        # 取前幾個職能基準名稱作為關鍵字
        keywords = names[:5]

        # 生成名稱
        if keywords:
            name = f"{main_type}群組 - {keywords[0][:10]}等"
        else:
            name = f"{main_type}群組 #{len(self.communities)}"

        return name, keywords

    def _generate_colors(self, n: int) -> List[str]:
        """生成 n 個不同的顏色"""
        colors = [
            "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
            "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5",
            "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f",
            "#e5c494", "#b3b3b3", "#1b9e77", "#d95f02", "#7570b3",
            "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"
        ]

        # 如果需要更多顏色，使用 HSL 生成
        if n > len(colors):
            for i in range(len(colors), n):
                hue = (i * 137.508) % 360  # 黃金角
                colors.append(f"hsl({hue}, 70%, 50%)")

        return colors

    def generate_summary(self, community_id: int) -> str:
        """生成社群摘要"""
        if community_id not in self.communities:
            return ""

        community = self.communities[community_id]

        # 統計資訊
        type_counts = defaultdict(int)
        standards = []
        skills = []
        knowledge = []

        for node in community.nodes:
            if self.original_graph.has_node(node):
                data = self.original_graph.nodes[node]
                node_type = data.get('node_type', '未知')
                type_counts[node_type] += 1

                name = data.get('name', '') or data.get('description', '')[:30]

                if node_type == '職能基準':
                    standards.append(name)
                elif node_type == '技能':
                    skills.append(name)
                elif node_type == '知識':
                    knowledge.append(name)

        # 生成摘要
        summary_parts = [f"此社群包含 {community.size} 個節點。"]

        if standards:
            summary_parts.append(f"主要職能基準: {', '.join(standards[:3])}")

        if type_counts:
            type_str = ', '.join([f"{t}: {c}" for t, c in sorted(type_counts.items(), key=lambda x: -x[1])[:5]])
            summary_parts.append(f"節點類型分布: {type_str}")

        if skills:
            summary_parts.append(f"核心技能: {', '.join(skills[:5])}")

        if knowledge:
            summary_parts.append(f"關鍵知識: {', '.join(knowledge[:5])}")

        community.summary = ' '.join(summary_parts)
        return community.summary


def generate_community_visualization(
    graph: nx.Graph,
    output_path: str,
    center_node: str = None,
    resolution: float = 1.0,
    max_nodes: int = 500,
    title: str = "職能基準知識圖譜 - 社群視覺化"
) -> str:
    """
    生成社群視覺化 HTML（圓形散佈佈局 + 社群顏色）

    Args:
        graph: NetworkX 圖譜
        output_path: 輸出路徑
        center_node: 中心職能基準節點（如果指定，只展開這個節點的子節點）
        resolution: 社群解析度
        max_nodes: 最大節點數
        title: 標題

    Returns:
        輸出檔案路徑
    """
    logger.info("開始生成社群視覺化（圓形散佈佈局）...")

    # 偵測社群
    detector = GraphCommunityDetector(graph)
    communities = detector.detect_communities(resolution=resolution)

    # 選擇要顯示的節點
    selected_nodes = set()

    if center_node and graph.has_node(center_node):
        # 如果指定了中心節點，從該節點展開
        selected_nodes.add(center_node)
        _expand_hierarchical_nodes(graph, center_node, selected_nodes, max_nodes, depth=3)
        logger.info(f"從 {center_node} 展開，選擇了 {len(selected_nodes)} 個節點")
    else:
        # 沒有指定中心節點，選擇較大社群的節點
        sorted_communities = sorted(communities.values(), key=lambda c: c.size, reverse=True)

        for comm in sorted_communities:
            if len(selected_nodes) >= max_nodes:
                break

            # 每個社群選擇一些節點
            comm_nodes = list(comm.nodes)

            # 優先選擇職能基準節點
            priority_nodes = []
            other_nodes = []

            for node in comm_nodes:
                if graph.has_node(node):
                    data = graph.nodes[node]
                    if data.get('node_type') == '職能基準':
                        priority_nodes.append(node)
                    else:
                        other_nodes.append(node)

            # 選擇節點
            nodes_to_add = priority_nodes[:10] + other_nodes[:20]
            selected_nodes.update(nodes_to_add)

        logger.info(f"選擇了 {len(selected_nodes)} 個節點")

    # 生成社群摘要
    for comm_id in communities:
        detector.generate_summary(comm_id)

    # 準備視覺化資料
    nodes_data = []
    edges_data = []
    community_data = []

    # 計算社群中心位置（使用圓形布局）
    sorted_communities = sorted(communities.values(), key=lambda c: c.size, reverse=True)
    num_communities = len(sorted_communities)
    community_positions = {}

    for i, comm in enumerate(sorted_communities[:20]):  # 最多顯示 20 個社群
        angle = 2 * math.pi * i / min(num_communities, 20)
        radius = 400
        cx = radius * math.cos(angle)
        cy = radius * math.sin(angle)
        community_positions[comm.id] = (cx, cy)

        community_data.append({
            "id": comm.id,
            "name": comm.name,
            "summary": comm.summary,
            "size": comm.size,
            "color": comm.color,
            "keywords": comm.keywords[:5],
            "x": cx,
            "y": cy
        })

    # 準備節點資料（圓形散佈位置）
    for node in selected_nodes:
        if not graph.has_node(node):
            continue

        data = graph.nodes[node]
        comm_id = detector.node_to_community.get(node, 0)
        comm = communities.get(comm_id)

        if not comm or comm_id not in community_positions:
            continue

        # 在社群中心附近散佈
        cx, cy = community_positions[comm_id]

        # 使用節點 hash 來產生穩定的隨機位置
        hash_val = hash(node)
        angle = (hash_val % 360) * math.pi / 180
        r = 50 + (hash_val % 100)

        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)

        node_type = data.get('node_type', '未知')
        name = data.get('name', '') or data.get('code', node)[:20]

        nodes_data.append({
            "id": node,
            "label": name[:15] + "..." if len(name) > 15 else name,
            "title": f"{node_type}: {name}",
            "group": comm_id,
            "color": comm.color if comm else "#888888",
            "size": 20 if node_type == '職能基準' else 10,
            "x": x,
            "y": y,
            "type": node_type
        })

    # 準備邊資料（使用社群顏色）
    for source, target, edge_data in graph.edges(data=True):
        if source in selected_nodes and target in selected_nodes:
            # 取得來源節點的社群顏色
            source_comm_id = detector.node_to_community.get(source, 0)
            target_comm_id = detector.node_to_community.get(target, 0)
            source_comm = communities.get(source_comm_id)
            target_comm = communities.get(target_comm_id)

            # 如果同社群，使用該社群顏色；否則使用灰色
            if source_comm_id == target_comm_id and source_comm:
                edge_color = source_comm.color
                edge_width = 2
            else:
                edge_color = "#888888"
                edge_width = 1

            edges_data.append({
                "from": source,
                "to": target,
                "color": edge_color,
                "width": edge_width
            })

    # 生成 HTML
    html_content = _generate_community_html(
        nodes_data,
        edges_data,
        community_data,
        title
    )

    # 寫入檔案
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.success(f"社群視覺化已儲存: {output_path}")
    return str(output_path)


def _expand_hierarchical_nodes(graph: nx.Graph, center_node: str, selected_nodes: set,
                                max_nodes: int, depth: int = 2):
    """沿著出邊方向展開階層節點"""
    current_level = {center_node}

    for d in range(depth):
        if len(selected_nodes) >= max_nodes:
            break

        next_level = set()
        for node in current_level:
            # 只沿著出邊方向展開
            if isinstance(graph, nx.DiGraph):
                neighbors = [t for _, t in graph.out_edges(node)]
            else:
                neighbors = list(graph.neighbors(node))

            for neighbor in neighbors:
                if len(selected_nodes) >= max_nodes:
                    break
                if neighbor not in selected_nodes:
                    neighbor_type = graph.nodes[neighbor].get('node_type', '')
                    # 不要把其他職能基準拉進來
                    if neighbor_type != '職能基準':
                        selected_nodes.add(neighbor)
                        next_level.add(neighbor)

        current_level = next_level


def _generate_community_html(
    nodes: List[Dict],
    edges: List[Dict],
    communities: List[Dict],
    title: str
) -> str:
    """生成社群視覺化 HTML"""

    nodes_json = json.dumps(nodes, ensure_ascii=False)
    edges_json = json.dumps(edges, ensure_ascii=False)
    communities_json = json.dumps(communities, ensure_ascii=False)

    html = f'''<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Microsoft JhengHei', 'PingFang TC', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}

        .container {{
            display: flex;
            height: 100vh;
        }}

        .sidebar {{
            width: 320px;
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            overflow-y: auto;
            box-shadow: 2px 0 10px rgba(0,0,0,0.1);
        }}

        .main {{
            flex: 1;
            position: relative;
        }}

        #network {{
            width: 100%;
            height: 100%;
            background: #f8f9fa;
        }}

        h1 {{
            font-size: 1.4em;
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}

        h2 {{
            font-size: 1.1em;
            color: #555;
            margin: 15px 0 10px 0;
        }}

        .community-card {{
            background: #fff;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: all 0.3s ease;
            border-left: 4px solid #888;
        }}

        .community-card:hover {{
            transform: translateX(5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}

        .community-card.active {{
            background: #f0f4ff;
        }}

        .community-header {{
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }}

        .community-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
        }}

        .community-name {{
            font-weight: bold;
            color: #333;
            flex: 1;
        }}

        .community-size {{
            font-size: 0.85em;
            color: #888;
        }}

        .community-summary {{
            font-size: 0.9em;
            color: #666;
            line-height: 1.5;
            margin-top: 8px;
        }}

        .community-keywords {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-top: 10px;
        }}

        .keyword-tag {{
            background: #e9ecef;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            color: #555;
        }}

        .legend {{
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}

        .legend-title {{
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
            font-size: 0.85em;
        }}

        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}

        .stats {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}

        .stats-row {{
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
        }}

        .tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 0.9em;
            max-width: 300px;
            z-index: 1000;
            pointer-events: none;
            display: none;
        }}

        .controls {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .controls button {{
            background: #667eea;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
            margin: 2px;
            font-size: 0.9em;
        }}

        .controls button:hover {{
            background: #5a6fd6;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h1>🔮 {title}</h1>

            <div class="stats">
                <div class="stats-row">
                    <span>節點數</span>
                    <span id="nodeCount">0</span>
                </div>
                <div class="stats-row">
                    <span>邊數</span>
                    <span id="edgeCount">0</span>
                </div>
                <div class="stats-row">
                    <span>社群數</span>
                    <span id="communityCount">0</span>
                </div>
            </div>

            <h2>📊 社群列表</h2>
            <div id="communityList"></div>

            <div class="legend">
                <div class="legend-title">說明</div>
                <div style="font-size: 0.85em; color: #666; line-height: 1.6;">
                    • <strong>節點顏色</strong>代表社群分組<br>
                    • <strong>同色節點</strong>屬於同一社群<br>
                    • <strong>大節點</strong>代表職能基準<br>
                    • 點擊左側社群卡片可聚焦該社群
                </div>
            </div>
        </div>

        <div class="main">
            <div id="network"></div>
            <div class="controls">
                <button onclick="resetView()">重置視圖</button>
                <button onclick="togglePhysics()">切換物理效果</button>
            </div>
            <div class="tooltip" id="tooltip"></div>
        </div>
    </div>

    <script>
        // 資料
        const nodesData = {nodes_json};
        const edgesData = {edges_json};
        const communitiesData = {communities_json};

        // 更新統計
        document.getElementById('nodeCount').textContent = nodesData.length;
        document.getElementById('edgeCount').textContent = edgesData.length;
        document.getElementById('communityCount').textContent = communitiesData.length;

        // 建立社群列表
        const communityList = document.getElementById('communityList');
        communitiesData.forEach(comm => {{
            const card = document.createElement('div');
            card.className = 'community-card';
            card.style.borderLeftColor = comm.color;
            card.innerHTML = `
                <div class="community-header">
                    <div class="community-color" style="background: ${{comm.color}}"></div>
                    <span class="community-name">${{comm.name}}</span>
                    <span class="community-size">${{comm.size}} 節點</span>
                </div>
                <div class="community-summary">${{comm.summary}}</div>
                <div class="community-keywords">
                    ${{comm.keywords.map(k => `<span class="keyword-tag">${{k}}</span>`).join('')}}
                </div>
            `;
            card.onclick = () => focusCommunity(comm.id);
            communityList.appendChild(card);
        }});

        // 建立網路（圓形散佈佈局）
        const nodes = new vis.DataSet(nodesData.map(n => ({{
            id: n.id,
            label: n.label,
            title: n.title,
            color: {{
                background: n.color,
                border: n.color,
                highlight: {{ background: '#fff', border: n.color }}
            }},
            size: n.size,
            x: n.x,
            y: n.y,
            group: n.group,
            font: {{ color: '#333', size: 12 }}
        }})));

        const edges = new vis.DataSet(edgesData.map(e => ({{
            from: e.from,
            to: e.to,
            color: {{ color: e.color, opacity: 0.8 }},
            width: e.width,
            smooth: {{ type: 'continuous' }}
        }})));

        const container = document.getElementById('network');
        const data = {{ nodes, edges }};

        const options = {{
            physics: {{
                enabled: true,
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: -50,
                    centralGravity: 0.005,
                    springLength: 100,
                    springConstant: 0.08
                }},
                stabilization: {{
                    iterations: 100
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                zoomView: true,
                dragView: true
            }},
            nodes: {{
                shape: 'dot',
                borderWidth: 2,
                shadow: true
            }},
            edges: {{
                smooth: {{
                    type: 'continuous'
                }}
            }}
        }};

        const network = new vis.Network(container, data, options);

        let physicsEnabled = true;

        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
        }}

        function resetView() {{
            network.fit();
        }}

        function focusCommunity(commId) {{
            const commNodes = nodesData.filter(n => n.group === commId).map(n => n.id);
            if (commNodes.length > 0) {{
                network.selectNodes(commNodes);
                network.fit({{ nodes: commNodes, animation: true }});
            }}

            // 更新卡片樣式
            document.querySelectorAll('.community-card').forEach((card, i) => {{
                card.classList.toggle('active', communitiesData[i].id === commId);
            }});
        }}

        // Tooltip
        const tooltip = document.getElementById('tooltip');

        network.on('hoverNode', function(params) {{
            const node = nodesData.find(n => n.id === params.node);
            if (node) {{
                tooltip.innerHTML = `<strong>${{node.type}}</strong><br>${{node.title}}`;
                tooltip.style.display = 'block';
            }}
        }});

        network.on('blurNode', function() {{
            tooltip.style.display = 'none';
        }});

        container.addEventListener('mousemove', function(e) {{
            tooltip.style.left = (e.pageX + 10) + 'px';
            tooltip.style.top = (e.pageY + 10) + 'px';
        }});

        // 穩定後停止物理效果
        network.once('stabilized', function() {{
            setTimeout(() => {{
                network.setOptions({{ physics: {{ enabled: false }} }});
                physicsEnabled = false;
            }}, 1000);
        }});
    </script>
</body>
</html>'''

    return html


if __name__ == "__main__":
    # 測試
    from graph_builder import CompetencyKnowledgeGraph
    from config import get_config

    config = get_config()

    kg = CompetencyKnowledgeGraph()
    kg.load(str(config.GRAPH_DB_DIR / config.GRAPH_FILE))

    output_path = config.OUTPUTS_DIR / "community_visualization.html"
    generate_community_visualization(
        kg.graph,
        str(output_path),
        resolution=1.0,
        max_nodes=300
    )

    print(f"視覺化已儲存: {output_path}")
