from graphviz import Digraph
import pandas as pd

def create_computational_graph(node):
    graph = CompGraph()
    graph.build_graph(node)
    return graph

def create_computational_table(node):
    table = CompTable()
    df = table.build_table(node)
    return df

class CompGraph(Digraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.added_nodes = set()
        self.added_edges = set()
        self.graph_attr.update(rankdir="BT")

    @staticmethod
    def id_str(node):
        return str(node.id)

    @staticmethod
    def get_label(node):
        if node.type == "Variable":
            return str(node.name)
        if node.type == "Constant":
            return str(node.value())
        return str(node.type)

    @staticmethod
    def get_color(node):
        if node.type == "Variable":
            return "red"
        elif node.type == "Constant":
            return "darkgreen"
        else:
            return "blue"

    @staticmethod
    def get_shape(node):
        if node.type == "Variable":
            return "box"
        elif node.type == "Constant":
            return "triangle"
        else:
            return "oval"

    def add_node(self, node):
        super().node(CompGraph.id_str(node),
                     label=CompGraph.get_label(node),
                     color=CompGraph.get_color(node),
                     shape=CompGraph.get_shape(node))
        self.added_nodes.add(CompGraph.id_str(node))

    def add_edge(self, child, parent):
        self.edge(CompGraph.id_str(child),
                  CompGraph.id_str(parent),
                  **{"style": "filled"})
        self.added_edges.add((CompGraph.id_str(child),
                  CompGraph.id_str(parent)))

    def build_graph(self, top_node):
        if CompGraph.id_str(top_node) not in self.added_nodes:
            self.add_node(top_node)
            # a
            for child in top_node.children:
                self.add_edge(child, top_node)

            # Make each of the children do the same, but skip duplicates
            for child in set(top_node.children):
                self.build_graph(child)

class CompTable():
    def __init__(self):
        self.rows = []
        self.added_nodes = set()
        self.nodes = []
        self.variables = []

    def add_nodes(self, top_node):
        if top_node.id not in self.added_nodes:
            self.added_nodes.add(top_node.id)
            self.nodes.append(top_node)
            if top_node.type == "Variable":
                self.variables.append(top_node.name)
            for child in set(top_node.children):
                self.add_nodes(child)
        self.nodes.sort(key = lambda x: x.id)

    def build_rows(self):
        for idx, node in enumerate(self.nodes):
            row = {}
            # Set Trace
            row["Trace"] = f"x_{idx + 1}"

            # Set Elementary Function
            if node.type == "Variable":
                row["Elementary Function"] = str(node.name)
            elif node.type == "Constant":
                row["Elementary Function"] = str(node.value())
            else:
                children = []
                for child in node.children:
                    child_node = list(filter(lambda node: node.id == child.id,self.nodes))[0]
                    try:
                        index_node = self.nodes.index(child_node)
                        c = f"x_{index_node + 1}"
                    except:
                        c = child_node.value()
                    children.append(c)
                children = ",".join(children)
                row["Elementary Function"] = f"{node.type}({children})"
            self.rows.append(row)

            # Set Current Value
            row["Current Value"]= node.value()

            # Set Gradients
            if node.type == "Variable":
                for variable in self.variables:
                    if variable == node.name:
                        row[f"Grad {variable} value"] = 1
                    else:
                        row[f"Grad {variable} value"] = 0
            elif node.type == "Constant":
                for variable in self.variables:
                    row[f"Grad {variable} value"] = 0
            else:
                for variable in self.variables:
                    try:
                        row[f"Grad {variable} value"] = round(node.derivative()[variable],2)
                    except:
                        row[f"Grad {variable} value"] = 0


    def build_table(self,top_node):
        self.add_nodes(top_node)
        self.build_rows()

        df = pd.DataFrame(self.rows)

        # Order Columns
        columns = ["Trace","Elementary Function", "Current Value"]
        for variable in self.variables:
            columns.append(f"Grad {variable} value")

        df = df[columns]
        return df