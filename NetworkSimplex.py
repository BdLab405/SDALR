class NetworkSimplex:
    def __init__(self):
        self.n_ = 0
        self.k_ = 0
        self.parent_ = []
        self.parent_edge_index_ = []
        self.parent_direction_ = []
        self.vis_ = []
        self.potential_ = []
        self.potential_tag_ = []
        self.edge_list_ = []
        self.min_cost_ = 0.0
        self.tag_ = 0

    def build_hard(self, costs, k, lower_bound, upper_bound):
        sum_flow = self.build_basic(costs, 1)
        for i in range(self.k_):
            edge = self.edge_list_[self.n_ * self.k_ + i]
            edge['from'] = self.n_ + 1 + i
            edge['to'] = 0
            edge['cap'] = upper_bound - lower_bound
            edge['flow'] = sum_flow[i] - lower_bound
            edge['cost'] = 0.0
            edge['in_tree'] = True
        self.build_tree()

    def regularized_f(self, i, a, b):
        if i == 0:
            return b * b
        else:
            print("error: function i-th not found")
            return 0

    def build(self, costs, f_th):
        sum_flow = self.build_basic(costs, len(costs))
        for i in range(self.k_):
            for j in range(self.n_):
                edge = self.edge_list_[self.n_ * self.k_ + i * self.n_ + j]
                edge['from'] = self.n_ + 1 + i
                edge['to'] = 0
                edge['flow'] = sum_flow[i] >= j + 1
                edge['in_tree'] = j == 0
                edge['cap'] = 1
                edge['cost'] = self.regularized_f(f_th, i, j + 1) - self.regularized_f(f_th, i, j)
        self.build_tree()

    def build_basic(self, costs, extra_edge_num_):
        self.n_ = len(costs)
        self.k_ = len(costs[0])
        sum_flow = [0] * self.k_
        for i in range(self.n_):
            sum_flow[i % self.k_] += 1
        vertex_num = self.n_ + self.k_ + 1
        edge_num = self.n_ * self.k_ + self.k_ * extra_edge_num_
        self.parent_ = [0] * vertex_num
        self.parent_edge_index_ = [0] * vertex_num
        self.parent_direction_ = [0] * vertex_num
        self.vis_ = [0] * vertex_num
        self.potential_ = [0.0] * vertex_num
        self.potential_tag_ = [-1] * vertex_num
        self.edge_list_ = [{} for _ in range(edge_num)]
        self.potential_tag_[0] = self.tag_ = 0
        for i in range(self.n_):
            for j in range(self.k_):
                edge = self.edge_list_[i * self.k_ + j]
                edge['from'] = i + 1
                edge['to'] = self.n_ + 1 + j
                edge['cap'] = 1
                edge['flow'] = 0
                edge['cost'] = costs[i][j]
                edge['in_tree'] = False
        for i in range(self.n_):
            j = i % self.k_
            edge = self.edge_list_[i * self.k_ + j]
            edge['flow'] = 1
            edge['in_tree'] = True
        return sum_flow

    def build_tree(self):
        self.min_cost_ = 0.0
        for i, edge in enumerate(self.edge_list_):
            self.min_cost_ += edge['flow'] * edge['cost']
            if edge['in_tree']:
                self.parent_[edge['from']] = edge['to']
                self.parent_edge_index_[edge['from']] = i
                self.parent_direction_[edge['from']] = 1

    def simplex(self, max_iterations=10000):
        num_edges = len(self.edge_list_)
        scaned = 0
        edge_index = 0
        iterations = 0
        while scaned < num_edges and iterations < max_iterations:
            if edge_index == num_edges:
                edge_index = 0
            edge = self.edge_list_[edge_index]
            if edge['in_tree'] or edge['cap'] == 0:
                edge_index += 1
                scaned += 1
                continue
            potential_from = self.get_potential(edge['from'])
            potential_to = self.get_potential(edge['to'])
            direction = 1 if edge['flow'] == 0 else -1
            delta = (potential_to - potential_from + edge['cost']) * direction
            if delta < -1e-6:
                self.pivot(edge_index, direction, delta)
                scaned = 0

            edge_index += 1
            scaned += 1
            iterations += 1

    def update_costs(self, costs):
        n_ = len(costs)
        for edge in self.edge_list_:
            if 1 <= edge['from'] <= n_:
                if edge['flow'] == 1:
                    self.min_cost_ += costs[edge['from'] - 1][edge['to'] - n_ - 1] - edge['cost']
                edge['cost'] = costs[edge['from'] - 1][edge['to'] - n_ - 1]
        self.tag_ += 1
        self.potential_tag_[0] = self.tag_

    def get_assignments(self):
        assignments = [-1] * self.n_
        for edge in self.edge_list_:
            if edge['flow'] == 1 and 1 <= edge['from'] <= self.n_:
                assignments[edge['from'] - 1] = edge['to'] - self.n_ - 1
        return assignments

    def min_cost(self):
        return self.min_cost_

    def pivot(self, edge_index, direction, delta):
        edge = self.edge_list_[edge_index]
        min_res_cap = edge['cap']
        min_res_cap_edge_index = -1
        min_res_direction = 0
        lca = self.find_lca(edge['from'], edge['to'])
        current_node = edge['from']
        while current_node != lca:
            res_cap = self.get_parent_res_cap(current_node, -direction)
            if res_cap < min_res_cap:
                min_res_cap = res_cap
                min_res_cap_edge_index = current_node
                min_res_direction = 1
            current_node = self.parent_[current_node]
        current_node = edge['to']
        while current_node != lca:
            res_cap = self.get_parent_res_cap(current_node, direction)
            if res_cap < min_res_cap:
                min_res_cap = res_cap
                min_res_cap_edge_index = current_node
                min_res_direction = -1
            current_node = self.parent_[current_node]
        if min_res_cap > 0:
            self.min_cost_ += min_res_cap * delta
            edge['flow'] += direction * min_res_cap
            current_node = edge['from']
            while current_node != lca:
                self.apply_parent_flow(current_node, -direction, min_res_cap)
                current_node = self.parent_[current_node]
            current_node = edge['to']
            while current_node != lca:
                self.apply_parent_flow(current_node, direction, min_res_cap)
                current_node = self.parent_[current_node]
        if min_res_direction != 0:
            self.tag_ += 1
            self.potential_tag_[0] = self.tag_
            self.edge_list_[self.parent_edge_index_[min_res_cap_edge_index]]['in_tree'] = False
            edge['in_tree'] = True
            current_node = edge['from'] if min_res_direction == 1 else edge['to']
            self.change_direction(current_node, min_res_cap_edge_index)
            self.parent_edge_index_[current_node] = edge_index
            self.parent_[current_node] = edge['from'] ^ edge['to'] ^ current_node
            self.parent_direction_[current_node] = min_res_direction

    def get_potential(self, u):
        if self.potential_tag_[u] != self.tag_:
            self.potential_[u] = self.get_potential(self.parent_[u]) + self.parent_direction_[u] * self.edge_list_[self.parent_edge_index_[u]]['cost']
        self.potential_tag_[u] = self.tag_
        return self.potential_[u]

    def get_parent_res_cap(self, u, direction):
        if direction * self.parent_direction_[u] > 0:
            return self.edge_list_[self.parent_edge_index_[u]]['cap'] - self.edge_list_[self.parent_edge_index_[u]]['flow']
        else:
            return self.edge_list_[self.parent_edge_index_[u]]['flow']

    def apply_parent_flow(self, u, direction, flow):
        self.edge_list_[self.parent_edge_index_[u]]['flow'] += direction * self.parent_direction_[u] * flow

    def change_direction(self, u, end):
        if u != end:
            self.change_direction(self.parent_[u], end)
            self.parent_[self.parent_[u]] = u
            self.parent_edge_index_[self.parent_[u]] = self.parent_edge_index_[u]
            self.parent_direction_[self.parent_[u]] = -self.parent_direction_[u]

    def find_lca(self, u, v):
        t = u
        while t:
            self.vis_[t] = True
            t = self.parent_[t]
        while v and not self.vis_[v]:
            v = self.parent_[v]
        while u:
            self.vis_[u] = False
            u = self.parent_[u]
        return v
