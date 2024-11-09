import math
import random


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

    def add_child(self, child_state):
        child = MCTSNode(child_state, self)
        self.children.append(child)

    def update(self, result):
        self.visits += 1
        self.wins += result

    def get_UCT_value(self):
        if self.visits == 0:
            return float('inf')

        exploitation = self.wins / self.visits
        exploration = 2 * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

class MonteCarloTreeSearch:
    def __init__(self, initial_state):
        self.root = MCTSNode(initial_state)

    def search(self, num_iterations):
        for i in range(num_iterations):
            node = self.select()  # selection
            node.add_child(self.get_random_string(i))  # the expansion
            result = self.simulate(node)  # simulation
            self.backpropagate(node, result)    # backpropagation

    def get_random_string(self, length):
        return  "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=length))

    def select(self):
        current_node = self.root
        while current_node.children:
            best_child = max(current_node.children, key=lambda child: child.get_UCT_value())
            current_node = best_child
        return current_node

    def simulate(self, node):
        # 这里使用随机结果来模拟
        return random.choice([True, False])

    def backpropagate(self, node, result):
        while node:
            node.update(result)
            node = node.parent

    def layer_traversal(self):
        results=[]
        q = [self.root]
        while q:
            node = q.pop(0)
            results.append({"state":node.state,"visits":node.visits,"wins":node.wins})
            q.extend(node.children)
        return results

    def get_depth(self, node):
        if len(node.children) ==0:
            return 1
        return max( [self.get_depth(n) for n in node.children] ) + 1

# 示例用法
if __name__ == "__main__":
    initial_state = "start"
    mcts = MonteCarloTreeSearch(initial_state)

    mcts.search(50)  # 执行1000次搜索迭代

    print("遍历结果：", mcts.layer_traversal())
    # for child in mcts.root.children:
    #     print(f"状态: {child.state}, 访问次数: {child.visits}, 胜利次数: {child.wins}")
