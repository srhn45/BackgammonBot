import numpy as np
import random
import torch
from backgammon import BackgammonBoard, Game
from BackModel import ResidualBlock, BackModel
import copy

class MCTSNode:
    def __init__(self, state=Game, parent=None):
        self.last_move = None
        self.state = state
        self.legal_moves = state.get_legal_moves()
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.prior = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_moves())

    def best_child(self, c_param=1.4):
        if not self.children:
            return None

        total_visits = sum(child.visits for child in self.children)

        scores = []
        for child in self.children:
            if child.visits > 0:
                q_value = child.wins / child.visits
            else:
                q_value = 0
            u_value = c_param * child.prior * np.sqrt(total_visits) / (1 + child.visits)
            scores.append(q_value + u_value)

        return self.children[np.argmax(scores)]

    def expand(self, model, add_dirichlet_noise=False, eps=0.25, alpha=0.3):
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        move_probs, _ = model(self.state.get_input_matrix().to("cuda" if torch.cuda.is_available() else "cpu").unsqueeze(0))
        move_probs = torch.exp(move_probs).squeeze().detach().cpu().numpy()

        legal_indices = []
        for (start, end, die) in self.legal_moves:
            idx = die - 1 if start == -1 else start * 6 + (die - 1)
            legal_indices.append(idx)

        probs = move_probs[legal_indices]
        probs /= probs.sum()
    
        if add_dirichlet_noise and self.parent is None:
            noise = np.random.dirichlet([alpha] * len(probs))
            probs = (1 - eps) * probs + eps * noise

        for move, prior in zip(self.legal_moves, probs):
            new_state = copy.deepcopy(self.state)
            new_state.play_one_move(*move)
            child_node = MCTSNode(new_state, parent=self)
            child_node.last_move = move
            child_node.prior = prior
            self.children.append(child_node)
        
        model.to("cpu")

    def update(self, result):
        self.visits += 1
        self.wins += result*self.state.current_player

class MCTS_Searcher:
    def __init__(self, model, n_simulations=1000):
        self.model = model
        self.n_simulations = n_simulations

    def search(self, initial_state):
        root = MCTSNode(initial_state)

        for _ in range(self.n_simulations):
            node = root
            state = copy.deepcopy(initial_state)

            while node.is_fully_expanded() and node.children:
                node = node.best_child()
                state = node.state

            if not node.is_fully_expanded():
                node.expand(self.model, add_dirichlet_noise=(node.parent is None))
                node = random.choice(node.children)
                state = node.state

            result = self.simulate(state)

            while node is not None:
                node.update(result)
                node = node.parent

        best_child = root.best_child(c_param=0)
        if best_child is None:
            return root, root
        return root, best_child

    def simulate(self, state):
        current_state = copy.deepcopy(state)
        while current_state.check_game_over() == 0:
            legal_moves = current_state.get_legal_moves()
            if not legal_moves:
                break
            move_probs, value = self.model(current_state.get_input_matrix().unsqueeze(0))
            move_probs = torch.exp(move_probs).squeeze().detach().numpy()

            legal_indices = []
            for (start, end, die) in legal_moves:
                if start == -1:
                    idx = die - 1  # bar move encoding
                else:
                    idx = start * 6 + (die - 1)
                legal_indices.append(idx)

            weights = move_probs[legal_indices]
            weights /= weights.sum()

            chosen_idx = np.random.choice(len(legal_moves), p=weights)  # index in legal_moves
            start, end, die = legal_moves[chosen_idx]

            
            start, end, die = legal_moves[chosen_idx]
            current_state.play_one_move(start, end, die)
        return current_state.check_game_over()