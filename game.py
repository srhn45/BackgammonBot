import numpy as np
import random
import torch
from backgammon import BackgammonBoard, Game
from BackModel import ResidualBlock, BackModel
from tqdm import tqdm

torch.set_grad_enabled(True)

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

        #with torch.no_grad():
        #    visits = torch.tensor([child.visits for child in self.children], dtype=torch.float32, device=device)
        #    wins = torch.tensor([child.wins for child in self.children], dtype=torch.float32, device=device)
        #    priors = torch.tensor([child.prior for child in self.children], dtype=torch.float32, device=device)

        #    total_visits = visits.sum()
        #    q_values = torch.where(visits > 0, wins / visits, torch.zeros_like(wins))
        #    u_values = c_param * priors * torch.sqrt(total_visits) / (1 + visits)  

        #scores = q_values + u_values
        #best_idx = torch.argmax(scores).item()
        
        #return self.children[best_idx]



    def expand(self, model, add_dirichlet_noise=False, eps=0.25, alpha=0.3):
        device = next(model.parameters()).device 
        
        with torch.no_grad():
            move_probs, _ = model(self.state.get_input_matrix().to(device).unsqueeze(0))
            move_probs = torch.exp(move_probs).squeeze().cpu().numpy()

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
            #new_state = copy.deepcopy(self.state)
            new_state = self.state.clone()
            new_state.play_one_move(*move)
            child_node = MCTSNode(new_state, parent=self)
            child_node.last_move = move
            child_node.prior = prior
            self.children.append(child_node)
        

    def update(self, result):
        self.visits += 1
        self.wins += result*self.state.current_player
        
class MCTS_Searcher:
    def __init__(self, model, n_simulations=1000):
        self.model = model
        self.device = next(model.parameters()).device 
        self.n_simulations = n_simulations

    def search(self, initial_state):
        root = MCTSNode(initial_state)

        for _ in range(self.n_simulations):
            node = root
            #state = copy.deepcopy(initial_state)
            state = initial_state.clone()

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
        #current_state = copy.deepcopy(state)
        current_state = state.clone()
        while current_state.check_game_over() == 0:
            legal_moves = current_state.get_legal_moves()
            if not legal_moves:
                break
            
            with torch.no_grad():
                move_probs, value = self.model(current_state.get_input_matrix().to(self.device).unsqueeze(0))   
                move_probs = torch.exp(move_probs).squeeze()

            legal_indices = []
            for (start, end, die) in legal_moves:
                if start == -1:
                    idx = die - 1  # bar move encoding
                else:
                    idx = start * 6 + (die - 1)
                legal_indices.append(idx)
            
            #legal_moves_arr = np.array(legal_moves)  # shape (num_moves, 3)
            #starts = legal_moves_arr[:, 0]
            #dies = legal_moves_arr[:, 2]

            #legal_indices = np.where(starts == -1, dies - 1, starts * 6 + (dies - 1))
            #legal_indices = legal_indices.tolist()

            weights = move_probs[legal_indices]
            if torch.any(torch.isnan(weights)) or weights.sum() <= 0:
                weights = torch.ones(len(legal_moves), device=self.device) / len(legal_moves)
            else:
                weights /= weights.sum()


            chosen_idx = torch.multinomial(weights, num_samples=1).item()
            start, end, die = legal_moves[chosen_idx]
            current_state.play_one_move(start, end, die)
        return current_state.check_game_over()
    
model = BackModel(num_resnets=4, num_skips=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.load_state_dict(torch.load("model.pth", map_location=device))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.999)

for epoch in range(1000):
    buffer = []
    searcher = MCTS_Searcher(model=model, n_simulations=400)

    loader = tqdm(range(0, 500))
    for i in loader:
        positions = []
        backgame = Game()
        while backgame.check_game_over() == 0:
            searcher.model.eval()
            root, next_board = searcher.search(backgame)

            if root == next_board:
                next_board.state.switch_player()
            else:
                board_state = root.state.get_input_matrix()
                probas = np.zeros(24 * 6)
                for child in root.children:
                    if child.last_move is not None:
                        start, end, die = child.last_move
                        index = start * 6 + (die - 1)
                        probas[index] = child.visits
                probas /= probas.sum()
                positions.append((board_state, probas, None))

            backgame = next_board.state

        positions = [(s, p, backgame.check_game_over()) for (s, p, v) in positions]
        buffer.extend(positions)

    random.shuffle(buffer)

    loader = tqdm(range(0, len(buffer), 32))
    model.train()
    model = model.to(device)
    total_loss = 0.0
    for idx, i in enumerate(loader, start=1):
        batch = buffer[i:i + 32]
        states, target_probs, target_values = zip(*batch)

        states = torch.stack([s.clone().to(device) for s in states])
        target_probs = torch.stack([
        torch.from_numpy(p).float().to(device) for p in target_probs
    ])
        target_values = torch.tensor(target_values, dtype=torch.float32, device=device)  # if it's already numeric

        pred_probs, pred_values = model(states)

        value_loss = torch.nn.functional.mse_loss(pred_values.squeeze(), target_values)
        policy_loss = -torch.mean(torch.sum(target_probs * pred_probs, dim=1))
        loss = value_loss + policy_loss


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loader.set_postfix(loss=loss.item())

    avg_loss = total_loss / idx
    scheduler.step()
    print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "model.pth")
