import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import random
from collections import defaultdict
import threading
import time

# --- MDP definitions --------------------------------------------------------

WINNING_POSITIONS = [
    (0,1,2), (3,4,5), (6,7,8),
    (0,3,6), (1,4,7), (2,5,8),
    (0,4,8), (2,4,6)
]

def init_q_table():
    """Q-table: state → Array(9) with action values."""
    return defaultdict(lambda: np.zeros(9))

def available_actions(state):
    """Returns available positions (Index 0–8)."""
    return [i for i, c in enumerate(state) if c == '_']

def check_winner(state):
    """Checks for win for X/O, Draw or None."""
    for (i,j,k) in WINNING_POSITIONS:
        line = state[i] + state[j] + state[k]
        if line == 'XXX': return 'X'
        if line == 'OOO': return 'O'
    if '_' not in state:
        return 'Draw'
    return None

def make_move(state, action, player):
    """Sets 'X' or 'O' in the string state."""
    lst = list(state)
    lst[action] = player
    return ''.join(lst)

def get_reward(winner, player='O'):
    """Reward for agent 'O': +1 win, 0 draw/ongoing, -1 loss."""
    if winner == player:    return +1
    if winner == 'Draw' or winner is None: return 0
    return -1

# --- New functions for enhanced training ----------------------------

def detect_blocking_opportunity(state, player='O'):
    """
    Detects if the opponent is about to win and a block is needed.
    Returns (True, block_position) if a block is needed, otherwise (False, None).
    """
    opponent = 'X' if player == 'O' else 'O'

    # Check all winning lines
    for (i, j, k) in WINNING_POSITIONS:
        positions = [i, j, k]
        line = [state[pos] for pos in positions]

        # If the opponent occupies two fields and the third is free
        if line.count(opponent) == 2 and line.count('_') == 1:
            # Find the position to block
            block_pos = positions[line.index('_')]
            return True, block_pos
            
    return False, None

def detect_winning_opportunity(state, player='O'):
    """
    Detects if the player is about to win.
    Returns (True, win_position) if a winning move is possible, otherwise (False, None).
    """
    # Check all winning lines
    for (i, j, k) in WINNING_POSITIONS:
        positions = [i, j, k]
        line = [state[pos] for pos in positions]

        # If the player occupies two fields and the third is free
        if line.count(player) == 2 and line.count('_') == 1:
            # Find the position to win
            win_pos = positions[line.index('_')]
            return True, win_pos
            
    return False, None

def get_enhanced_reward(state, action, next_state, winner, player='O'):
    """
    Enhanced reward function that rewards blocking and strategic moves.
    """
    # Base reward for win/loss/draw
    base_reward = get_reward(winner, player)

    # If the game is already decided, return only base reward
    if winner is not None:
        return base_reward
    
    additional_reward = 0
    opponent = 'X' if player == 'O' else 'O'

    # Reward for blocking an opponent's winning move
    needs_block, block_pos = detect_blocking_opportunity(state, player)
    if needs_block and action == block_pos:
        additional_reward += 0.7  # High reward for blocking

    # Reward for taking advantage of one's own winning opportunity
    can_win, win_pos = detect_winning_opportunity(state, player)
    if can_win and action == win_pos:
        additional_reward += 0.8 # Very high reward for winning moves

    # Reward for moves in the center (strategically valuable)
    if action == 4 and state[4] == '_':
        additional_reward += 0.3

    # Reward for moves in the corners (strategically valuable)
    if action in [0, 2, 6, 8] and state[action] == '_':
        additional_reward += 0.2
        
    return base_reward + additional_reward

# --- Improved opponent strategies -----------------------------------------

def opponent_smart(state):
    """
    Intelligent opponent for better training:
    1. Wins if possible
    2. Blocks if necessary
    3. Otherwise plays randomly
    """
    # Check if opponent can win
    can_win, win_pos = detect_winning_opportunity(state, 'X')
    if can_win:
        return win_pos

    # Check if agent needs to be blocked
    needs_block, block_pos = detect_blocking_opportunity(state, 'X')
    if needs_block:
        return block_pos

    # Prefer the center if free
    if state[4] == '_':
        return 4

    # Prefer corners if free
    corners = [0, 2, 6, 8]
    free_corners = [c for c in corners if state[c] == '_']
    if free_corners:
        return random.choice(free_corners)

    # Otherwise random move
    return random.choice(available_actions(state))

def opponent_mixed(state):
    """
    Mixes between random and intelligent play.
    Helps the agent learn against different play styles.
    """
    if random.random() < 0.7:  # 70% intelligent, 30% random
        return opponent_smart(state)
    else:
        return opponent_random(state)

# --- Q-Learning -------------------------------------------------------------

def choose_action(state, q_table, epsilon):
    """ε-greedy policy on Q-table."""
    legal = available_actions(state)
    
    # With probability ε choose a random move
    if random.random() < epsilon:
        return random.choice(legal)
    
    # Otherwise choose the best move
    qs = np.copy(q_table[state])
    # Mask illegal actions
    for i in range(9):
        if i not in legal:
            qs[i] = -np.inf
    return int(np.argmax(qs))

def update_q(q_table, state, action, reward, next_state, alpha, gamma):
    """Q(s,a) ← Q + α [r + γ·max Q(s',·) − Q]."""
    max_next = np.max(q_table[next_state])
    q_old = q_table[state][action]
    q_table[state][action] += alpha * (reward + gamma*max_next - q_old)

# --- Opponent strategies ----------------------------------------------------

def opponent_random(state):
    """Random opponent for evaluation/training."""
    return random.choice(available_actions(state))

# --- Simulate a single episode ---------------------------------------------

def play_episode(q_table, opponent_fn, epsilon=0.0):
    """
    Plays exactly one game:
    Agent 'O' vs. opponent_fn (X).
    ε=0 → purely greedy.
    Returns: winner 'O', 'X' or 'Draw'
    """
    state = '_________'
    player = 'O'  # Agent starts
    while True:
        if player == 'O':
            a = choose_action(state, q_table, epsilon)
            state = make_move(state, a, 'O')
        else:
            a = opponent_fn(state)
            state = make_move(state, a, 'X')

        winner = check_winner(state)
        if winner is not None:
            return winner

        player = 'X' if player == 'O' else 'O'

# --- Improved training with regular evaluation -----------------------------

def train_and_evaluate(alpha, gamma, decay,
                       episodes=5000,
                       eval_interval=500,
                       eval_games=100,
                       epsilon_start=1.0,
                       epsilon_end=0.1,
                       progress_callback=None,
                       use_enhanced_reward=True,
                       use_smart_opponent=True):
    """
    Trains agent 'O' with improved methods.
    Returns:
      q_table, list_of_episodes, list_of_win_rates
    """
    q_table = init_q_table()
    epsilon = epsilon_start

    eval_points = []
    win_rates   = []
    
    # Choose opponent function based on parameter
    opponent_fn = opponent_mixed if use_smart_opponent else opponent_random

    for ep in range(1, episodes+1):
        # --- one training episode ---
        state = '_________'
        done = False
        while not done:
            # Agent move
            action = choose_action(state, q_table, epsilon)
            next_state = make_move(state, action, 'O')
            winner = check_winner(next_state)

            if winner is not None:
                # Use enhanced reward function if enabled
                if use_enhanced_reward:
                    reward = get_enhanced_reward(state, action, next_state, winner, 'O')
                else:
                    reward = get_reward(winner, 'O')
                    
                update_q(q_table, state, action, reward, next_state, alpha, gamma)
                done = True
            else:
                # Opponent move
                opp_a = opponent_fn(next_state)
                next_state2 = make_move(next_state, opp_a, 'X')
                winner2 = check_winner(next_state2)

                # Use enhanced reward function if enabled
                if use_enhanced_reward:
                    reward = get_enhanced_reward(state, action, next_state2, winner2, 'O')
                else:
                    reward = get_reward(winner2, 'O')
                    
                update_q(q_table, state, action, reward, next_state2, alpha, gamma)

                state = next_state2
                if winner2 is not None:
                    done = True

        # ε-decay
        epsilon = max(epsilon_end, epsilon - decay)

        # --- Evaluation ---
        if ep % eval_interval == 0:
            wins = 0
            for _ in range(eval_games):
                if play_episode(q_table, opponent_smart, epsilon=0.0) == 'O':
                    wins += 1
            eval_points.append(ep)
            win_rates.append(wins / eval_games)
            
            # Callback for GUI updates
            if progress_callback:
                progress_callback(ep, episodes, epsilon, wins / eval_games, eval_points, win_rates)

    return q_table, eval_points, win_rates

# --- GUI class -------------------------------------------------------------

class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe with Q-Learning")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Q-table and training variables
        self.q_table = init_q_table()
        self.training_in_progress = False
        self.training_thread = None
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side: game board
        self.game_frame = ttk.LabelFrame(self.main_frame, text="Game Board", padding="10")
        self.game_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Board buttons
        self.board_buttons = []
        self.board_frame = ttk.Frame(self.game_frame)
        self.board_frame.pack(fill=tk.BOTH, expand=True)
        
        for i in range(3):
            self.board_frame.columnconfigure(i, weight=1)
            self.board_frame.rowconfigure(i, weight=1)
            
        for row in range(3):
            for col in range(3):
                idx = row * 3 + col
                btn = tk.Button(self.board_frame, text="", font=("Arial", 24, "bold"),
                               width=3, height=1, command=lambda idx=idx: self.make_player_move(idx))
                btn.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)
                self.board_buttons.append(btn)
        
        # Game controls
        self.control_frame = ttk.Frame(self.game_frame)
        self.control_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = ttk.Label(self.control_frame, text="Start game or train")
        self.status_label.pack(side=tk.LEFT, pady=5)
        
        self.new_game_btn = ttk.Button(self.control_frame, text="New Game", command=self.new_game)
        self.new_game_btn.pack(side=tk.RIGHT, pady=5)
        
        # Right side: training and statistics
        self.training_frame = ttk.LabelFrame(self.main_frame, text="Training & Statistics", padding="10")
        self.training_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Training parameters
        self.param_frame = ttk.Frame(self.training_frame)
        self.param_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Parameter labels and entry fields
        param_labels = ["Alpha:", "Gamma:", "Decay:", "Episodes:", "Eval-Interval:", "Eval-Games:"]
        self.param_entries = {}
        default_values = {"Alpha:": "0.2", "Gamma:": "0.95", "Decay:": "0.0001", 
                         "Episodes:": "10000", "Eval-Interval:": "500", "Eval-Games:": "50"}
        
        for i, label_text in enumerate(param_labels):
            row = i // 2
            col = i % 2 * 2
            
            label = ttk.Label(self.param_frame, text=label_text)
            label.grid(row=row, column=col, sticky="e", padx=(5, 2), pady=2)
            
            entry = ttk.Entry(self.param_frame, width=10)
            entry.insert(0, default_values[label_text])
            entry.grid(row=row, column=col+1, sticky="w", padx=(2, 5), pady=2)
            
            self.param_entries[label_text] = entry

        # Advanced training options
        self.options_frame = ttk.Frame(self.training_frame)
        self.options_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.enhanced_reward_var = tk.BooleanVar(value=True)
        self.enhanced_reward_check = ttk.Checkbutton(
            self.options_frame,
            text="Enhanced Reward Function",
            variable=self.enhanced_reward_var
        )
        self.enhanced_reward_check.pack(anchor="w")
        
        self.smart_opponent_var = tk.BooleanVar(value=True)
        self.smart_opponent_check = ttk.Checkbutton(
            self.options_frame, 
            text="Intelligent training opponent", 
            variable=self.smart_opponent_var
        )
        self.smart_opponent_check.pack(anchor="w")
        
        # Training button
        self.train_button = ttk.Button(self.training_frame, text="Train agent", command=self.start_training)
        self.train_button.pack(fill=tk.X, pady=(0, 10))
        
        # Progress display
        self.progress_frame = ttk.Frame(self.training_frame)
        self.progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_label = ttk.Label(self.progress_frame, text="Progress: 0%")
        self.progress_label.pack(side=tk.TOP, anchor="w")
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", length=100, mode="determinate")
        self.progress_bar.pack(fill=tk.X)
        
        self.win_rate_label = ttk.Label(self.progress_frame, text="Win rate: -")
        self.win_rate_label.pack(side=tk.TOP, anchor="w", pady=(5, 0))
        
        # Learning curve plot
        self.fig, self.ax = plt.subplots(figsize=(4, 3), dpi=80)
        self.ax.set_xlabel("Training episodes")
        self.ax.set_ylabel("Win rate vs. random")
        self.ax.set_title("Learning curve (Q-Learning)")
        self.ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.training_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize game state
        self.state = '_________'
        self.game_active = False
        
        # Start new game
        self.new_game()
    
    def new_game(self):
        """Starts a new game."""
        self.state = '_________'
        self.game_active = True
        
        # Reset board
        for btn in self.board_buttons:
            btn.config(text="", state=tk.NORMAL, bg="SystemButtonFace")
        
        self.status_label.config(text="Your move (X)")
    
    def make_player_move(self, index):
        """Processes the player's move."""
        if not self.game_active or self.state[index] != '_':
            return
        
        # Player move (X)
        self.state = make_move(self.state, index, 'X')
        self.board_buttons[index].config(text="X", state=tk.DISABLED)
        
        # Check if game is over
        winner = check_winner(self.state)
        if winner:
            self.end_game(winner)
            return
        
        # Agent move (O)
        self.status_label.config(text="Agent (O) is thinking...")
        self.root.update()
        time.sleep(0.5)  # Short delay for better UX
        
        # Check if agent needs to block
        needs_block, block_pos = detect_blocking_opportunity(self.state, 'O')
        
        # Check if agent can win
        can_win, win_pos = detect_winning_opportunity(self.state, 'O')
        
        # Show hint if agent blocks or wins
        if can_win:
            agent_move = win_pos
            self.status_label.config(text="Agent makes winning move!")
        elif needs_block:
            agent_move = block_pos
            self.status_label.config(text="Agent blocks!")
        else:
            agent_move = choose_action(self.state, self.q_table, epsilon=0.0)
        
        self.root.update()
        time.sleep(0.5)
        
        self.state = make_move(self.state, agent_move, 'O')
        self.board_buttons[agent_move].config(text="O", state=tk.DISABLED, bg="#f0f0ff")
        
        # Check if game is over
        winner = check_winner(self.state)
        if winner:
            self.end_game(winner)
        else:
            self.status_label.config(text="Your move (X)")
    
    def end_game(self, winner):
        """Ends the current game."""
        self.game_active = False
        
        if winner == 'Draw':
            self.status_label.config(text="Draw!")
        else:
            if winner == 'X':
                self.status_label.config(text="You won!")
            else:
                self.status_label.config(text="The agent won!")
            
            # Highlight winning line
            for pos in WINNING_POSITIONS:
                line = self.state[pos[0]] + self.state[pos[1]] + self.state[pos[2]]
                if line == winner * 3:
                    for idx in pos:
                        self.board_buttons[idx].config(bg="#aaffaa" if winner == 'X' else "#aaaaff")
    
    def start_training(self):
        """Starts training in a separate thread."""
        if self.training_in_progress:
            messagebox.showinfo("Training running", "Training is already running. Please wait until it is finished.")
            return
        
        # Read parameters
        try:
            alpha = float(self.param_entries["Alpha:"].get())
            gamma = float(self.param_entries["Gamma:"].get())
            decay = float(self.param_entries["Decay:"].get())
            episodes= int(self.param_entries["Episodes:"].get())
            eval_interval= int(self.param_entries["Eval-Interval:"].get())
            eval_games = int(self.param_entries["Eval-Games:"].get())
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid numbers for all parameters.")
            return
        
        # Read advanced options
        use_enhanced_reward = self.enhanced_reward_var.get()
        use_smart_opponent = self.smart_opponent_var.get()
        
        # Start training thread
        self.training_in_progress = True
        self.train_button.config(state=tk.DISABLED)
        self.progress_bar["maximum"] = episodes
        self.progress_bar["value"] = 0
        
        self.training_thread = threading.Thread(
            target=self.run_training,
            args=(alpha, gamma, decay, episodes, eval_interval, eval_games, 
                  use_enhanced_reward, use_smart_opponent)
        )
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def run_training(self, alpha, gamma, decay, episodes, eval_interval, eval_games,
                    use_enhanced_reward, use_smart_opponent):
        """Runs training in the background."""
        try:
            self.q_table, eval_points, win_rates = train_and_evaluate(
                alpha=alpha,
                gamma=gamma,
                decay=decay,
                episodes=episodes,
                eval_interval=eval_interval,
                eval_games=eval_games,
                progress_callback=self.update_training_progress,
                use_enhanced_reward=use_enhanced_reward,
                use_smart_opponent=use_smart_opponent
            )
            
            # Training completed
            self.root.after(0, self.training_completed, eval_points, win_rates)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Training error", f"Error during training: {str(e)}"))
            self.root.after(0, self.reset_training_ui)
    
    def update_training_progress(self, current_episode, total_episodes, epsilon, win_rate, eval_points, win_rates):
        """Updates the UI with training progress."""
        progress = (current_episode / total_episodes) * 100
        
        def update_ui():
            self.progress_bar["value"] = current_episode
            self.progress_label.config(text=f"Progress: {progress:.1f}% (Episode {current_episode}/{total_episodes})")
            self.win_rate_label.config(text=f"Win rate: {win_rate:.2f}, Epsilon: {epsilon:.3f}")
            
            # Update plot
            self.ax.clear()
            self.ax.plot(eval_points, win_rates, marker='o')
            self.ax.set_xlabel("Training episodes")
            self.ax.set_ylabel("Win rate vs. intelligent opponent")
            self.ax.set_title("Learning curve (Q-Learning)")
            self.ax.grid(True)
            self.canvas.draw()
        
        self.root.after(0, update_ui)
    
    def training_completed(self, eval_points, win_rates):
        """Called when training is completed."""
        messagebox.showinfo("Training completed", 
                           f"Training finished successfully!\nFinal win rate: {win_rates[-1]:.2f}")
        self.reset_training_ui()
    
    def reset_training_ui(self):
        """Resets the training UI."""
        self.training_in_progress = False
        self.train_button.config(state=tk.NORMAL)

# --- Main program -----------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()
