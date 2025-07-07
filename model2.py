import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyValueNetwork(nn.Module):
    def __init__(self):
        super(PolicyValueNetwork, self).__init__()
        # Input: 9 (original board) + 9 (player-perspective board) + 1 (current_player)
        self.fc1 = nn.Linear(19, 64)
        self.fc2 = nn.Linear(64, 32)

        # Policy head
        self.policy_head = nn.Linear(32, 9)  # 9 output logits

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()  # Output between -1 and 1
        )

    def forward(self, board, player):
        # board: (batch, 9), player: (batch, 1)
        player = player.view(-1, 1)
        board_perspective = board * player  # Transform to player perspective
        features = torch.cat([board, board_perspective, player], dim=1)  # (batch, 19)

        # Shared layers
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))

        # Policy head
        policy_logits = self.policy_head(x)
        # Mask invalid moves (non-empty cells)
        mask = (board == 0).float()
        masked_logits = policy_logits + (mask + 1e-8).log()
        policy = F.softmax(masked_logits, dim=1)

        # Value head
        value = self.value_head(x)

        return policy, value


def prepare_data(data):
    boards, players, policies, values = [], [], [], []

    for sample in data:
        # Flatten 3x3 board to 9-element vector
        board_flat = torch.tensor(
            [cell for row in sample["board"] for cell in row],
            dtype=torch.float32
        )
        player = torch.tensor([sample["current_player"]], dtype=torch.float32)
        policy = torch.tensor(sample["policy"], dtype=torch.float32)

        # Get value if provided, otherwise default to 0
        value = torch.tensor([sample.get("value", 0.0)], dtype=torch.float32)

        boards.append(board_flat)
        players.append(player)
        policies.append(policy)
        values.append(value)

    return (
        torch.stack(boards),
        torch.stack(players),
        torch.stack(policies),
        torch.stack(values)
    )
