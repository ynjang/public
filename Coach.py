import MCTS
import numpy as np
import torch
from model import ResNet, ResidualBlock, load_model, save_model

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = np.zeros((1, 32))
    model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)
    turn = 1
    n_run = 100

    # 다음 두줄로 MCTS 처리
    mcts = MCTS.MCTS(model, state, turn, n_run)
    action_prob = mcts.getActionProb(temp=1)

    print(action_prob)
    print(np.sum(action_prob))





