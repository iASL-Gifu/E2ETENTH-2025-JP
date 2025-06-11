import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy

class Actor(nn.Module):
    def __init__(self, lidar_dim, action_dim=2, hidden_dim=256):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        
        # LiDARデータを直接処理する全結合層
        self.fc1 = nn.Linear(lidar_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # 平均と対数標準偏差の出力ヘッド
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # ✅ パラメータとして登録（register_bufferを使用）
        self.register_buffer('action_scale', torch.FloatTensor([1.0, 1.0]))
        self.register_buffer('action_bias', torch.FloatTensor([0.0, 0.0]))
        
    def forward(self, lidar_data):
        # LiDARデータを直接処理
        x = F.relu(self.fc1(lidar_data))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # 安定性のためクリッピング
        
        return mean, log_std
    
    def sample(self, lidar_data):
        """アクションをサンプリング"""
        mean, log_std = self.forward(lidar_data)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        
        # 再パラメータ化トリック
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # ログ確率を計算
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # 決定論的な平均アクション
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean

class Critic(nn.Module):
    """SACクリティックネットワーク（Q値ネットワーク）"""
    def __init__(self, lidar_dim, action_dim=2, hidden_dim=256):
        super(Critic, self).__init__()
        
        # LiDARデータを直接処理する全結合層
        self.lidar_fc = nn.Linear(lidar_dim, hidden_dim)
        
        # Q1ネットワーク
        self.q1_fc1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, 1)
        
        # Q2ネットワーク（ツインQ学習）
        self.q2_fc1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, lidar_data, action):
        # LiDARデータを直接処理
        lidar_features = F.relu(self.lidar_fc(lidar_data))
        
        # 特徴とアクションを結合
        xu = torch.cat([lidar_features, action], 1)
        
        # Q1の前向き計算
        x1 = F.relu(self.q1_fc1(xu))
        x1 = F.relu(self.q1_fc2(x1))
        q1 = self.q1_fc3(x1)
        
        # Q2の前向き計算
        x2 = F.relu(self.q2_fc1(xu))
        x2 = F.relu(self.q2_fc2(x2))
        q2 = self.q2_fc3(x2)
        
        return q1, q2

class ReplayBuffer:
    """経験再生バッファ"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """経験をバッファに追加"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """バッチサイズ分の経験をランダムサンプリング"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class SACAgent:
    """LiDARベース自動運転用Soft Actor-Criticエージェント"""
    
    def __init__(self, lidar_dim, action_dim=2, actor_lr=3e-4, critic_lr=3e-4, gamma=0.99, tau=0.005, 
                 alpha=0.2, automatic_entropy_tuning=True, device='cuda',capacity=500000):
        self.device = device
        self.gamma = gamma  # 割引率
        self.tau = tau      # ソフト更新率
        self.action_dim = action_dim
        
        # 🔧 automatic_entropy_tuningフラグを最初に設定
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # ネットワーク初期化
        self.actor = Actor(lidar_dim, action_dim).to(device)
        self.critic = Critic(lidar_dim, action_dim).to(device)
        self.critic_target = Critic(lidar_dim, action_dim).to(device)
        
        # ターゲットネットワークにパラメータをコピー
        self.hard_update(self.critic_target, self.critic)
        
        # オプティマイザー（異なる学習率を設定）
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 🔧 エントロピー調整の設定を修正
        if self.automatic_entropy_tuning:
            # 自動エントロピー調整
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=actor_lr)
            # alphaは学習可能パラメータとして管理
            self.alpha = self.log_alpha.exp().detach()
        else:
            # 固定アルファ値
            self.alpha = alpha
            self.log_alpha = None
            self.alpha_optimizer = None
            self.target_entropy = None
        
        # リプレイバッファ
        self.replay_buffer = ReplayBuffer(capacity=capacity)
        
    def select_action(self, lidar_data, evaluate=False):
        """方策からアクションを選択"""
        lidar_data = torch.FloatTensor(lidar_data).unsqueeze(0).to(self.device)
        
        if evaluate:
            # 評価時は決定論的なアクション
            _, _, action = self.actor.sample(lidar_data)
        else:
            # 学習時は確率的なアクション
            action, _, _ = self.actor.sample(lidar_data)
            
        return action.detach().cpu().numpy()[0]
    
    def update_parameters(self, batch_size=256):
        """ネットワークパラメータを更新"""
        if len(self.replay_buffer) < batch_size:
            return {}  # 空の辞書を返す
            
        # バッチサンプリング
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.replay_buffer.sample(batch_size)
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)
        
        # 🔧 alphaを現在の値で取得（自動調整の場合は更新される）
        if self.automatic_entropy_tuning:
            current_alpha = self.log_alpha.exp()
        else:
            current_alpha = self.alpha
        
        # ターゲットQ値計算（勾配計算無し）
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - current_alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * min_qf_next_target
        
        # クリティック更新
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)  # Q1の損失
        qf2_loss = F.mse_loss(qf2, next_q_value)  # Q2の損失
        qf_loss = qf1_loss + qf2_loss
        
        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()
        
        # アクター更新
        pi, log_pi, _ = self.actor.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
        # 方策損失（最大エントロピー強化学習）
        policy_loss = ((current_alpha * log_pi) - min_qf_pi).mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        # 損失辞書を準備
        loss_dict = {
            'qf1_loss': qf1_loss.item(),
            'qf2_loss': qf2_loss.item(),
            'policy_loss': policy_loss.item(),
        }
        
        # 🔧 α（エントロピー重み）更新 - 自動調整の場合のみ
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # 🔧 alphaの値を更新（学習された値を反映）
            self.alpha = self.log_alpha.exp().detach()
            
            loss_dict['alpha_loss'] = alpha_loss.item()
            loss_dict['alpha'] = self.alpha.item()
        else:
            # 固定アルファの場合も記録
            loss_dict['alpha'] = self.alpha
        
        # ターゲットネットワークのソフト更新
        self.soft_update(self.critic_target, self.critic, self.tau)
        
        return loss_dict
    
    def soft_update(self, target, source, tau):
        """ターゲットネットワークのソフト更新"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def hard_update(self, target, source):
        """ターゲットネットワークのハード更新"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def save_model(self, filepath):
        """モデルのチェックポイントを保存"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'automatic_entropy_tuning': self.automatic_entropy_tuning,
            'alpha': self.alpha,
        }
        
        # 🔧 自動エントロピー調整の場合は追加パラメータを保存
        if self.automatic_entropy_tuning:
            checkpoint['log_alpha'] = self.log_alpha
            checkpoint['alpha_optimizer'] = self.alpha_optimizer.state_dict()
            checkpoint['target_entropy'] = self.target_entropy
            
        torch.save(checkpoint, filepath)
    
    def load_model(self, filepath):
        """モデルのチェックポイントを読み込み"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        # 🔧 アルファ関連パラメータの読み込み
        self.automatic_entropy_tuning = checkpoint.get('automatic_entropy_tuning', True)
        self.alpha = checkpoint.get('alpha', 0.2)
        
        if self.automatic_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            self.target_entropy = checkpoint.get('target_entropy', -self.action_dim)


# 使用例（修正版）
def main():
    # パラメータ設定
    lidar_dim = 100  # LiDARデータの次元数（1080から100に削減済み）
    action_dim = 2   # ステアリングとスピード
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # エージェント初期化（異なる学習率を指定）
    agent = SACAgent(
        lidar_dim=lidar_dim, 
        action_dim=action_dim, 
        actor_lr=1e-4,    # Actorの学習率
        critic_lr=3e-4,   # Criticの学習率（通常はActorより高め）
        alpha=0.2,        # 固定alpha値（automatic_entropy_tuning=Falseの場合）
        automatic_entropy_tuning=True,  # 自動エントロピー調整を有効/無効
        device=device
    )
    
    # 学習ループの例
    num_episodes = 1000  # エピソード数
    max_steps = 1000     # 1エピソードあたりの最大ステップ数
    
    for episode in range(num_episodes):
        # 環境初期化（ここでは仮想的）
        lidar_data = np.random.random(lidar_dim)  # 実際のLiDARデータに置き換え
        total_reward = 0
        
        for step in range(max_steps):
            # アクション選択
            action = agent.select_action(lidar_data)
            steer, speed = action[0], action[1]
            
            # 環境との相互作用（実装必要）
            # next_lidar_data, reward, done, info = env.step([steer, speed])
            
            # 仮想的な次状態と報酬
            next_lidar_data = np.random.random(lidar_dim)
            reward = np.random.random() - 0.5
            done = np.random.random() > 0.99
            
            # リプレイバッファに追加
            agent.replay_buffer.push(lidar_data, action, reward, next_lidar_data, done)
            
            # パラメータ更新
            if len(agent.replay_buffer) > 256:
                loss_info = agent.update_parameters()
                
                # 学習状況の表示（1000ステップごと）
                if step % 1000 == 0 and loss_info:
                    print(f"Episode {episode}, Step {step}")
                    print(f"  Policy Loss: {loss_info.get('policy_loss', 0):.4f}")
                    print(f"  Q1 Loss: {loss_info.get('qf1_loss', 0):.4f}")
                    print(f"  Q2 Loss: {loss_info.get('qf2_loss', 0):.4f}")
                    print(f"  Alpha: {loss_info.get('alpha', 0):.4f}")
                    if 'alpha_loss' in loss_info:
                        print(f"  Alpha Loss: {loss_info.get('alpha_loss', 0):.4f}")
            
            total_reward += reward
            lidar_data = next_lidar_data
            
            if done:
                break
        
        # 進捗表示とモデル保存
        if episode % 100 == 0:
            print(f"エピソード {episode}, 累積報酬: {total_reward:.2f}")
            agent.save_model(f"sac_model_episode_{episode}.pth")


if __name__ == "__main__":
    main()