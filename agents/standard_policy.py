import torch
from torch.nn import Linear, LSTMCell, Module, Embedding
from torch.nn.init import uniform_
import torch.nn.functional as F
import numpy as np

from utils import BetterAnomalyDetection

class CriticNet(Module):
    def __init__(self, hidden_size):
        super(CriticNet, self).__init__()
        self.l1 = Linear(hidden_size, hidden_size//2)
        self.l2 = Linear(hidden_size//2, 1)

    def forward(self, hidden_state):
        x = F.relu(self.l1(hidden_state))
        x = torch.tanh(self.l2(x))
        return x


class ActorNet(Module):
    def __init__(self, hidden_size, num_programs):
        super(ActorNet, self).__init__()
        self.l1 = Linear(hidden_size, hidden_size//2)
        self.l2 = Linear(hidden_size//2, num_programs)

    def forward(self, hidden_state):
        x = F.relu(self.l1(hidden_state))
        x = F.softmax(self.l2(x), dim=-1)
        return x


class ArgumentsNet(Module):
    def __init__(self, hidden_size, num_arguments=8):
        super(ArgumentsNet, self).__init__()
        self.l1 = Linear(hidden_size, hidden_size//2)
        self.l2 = Linear(hidden_size//2, num_arguments)

    def forward(self, hidden_state):
        x = F.relu(self.l1(hidden_state))
        x = F.softmax(self.l2(x), dim=-1)
        return x


class StandardPolicy(Module):
    def __init__(self, encoder, hidden_size, num_programs, num_non_primary_programs, embedding_dim, encoding_dim,
                 indices_non_primary_programs, learning_rate=1e-3, use_args=True, use_gpu=False):

        super().__init__()

        self.device = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"

        self._uniform_init = (-0.1, 0.1)

        self._hidden_size = hidden_size
        self.num_programs = num_programs
        self.num_arguments = 8
        self.num_non_primary_programs = num_non_primary_programs

        self.embedding_dim = embedding_dim
        self.encoding_dim = encoding_dim

        # Initialize networks
        self.Mprog = Embedding(num_non_primary_programs, embedding_dim).to(self.device)
        self.encoder = encoder.to(self.device)

        self.lstm = LSTMCell(self.encoding_dim + self.embedding_dim, self._hidden_size).to(self.device)
        self.critic = CriticNet(self._hidden_size).to(self.device)
        self.actor = ActorNet(self._hidden_size, self.num_programs).to(self.device)

        self.use_args = use_args
        self.arguments = ArgumentsNet(self._hidden_size, num_arguments=self.num_arguments).to(self.device)

        self.init_networks()
        self.init_optimizer(lr=learning_rate)

        self.args_criterion = torch.nn.BCELoss()

        # Small epsilon used to preven torch.log() to produce nan
        self.epsilon = np.finfo(np.float32).eps

        # Compute relative indices of non primary programs (to deal with task indices)
        self.relative_indices = dict((prog_idx, relat_idx) for relat_idx, prog_idx in enumerate(indices_non_primary_programs))

    def init_networks(self):

        for p in self.encoder.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

        for p in self.lstm.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

        for p in self.critic.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

        for p in self.actor.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

        for p in self.arguments.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

    def init_optimizer(self, lr):
        '''Initialize the optimizer.
        Args:
            lr (float): learning rate
        '''
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def _one_hot_encode(self, digits, basis=6):
        """One hot encode a digit with basis. The digit may be None,
        the encoding associated to None is a vector full of zeros.
        Args:
          digits: batch (list) of digits
          basis:  (Default value = 6)
        Returns:
          a numpy array representing the 10-hot-encoding of the digit
        """
        encoding = torch.zeros(len(digits), basis)
        digits_filtered = list(filter(lambda x: x is not None, digits))

        if len(digits_filtered) != 0:
            tmp = [[idx for idx, digit in enumerate(digits) if digit is not None], digits_filtered]
            encoding[tmp] = 1.0
        return encoding

    def predict_on_batch(self, e_t, i_t, h_t, c_t):
        """Run one NPI inference.
        Args:
          e_t: batch of environment observation
          i_t: batch of calling program
          h_t: batch of lstm hidden state
          c_t: batch of lstm cell state
        Returns:
          probabilities over programs, value, new hidden state, new cell state
        """
        batch_size = len(i_t)
        s_t = self.encoder(e_t.view(batch_size, -1))
        relative_prog_indices = [self.relative_indices[idx] for idx in i_t]
        p_t = self.Mprog(torch.LongTensor(relative_prog_indices).to(self.device)).view(batch_size, -1)

        new_h, new_c = self.lstm(torch.cat([s_t, p_t], -1), (h_t, c_t))

        actor_out = self.actor(new_h)
        critic_out = self.critic(new_h)
        if self.use_args:
            args_out = self.arguments(new_h)
        else:
            args_out = 0

        return actor_out, critic_out, args_out, new_h, new_c

    def train_on_batch(self, batch, check_autograd=False):
        """perform optimization step.
        Args:
          batch (tuple): tuple of batches of environment observations, calling programs, lstm's hidden and cell states
        Returns:
          policy loss, value loss, total loss combining policy and value losses
        """
        e_t = torch.FloatTensor(np.stack(batch[0]))
        i_t = batch[1]
        lstm_states = batch[2]
        h_t, c_t = zip(*lstm_states)
        h_t, c_t = torch.squeeze(torch.stack(list(h_t))), torch.squeeze(torch.stack(list(c_t)))

        policy_labels = torch.squeeze(torch.stack(batch[3]))
        value_labels = torch.stack(batch[4]).view(-1, 1)

        # Use a better anomaly detection
        with BetterAnomalyDetection(check_autograd):

            self.optimizer.zero_grad()
            policy_predictions, value_predictions, args_predictions, _, _ = self.predict_on_batch(e_t, i_t, h_t, c_t)

            policy_loss = -torch.mean(
                policy_labels[:, 0:self.num_programs] * torch.log(policy_predictions + self.epsilon), dim=-1
            ).mean()

            if self.use_args:
                args_loss = -torch.mean(
                    policy_labels[:, self.num_programs:] * torch.log(args_predictions + self.epsilon), dim=-1
                ).mean()
            else:
                args_loss = torch.FloatTensor([0])

            value_loss = torch.pow(value_predictions - value_labels, 2).mean()

            if self.use_args:
                total_loss = (policy_loss + args_loss + value_loss)/3
            else:
                total_loss = (policy_loss + value_loss) / 2

            total_loss.backward()
            self.optimizer.step()

        return policy_loss.item(), value_loss.item(), args_loss.item(), total_loss.item()

    def forward_once(self, e_t, i_t, h, c):
        """Run one NPI inference using predict.
        Args:
          e_t: current environment observation
          i_t: current program calling
          h: previous lstm hidden state
          c: previous lstm cell state
        Returns:
          probabilities over programs, value, new hidden state, new cell state, a program index sampled according to
          the probabilities over programs)
        """
        e_t = torch.FloatTensor(e_t)
        e_t, h, c = e_t.view(1, -1), h.view(1, -1), c.view(1, -1)
        with torch.no_grad():
            e_t = e_t.to(self.device)
            actor_out, critic_out, args_out, new_h, new_c = self.predict_on_batch(e_t, [i_t], h, c)
        return actor_out, critic_out, args_out, new_h, new_c

    def init_tensors(self):
        """Creates tensors representing the internal states of the lstm filled with zeros.

        Returns:
            instantiated hidden and cell states
        """
        h = torch.zeros(1, self._hidden_size)
        c = torch.zeros(1, self._hidden_size)
        h, c = h.to(self.device), c.to(self.device)
        return h, c