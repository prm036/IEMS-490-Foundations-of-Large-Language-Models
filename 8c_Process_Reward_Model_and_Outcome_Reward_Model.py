"""
Process and Outcome Reward Models:

PRM (Process Reward Model): Evaluates the step-by-step reasoning of a model.
It extracts logits only at specific placeholder tokens, restricts them to a
defined set of reward token IDs, and computes a cross-entropy loss against
the labels.

ORM (Outcome Reward Model): Evaluates the final result of a generation.
It extracts a scalar reward from the final state (using the EOS token position)
and trains using a pairwise Bradley-Terry loss (log-sigmoid) between chosen
and rejected outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PRM(nn.Module):
    def __init__(self, place_token_id, reward_id_list):
        super().__init__()
        self.place_token_id = place_token_id
        self.reward_id_list = reward_id_list
        self.IGNORE_ID = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_ID)

    def forward(self, inputs, logits, labels):
        # 1. Create mask to find specific placeholder tokens to evaluate
        place_index = (inputs == self.place_token_id)
        
        # 2. Filter logits and labels down to only those placeholder positions
        logits = logits[place_index]
        labels = labels[place_index]
        
        # 3. Select only the columns corresponding to the specific reward token IDs
        logits = logits[:, self.reward_id_list]
        
        # 4. Remap the actual label token IDs to indices (0 to R-1) 
        #    to match the new narrowed logit dimension
        for i, token in enumerate(self.reward_id_list):
            labels = torch.where(
                labels == token,
                torch.tensor(i, device=labels.device),
                labels
            )
            
        # 5. Compute and return cross-entropy loss
        loss = self.loss(logits, labels)
        return loss


class ORM(nn.Module):
    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model
        self.config = config
        # Linear head to map the hidden state down to a single scalar reward
        self.v_head = nn.Linear(config.d_models, 1)

    def forward(self, input_ids, attention_mask, eos_indices):
        # 1. Get the final hidden states from the base transformer model
        output = self.base_model(
            input_ids, 
            attention_mask, 
            output_hidden_states=True
        )
        last_hidden_state = output["hidden_states"][-1]
        
        # 2. Apply the value head to compute per-token rewards
        rewards = self.v_head(last_hidden_state)
        
        # 3. Extract the specific scalar reward at the EOS token position
        rewards = torch.gather(
            rewards, 
            dim=1, 
            index=eos_indices.unsqueeze(-1) # Align dimensions for gather
        ).squeeze(-1).squeeze(-1) 
        
        # 4. Split the batch into chosen and rejected halves
        bs = rewards.shape[0] // 2
        chosen_rewards = rewards[:bs]
        reject_rewards = rewards[bs:]
        
        # 5. Compute the pairwise (Bradley-Terry) preference loss
        pairwise_loss = -nn.functional.logsigmoid(chosen_rewards - reject_rewards)
        
        return pairwise_loss.mean()