import torch
from collections import defaultdict
from trl import ORPOTrainer
from contextlib import nullcontext
import torch.nn as nn
import torch.nn.functional as F

class Custom3ORPOTrainer(ORPOTrainer):
    # def compute_loss(self, logits, labels, rewards):
    #     """
    #     Override the loss computation.
        
    #     Args:
    #         logits: Logits from the model.
    #         labels: Ground truth labels.
    #         rewards: Rewards obtained from the environment.

    #     Returns:
    #         A modified loss tensor.
    #     """
    #     # Implement your custom loss function here
    #     # This is just a placeholder example
    #     custom_loss = super().compute_loss(logits, labels, rewards)  # Call the original loss computation
    #     modified_loss = custom_loss + your_modification_here  # Modify the loss as needed
    #     return modified_loss
    def odds_ratio_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        chosen_rejected_difference,
        length_penalty_coefficient = 0.1
    ): # -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute ORPO's odds ratio (OR) loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the ORPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
            The log odds ratio of the chosen responses over the rejected responses ratio for logging purposes.
            The `log(sigmoid(log_odds_chosen))` for logging purposes.
        """

        # Derived from Eqs. (4) and (7) from https://arxiv.org/abs/2403.07691 by using log identities and exp(log(P(y|x)) = P(y|x)
        log_odds = (policy_chosen_logps - policy_rejected_logps) - (
            torch.log1p(-torch.exp(policy_chosen_logps)) - torch.log1p(-torch.exp(policy_rejected_logps))
        )

        # CoLoR Optimization ---------------------------------
        length_penalty_loss = length_penalty_coefficient * chosen_rejected_difference * -1
        length_penalty_loss = torch.max(length_penalty_loss, torch.tensor(1.0))
        log_odds = log_odds * length_penalty_loss

        sig_ratio = F.sigmoid(log_odds)
        ratio = torch.log(sig_ratio)
        losses = self.beta * ratio

        chosen_rewards = self.beta * (policy_chosen_logps.to(self.accelerator.device)).detach()
        rejected_rewards = self.beta * (policy_rejected_logps.to(self.accelerator.device)).detach()

        return losses, chosen_rewards, rejected_rewards, torch.mean(ratio).item(), torch.mean(log_odds).item()
    
    def concatenated_forward(
        self, model, # : nn.Module, 
        batch, #: Dict[str, Union[List, torch.LongTensor]]
    ): # -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """

        concatenated_batch = super().concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "decoder_input_ids": super()._shift_right(concatenated_batch["concatenated_labels"]),
            }
            if self.is_encoder_decoder
            else {}
        )

        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        if self.is_encoder_decoder:
            labels = concatenated_batch["concatenated_labels"].clone()
        else:
            labels = concatenated_batch["concatenated_input_ids"].clone()
            attention_mask = concatenated_batch["concatenated_attention_mask"]
            labels = torch.where(attention_mask == 1, labels, self.label_pad_token_id)

        chosen_nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        all_logps = super().get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=True,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_nll_loss)
    
    def count_non_negatives_and_difference(self,chosen_labels, rejected_labels):
        # Count non-negative elements in each tensor
        count_chosen_labels = (chosen_labels != -100).sum(dim=1)
        count_rejected_labels = (rejected_labels != -100).sum(dim=1)
        
        # Calculate the difference between counts
        difference = count_chosen_labels - count_rejected_labels
        return difference


    def get_batch_loss_metrics(
        self,
        model,
        batch, # : Dict[str, Union[List, torch.LongTensor]],
        train_eval, # : Literal["train", "eval"] = "train",
        length_penalty=0.001 # Adding length penalty as an optional parameter MINJU
    ):
        """Compute the ORPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
        ) = self.concatenated_forward(model, batch)

        chosen_rejected_difference = self.count_non_negatives_and_difference(batch['chosen_labels'], batch['rejected_labels'])

        losses, chosen_rewards, rejected_rewards, log_odds_ratio, log_odds_chosen = self.odds_ratio_loss(
            policy_chosen_logps, policy_rejected_logps, chosen_rejected_difference
        )
        
        # full ORPO loss
        loss = policy_nll_loss - losses.mean()
        
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().mean().cpu()
        metrics[f"{prefix}log_odds_ratio"] = log_odds_ratio
        metrics[f"{prefix}log_odds_chosen"] = log_odds_chosen

        return loss, metrics
    
    def compute_loss(
        self,
        model, # : Union[PreTrainedModel, nn.Module],
        inputs, # : Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ): #  -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # force log the metrics
        super().store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss