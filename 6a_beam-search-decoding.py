import torch
import torch.nn.functional as F

def beam_search(model, tokenizer, prompt, beam_width=3, max_length=50, temperature=1.0, device='cpu'):
    """
    Implement beam search decoding for a pre-trained language model.
    Args:
        model: The pre-trained language model to use for decoding.
        tokenizer: The tokenizer to use for encoding and decoding tokens.
        prompt: The initial prompt to start decoding from.
        beam_width: The number of beams to maintain during decoding.
        max_length: The maximum length of the generated sequence.
        temperature: The temperature to use for sampling.
        device: The device to use for decoding.
    Returns:
        A list of tuples containing the decoded sequence and its log probability.

    The implementation maintains a set of beam_width candidate sequences
    throughout the generation process. For each beam, the model evaluates
    the logits at the last position. The log_softmax is computed to select
    the top candidates based on the beam_width.
    New candidate sequences are formed by updating both the sequence and the
    cumulative log-probability scores. The top overall beam_width candidates
    are retained. An early stopping condition is checked if all beams generate
    the end-of-sequence token. The final results are returned normalized by
    the length of the sequences.  
    """
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Each beam is (token_sequence, cumulative_log_prob)
    beams = [(input_ids[0].tolist(), 0.0)]
    
    for _ in range(max_length - input_ids.shape[1]):
        new_beams = []
        for seq, score in beams:
            # 1. For each beam, run the model to get logits for the last position
            seq_tensor = torch.tensor([seq]).to(device)
            with torch.no_grad():
                outputs = model(seq_tensor) 
                logits = outputs.logits[:, -1, :] / temperature
            
            # 2. Compute log_softmax and select the top beam_width tokens
            log_probs = F.log_softmax(logits, dim=-1)
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)
            
            # 3. Create candidate beams with updated sequences and cumulative log-prob scores
            for i in range(beam_width):
                token = topk_indices[0, i].item()
                prob = topk_log_probs[0, i].item()
                new_beams.append((seq + [token], score + prob))
        
        # 4. Keep the top beam_width candidates overall
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # 5. Optionally stop early if all beams have hit the EOS token
        if hasattr(tokenizer, 'eos_token_id') and all(beam[0][-1] == tokenizer.eos_token_id for beam in beams):
            break
            
    results = []
    # 6. Return results normalized by sequence length
    for seq, score in beams:
        text = tokenizer.decode(seq)
        normalized_score = score / len(seq)
        results.append((text, normalized_score))
        
    return sorted(results, key=lambda x: x[1], reverse=True)