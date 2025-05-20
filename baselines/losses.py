import torch
import torch.nn as nn
import torch.nn.functional as F


def npo(updated_model, ref_model, unlearn_inputs, beta, nu=0.0):
    unlearn_outputs = updated_model(**unlearn_inputs, labels=unlearn_inputs.input_ids)
    current_unlearn_loss = unlearn_outputs.loss

    with torch.no_grad():
        ref_outputs = ref_model.forward_with_injected_noise(input_ids=unlearn_inputs.input_ids, nu=nu)
        ref_unlearn_loss = ref_outputs.loss
    
    neg_log_ratios = current_unlearn_loss - ref_unlearn_loss

    loss = -F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta
    return loss

def sim_npo(updated_model, unlearn_inputs, beta):
    unlearn_outputs = updated_model(**unlearn_inputs, labels=unlearn_inputs.input_ids)
    current_unlearn_loss = unlearn_outputs.loss

    loss = -F.logsigmoid(beta * current_unlearn_loss).mean() * 2 / beta
    return loss


def dpo(updated_model, ref_model, unlearn_inputs, idk_inputs, beta, nu=0.0):
    unlearn_outputs = updated_model(**unlearn_inputs, labels=unlearn_inputs.input_ids)
    idk_outputs = updated_model(**idk_inputs, labels=idk_inputs.input_ids)

    idk_current_loss = -1.0 * idk_outputs.loss 
    unlearn_current_loss = -1.0 * unlearn_outputs.loss

    with torch.no_grad():
        idk_ref_outputs = ref_model.forward_with_injected_noise(input_ids=idk_inputs.input_ids, nu=nu)
        unlearn_ref_outputs = ref_model.forward_with_injected_noise(input_ids=unlearn_inputs.input_ids, nu=nu)
        idk_ref_loss = -1.0 * idk_ref_outputs.loss
        unlearn_ref_loss = -1.0 * unlearn_ref_outputs.loss
    
    pi_log_rarios = idk_current_loss - unlearn_current_loss
    ref_log_ratios = idk_current_loss - unlearn_ref_loss

    loss = - F.logsigmoid(beta * (pi_log_rarios - ref_log_ratios)).mean() * 2 / beta

    return loss

def mse(updated_model, ref_model, retain_inputs, nu):
    retain_outputs = updated_model(**retain_inputs, labels=retain_inputs["input_ids"])
    retain_logits = retain_outputs.logits

    ref_outputs = ref_model.forward_with_injected_noise(input_ids=retain_inputs.input_ids, nu=nu)
    ref_logits = ref_outputs.logits

    probs = F.log_softmax(retain_logits, dim=-1).view(-1, retain_logits.shape[-1]).to(torch.bfloat16)
    ref_probs = F.log_softmax(ref_logits, dim=-1).view(-1, ref_logits.shape[-1]).to(torch.bfloat16)

    retain_loss = F.mse_loss(probs, ref_probs, reduction='mean')
    return retain_loss

def kl(updated_model, ref_model, retain_inputs, nu):
    retain_outputs = updated_model(**retain_inputs, labels=retain_inputs["input_ids"])
    retain_logits = retain_outputs.logits

    ref_outputs = ref_model.forward_with_injected_noise(input_ids=retain_inputs.input_ids, nu=nu)
    ref_logits = ref_outputs.logits

    probs = F.log_softmax(retain_logits, dim=-1).view(-1, retain_logits.shape[-1]).to(torch.bfloat16)
    ref_probs = F.log_softmax(ref_logits, dim=-1).view(-1, ref_logits.shape[-1])

    retain_loss = F.kl_div(probs, ref_probs, reduction='batchmean', log_target=True)
    return retain_loss