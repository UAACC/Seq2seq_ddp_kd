import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Net(nn.Module):
    def __init__(self, seq2seq_model, teacher=None):
        super(Net, self).__init__()
        self.seq2seq_model = seq2seq_model
        #fixed teacher
        self.teacher = teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.pad_token_id = -100
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=-100)#[logits([batch_size * sequence_length, vocab_size]),labels([batch_size * sequence_length])]
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')#[logprob([batch_size * sequence_length, vocab_size]), prob([batch_size * sequence_length, vocab_size])]
    
    def forward(self, input_ids, attention_mask, labels):
        # print(f'labels are {labels} size is {labels.size()}')
        # Forward pass for seq2seq_model
        real_stu_output = self.seq2seq_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        real_stu_logits = real_stu_output.logits
        #copy?
        copy_student = copy.deepcopy(self.seq2seq_model)
        

        llm_loss=real_stu_output.loss
        # print(f"LLM Loss: {llm_loss.item()}")

        # logits=real_stu_output.logits
        # past_key_values=real_stu_output.past_key_values
        # decoder_hidden_states=real_stu_output.hidden_states
        # decoder_attentions=real_stu_output.attentions
        # cross_attentions=real_stu_output.cross_attentions
        # encoder_last_hidden_state=real_stu_output.last_hidden_state
        # encoder_hidden_states=real_stu_output.hidden_states
        # encoder_attentions=real_stu_output.attentions

        # Forward pass for teacher (fixed model)
        with torch.no_grad():  
            teacher_output = self.teacher(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            teacher_logits = teacher_output.logits
        #copy_student output
        with torch.no_grad():  # Ensure no gradients are calculated
            copy_stu_output = copy_student(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            copy_stu_logits = copy_stu_output.logits

        

        # Reshape logits and labels for CrossEntropyLoss
        real_stu_logits = real_stu_logits.view(-1, real_stu_logits.size(-1))  # [batch_size * sequence_length, vocab_size]
        labels = labels.view(-1)  # [batch_size, sequence_length] to [batch_size * sequence_length]
        
        # Calculate Cross-Entropy Loss (real student vs labels)
        # print(f"Labels: {labels}")
        ce_loss = self.loss_ce(real_stu_logits, labels)
        # print(f"Cross-Entropy Loss: {ce_loss.item()}")

        
        # Reshape teacher logits for KL Divergence Loss
        # print(f'teacher logits size {teacher_logits.size()}')
        teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))  # [batch_size * sequence_length, vocab_size]
        # print(f'teacher logits size {teacher_logits.size()}')
        # print(f"Teacher Logits: {teacher_logits}")
        # print(f"Teacher Logits min: {teacher_logits.min()}, max: {teacher_logits.max()}")
        

        #create mask for kl
        mask = (labels != self.pad_token_id).unsqueeze(1).expand_as(real_stu_logits)
        # Apply the mask to logits
        masked_real_stu_logits = real_stu_logits * mask
        masked_teacher_logits = teacher_logits * mask
        
        # Apply log_softmax and softmax after masking
        log_probs = F.log_softmax(masked_real_stu_logits, dim=-1)
        probs = F.softmax(masked_teacher_logits, dim=-1)
        # Check for nans before computing KL loss
        
        # Calculate KL Divergence Loss
        kl_loss = self.kl_loss(log_probs, probs)

        # print(f"KL Divergence Loss: {kl_loss.item()}")
        
        # Combine the model's internal loss (if any) with the calculated losses
        llm_loss = real_stu_output.loss
        # total_loss = llm_loss + ce_loss + kl_loss

        #only ce loss default in model.forward()
        total_loss = llm_loss
        
        return real_stu_output, total_loss
    
    def generate(self, input_ids, **gen_kwargs):
        # generate?
        generated_tokens = self.seq2seq_model.generate(
                    input_ids, **gen_kwargs,
                )
        return generated_tokens
        