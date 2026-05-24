import os
import random, re
import pickle
import gzip
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import numpy as np
import json
from models.recsys_model import *
from models.llm4rec import *
from sentence_transformers import SentenceTransformer

class two_layer_mlp(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.fc1 = nn.Linear(dims, 128)
        self.fc2 = nn.Linear(128, dims)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x1 = self.fc2(x)
        return x, x1

class A_llmrec_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        rec_pre_trained_data = args.rec_pre_trained_data
        self.args = args
        self.device = args.device
       
        with gzip.open(f'./data/{args.rec_pre_trained_data}/{args.rec_pre_trained_data}_text_name_dict.json.gz','rb') as ft:
            self.text_name_dict = pickle.load(ft)
        
        self.recsys = RecSys(args.recsys, rec_pre_trained_data, self.device)
        self.item_num = self.recsys.item_num
        self.rec_sys_dim = self.recsys.hidden_units
        self.sbert_dim = 768
        
        self.mlp = two_layer_mlp(self.rec_sys_dim)
        if args.phase == 1:
            self.sbert = SentenceTransformer('nq-distilbert-base-v1')
            self.mlp2 = two_layer_mlp(self.sbert_dim)
        
        self.mse = nn.MSELoss()
        
        self.maxlen = args.maxlen
        self.NDCG = 0
        self.HIT = 0
        self.rec_NDCG = 0
        self.rec_HIT = 0
        self.lan_NDCG=0
        self.lan_HIT=0
        self.num_user = 0
        self.yes = 0
        
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        
        if args.phase == 2 or args.phase == 3:
            self.llm = llm4rec(device=self.device, llm_model=args.llm)
            
            self.log_emb_proj = nn.Sequential(
                nn.Linear(self.rec_sys_dim, self.llm.llm_model.config.hidden_size),
                nn.LayerNorm(self.llm.llm_model.config.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.llm.llm_model.config.hidden_size, self.llm.llm_model.config.hidden_size)
            )
            nn.init.xavier_normal_(self.log_emb_proj[0].weight)
            nn.init.xavier_normal_(self.log_emb_proj[3].weight)

            self.item_emb_proj = nn.Sequential(
                nn.Linear(128, self.llm.llm_model.config.hidden_size),
                nn.LayerNorm(self.llm.llm_model.config.hidden_size),
                nn.GELU(),
                nn.Linear(self.llm.llm_model.config.hidden_size, self.llm.llm_model.config.hidden_size)
            )
            nn.init.xavier_normal_(self.item_emb_proj[0].weight)
            nn.init.xavier_normal_(self.item_emb_proj[3].weight)
            
    def save_model(self, args, epoch1=None, epoch2=None):
        out_dir = f'./A-LLMRec/models/saved_models/'
        create_dir(out_dir)
        out_dir += f'{args.rec_pre_trained_data}_{args.recsys}_{epoch1}_'
        if args.phase == 1:
            torch.save(self.sbert.state_dict(), out_dir + 'sbert.pt')
            torch.save(self.mlp.state_dict(), out_dir + 'mlp.pt')
            torch.save(self.mlp2.state_dict(), out_dir + 'mlp2.pt') 
        
        out_dir += f'{args.llm}_{epoch2}_'
        if args.phase == 2:
            torch.save(self.log_emb_proj.state_dict(), out_dir + 'log_proj.pt')
            torch.save(self.item_emb_proj.state_dict(), out_dir + 'item_proj.pt')
            
    def load_model(self, args, phase1_epoch=None, phase2_epoch=None):
        out_dir = f'./A-LLMRec/models/saved_models/{args.rec_pre_trained_data}_{args.recsys}_{phase1_epoch}_'
        
        mlp = torch.load(out_dir + 'mlp.pt', map_location = args.device)
        self.mlp.load_state_dict(mlp)
        del mlp
        for name, param in self.mlp.named_parameters():
            param.requires_grad = False

        if args.phase == 3:
            out_dir += f'{args.llm}_{phase2_epoch}_'
            
            log_emb_proj_dict = torch.load(out_dir + 'log_proj.pt', map_location = args.device)
            self.log_emb_proj.load_state_dict(log_emb_proj_dict)
            del log_emb_proj_dict
            
            item_emb_proj_dict = torch.load(out_dir + 'item_proj.pt', map_location = args.device)
            self.item_emb_proj.load_state_dict(item_emb_proj_dict)
            del item_emb_proj_dict

    def find_item_text(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'
        if title_flag and description_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}, {self.text_name_dict[d].get(i,d_)}"' for i in item]
        elif title_flag and not description_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}"' for i in item]
        elif not title_flag and description_flag:
            return [f'"{self.text_name_dict[d].get(i,d_)}"' for i in item]
    
    def find_item_text_single(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'
        if title_flag and description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}, {self.text_name_dict[d].get(item,d_)}"'
        elif title_flag and not description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}"'
        elif not title_flag and description_flag:
            return f'"{self.text_name_dict[d].get(item,d_)}"'
        
    def get_item_emb(self, item_ids):
        with torch.no_grad():
            item_embs = self.recsys.model.item_emb(torch.LongTensor(item_ids).to(self.device))
            item_embs, _ = self.mlp(item_embs)
        
        return item_embs
    
    def _extract_last_layer_mean_attention(
        self,
        llm_tokens,
        attention_mask,
        attn_tuple,
        text_input,
        history_texts,
        history_ids_batch,
        candidate_texts,
        candidate_ids_batch,
    ):
        last_attn = attn_tuple[-1]              # (B, H, T, T), last layer
        last_attn_mean = last_attn.mean(dim=1)  # (B, T, T)
        
        # # 마지막 4개 레이어를 stack: (4, B, H, T, T)
        # last4_attn = torch.stack(attn_tuple[-4:], dim=0)
        # # 레이어 평균 후 shape: (B, H, T, T)
        # last4_attn_mean_over_layers = last4_attn.mean(dim=0)
        # # head 평균 후 shape: (B, T, T)
        # last4_attn_mean = last4_attn_mean_over_layers.mean(dim=1)
        # last_attn_mean = last4_attn_mean

        results = []

        token_strs_batch = [
            self.llm.llm_tokenizer.convert_ids_to_tokens(ids.tolist())
            for ids in llm_tokens.input_ids
        ]
        token_strs_batch = [["<USER_EMB>"] + toks for toks in token_strs_batch]

        valid_lens = attention_mask.sum(dim=1).tolist()

        for b in range(len(token_strs_batch)):
            q_idx = int(valid_lens[b]) - 1
            attn_row = last_attn_mean[b, q_idx, :].detach().float().cpu().numpy().tolist()
            valid_len = int(valid_lens[b])

            results.append({
                "prompt": text_input[b],
                "history_text": history_texts[b],
                "history_ids": [int(x) for x in history_ids_batch[b]],
                "candidate_text": candidate_texts[b],
                "candidate_ids": [int(x) for x in candidate_ids_batch[b]],
                "query_index": q_idx,
                "tokens": token_strs_batch[b][:valid_len],
                "attention_scores": attn_row[:valid_len],
            })

        return results

    def _save_attention_debug(self, attn_results, out_path):
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(attn_results, f, ensure_ascii=False, indent=2)
    
    def forward(self, data, optimizer=None, batch_iter=None, mode='phase1', seed=None):
        if mode == 'phase1':
            self.pre_train_phase1(data, optimizer, batch_iter)
        if mode == 'phase2':
            self.pre_train_phase2(data, optimizer, batch_iter)
        if mode =='generate':
            #self.generate(data)
            return self.generate_solo(data, seed)

    def pre_train_phase1(self,data,optimizer, batch_iter):
        epoch, total_epoch, step, total_step = batch_iter
        
        self.sbert.train()
        optimizer.zero_grad()

        u, seq, pos, neg = data
        indices = [self.maxlen*(i+1)-1 for i in range(u.shape[0])]
        
        with torch.no_grad():
            log_emb, pos_emb, neg_emb = self.recsys.model(u, seq, pos, neg, mode='item')
            
        log_emb_ = log_emb[indices]
        pos_emb_ = pos_emb[indices]
        neg_emb_ = neg_emb[indices]
        pos_ = pos.reshape(pos.size)[indices]
        neg_ = neg.reshape(neg.size)[indices]
        
        start_inx = 0
        end_inx = 60
        iterss = 0
        mean_loss = 0
        bpr_loss = 0
        gt_loss = 0
        rc_loss = 0
        text_rc_loss = 0
        original_loss = 0
        while start_inx < len(log_emb_):
            log_emb = log_emb_[start_inx:end_inx]
            pos_emb = pos_emb_[start_inx:end_inx]
            neg_emb = neg_emb_[start_inx:end_inx]
            
            pos__ = pos_[start_inx:end_inx]
            neg__ = neg_[start_inx:end_inx]
            
            start_inx = end_inx
            end_inx += 60
            iterss +=1
            
            pos_text = self.find_item_text(pos__)
            neg_text = self.find_item_text(neg__)
            
            pos_token = self.sbert.tokenize(pos_text)
            pos_text_embedding= self.sbert({'input_ids':pos_token['input_ids'].to(self.device),'attention_mask':pos_token['attention_mask'].to(self.device)})['sentence_embedding']
            neg_token = self.sbert.tokenize(neg_text)
            neg_text_embedding= self.sbert({'input_ids':neg_token['input_ids'].to(self.device),'attention_mask':neg_token['attention_mask'].to(self.device)})['sentence_embedding']
            
            pos_text_matching, pos_proj = self.mlp(pos_emb)
            neg_text_matching, neg_proj = self.mlp(neg_emb)
            
            pos_text_matching_text, pos_text_proj = self.mlp2(pos_text_embedding)
            neg_text_matching_text, neg_text_proj = self.mlp2(neg_text_embedding)
            
            pos_logits, neg_logits = (log_emb*pos_proj).mean(axis=1), (log_emb*neg_proj).mean(axis=1)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=pos_logits.device), torch.zeros(neg_logits.shape, device=pos_logits.device)

            loss = self.bce_criterion(pos_logits, pos_labels)
            loss += self.bce_criterion(neg_logits, neg_labels)
            
            matching_loss = self.mse(pos_text_matching,pos_text_matching_text) + self.mse(neg_text_matching,neg_text_matching_text)
            reconstruction_loss = self.mse(pos_proj,pos_emb) + self.mse(neg_proj,neg_emb)
            text_reconstruction_loss = self.mse(pos_text_proj,pos_text_embedding.data) + self.mse(neg_text_proj,neg_text_embedding.data)
            
            total_loss = loss + matching_loss + 0.5*reconstruction_loss + 0.2*text_reconstruction_loss
            total_loss.backward()
            optimizer.step()
            
            mean_loss += total_loss.item()
            bpr_loss += loss.item()
            gt_loss += matching_loss.item()
            rc_loss += reconstruction_loss.item()
            text_rc_loss += text_reconstruction_loss.item()
            
        print(
            f"\rloss in epoch {epoch}/{total_epoch} iteration {step}/{total_step}: "
            f"{mean_loss/iterss} / BPR loss: {bpr_loss/iterss} / Matching loss: {gt_loss/iterss} "
            f"/ Item reconstruction: {rc_loss/iterss} / Text reconstruction: {text_rc_loss/iterss}",
            end=""
        )

    
    def make_interact_text(self, interact_ids, interact_max_num):
        interact_item_titles_ = self.find_item_text(interact_ids, title_flag=True, description_flag=False)
        interact_text = []
        if interact_max_num == 'all':
            for title in interact_item_titles_:
                interact_text.append(title + '[HistoryEmb]')
        else:
            for title in interact_item_titles_[-interact_max_num:]:
                interact_text.append(title + '[HistoryEmb]')
            interact_ids = interact_ids[-interact_max_num:]
            
        interact_text = ','.join(interact_text)
        return interact_text, interact_ids
    
    def make_candidate_text(self, interact_ids, candidate_num, target_item_id, target_item_title, user_gt=None, seed=None):
        neg_item_id = []
        user_gt_set = set(user_gt) if user_gt is not None else set()
        rng = np.random.default_rng(seed)

        while len(neg_item_id)<50:
            t = rng.integers(0, self.item_num)
            if not (t in interact_ids or t in neg_item_id or t in user_gt_set):
                neg_item_id.append(t)
        neg_item_id = list(rng.permutation(neg_item_id))
        
        if hasattr(target_item_id, '__len__') and len(target_item_id) >= 3:
            candidate_ids = list(target_item_id)
            candidate_text = []
            for title in target_item_title:
                candidate_text.append(title + '[CandidateEmb]')
        else:
            candidate_ids = [target_item_id]
            candidate_text = [target_item_title + '[CandidateEmb]']

        for neg_candidate in neg_item_id[:candidate_num-1]:
            candidate_text.append(self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False) + '[CandidateEmb]')
            candidate_ids.append(neg_candidate)
                
        random_ = rng.permutation(len(candidate_text))
        candidate_text = np.array(candidate_text)[random_]
        candidate_ids = np.array(candidate_ids)[random_]
            
        return ','.join(candidate_text), candidate_ids
    
    def pre_train_phase2(self, data, optimizer, batch_iter):
        epoch, total_epoch, step, total_step = batch_iter
        
        optimizer.zero_grad()
        u, seq, pos, neg = data
        mean_loss = 0
        
        text_input = []
        text_output = []
        interact_embs = []
        candidate_embs = []
        self.llm.eval()
        
        with torch.no_grad():
            log_emb = self.recsys.model(u,seq,pos,neg, mode = 'log_only')
            
        for i in range(len(u)):
            target_item_id = pos[i][-1]
            target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
            
            interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10)
            candidate_num = 20
            candidate_text, candidate_ids = self.make_candidate_text(seq[i][seq[i]>0], candidate_num, target_item_id, target_item_title)
            
            input_text = ''
            input_text += ' is a user representation.'

            if self.args.rec_pre_trained_data in ['Movies_and_TV', 'ml-100k', 'ml-1m']:
                input_text += 'This user has watched '
            elif self.args.rec_pre_trained_data == 'Video_Games':
                input_text += 'This user has played '
            elif self.args.rec_pre_trained_data in ['Luxury_Beauty', 'Toys_and_Games']:
                input_text += 'This user has bought '
            elif self.args.rec_pre_trained_data in ['Books', 'books', 'Amazon_Books']:
                input_text += 'This user has read '
            elif self.args.rec_pre_trained_data in ['yelp', 'Yelp']:
                input_text += 'This user has visited '

            input_text += interact_text

            if self.args.rec_pre_trained_data in ['Movies_and_TV', 'ml-100k', 'ml-1m']:
                input_text += ' in the previous. Recommend one next movie for this user to watch next from the following movie title set, '
            elif self.args.rec_pre_trained_data == 'Video_Games':
                input_text += ' in the previous. Recommend one next game for this user to play next from the following game title set, '
            elif self.args.rec_pre_trained_data in ['Luxury_Beauty', 'Toys_and_Games']:
                input_text += ' in the previous. Recommend one next item for this user to buy next from the following item title set, '
            elif self.args.rec_pre_trained_data in ['Books', 'books', 'Amazon_Books']:
                input_text += ' in the previous. Recommend one next book for this user to read next from the following book title set, '
            elif self.args.rec_pre_trained_data in ['yelp', 'Yelp']:
                input_text += ' in the previous. Recommend one next business for this user to visit next from the following business title set, '

            input_text += candidate_text
            input_text += '. Do not write any words other than the only one title. The recommendation is'

            text_input.append(input_text)
            text_output.append(target_item_title)

            interact_embs.append(self.item_emb_proj(self.get_item_emb(interact_ids)))
            candidate_embs.append(self.item_emb_proj(self.get_item_emb(candidate_ids)))
            
        samples = {'text_input': text_input, 'text_output': text_output, 'interact': interact_embs, 'candidate':candidate_embs}
        log_emb = self.log_emb_proj(log_emb)
        loss_rm = self.llm(log_emb, samples)
        loss_rm.backward()
        optimizer.step()
        mean_loss += loss_rm.item()
        print(f"\rA-LLMRec epoch {epoch}/{total_epoch} step {step}/{total_step}: {mean_loss}", end="")

        
    def generate(self, data):
        u, seq, pos, neg, rank = data
        
        answer = []
        text_input = []
        interact_embs = []
        candidate_embs = []
        candidates = []
        with torch.no_grad():
            log_emb = self.recsys.model(u,seq,pos,neg, mode = 'log_only')
            for i in range(len(u)):
                target_item_id = pos[i]
                if len(target_item_id)>=3:
                    target_item_titles = []
                    candidate_num = 95
                    for id in target_item_id:
                        target_item_titles.append(self.find_item_text_single(id, title_flag=True, description_flag=False))
                    candidate_text, candidate_ids = self.make_candidate_text(seq[i][seq[i]>0], candidate_num, target_item_id, target_item_titles)
                    candidates.append(candidate_ids)
                else:
                    candidate_num = 20
                    target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
                    candidate_text, candidate_ids = self.make_candidate_text(seq[i][seq[i]>0], candidate_num, target_item_id, target_item_title)
                    candidates.append(candidate_ids)
                interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10)
                
                input_text = ''
                input_text += ' is a user representation.'
                if self.args.rec_pre_trained_data in ['Movies_and_TV', 'ml-100k', 'ml-1m']:
                    input_text += 'This user has watched '
                elif self.args.rec_pre_trained_data == 'Video_Games':
                    input_text += 'This user has played '
                elif self.args.rec_pre_trained_data in ['Luxury_Beauty', 'Toys_and_Games']:
                    input_text += 'This user has bought '
                elif self.args.rec_pre_trained_data in ['Books', 'books', 'Amazon_Books']:
                    input_text += 'This user has read '
                elif self.args.rec_pre_trained_data in ['yelp', 'Yelp']:
                    input_text += 'This user has visited '

                input_text += interact_text

                if self.args.rec_pre_trained_data in ['Movies_and_TV', 'ml-100k', 'ml-1m']:
                    input_text += ' in the previous. Recommend one next movie for this user to watch next from the following movie title set, '
                elif self.args.rec_pre_trained_data == 'Video_Games':
                    input_text += ' in the previous. Recommend one next game for this user to play next from the following game title set, '
                elif self.args.rec_pre_trained_data in ['Luxury_Beauty', 'Toys_and_Games']:
                    input_text += ' in the previous. Recommend one next item for this user to buy next from the following item title set, '
                elif self.args.rec_pre_trained_data in ['Books', 'books', 'Amazon_Books']:
                    input_text += ' in the previous. Recommend one next book for this user to read next from the following book title set, '
                elif self.args.rec_pre_trained_data in ['yelp', 'Yelp']:
                    input_text += ' in the previous. Recommend one next business for this user to visit next from the following business title set, '

                input_text += candidate_text
                input_text += '. Do not write any words other than the only one title. The recommendation is '
                
                if len(target_item_id)>=3:
                    answer.append(target_item_titles)
                    text_input.append(input_text)
                else:
                    answer.append(target_item_title)
                
                interact_embs.append(self.item_emb_proj(self.get_item_emb(interact_ids)))
                candidate_embs.append(self.item_emb_proj(self.get_item_emb(candidate_ids)))
        
        log_emb = self.log_emb_proj(log_emb)
        atts_llm = torch.ones(log_emb.size()[:-1], dtype=torch.long).to(self.device)
        atts_llm = atts_llm.unsqueeze(1)
        log_emb = log_emb.unsqueeze(1)
        
        with torch.no_grad():
            self.llm.llm_tokenizer.padding_side = "left"
            llm_tokens = self.llm.llm_tokenizer(
                text_input,
                padding="longest",
                return_tensors="pt"
            ).to(self.device)
            
            with torch.cuda.amp.autocast():
                inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens.input_ids)
        
                llm_tokens, inputs_embeds = self.llm.replace_hist_candi_token(llm_tokens, inputs_embeds, interact_embs, candidate_embs)
                    
                attention_mask = llm_tokens.attention_mask
                inputs_embeds = torch.cat([log_emb, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)
                    
                outputs = self.llm.llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=False,
                    #top_p=0.9,
                    temperature=0.7,
                    num_beams=1,
                    max_new_tokens=64,
                    #max_length=64,
                    min_length=1,
                    pad_token_id=self.llm.llm_tokenizer.eos_token_id,
                    repetition_penalty=1.5,
                    length_penalty=1,
                    num_return_sequences=1,
                )

            outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
            output_text = self.llm.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]
            
        def clean_title(text):
            candidates = re.findall(r'"([^"]+)"', text)
            if candidates:
                return candidates[0].strip()
            return text.strip()


        for i in range(len(text_input)):
            predicted_title = clean_title(output_text[i])

            # 🎯 비교 대상: candidate_title 리스트
            candidate_titles = [
                self.find_item_text_single(cid, title_flag=True, description_flag=False)
                for cid in candidates[i]
            ]
            candidate_titles = [clean_title(t) for t in candidate_titles]

            # ✅ hallucination 여부
            is_hallucinated = predicted_title not in candidate_titles
            
            
            f = open(f'./A-LLMRec/recommendation_output.txt','a')
            f.write(text_input[i])
            f.write('\n\n')
        
            a = [t.strip().strip('"') for t in answer[i]]
            ans = ""
            for st in a:
                ans += st + " "
            
            f.write('Answer: '+ ans)
            f.write('\n\n')
            
            f.write('LLM: '+ predicted_title)
            f.write('\n\n')
            
            f.write(f"Hallucination: {is_hallucinated}\n\n")
            
            f.close()

        return output_text
    
    def generate_solo(self, data, seed):
        self.eval()
        u, seq, pos, neg, user_gt, rank = data
        answer = []
        text_input = []
        interact_embs = []
        candidate_embs = []
        candidates = []

        history_texts = []
        history_ids_batch = []
        candidate_texts = []
        candidate_ids_batch = []
        with torch.no_grad():
            log_emb = self.recsys.model(u,seq,pos,neg, mode = 'log_only')
            for i in range(len(u)):
                target_item_id = pos[i]
                target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
                
                interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10)
                candidate_num = 20
                candidate_text, candidate_ids = self.make_candidate_text(
                    seq[i][seq[i]>0], candidate_num, target_item_id, target_item_title, user_gt, seed
                )

                history_texts.append(interact_text)
                candidate_texts.append(candidate_text)
                history_ids_batch.append([int(x) for x in list(interact_ids)])
                candidate_ids_batch.append([int(x) for x in list(candidate_ids)])
                
                # 여기서 로그 저장
                log = {
                    "user": int(u[i]),   # 현재 배치의 u[i]
                    "k": int(i),         # i를 반복 index로 그대로 써도 됨 (혹은 step/k 값 넘겨받기)
                    "target_item_id": int(target_item_id),
                    "cands": sorted([int(x) for x in candidate_ids.tolist()])
                }
                with open("./A-LLMRec/candidate_debug.jsonl", "a") as f:
                    f.write(json.dumps(log) + "\n")
                
                candidates.append(candidate_ids)
                
                input_text = ''
                input_text += ' is a user representation.'
                if self.args.rec_pre_trained_data in ['Movies_and_TV', 'ml-100k', 'ml-1m']:
                    input_text += 'This user has watched '
                elif self.args.rec_pre_trained_data == 'Video_Games':
                    input_text += 'This user has played '
                elif self.args.rec_pre_trained_data in ['Luxury_Beauty', 'Toys_and_Games']:
                    input_text += 'This user has bought '
                elif self.args.rec_pre_trained_data in ['Books', 'books', 'Amazon_Books']:
                    input_text += 'This user has read '
                elif self.args.rec_pre_trained_data in ['yelp', 'Yelp']:
                    input_text += 'This user has visited '

                input_text += interact_text

                if self.args.rec_pre_trained_data in ['Movies_and_TV', 'ml-100k', 'ml-1m']:
                    input_text += ' in the previous. Recommend one next movie for this user to watch next from the following movie title set, '
                elif self.args.rec_pre_trained_data == 'Video_Games':
                    input_text += ' in the previous. Recommend one next game for this user to play next from the following game title set, '
                elif self.args.rec_pre_trained_data in ['Luxury_Beauty', 'Toys_and_Games']:
                    input_text += ' in the previous. Recommend one next item for this user to buy next from the following item title set, '
                elif self.args.rec_pre_trained_data in ['Books', 'books', 'Amazon_Books']:
                    input_text += ' in the previous. Recommend one next book for this user to read next from the following book title set, '
                elif self.args.rec_pre_trained_data in ['yelp', 'Yelp']:
                    input_text += ' in the previous. Recommend one next business for this user to visit next from the following business title set, '

                input_text += candidate_text
                input_text += '. Do not write any words other than the only one title. The recommendation is '
                
                answer.append(target_item_title)
                text_input.append(input_text)
                
                interact_embs.append(self.item_emb_proj(self.get_item_emb(interact_ids)))
                candidate_embs.append(self.item_emb_proj(self.get_item_emb(candidate_ids)))
        
        log_emb = self.log_emb_proj(log_emb)
        atts_llm = torch.ones(log_emb.size()[:-1], dtype=torch.long).to(self.device)
        atts_llm = atts_llm.unsqueeze(1)
        log_emb = log_emb.unsqueeze(1)
        
        with torch.inference_mode():
            self.llm.llm_tokenizer.padding_side = "left"
            llm_tokens = self.llm.llm_tokenizer(
                text_input,
                padding="longest",
                return_tensors="pt"
            ).to(self.device)

            with torch.cuda.amp.autocast():
                inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens.input_ids)

                llm_tokens, inputs_embeds = self.llm.replace_hist_candi_token(
                    llm_tokens, inputs_embeds, interact_embs, candidate_embs
                )

                attention_mask = llm_tokens.attention_mask
                inputs_embeds = torch.cat([log_emb, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

                # =========================
                # 1) attention 추출용 forward
                # =========================
                attn_out = self.llm.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    return_dict=True,
                    use_cache=False,
                )

                attn_results = self._extract_last_layer_mean_attention(
                    llm_tokens=llm_tokens,
                    attention_mask=attention_mask,
                    attn_tuple=attn_out.attentions,
                    text_input=text_input,
                    history_texts=history_texts,
                    history_ids_batch=history_ids_batch,
                    candidate_texts=candidate_texts,
                    candidate_ids_batch=candidate_ids_batch,
                )

                alpha_tag = f"{self.args.alpha:g}"
                attn_save_path = (
                    f'./data/{self.args.rec_pre_trained_data}/'
                    f'A-LLMRec_format/attention_score/'
                    f'part{self.args.feedback_loop_parts}_alpha{alpha_tag}/'
                    f'step{self.args.feedback_loop_step}/'
                )
                os.makedirs(attn_save_path, exist_ok=True)
                
                self._save_attention_debug(attn_results, attn_save_path+f"attention_debug_user{int(u[0])}.json")

                # =========================
                # 2) 기존 생성
                # =========================
                outputs = self.llm.llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8,
                    num_beams=1,
                    max_new_tokens=32,
                    min_length=1,
                    pad_token_id=self.llm.llm_tokenizer.eos_token_id,
                    repetition_penalty=1.5,
                    length_penalty=1,
                    num_return_sequences=1,
                )

            outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
            output_text = self.llm.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]
            
        def clean_title(text):
            candidates = re.findall(r'"([^"]+)"', text)
            if candidates:
                return candidates[0].strip()
            return text.strip()

        predicted_title = clean_title(output_text[0])
        candidates_include_target = [int(target_item_id)] + [int(x) for x in candidate_ids.tolist()]
        candidates_include_target_titles = [
            self.find_item_text_single(cid, title_flag=True, description_flag=False)
            for cid in candidates_include_target
        ]
        
        return predicted_title, candidates_include_target, candidates_include_target_titles

    
if __name__ == "__main__":
    print("start")

    
