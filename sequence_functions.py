# sequence

def gen_sequence_maps(seq_len, n_seq, n=20):
    
    l = []
    for key, val in stage_weights.items():
        l.extend([key] * val)
    
    all_seq = []
    
    for _ in range(n_seq):
        seq_abs = []
        
        c = 1 # caret of the starting state
        i = 0
        while True: # emulate do while loop
        
            r = torch.randint(len(l), (1,)).item()
            d = l[r]
            if 0 < c + d < n:
                c += d
                seq_abs.append(c)
        
                i += 1
            
            if i >= seq_len:
                break
                
        all_seq.append(seq_abs)

    return torch.tensor(all_seq)


def sequence(seq_len, n_seq, S):

    s = S[0] # take any sample
    n = s.orig_v.shape[0] # original number of stages: 20
    
    for s in S:
    
        v = s.vs     # (20, 16, 120)
        p = s.orig_p # (20, 16)
    
        i_seqs = gen_sequence_maps(seq_len, n_seq, n=n)
    
        v_seqs = v[i_seqs]
        p_seqs = p[i_seqs]
    
        dv_seqs = v_seqs[:,1:] - v_seqs[:,:-1]
        dp_seqs = p_seqs[:,1:] - p_seqs[:,:-1]
    
        s.vs = dv_seqs
        s.ps = dp_seqs
    
    
    #print(s.vs.shape)
    #print(s.ps.shape)


seq_len = RANDOM_CRAWLER_SEQUENCE_LEN
n_seq   = NUMBER_OF_RANDOM_SEQUENCES

sequence(seq_len, n_seq, S)
