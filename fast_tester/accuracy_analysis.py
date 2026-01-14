import pickle
import torch
import numpy as np

save_path = "./attack_records/datasetimagenet_modeldensenet121_targeted_pool1_num1000_queries200_norminf_fast_test.pkl"
with open(save_path, 'rb') as f:
    attack_record = pickle.load(f)

true_flag = attack_record["true_flag"]
try_flag = attack_record["try_flag"]
lin_queries = attack_record["lin_queries"]

#print(true_flag)
#print(try_flag)
#print(lin_queries)

'''
if not isinstance(true_flag, torch.Tensor):
    true_flag = torch.tensor(true_flag)

    
if not isinstance(try_flag, torch.Tensor):
    try_flag = torch.tensor(try_flag)

assert true_flag.shape == try_flag.shape, f"Shape mismatch: {true_flag.shape} vs {try_flag.shape}"
'''

#try_flag = try_flag.int()


tp_count=0
tn_count=0
fn_count=0
fp_count=0
path_count = 0

T_count=0
F_count=0

for i in range(len(true_flag)):
    for j in range(len(true_flag[i])):
        num_val = 1 if try_flag[i][j] else 0 
        
        
        if true_flag[i][j]==1 and num_val == 1:
            tp_count+=1
            path_count += lin_queries[i][j]
        elif true_flag[i][j]==0 and num_val==0:
            tn_count+=1
        elif true_flag[i][j]==1 and num_val==0:
            fn_count+=1
            path_count += lin_queries[i][j]
        elif true_flag[i][j]==0 and num_val==1:
            fp_count+=1
        '''
        if true_flag[i][j]==1 and T_count<5000:
            T_count += 1
            path_count += lin_queries[i][j]
            if num_val == 1:
                tp_count+=1
            else:
                fn_count+=1
        if true_flag[i][j]==0 and F_count<5000:
            F_count+=1
            if num_val ==0:
                tn_count+=1
            else:
                fp_count+=1
        '''

print(f"tp_count: {tp_count}")
print(f"tn_count: {tn_count}")
print(f"fn_count: {fn_count}")
print(f"fp_count: {fp_count}")
print(f"avg_query: {path_count/(tp_count+fn_count)}")
print(f"Accuracy is about: {tp_count/(tp_count+fn_count)}")
